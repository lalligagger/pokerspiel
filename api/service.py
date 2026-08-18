from __future__ import annotations

import threading
from typing import Dict, Iterable, List, Optional

import pyspiel

from app_solver import (
    GAME_CONFIGS,
    aggregate_selected_node_ranges,
    make_solver,
    prepare_selected_node_probes,
    resolve_node_specs,
    snapshot_probe_states,
    summarize_selected_node_stability,
)

from .contracts import (
    BulkProbeRequest,
    BulkProbeResponse,
    HandPolicy,
    HealthStatus,
    ProbeRequest,
    ProbeResponse,
    SolverStatusResponse,
    StabilitySummary,
)
from .state_machine import SolverRuntimeState, SolverState


class SolverService:
    """Live solver adapter for the read-only API hooks."""

    def __init__(
        self,
        *,
        solver_name: str = "external",
        max_iterations: int = 1_000_000,
        checkpoint_every: int = 4000,
        stability_threshold: float = 0.01,
        stop_patience: int = 3,
        min_iterations: int = 1_000_000,
        probe_min_iteration: int = 0,
        range_samples: int = 1326,
    ):
        self.solver_name = solver_name
        self.max_iterations = max_iterations
        self.checkpoint_every = checkpoint_every
        self.stability_threshold = stability_threshold
        self.stop_patience = stop_patience
        self.min_iterations = min_iterations
        self.probe_min_iteration = probe_min_iteration
        self.range_samples = range_samples

        self.lock = threading.RLock()
        self.runtime = SolverRuntimeState(state=SolverState.RUNNING)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._game = None
        self._solver = None
        self._selected_specs = []
        self._probes = []
        self._current_ranges = {"nodes": []}
        self._last_stability: Optional[Dict[str, object]] = None
        self._last_error: Optional[str] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_live_solver, daemon=True, name="live-solver")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self.runtime.state = SolverState.STOPPED
        self.runtime.ready_for_queries = False

    def health(self) -> HealthStatus:
        status = self.runtime.state.value if self.runtime.state else "running"
        return HealthStatus(
            status=status,
            iteration=self.runtime.iteration,
            stable=self.runtime.stable,
            last_probe_at=self.runtime.last_probe_at,
            ready_for_queries=self.runtime.ready_for_queries,
            message=self._last_error or "solver is running; read-only probe APIs are enabled",
        )

    def status(self) -> SolverStatusResponse:
        return SolverStatusResponse(
            solver=self.solver_name,
            iteration=self.runtime.iteration,
            stable=self.runtime.stable,
            stability=self._stability_summary(),
            ready_for_queries=self.runtime.ready_for_queries,
            last_probe_at=self.runtime.last_probe_at,
            min_iteration=self.min_iterations,
            probe_budget_remaining=self.range_samples,
        )

    def _stability_summary(self) -> Optional[StabilitySummary]:
        summary = self._last_stability
        if not isinstance(summary, dict):
            return None
        return StabilitySummary(
            passed=bool(summary.get("passed")),
            max_abs_delta=summary.get("max_abs_delta"),
            avg_abs_delta=summary.get("avg_abs_delta"),
            threshold=summary.get("threshold"),
            consecutive_passes=0,
            matched_nodes=summary.get("matched_nodes"),
            top_moving=list(summary.get("top_moving") or []),
        )

    def _run_live_solver(self) -> None:
        self.runtime.state = SolverState.RUNNING
        self.runtime.ready_for_queries = True
        self.runtime.stable = False
        self._last_error = None
        try:
            self._game = pyspiel.load_game("python_pokerkit_wrapper", GAME_CONFIGS["hulh"])
            self._solver = make_solver(self._game, self.solver_name)
            self._selected_specs = resolve_node_specs("hulh-preflop", ())
            self._probes = prepare_selected_node_probes(
                self._game,
                self._selected_specs,
                samples_per_node=self.range_samples,
            )

            previous_ranges = None
            consecutive_stable = 0

            for iteration in range(1, self.max_iterations + 1):
                if self._stop_event.is_set():
                    self.runtime.state = SolverState.PAUSED
                    self.runtime.ready_for_queries = True
                    break

                self._solver.run_iteration()
                self.runtime.iteration = iteration

                if iteration % self.checkpoint_every == 0:
                    policy = self._solver.average_policy()
                    checkpoint_records = snapshot_probe_states(policy, self._probes)
                    for record in checkpoint_records:
                        record["iteration"] = iteration

                    current_ranges = aggregate_selected_node_ranges(checkpoint_records)
                    checkpoint_summary = summarize_selected_node_stability(
                        current_ranges,
                        previous_ranges,
                        threshold=self.stability_threshold,
                    )
                    previous_ranges = current_ranges
                    self._current_ranges = current_ranges
                    self._last_stability = checkpoint_summary
                    self.runtime.last_probe_at = iteration
                    self.runtime.current_average_policy = policy

                    if checkpoint_summary.get("passed"):
                        consecutive_stable += 1
                    else:
                        consecutive_stable = 0

                    if iteration >= self.min_iterations and consecutive_stable >= self.stop_patience:
                        self.runtime.stable = True
                        self.runtime.state = SolverState.QUERYABLE
                        self.runtime.ready_for_queries = True
                        self.runtime.latest_stable_snapshot = {
                            "iteration": iteration,
                            "stability": checkpoint_summary,
                        }

            if self._stop_event.is_set():
                self.runtime.state = SolverState.PAUSED
                self.runtime.ready_for_queries = True
            elif self.runtime.iteration >= self.max_iterations:
                self.runtime.state = SolverState.STOPPED
                self.runtime.ready_for_queries = True
            elif self.runtime.state not in {SolverState.QUERYABLE, SolverState.STABLE, SolverState.RUNNING, SolverState.PAUSED}:
                self.runtime.state = SolverState.RUNNING
                self.runtime.ready_for_queries = True

            if self._solver is not None:
                self.runtime.current_average_policy = self._solver.average_policy()

            if self.runtime.stable and self.runtime.state != SolverState.STOPPED:
                self.runtime.state = SolverState.STABLE
                self.runtime.ready_for_queries = True

        except Exception as exc:  # pragma: no cover - runtime path
            self._last_error = str(exc)
            self.runtime.state = SolverState.ERROR
            self.runtime.ready_for_queries = False

    def request_probe(self, request: ProbeRequest) -> ProbeResponse:
        with self.lock:
            if self._solver is None or self._game is None:
                return ProbeResponse(
                    iteration=self.runtime.iteration,
                    node=request.node,
                    display_name=request.node,
                    history=request.history,
                    sample_count=0,
                    action_frequencies={},
                    hands=[],
                    stability=self._stability_summary(),
                    ready=False,
                    message="live solver has not started yet",
                )

            effective_min_iteration = request.min_iteration if request.min_iteration is not None else self.probe_min_iteration
            if request.min_iteration is not None and self.runtime.iteration < effective_min_iteration:
                return ProbeResponse(
                    iteration=self.runtime.iteration,
                    node=request.node,
                    display_name=request.node,
                    history=request.history,
                    sample_count=0,
                    action_frequencies={},
                    hands=[],
                    stability=self._stability_summary(),
                    ready=False,
                    message=(
                        f"solver iteration {self.runtime.iteration} is below min_iteration "
                        f"{effective_min_iteration}"
                    ),
                )

            lookup = {spec["name"]: spec for spec in self._selected_specs}
            resolved = lookup.get(request.node)
            if resolved is None:
                for spec in self._selected_specs:
                    if spec.get("display_name") == request.node:
                        resolved = spec
                        break
            if resolved is None:
                return ProbeResponse(
                    iteration=self.runtime.iteration,
                    node=request.node,
                    display_name=request.node,
                    history=request.history,
                    sample_count=0,
                    action_frequencies={},
                    hands=[],
                    stability=self._stability_summary(),
                    ready=False,
                    message=f"unknown selected node '{request.node}'",
                )

            sample_count = max(int(request.samples or self.range_samples), 1)
            probes = prepare_selected_node_probes(self._game, [resolved], samples_per_node=sample_count)
            records = snapshot_probe_states(self._solver.average_policy(), probes)
            if not records:
                return ProbeResponse(
                    iteration=self.runtime.iteration,
                    node=request.node,
                    display_name=resolved.get("display_name") or request.node,
                    history=list(resolved.get("history") or []),
                    sample_count=0,
                    action_frequencies={},
                    hands=[],
                    stability=self._stability_summary(),
                    ready=False,
                    message=f"no probe records available for node '{request.node}'",
                )

            aggregated = aggregate_selected_node_ranges(records)
            node_data = next(
                (node for node in aggregated.get("nodes", []) if node.get("name") == resolved.get("name")),
                aggregated.get("nodes", [{}])[0],
            )
            action_frequencies = dict(node_data.get("action_frequencies") or {})
            hand_policies = [
                HandPolicy(
                    hand=hand.get("hand"),
                    policy={str(action): float(prob) for action, prob in (hand.get("policy") or {}).items()},
                )
                for hand in node_data.get("hands", [])
                if hand.get("hand")
            ]

            return ProbeResponse(
                iteration=self.runtime.iteration,
                node=request.node,
                display_name=node_data.get("display_name") or resolved.get("display_name") or request.node,
                history=list(request.history or resolved.get("history") or []),
                sample_count=int(node_data.get("sample_count") or len(records)),
                action_frequencies={str(key): float(value) for key, value in action_frequencies.items()},
                hands=hand_policies,
                stability=self._stability_summary(),
                ready=True,
                message="live selected-node snapshot from current in-memory policy",
            )

    def request_bulk_probe(self, request: BulkProbeRequest) -> BulkProbeResponse:
        results = [self.request_probe(item) for item in request.requests]
        failed = [result.node for result in results if not result.ready]
        return BulkProbeResponse(results=results, failed=failed)


service = SolverService()
