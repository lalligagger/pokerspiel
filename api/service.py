from __future__ import annotations

import hashlib
import os
import re
import tempfile
import threading
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pyspiel

from app_solver import (
    GAME_CONFIGS,
    FlatMCCFRTables,
    FlatStateIndex,
    aggregate_selected_node_ranges,
    build_selected_node_summary,
    canonical_action_family,
    decode_state_key_history_code,
    decode_state_key_player,
    encode_state_key,
    exact_hole_board_signature,
    make_solver,
    prepare_selected_node_probes,
    resolve_node_specs,
    runtime_telemetry_snapshot,
    snapshot_probe_states,
    summarize_selected_node_stability,
)

from .contracts import (
    BulkProbeRequest,
    BulkProbeResponse,
    HandPolicy,
    HealthStatus,
    PostflopExactRequest,
    PostflopExactResponse,
    PostflopRangeRequest,
    PostflopRangeResponse,
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
        solver_name: Optional[str] = None,
        max_iterations: int = 1_000_000,
        checkpoint_every: int = 4000,
        stability_threshold: float = 0.01,
        stop_threshold: float = 0.85,
        memory_threshold: float = 0.85,
        stop_patience: int = 3,
        min_iterations: int = 1_000_000,
        probe_min_iteration: int = 0,
        range_samples: Optional[int] = None,
        postflop_samples: Optional[int] = None,
    ):
        configured_solver_name = os.getenv("POKERSPIEL_SOLVER")
        if solver_name is None:
            solver_name = configured_solver_name or "external"
        solver_name = str(solver_name).lower()

        configured_range_samples = os.getenv("POKERSPIEL_RANGE_SAMPLES")
        if range_samples is None:
            range_samples = int(configured_range_samples) if configured_range_samples is not None else 1326

        configured_postflop_samples = os.getenv("POKERSPIEL_POSTFLOP_SAMPLES")
        if postflop_samples is None:
            postflop_samples = int(configured_postflop_samples) if configured_postflop_samples is not None else 32

        configured_max_iterations = os.getenv("POKERSPIEL_MAX_ITERATIONS") or os.getenv("POKERSPIEL_ITERATIONS")
        if configured_max_iterations is not None:
            max_iterations = int(configured_max_iterations)

        configured_min_iterations = os.getenv("POKERSPIEL_MIN_ITERATIONS")
        if configured_min_iterations is not None:
            min_iterations = int(configured_min_iterations)

        configured_checkpoint_every = os.getenv("POKERSPIEL_CHECKPOINT_EVERY")
        if configured_checkpoint_every is not None:
            checkpoint_every = int(configured_checkpoint_every)

        configured_stability_threshold = os.getenv("POKERSPIEL_STABILITY_THRESHOLD")
        if configured_stability_threshold is not None:
            stability_threshold = float(configured_stability_threshold)

        configured_stop_threshold = os.getenv("POKERSPIEL_STOP_THRESHOLD")
        if configured_stop_threshold is not None:
            stop_threshold = float(configured_stop_threshold)

        configured_memory_threshold = os.getenv("POKERSPIEL_MEMORY_THRESHOLD")
        if configured_memory_threshold is not None:
            memory_threshold = float(configured_memory_threshold)

        configured_stop_patience = os.getenv("POKERSPIEL_STOP_PATIENCE")
        if configured_stop_patience is not None:
            stop_patience = int(configured_stop_patience)

        self.solver_name = solver_name
        self.max_iterations = max_iterations
        self.checkpoint_every = checkpoint_every
        self.stability_threshold = stability_threshold
        self.stop_threshold = stop_threshold
        self.memory_threshold = memory_threshold
        self.stop_patience = stop_patience
        self.min_iterations = min_iterations
        self.probe_min_iteration = probe_min_iteration
        self.range_samples = range_samples
        self.postflop_samples = max(int(postflop_samples or 1), 1)

        self.lock = threading.RLock()
        self.runtime = SolverRuntimeState(state=SolverState.TRAINING)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._game = None
        self._solver = None
        self._selected_specs = []
        self._probes = []
        self._current_ranges = {"nodes": []}
        self._preflop_range_cache: Dict[str, Dict[str, Any]] = {}
        self._last_stability: Optional[Dict[str, object]] = None
        self._last_checkpoint_telemetry: Dict[str, Any] = {}
        self._last_selected_node_summary: List[Dict[str, Any]] = []
        self._last_memory_sample_iteration: int = 0
        self.sampling_policy: Dict[str, str] = {
            "preflop": "exact_only",
            "postflop": "diagnostic_only",
        }
        self._last_error: Optional[str] = None
        self._flat_state_index: Optional[FlatStateIndex] = None
        self._flat_tables: Optional[FlatMCCFRTables] = None
        self._flat_state_index_by_player: Dict[int, FlatStateIndex] = {0: None, 1: None}
        self._flat_tables_by_player: Dict[int, FlatMCCFRTables] = {0: None, 1: None}

    def _init_flat_kernel(self) -> None:
        if self._flat_state_index is not None and self._flat_tables is not None and self._flat_state_index_by_player[0] is not None and self._flat_tables_by_player[0] is not None:
            return

        memmap_dir = os.getenv("POKERSPIEL_MEMMAP_DIR") or os.path.join(tempfile.gettempdir(), "pokerspiel_live_solver")
        os.makedirs(memmap_dir, exist_ok=True)

        max_states = max(4096, min(self.max_iterations * 8, 2_000_000))
        for player in (0, 1):
            state_path = os.path.join(memmap_dir, f"state_player_{player}")
            tables_path = os.path.join(memmap_dir, f"solver_player_{player}")
            self._flat_state_index_by_player[player] = FlatStateIndex(state_path, max_states=max_states)
            self._flat_tables_by_player[player] = FlatMCCFRTables(tables_path, max_states=max_states, max_actions=3)

        self._flat_state_index = self._flat_state_index_by_player[0]
        self._flat_tables = self._flat_tables_by_player[0]

    def _state_kernel_for_player(self, player: int) -> Tuple[FlatStateIndex, FlatMCCFRTables]:
        if self._flat_state_index is None or self._flat_tables is None:
            self._init_flat_kernel()
        player = int(player) & 1
        state_index = self._flat_state_index_by_player.get(player)
        tables = self._flat_tables_by_player.get(player)
        if state_index is None or tables is None:
            self._init_flat_kernel()
            state_index = self._flat_state_index_by_player.get(player)
            tables = self._flat_tables_by_player.get(player)
        if state_index is None or tables is None:
            raise RuntimeError(f"flat state kernel missing for player {player}")
        return state_index, tables

    def _state_bucket_for_state(self, state, history=None) -> int:
        if state is None:
            return 0

        signature = exact_hole_board_signature(state)
        token = signature or "unknown|"
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) & 0xFFFFFFFF

    def _flat_action_code(self, action) -> Optional[int]:
        family = canonical_action_family(int(action))
        if family == "fold":
            return 0
        if family == "check_call":
            return 1
        if family == "bet_raise":
            return 2
        return None

    def _record_flat_policy_state(self, state, history, policy) -> Optional[int]:
        if state is None:
            return None

        player = int(state.current_player())
        state_index, tables = self._state_kernel_for_player(player)
        state_key = encode_state_key(
            history or [],
            bucket=self._state_bucket_for_state(state, history),
            player=player,
        )
        state_id = state_index.lookup_or_insert(state_key)

        try:
            legal = set(int(action) for action in state.legal_actions())
            raw_policy = policy.get_state_policy(state, player)
        except Exception:
            return state_id

        for action, probability in raw_policy:
            compact_action = self._flat_action_code(action)
            if compact_action is None or int(action) not in legal:
                continue
            tables.strategy[state_id, compact_action] = float(probability)
            tables.avg_strategy[state_id, compact_action] += float(probability)
            tables.visits[state_id] += 1.0

        return state_id

    def _register_probe_state(self, probe: Dict[str, object]) -> Optional[int]:
        if self._flat_state_index is None or self._flat_tables is None:
            self._init_flat_kernel()

        state = probe.get("state")
        if state is None:
            return None

        history = list(probe.get("history") or [])
        return self._record_flat_policy_state(state, history, self._solver.average_policy())

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_live_solver, daemon=True, name="live-solver")
        self._thread.start()

    def stop(self) -> None:
        """Gracefully stop training without discarding the solver object or cached policy state.

        The loop exits on the next safe iteration boundary, but the live service keeps its
        in-memory ranges and last stability snapshot available for API inspection.
        """
        self._stop_event.set()
        self._last_checkpoint_telemetry = runtime_telemetry_snapshot()
        self.runtime.state = SolverState.PAUSED
        self.runtime.ready_for_queries = bool(
            self._current_ranges.get("nodes")
            or self._preflop_range_cache
            or (self._last_stability is not None)
        )
        self.runtime.stable = bool(self._last_stability and self._last_stability.get("passed"))
        self.runtime.last_stability_check = {
            "iteration": self.runtime.iteration,
            "passed": bool(self._last_stability and self._last_stability.get("passed")),
            "threshold": self.stop_threshold,
            "telemetry": self._last_checkpoint_telemetry,
        }

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

    def _history_code_for_spec(self, history: Iterable[str], player: int = 0) -> int:
        action_code = 0
        for depth, action in enumerate((history or [])[:8]):
            normalized = str(action).strip().lower()
            if normalized in {"fold"}:
                compact = 0
            elif normalized in {"check", "call"}:
                compact = 1
            elif normalized in {"bet", "raise"}:
                compact = 2
            else:
                compact = int(action) % 4
            action_code |= int(compact) << (2 * depth)
        return action_code

    def _flat_action_frequencies_for_history(self, history: Iterable[str], player: int = 0) -> Optional[Dict[str, float]]:
        if self._flat_state_index is None or self._flat_tables is None:
            self._init_flat_kernel()

        state_index, tables = self._state_kernel_for_player(player)
        keys = state_index.state_keys[: state_index.state_count]
        target_history_code = self._history_code_for_spec(history, player=player)
        player_code = int(player) & 0x3
        matches = [
            idx
            for idx, key in enumerate(keys)
            if decode_state_key_player(int(key)) == player_code
            and decode_state_key_history_code(int(key)) == target_history_code
        ]
        if not matches:
            return None

        action_frequencies = {
            "fold": 0.0,
            "check_call": 0.0,
            "bet_raise": 0.0,
        }
        sample_count = 0
        for idx in matches:
            action_frequencies["fold"] += float(tables.avg_strategy[idx, 0])
            action_frequencies["check_call"] += float(tables.avg_strategy[idx, 1])
            action_frequencies["bet_raise"] += float(tables.avg_strategy[idx, 2])
            sample_count += int(tables.visits[idx])

        return {
            "action_frequencies": action_frequencies,
            "sample_count": sample_count,
        }

    def _selected_summary_from_flat_kernel(self) -> List[Dict[str, object]]:
        if self._flat_state_index is None or self._flat_tables is None:
            return []

        summary_rows = []
        for spec in self._selected_specs:
            history = list(spec.get("history") or [])
            flat_summary = self._flat_action_frequencies_for_history(history)
            if flat_summary is None:
                continue

            row = {
                "node_name": spec.get("name"),
                "display_name": spec.get("display_name") or spec.get("name"),
                "sample_count": flat_summary["sample_count"],
            }
            summary_rows.append(row)
        return summary_rows

    def _materialize_selected_preflop_reference(self, current_ranges: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Materialize selected preflop spots at checkpoint time.

        The selected spots are always represented in the cache so the read-only API has a stable
        reference for the current checkpoint, even before those spots have accumulated concrete hand
        rows. When a spot is empty, we prefer the aggregate policy from the populated sibling spots
        over a hardcoded uniform seed so the API remains informative rather than silently degenerating
        every empty node to a neutral 1/3 policy.
        """
        source = current_ranges if isinstance(current_ranges, dict) else self._current_ranges
        cache: Dict[str, Dict[str, Any]] = {}
        compact_nodes: List[Dict[str, Any]] = []
        current_by_name: Dict[str, Dict[str, Any]] = {}

        for node in (source or {}).get("nodes", []) or []:
            name = str(node.get("name") or node.get("display_name") or "").strip()
            if not name:
                continue
            current_by_name[name] = node

        aggregate_policy: Dict[str, float] = {
            "fold": 0.0,
            "check_call": 0.0,
            "bet_raise": 0.0,
        }
        filled_rows = 0
        for node in current_by_name.values():
            for hand_entry in (node.get("hands") or []) or []:
                if not hand_entry.get("hand"):
                    continue
                policy = hand_entry.get("policy") or {}
                for key in ("fold", "check_call", "bet_raise"):
                    aggregate_policy[key] += float(policy.get(key, 0.0) or 0.0)
                filled_rows += 1

        if filled_rows > 0:
            for key in aggregate_policy:
                aggregate_policy[key] /= max(filled_rows, 1)

        for spec in getattr(self, "_selected_specs", []) or []:
            name = str(spec.get("name") or spec.get("display_name") or "").strip()
            if not name:
                continue

            node = current_by_name.get(name) or {}
            hands: List[Dict[str, Any]] = []
            for hand_entry in node.get("hands", []) or []:
                hand = hand_entry.get("hand")
                if not hand:
                    continue
                policy = hand_entry.get("policy") or {}
                hands.append(
                    {
                        "hand": str(hand),
                        "policy": {str(action): float(prob) for action, prob in policy.items()},
                    }
                )

            reference_policy = None
            status = "materialized"
            if not hands:
                if filled_rows > 0:
                    reference_policy = {key: float(value) for key, value in aggregate_policy.items()}
                    status = "fallback_seed"
                else:
                    reference_policy = {
                        "fold": 1.0 / 3.0,
                        "check_call": 1.0 / 3.0,
                        "bet_raise": 1.0 / 3.0,
                    }
                    status = "uniform_seed"

            cache[name] = {
                "spot": name,
                "iteration": self.runtime.iteration,
                "status": status,
                "hands": hands,
                "hand_count": len(hands),
                "ready": bool(hands or reference_policy is not None),
                "message": "checkpoint preflop range snapshot",
                "reference_policy": reference_policy,
            }
            compact_nodes.append(
                {
                    "name": name,
                    "display_name": spec.get("display_name") or name,
                    "sample_count": int(node.get("sample_count") or 0),
                }
            )

        for node in (source or {}).get("nodes", []) or []:
            name = str(node.get("name") or node.get("display_name") or "").strip()
            if not name or name in cache:
                continue
            hands = [
                {
                    "hand": str(hand_entry.get("hand")),
                    "policy": {str(action): float(prob) for action, prob in (hand_entry.get("policy") or {}).items()},
                }
                for hand_entry in (node.get("hands") or [])
                if hand_entry.get("hand")
            ]
            cache[name] = {
                "spot": name,
                "iteration": self.runtime.iteration,
                "hands": hands,
                "hand_count": len(hands),
                "ready": bool(hands or (node.get("sample_count") or 0) > 0),
                "message": "checkpoint preflop range snapshot",
                "reference_policy": None,
            }
            compact_nodes.append(
                {
                    "name": name,
                    "display_name": node.get("display_name") or name,
                    "sample_count": node.get("sample_count"),
                }
            )

        self._preflop_range_cache = cache
        self._current_ranges = {"nodes": compact_nodes}
        return cache

    def _refresh_preflop_range_cache(self, current_ranges: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        self._materialize_selected_preflop_reference(current_ranges)
        return self._preflop_range_cache

    def _stop_recommendation(self) -> bool:
        if not isinstance(self._last_stability, dict):
            return False
        avg_delta = float(self._last_stability.get("avg_abs_delta") or 0.0)
        max_delta = float(self._last_stability.get("max_abs_delta") or 0.0)
        threshold = float(self.stop_threshold)
        passed = bool(self._last_stability.get("passed"))
        return passed and avg_delta <= threshold and max_delta <= threshold

    def _refresh_memory_telemetry(self, iteration: Optional[int] = None) -> Dict[str, Any]:
        current_iteration = int(iteration if iteration is not None else self.runtime.iteration)
        cadence = max(100, min(1000, int(self.checkpoint_every or 100)))
        if (
            not self._last_checkpoint_telemetry
            or current_iteration - self._last_memory_sample_iteration >= cadence
            or current_iteration == 0
        ):
            self._last_checkpoint_telemetry = runtime_telemetry_snapshot()
            self._last_memory_sample_iteration = current_iteration
        return self._last_checkpoint_telemetry

    def status(self) -> SolverStatusResponse:
        telemetry = self._refresh_memory_telemetry()
        selected_summary = self._last_selected_node_summary or []
        if not selected_summary:
            current_nodes = (self._current_ranges or {}).get("nodes") or []
            if current_nodes:
                selected_summary = build_selected_node_summary({"nodes": current_nodes})
        return SolverStatusResponse(
            solver=self.solver_name,
            iteration=self.runtime.iteration,
            stable=self.runtime.stable,
            stability=self._stability_summary(),
            ready_for_queries=self.runtime.ready_for_queries,
            last_probe_at=self.runtime.last_probe_at,
            min_iteration=self.min_iterations,
            probe_budget_remaining=self.range_samples,
            selected_node_summary=selected_summary,
            telemetry=telemetry,
            sampling_policy=dict(self.sampling_policy),
            stability_threshold=float(self.stability_threshold),
            stop_threshold=float(self.stop_threshold),
            memory_threshold=float(self.memory_threshold),
            stop_recommended=self._stop_recommendation(),
            memory_stop_recommended=bool(self.memory_threshold is not None and self.memory_threshold > 0),
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
            consecutive_passes=summary.get("consecutive_passes", 0),
            matched_nodes=summary.get("matched_nodes"),
            top_moving=list(summary.get("top_moving") or []),
        )

    def _run_live_solver(self) -> None:
        self.runtime.state = SolverState.TRAINING
        self.runtime.ready_for_queries = False
        self.runtime.stable = False
        self._last_error = None
        try:
            self._game = pyspiel.load_game("python_pokerkit_wrapper", GAME_CONFIGS["hulh"])
            self._solver = make_solver(self._game, self.solver_name)
            self._selected_specs = resolve_node_specs("hulh-preflop", ())

            previous_ranges = None
            consecutive_stable = 0

            for iteration in range(1, self.max_iterations + 1):
                if self._stop_event.is_set():
                    self.runtime.state = SolverState.PAUSED
                    self.runtime.ready_for_queries = bool(
                        self._current_ranges.get("nodes")
                        or self._preflop_range_cache
                        or (self._last_stability is not None)
                    )
                    self.runtime.stable = bool(self._last_stability and self._last_stability.get("passed"))
                    break

                self.runtime.ready_for_queries = bool(
                    self._current_ranges.get("nodes")
                    or self._preflop_range_cache
                    or (self._last_stability is not None)
                )
                self.runtime.stable = bool(self._last_stability and self._last_stability.get("passed"))
                self._refresh_memory_telemetry(iteration)
                self._solver.run_iteration()
                self.runtime.iteration = iteration
                if iteration % max(100, int(self.checkpoint_every or 100)) == 0:
                    self.runtime.last_probe_at = iteration

                if iteration >= self.min_iterations:
                    if self.runtime.state == SolverState.TRAINING:
                        self.runtime.state = SolverState.SCORING
                    if not self._probes and self.checkpoint_every:
                        self._probes = prepare_selected_node_probes(
                            self._game,
                            self._selected_specs,
                            samples_per_node=self.range_samples,
                        )
                        for probe in self._probes:
                            self._register_probe_state(probe)

                if self.checkpoint_every and iteration >= self.min_iterations and iteration % self.checkpoint_every == 0:
                    self._init_flat_kernel()
                    for probe in self._probes:
                        self._register_probe_state(probe)
                    if self._solver is not None:
                        self.runtime.current_average_policy = self._solver.average_policy()
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
                    self._last_selected_node_summary = build_selected_node_summary(current_ranges)
                    self._refresh_preflop_range_cache(current_ranges)
                    self._last_stability = checkpoint_summary
                    self._last_checkpoint_telemetry = runtime_telemetry_snapshot()
                    self._last_memory_sample_iteration = iteration
                    self.runtime.last_probe_at = iteration
                    self.runtime.current_average_policy = policy
                    self.runtime.last_stability_check = {
                        "iteration": iteration,
                        "passed": bool(checkpoint_summary.get("passed")),
                        "threshold": self.stop_threshold,
                        "avg_abs_delta": checkpoint_summary.get("avg_abs_delta"),
                        "max_abs_delta": checkpoint_summary.get("max_abs_delta"),
                        "telemetry": self._last_checkpoint_telemetry,
                    }

                    if checkpoint_summary.get("passed"):
                        consecutive_stable += 1
                    else:
                        consecutive_stable = 0

                    if iteration >= self.min_iterations and consecutive_stable >= self.stop_patience:
                        self.runtime.stable = True
                        self.runtime.state = SolverState.AVAILABLE
                        self.runtime.ready_for_queries = True
                        self.runtime.latest_stable_snapshot = {
                            "iteration": iteration,
                            "stability": checkpoint_summary,
                        }

            if self._stop_event.is_set():
                self.runtime.state = SolverState.PAUSED
                self.runtime.ready_for_queries = bool(
                    self._current_ranges.get("nodes")
                    or self._preflop_range_cache
                    or (self._last_stability is not None)
                )
                self.runtime.stable = bool(self._last_stability and self._last_stability.get("passed"))
            elif self.runtime.iteration >= self.max_iterations:
                self.runtime.state = SolverState.STOPPED
                self.runtime.ready_for_queries = False
            elif self.runtime.state == SolverState.TRAINING:
                self.runtime.state = SolverState.TRAINING
                self.runtime.ready_for_queries = False
            elif self.runtime.state == SolverState.SCORING and not self.runtime.stable:
                self.runtime.state = SolverState.SCORING
                self.runtime.ready_for_queries = False

            if self._solver is not None:
                self.runtime.current_average_policy = self._solver.average_policy()

            if self.runtime.stable and self.runtime.state not in {SolverState.STOPPED, SolverState.PAUSED}:
                self.runtime.state = SolverState.AVAILABLE
                self.runtime.ready_for_queries = True

        except Exception as exc:  # pragma: no cover - runtime path
            import traceback

            print("[solver-start] ERROR in live solver thread", flush=True)
            traceback.print_exc()
            self._last_error = str(exc)
            self.runtime.state = SolverState.ERROR
            self.runtime.ready_for_queries = False

    def request_probe(self, request: ProbeRequest) -> ProbeResponse:
        with self.lock:
            flat_ready = (
                self._flat_state_index is not None
                and self._flat_tables is not None
                and self._flat_state_index.state_count > 0
            )
            if (self._solver is None or self._game is None) and not flat_ready:
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

            if self.runtime.state not in {SolverState.AVAILABLE, SolverState.QUERYABLE, SolverState.STABLE} and not flat_ready:
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
                        f"solver is in phase '{self.runtime.state.value if self.runtime.state else 'unknown'}'; "
                        "preflop probes are unavailable until min_iterations and stability are both satisfied"
                    ),
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

            flat_ready = (
                self._flat_state_index is not None
                and self._flat_tables is not None
                and self._flat_state_index.state_count > 0
            )

            def flat_probe_response_for_history(history: Iterable[str], *, message: str) -> ProbeResponse:
                summary = self._flat_action_frequencies_for_history(history)
                if summary is None:
                    return ProbeResponse(
                        iteration=self.runtime.iteration,
                        node=request.node,
                        display_name=resolved.get("display_name") or request.node,
                        history=list(history),
                        sample_count=0,
                        action_frequencies={},
                        hands=[],
                        stability=self._stability_summary(),
                        ready=False,
                        message=f"no probe records available for node '{request.node}'",
                    )
                return ProbeResponse(
                    iteration=self.runtime.iteration,
                    node=request.node,
                    display_name=resolved.get("display_name") or request.node,
                    history=list(history),
                    sample_count=summary["sample_count"],
                    action_frequencies=summary["action_frequencies"],
                    hands=[],
                    stability=self._stability_summary(),
                    ready=True,
                    message=message,
                )

            if self._game is None or self._solver is None:
                if flat_ready:
                    history = list(resolved.get("history") or [])
                    return flat_probe_response_for_history(
                        history,
                        message="live selected-node snapshot from flat memmap-backed state table",
                    )
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

            if not flat_ready:
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
                    message=(
                        f"no exact selected-node state is available for '{request.node}'; "
                        "realtime sampled probes were removed from the preflop path"
                    ),
                )

            history = list(resolved.get("history") or [])
            return flat_probe_response_for_history(
                history,
                message="live selected-node snapshot from exact memmap-backed state table",
            )

    def request_bulk_probe(self, request: BulkProbeRequest) -> BulkProbeResponse:
        results = [self.request_probe(item) for item in request.requests]
        failed = [result.node for result in results if not result.ready]
        return BulkProbeResponse(results=results, failed=failed)

    def _normalize_preflop_spot(self, spot: str) -> str:
        alias_map = {
            "first": "first_to_act",
            "first_to_act": "first_to_act",
            "limp": "response_to_limp",
            "response_to_limp": "response_to_limp",
            "response_to_limp_raise": "response_to_limp_raise",
            "limp_raise": "response_to_limp_raise",
            "open": "response_to_open",
            "response_to_open": "response_to_open",
            "response_to_open_3bet": "response_to_open_3bet",
            "threebet": "response_to_open_3bet",
            "3bet": "response_to_open_3bet",
            "opener_response_to_3bet": "response_to_open_3bet",
            "response_to_open_4bet": "response_to_open_4bet",
            "fourbet": "response_to_open_4bet",
            "4bet": "response_to_open_4bet",
            "opener_response_to_4bet": "response_to_open_4bet",
            "response_to_open_5bet": "response_to_open_5bet",
            "fivebet": "response_to_open_5bet",
            "5bet": "response_to_open_5bet",
            "opener_response_to_5bet": "response_to_open_5bet",
        }
        key = (spot or "").strip().lower().replace("-", "_").replace(" ", "_")
        return alias_map.get(key, key)

    def _normalize_hand_key(self, hand: str) -> str:
        key = (hand or "").strip()
        if not key:
            return ""
        return key.replace(" ", "")

    def _current_average_policy(self):
        if self._solver is None:
            return None
        average_policy = getattr(self._solver, "average_policy", None)
        if callable(average_policy):
            return average_policy()
        return self._solver

    def _postflop_access_block_reason(self, *, request_min_iteration: Optional[int] = None) -> Optional[str]:
        """Return a blocking reason when post-flop probes are not yet allowed."""
        if self._solver is None or self._game is None:
            return "live solver has not started yet"

        effective_min_iteration = self.min_iterations
        if request_min_iteration is not None:
            effective_min_iteration = max(int(request_min_iteration), effective_min_iteration)

        if self.runtime.state not in {SolverState.AVAILABLE, SolverState.QUERYABLE, SolverState.STABLE}:
            return (
                f"solver is in phase '{self.runtime.state.value if self.runtime.state else 'unknown'}'; "
                "postflop probes are unavailable until min_iterations and stability are both satisfied"
            )

        if effective_min_iteration is not None and self.runtime.iteration < int(effective_min_iteration):
            return (
                f"solver iteration {self.runtime.iteration} is below min_iteration "
                f"{int(effective_min_iteration)}"
            )

        if self._last_stability is not None and not bool(self._last_stability.get("passed")):
            return "postflop probes are blocked until stability criteria pass"

        if not self.runtime.stable:
            return "postflop probes are blocked until the solver reaches minimum iteration and stability"

        return None

    def _canonical_postflop_action_names(self, action_probabilities):
        outputs = {}
        for action, probability in (action_probabilities or []):
            key = int(action)
            if key == 0:
                outputs["fold"] = float(probability)
            elif key == 1:
                outputs["check_call"] = float(probability)
            elif key == 4:
                outputs["bet_raise"] = float(probability)
            else:
                outputs[str(action)] = float(probability)
        return outputs

    def _canonicalize_card_token(self, token: str) -> Optional[str]:
        text = str(token or "").strip().replace(" ", "").replace("|", "")
        if not text:
            return None
        match = re.fullmatch(r"([2-9TJQKA])([cdhs])", text, flags=re.IGNORECASE)
        if match is None:
            return None
        rank = match.group(1).upper()
        suit = match.group(2).lower()
        return f"{rank}{suit}"

    def _canonicalize_hand_subset(self, hand: str) -> List[str]:
        text = str(hand or "").strip()
        if not text:
            return []
        if "[" in text or "(" in text or "," in text:
            values = re.findall(r"[2-9TJQKA][cdhs]", text, flags=re.IGNORECASE)
            return sorted({self._canonicalize_card_token(value) for value in values if self._canonicalize_card_token(value)})
        values = re.findall(r"[2-9TJQKA][cdhs]", text, flags=re.IGNORECASE)
        if len(values) >= 2:
            return sorted({self._canonicalize_card_token(value) for value in values[:2] if self._canonicalize_card_token(value)})
        return [text]

    def _canonical_infoset_id(self, *, board, history, hole_cards, player=None, game_name="hulh") -> str:
        canonical_board = sorted({self._canonicalize_card_token(card) for card in (board or []) if self._canonicalize_card_token(card)})
        canonical_hole = sorted({self._canonicalize_card_token(card) for card in (hole_cards or []) if self._canonicalize_card_token(card)})
        return (
            f"game={game_name}|player={player}|board={canonical_board}|hole={canonical_hole}"
            f"|history={list(history or [])}"
        )

    def _sample_postflop_states(self, *, board, history, hole_cards=None, player=None, samples=32):
        if self._game is None:
            return []

        history = list(history or [])
        hole_cards = list(hole_cards or [])
        states = []
        seen = set()
        attempts = 0
        max_attempts = max(samples * 20, 200)

        while len(states) < max(samples, 1) and attempts < max_attempts:
            attempts += 1
            state = self._game.new_initial_state()
            wrapped = getattr(state, "_wrapped_state", None)
            if wrapped is None:
                continue

            if board:
                board_cards = [str(card) for card in getattr(wrapped, "board_cards", []) or []]
                if sorted(board_cards) != sorted(board):
                    continue

            if hole_cards:
                actual_hole = []
                try:
                    actual_hole_values = getattr(wrapped, "hole_cards", []) or []
                    if actual_hole_values:
                        if player is not None and player < len(actual_hole_values):
                            actual_hole = [str(card) for card in actual_hole_values[player]]
                        else:
                            actual_hole = [str(card) for card in actual_hole_values[0]]
                except Exception:
                    actual_hole = []
                if actual_hole and sorted(actual_hole) != sorted(hole_cards):
                    continue

            resolved = state
            for action in history:
                legal = list(resolved.legal_actions())
                if not legal:
                    resolved = None
                    break
                normalized = str(action).lower()
                if normalized in {"check", "call"}:
                    chosen = 1 if 1 in legal else legal[0]
                elif normalized in {"bet", "raise"}:
                    chosen = 4 if 4 in legal else legal[0]
                elif normalized == "fold":
                    chosen = 0 if 0 in legal else legal[0]
                else:
                    try:
                        chosen = int(action)
                    except Exception:
                        chosen = legal[0]
                if chosen not in legal:
                    chosen = legal[0]
                resolved = resolved.child(chosen)
                if resolved is None:
                    break

            if resolved is None:
                continue

            wrapped_resolved = getattr(resolved, "_wrapped_state", None)
            signature = (
                tuple(sorted(str(card) for card in getattr(wrapped_resolved, "board_cards", []) or [])),
                tuple(sorted(str(card) for card in (getattr(wrapped_resolved, "hole_cards", []) or [])[0])) if getattr(wrapped_resolved, "hole_cards", None) else (),
                tuple(history),
            )
            if signature in seen:
                continue
            seen.add(signature)
            states.append(resolved)

        return states

    def _postflop_action_summary(self, state, policy):
        if state is None or policy is None:
            return None

        wrapped = getattr(state, "_wrapped_state", None)
        player = int(state.current_player()) if hasattr(state, "current_player") else 0
        legal_actions = list(state.legal_actions())
        raw_entries = policy.get_state_policy(state, player)
        entries = [(int(action), float(probability)) for action, probability in raw_entries]
        summary = {"player": player, "legal_actions": legal_actions, "entries": entries}
        if wrapped is not None:
            summary["board_cards"] = [str(card) for card in getattr(wrapped, "board_cards", []) or []]
            hole_cards = getattr(wrapped, "hole_cards", []) or []
            if hole_cards and player < len(hole_cards):
                summary["hole_cards"] = [str(card) for card in hole_cards[player]]
        return summary

    def request_postflop_exact(self, request: PostflopExactRequest) -> PostflopExactResponse:
        block_reason = self._postflop_access_block_reason(request_min_iteration=request.min_iteration)
        if block_reason is not None:
            return PostflopExactResponse(
                iteration=self.runtime.iteration,
                board=list(request.board or []),
                history=list(request.history or []),
                hole_cards=list(request.hole_cards or []),
                player=request.player,
                ready=False,
                message=block_reason,
            )

        state_sample_budget = max(int(request.samples if request.samples is not None else self.postflop_samples), 1)
        states = self._sample_postflop_states(
            board=list(request.board or []),
            history=list(request.history or []),
            hole_cards=list(request.hole_cards or []),
            player=request.player,
            samples=state_sample_budget,
        )
        if not states:
            return PostflopExactResponse(
                iteration=self.runtime.iteration,
                board=list(request.board or []),
                history=list(request.history or []),
                hole_cards=list(request.hole_cards or []),
                player=request.player,
                ready=False,
                message="no matching postflop state could be sampled for the requested exact infoset",
            )

        summary = self._postflop_action_summary(states[0], self._current_average_policy())
        if summary is None:
            return PostflopExactResponse(
                iteration=self.runtime.iteration,
                board=list(request.board or []),
                history=list(request.history or []),
                hole_cards=list(request.hole_cards or []),
                player=request.player,
                ready=False,
                message="no policy entries available for the requested exact postflop infoset",
            )

        probabilities = self._canonical_postflop_action_names(summary["entries"])
        exact_infoset_key = self._canonical_infoset_id(
            board=summary.get("board_cards") or request.board or [],
            history=request.history or [],
            hole_cards=summary.get("hole_cards") or request.hole_cards or [],
            player=summary["player"],
            game_name="hulh",
        )
        return PostflopExactResponse(
            iteration=self.runtime.iteration,
            board=list(summary.get("board_cards") or request.board or []),
            history=list(request.history or []),
            hole_cards=list(summary.get("hole_cards") or request.hole_cards or []),
            player=summary["player"],
            exact_infoset_key=exact_infoset_key,
            action_probabilities=probabilities,
            sample_count=len(states),
            ready=True,
            message="live exact postflop infoset policy from current in-memory solver policy",
        )

    def request_postflop_range(self, request: PostflopRangeRequest) -> PostflopRangeResponse:
        block_reason = self._postflop_access_block_reason(request_min_iteration=request.min_iteration)
        if block_reason is not None:
            return PostflopRangeResponse(
                iteration=self.runtime.iteration,
                board=list(request.board or []),
                history=list(request.history or []),
                hands=list(request.hands or []),
                player=request.player,
                ready=False,
                message=block_reason,
            )

        if not request.hands:
            return PostflopRangeResponse(
                iteration=self.runtime.iteration,
                board=list(request.board or []),
                history=list(request.history or []),
                hands=[],
                player=request.player,
                ready=False,
                message="range estimate requires a non-empty hand subset",
            )

        totals = {"fold": 0.0, "check_call": 0.0, "bet_raise": 0.0}
        processed_hands = 0
        for hand in request.hands:
            parsed = self._canonicalize_hand_subset(hand)
            if not parsed:
                continue
            state_sample_budget = max(int(request.samples if request.samples is not None else self.postflop_samples), 1)
            states = self._sample_postflop_states(
                board=list(request.board or []),
                history=list(request.history or []),
                hole_cards=parsed,
                player=request.player,
                samples=state_sample_budget,
            )
            if not states:
                continue
            summary = self._postflop_action_summary(states[0], self._current_average_policy())
            if summary is None:
                continue
            probabilities = self._canonical_postflop_action_names(summary["entries"])
            for key in totals:
                totals[key] += float(probabilities.get(key, 0.0))
            processed_hands += 1

        if processed_hands == 0:
            return PostflopRangeResponse(
                iteration=self.runtime.iteration,
                board=list(request.board or []),
                history=list(request.history or []),
                hands=list(request.hands or []),
                player=request.player,
                ready=False,
                message="no postflop states were available for any hand in the requested range subset",
            )

        normalized = {key: totals[key] / processed_hands for key in totals}
        return PostflopRangeResponse(
            iteration=self.runtime.iteration,
            board=list(request.board or []),
            history=list(request.history or []),
            hands=list(request.hands or []),
            player=request.player,
            hand_count=processed_hands,
            action_frequencies=normalized,
            sample_count=processed_hands,
            ready=True,
            message="live postflop range estimate over the chosen hand subset",
        )

    def _preflop_range_metadata(self, spot: str, *, history: Optional[List[str]] = None, prior_fold_mass: float = 0.0) -> Dict[str, Any]:
        """Return explicit branch metadata so the grid can distinguish unseen / prior-fold cells from zero action mass.

        The solver records the exact historical branch path, but the UI needs a stable flag telling it
        that a zero-mass cell means "not in range" rather than "fold action probability is 1.0".
        """
        normalized_history = list(history or [])
        normalized_spot = self._normalize_preflop_spot(spot)
        branch_valid = normalized_spot == "first_to_act" or bool(normalized_history)
        return {
            "spot": normalized_spot,
            "history": normalized_history,
            "prior_fold_mass": float(prior_fold_mass),
            "branch_valid": bool(branch_valid),
            "zero_means_not_in_range": True,
            "branch_model": "conditional_after_prior_folds",
        }

    def get_preflop_range(self, spot: str) -> "PreflopRangeResponse":
        from .contracts import PreflopRangeResponse

        start_time = time.perf_counter()
        resolved_spot = self._normalize_preflop_spot(spot)
        spot_history: List[str] = []
        selected_spec = None
        for spec in getattr(self, "_selected_specs", []) or []:
            if spec.get("name") == resolved_spot or spec.get("display_name") == resolved_spot:
                selected_spec = spec
                spot_history = list(spec.get("history") or [])
                break

        metadata = self._preflop_range_metadata(resolved_spot, history=spot_history)

        if self._solver is None or self._game is None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            print(
                f"[preflop-range] spot={resolved_spot} cache=miss elapsed_ms={elapsed_ms:.1f} "
                "status=solver-not-started",
                flush=True,
            )
            return PreflopRangeResponse(
                spot=resolved_spot,
                iteration=self.runtime.iteration,
                ready=False,
                message="live solver has not started yet",
                metadata=metadata,
            )

        cached = self._preflop_range_cache.get(resolved_spot)
        if cached is None and self.runtime.state not in {SolverState.AVAILABLE, SolverState.QUERYABLE, SolverState.STABLE}:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            print(
                f"[preflop-range] spot={resolved_spot} cache=miss elapsed_ms={elapsed_ms:.1f} "
                f"runtime_state={self.runtime.state.value if self.runtime.state else 'unknown'} "
                "status=not-ready",
                flush=True,
            )
            return PreflopRangeResponse(
                spot=resolved_spot,
                iteration=self.runtime.iteration,
                ready=False,
                message=(
                    f"solver is in phase '{self.runtime.state.value if self.runtime.state else 'unknown'}'; "
                    "preflop range data is unavailable until min_iterations and stability are both satisfied"
                ),
                metadata=metadata,
            )

        if cached is not None:
            hands = []
            for hand_entry in cached.get("hands", []) or []:
                hand = hand_entry.get("hand")
                if not hand:
                    continue
                policy = hand_entry.get("policy") or {}
                hands.append(
                    HandPolicy(
                        hand=str(hand),
                        policy={str(action): float(prob) for action, prob in policy.items()},
                    )
                )
            reference_policy = cached.get("reference_policy")
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if not hands and reference_policy is not None:
                status = str(cached.get("status") or "fallback_seed")
                if status == "fallback_seed":
                    message = (
                        f"preflop spot '{resolved_spot}' is materialized from the sibling aggregate checkpoint "
                        "policy until exact hand rows are available"
                    )
                else:
                    message = (
                        f"preflop spot '{resolved_spot}' is materialized as a checkpoint reference "
                        "with the default uniform action policy until exact hand rows are available"
                    )
                print(
                    f"[preflop-range] spot={resolved_spot} cache=hit elapsed_ms={elapsed_ms:.1f} "
                    f"iteration={int(cached.get('iteration', self.runtime.iteration))} "
                    f"hand_count=0 status={status} fallback_policy=True",
                    flush=True,
                )
                return PreflopRangeResponse(
                    spot=resolved_spot,
                    iteration=int(cached.get("iteration", self.runtime.iteration)),
                    hands=[],
                    hand_count=0,
                    ready=True,
                    message=message,
                    metadata={**metadata, "prior_fold_mass": 0.0, "reference_policy": reference_policy},
                )
            print(
                f"[preflop-range] spot={resolved_spot} cache=hit elapsed_ms={elapsed_ms:.1f} "
                f"iteration={int(cached.get('iteration', self.runtime.iteration))} hand_count={len(hands)} "
                f"ready={bool(cached.get('ready', bool(hands)))}",
                flush=True,
            )
            return PreflopRangeResponse(
                spot=resolved_spot,
                iteration=int(cached.get("iteration", self.runtime.iteration)),
                hands=hands,
                hand_count=len(hands),
                ready=bool(cached.get("ready", bool(hands))),
                message=cached.get("message") or "preflop range from latest checkpoint snapshot",
                metadata={**metadata, "reference_policy": reference_policy},
            )

        current_ranges = getattr(self, "_current_ranges", {}) or {}
        node_data = None
        for node in current_ranges.get("nodes", []):
            if node.get("name") == resolved_spot or node.get("display_name") == resolved_spot:
                node_data = node
                break

        if node_data is not None and (node_data.get("hands") or []):
            hands = []
            for hand_entry in node_data.get("hands", []):
                if not hand_entry.get("hand"):
                    continue
                policy = hand_entry.get("policy") or {}
                hands.append(
                    HandPolicy(
                        hand=str(hand_entry.get("hand")),
                        policy={str(action): float(prob) for action, prob in policy.items()},
                    )
                )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            print(
                f"[preflop-range] spot={resolved_spot} cache=miss-live elapsed_ms={elapsed_ms:.1f} "
                f"iteration={self.runtime.iteration} hand_count={len(hands)} status=current_ranges",
                flush=True,
            )
            return PreflopRangeResponse(
                spot=resolved_spot,
                iteration=self.runtime.iteration,
                hands=hands,
                hand_count=len(hands),
                ready=True,
                message="live preflop range from current in-memory policy",
                metadata=metadata,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        print(
            f"[preflop-range] spot={resolved_spot} cache=miss elapsed_ms={elapsed_ms:.1f} "
            f"iteration={self.runtime.iteration} status=unavailable",
            flush=True,
        )
        return PreflopRangeResponse(
            spot=resolved_spot,
            iteration=self.runtime.iteration,
            hands=[],
            hand_count=0,
            ready=False,
            message=(
                f"preflop range for spot '{resolved_spot}' is unavailable because the exact-state lookup "
                "has no materialized data; realtime sampled probes are intentionally disabled on this path"
            ),
            metadata={**metadata, "prior_fold_mass": 0.0},
        )

    def get_preflop_spot(self, spot: str, hand: str) -> "SpotFrequencyResponse":
        from .contracts import SpotFrequencyResponse

        resolved_spot = self._normalize_preflop_spot(spot)
        normalized_hand = self._normalize_hand_key(hand)
        if not normalized_hand:
            return SpotFrequencyResponse(
                spot=resolved_spot,
                hand=normalized_hand,
                iteration=self.runtime.iteration,
                frequencies={},
                ready=False,
                message="missing hand key",
            )

        if self._solver is None or self._game is None:
            return SpotFrequencyResponse(
                spot=resolved_spot,
                hand=normalized_hand,
                iteration=self.runtime.iteration,
                frequencies={},
                ready=False,
                message="live solver has not started yet",
            )

        if self.runtime.state not in {SolverState.AVAILABLE, SolverState.QUERYABLE, SolverState.STABLE}:
            return SpotFrequencyResponse(
                spot=resolved_spot,
                hand=normalized_hand,
                iteration=self.runtime.iteration,
                frequencies={},
                ready=False,
                message=(
                    f"solver is in phase '{self.runtime.state.value if self.runtime.state else 'unknown'}'; "
                    "preflop data is unavailable until min_iterations and stability are both satisfied"
                ),
            )

        node_name = resolved_spot
        node_data = None
        current_ranges = getattr(self, "_current_ranges", {}) or {}
        for node in current_ranges.get("nodes", []):
            if node.get("name") == node_name or node.get("display_name") == node_name:
                node_data = node
                break

        cached_range = self._preflop_range_cache.get(node_name)
        if cached_range is not None:
            hand_policy = next(
                (
                    hand_entry
                    for hand_entry in (cached_range.get("hands") or [])
                    if str(hand_entry.get("hand", "")).strip() == normalized_hand
                ),
                None,
            )
            if hand_policy is not None:
                policy = hand_policy.get("policy") or {}
                frequencies = {
                    "fold": float(policy.get("fold", 0.0)),
                    "check_call": float(policy.get("check_call", policy.get("call", 0.0))),
                    "bet_raise": float(policy.get("bet_raise", policy.get("bet", 0.0))),
                }
                return SpotFrequencyResponse(
                    spot=resolved_spot,
                    hand=normalized_hand,
                    iteration=int(cached_range.get("iteration", self.runtime.iteration)),
                    frequencies=frequencies,
                    ready=True,
                    message="live preflop spot lookup from the latest checkpoint snapshot",
                )

        selected_spec = None
        for spec in getattr(self, "_selected_specs", []) or []:
            if spec.get("name") == node_name or spec.get("display_name") == node_name:
                selected_spec = spec
                break

        hand_policy = None
        if node_data is not None:
            hand_policy = next(
                (
                    hand_entry
                    for hand_entry in (node_data.get("hands") or [])
                    if str(hand_entry.get("hand", "")).strip() == normalized_hand
                ),
                None,
            )

        if hand_policy is None:
            probe = self.request_probe(
                ProbeRequest(
                    node=node_name,
                    history=list((selected_spec or {}).get("history") or []),
                    samples=max(int(self.range_samples), 1),
                    min_iteration=0,
                    include_stability=False,
                    include_hands=True,
                )
            )
            if probe.ready:
                def _hand_key(hand_entry):
                    if isinstance(hand_entry, dict):
                        return str(hand_entry.get("hand", "")).strip()
                    return str(getattr(hand_entry, "hand", "")).strip()

                hand_policy = next(
                    (
                        hand_entry
                        for hand_entry in (probe.hands or [])
                        if _hand_key(hand_entry) == normalized_hand
                    ),
                    None,
                )

        if hand_policy is None:
            return SpotFrequencyResponse(
                spot=resolved_spot,
                hand=normalized_hand,
                iteration=self.runtime.iteration,
                frequencies={},
                ready=False,
                message=(
                    f"spot '{resolved_spot}' is not available in the current preflop runtime state"
                    if node_data is None
                    else f"hand '{normalized_hand}' not found for preflop spot '{resolved_spot}'"
                ),
            )

        if isinstance(hand_policy, dict):
            policy = hand_policy.get("policy") or {}
        else:
            policy = getattr(hand_policy, "policy", {}) or {}

        frequencies = {
            "fold": float(policy.get("fold", 0.0)),
            "check_call": float(policy.get("check_call", policy.get("call", 0.0))),
            "bet_raise": float(policy.get("bet_raise", policy.get("bet", 0.0))),
        }
        return SpotFrequencyResponse(
            spot=resolved_spot,
            hand=normalized_hand,
            iteration=self.runtime.iteration,
            frequencies=frequencies,
            ready=True,
            message="live preflop spot lookup from current in-memory policy",
        )


service = SolverService()
