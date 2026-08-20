from __future__ import annotations

import os
import re
import threading
from typing import Dict, Iterable, List, Optional

import pyspiel

from app_solver import (
    GAME_CONFIGS,
    aggregate_selected_node_ranges,
    build_selected_node_summary,
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

        configured_min_iterations = os.getenv("POKERSPIEL_MIN_ITERATIONS")
        if configured_min_iterations is not None:
            min_iterations = int(configured_min_iterations)

        configured_checkpoint_every = os.getenv("POKERSPIEL_CHECKPOINT_EVERY")
        if configured_checkpoint_every is not None:
            checkpoint_every = int(configured_checkpoint_every)

        configured_stability_threshold = os.getenv("POKERSPIEL_STABILITY_THRESHOLD")
        if configured_stability_threshold is not None:
            stability_threshold = float(configured_stability_threshold)

        configured_stop_patience = os.getenv("POKERSPIEL_STOP_PATIENCE")
        if configured_stop_patience is not None:
            stop_patience = int(configured_stop_patience)

        self.solver_name = solver_name
        self.max_iterations = max_iterations
        self.checkpoint_every = checkpoint_every
        self.stability_threshold = stability_threshold
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
        selected_summary = build_selected_node_summary(self._current_ranges)
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
        self.runtime.state = SolverState.TRAINING
        self.runtime.ready_for_queries = False
        self.runtime.stable = False
        self._last_error = None
        print("[solver-start] entering live solver thread", flush=True)
        try:
            print("[solver-start] loading game", flush=True)
            self._game = pyspiel.load_game("python_pokerkit_wrapper", GAME_CONFIGS["hulh"])
            print("[solver-start] creating solver", flush=True)
            self._solver = make_solver(self._game, self.solver_name)
            print("[solver-start] resolving node specs", flush=True)
            self._selected_specs = resolve_node_specs("hulh-preflop", ())

            previous_ranges = None
            consecutive_stable = 0

            for iteration in range(1, self.max_iterations + 1):
                if self._stop_event.is_set():
                    self.runtime.state = SolverState.PAUSED
                    self.runtime.ready_for_queries = False
                    break

                print(f"[solver-start] entering iteration {iteration}/{self.max_iterations}", flush=True)
                self._solver.run_iteration()
                self.runtime.iteration = iteration
                print(f"[solver-start] completed iteration {iteration}", flush=True)

                if iteration >= self.min_iterations:
                    if self.runtime.state == SolverState.TRAINING:
                        self.runtime.state = SolverState.SCORING
                    if not self._probes and self.checkpoint_every:
                        print("[solver-start] preparing scoring probes at min iteration", flush=True)
                        self._probes = prepare_selected_node_probes(
                            self._game,
                            self._selected_specs,
                            samples_per_node=self.range_samples,
                        )

                if self.checkpoint_every and iteration >= self.min_iterations and iteration % self.checkpoint_every == 0:
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
                        self.runtime.state = SolverState.AVAILABLE
                        self.runtime.ready_for_queries = True
                        self.runtime.latest_stable_snapshot = {
                            "iteration": iteration,
                            "stability": checkpoint_summary,
                        }

            if self._stop_event.is_set():
                self.runtime.state = SolverState.PAUSED
                self.runtime.ready_for_queries = False
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

            if self.runtime.state not in {SolverState.AVAILABLE, SolverState.QUERYABLE, SolverState.STABLE}:
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

    def _normalize_preflop_spot(self, spot: str) -> str:
        alias_map = {
            "first": "first_to_act",
            "first_to_act": "first_to_act",
            "open": "response_to_open",
            "response_to_open": "response_to_open",
            "response_to_open_3bet": "response_to_open_3bet",
            "threebet": "response_to_open_3bet",
            "3bet": "response_to_open_3bet",
            "response_to_open_4bet": "response_to_open_4bet",
            "fourbet": "response_to_open_4bet",
            "4bet": "response_to_open_4bet",
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

    def get_preflop_range(self, spot: str) -> "PreflopRangeResponse":
        from .contracts import PreflopRangeResponse

        resolved_spot = self._normalize_preflop_spot(spot)
        if self._solver is None or self._game is None:
            return PreflopRangeResponse(
                spot=resolved_spot,
                iteration=self.runtime.iteration,
                ready=False,
                message="live solver has not started yet",
            )

        if self.runtime.state not in {SolverState.AVAILABLE, SolverState.QUERYABLE, SolverState.STABLE}:
            return PreflopRangeResponse(
                spot=resolved_spot,
                iteration=self.runtime.iteration,
                ready=False,
                message=(
                    f"solver is in phase '{self.runtime.state.value if self.runtime.state else 'unknown'}'; "
                    "preflop range data is unavailable until min_iterations and stability are both satisfied"
                ),
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
            return PreflopRangeResponse(
                spot=resolved_spot,
                iteration=self.runtime.iteration,
                hands=hands,
                hand_count=len(hands),
                ready=True,
                message="live preflop range from current in-memory policy",
            )

        selected_spec = None
        for spec in getattr(self, "_selected_specs", []) or []:
            if spec.get("name") == resolved_spot or spec.get("display_name") == resolved_spot:
                selected_spec = spec
                break

        probe = self.request_probe(
            ProbeRequest(
                node=resolved_spot,
                history=list((selected_spec or {}).get("history") or []),
                samples=max(int(self.range_samples), 1),
                min_iteration=0,
                include_stability=False,
                include_hands=True,
            )
        )
        if not probe.ready:
            return PreflopRangeResponse(
                spot=resolved_spot,
                iteration=self.runtime.iteration,
                ready=False,
                message=probe.message or f"preflop range for spot '{resolved_spot}' is not available",
            )

        hands = [
            HandPolicy(
                hand=str(hand.hand),
                policy={str(action): float(prob) for action, prob in (getattr(hand, "policy", {}) or {}).items()},
            )
            for hand in (probe.hands or [])
            if getattr(hand, "hand", None)
        ]
        return PreflopRangeResponse(
            spot=resolved_spot,
            iteration=self.runtime.iteration,
            hands=hands,
            hand_count=len(hands),
            ready=True,
            message="live preflop range from current in-memory policy",
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
