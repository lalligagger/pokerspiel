import os
import subprocess
import sys

import numpy as np
import pyspiel
import pytest

from open_spiel.python.games import pokerkit_wrapper  # noqa: F401
from range_export import (
    aggregate_flattened_preflop_ranges,
    aggregate_range_profiles,
    aggregate_selected_node_ranges,
    canonical_preflop_label,
    flatten_preflop_bucket,
)

if "python_pokerkit_wrapper" not in {game.short_name for game in pyspiel.registered_games()}:
    pytest.skip("PokerKit OpenSpiel wrapper not registered in this environment", allow_module_level=True)
from api.router import service as router_service
from api.service import SolverService, service as app_service
from api.state_machine import SolverState
from api.contracts import PostflopExactRequest, PostflopExactResponse, PostflopRangeRequest, PostflopRangeResponse
from app_solver import (
    GAME_CONFIGS,
    FlatMCCFRTables,
    FlatStateIndex,
    decode_state_key_history_code,
    decode_state_key_player,
    encode_state_key,
    exact_hole_board_signature,
    filter_recent_iteration_records,
    format_hulh_history_label,
    infer_state_context,
    is_meaningful_state,
    prepare_selected_node_probes,
    profile_variant,
    replay_history_matches_spot,
    resolve_node_specs,
    runtime_telemetry_snapshot,
    sample_distinct_deal_states,
    summarize_policy_profiles,
    summarize_selected_node_stability,
)


def test_flat_state_index_and_tables_use_integer_ids_and_memmap(tmp_path):
    base = tmp_path / "solver"
    state_index = FlatStateIndex(str(base), max_states=8)
    tables = FlatMCCFRTables(str(base), max_states=8, max_actions=3)

    key_a = encode_state_key(history=(1, 4), bucket=7, player=0)
    key_b = encode_state_key(history=(1, 4), bucket=7, player=0)
    key_c = encode_state_key(history=(1, 4), bucket=7, player=1)
    key_d = encode_state_key(history=(0, 1), bucket=4, player=0)

    assert state_index.lookup_or_insert(key_a) == 0
    assert state_index.lookup_or_insert(key_b) == 0
    assert key_a != key_c
    assert decode_state_key_player(key_a) == 0
    assert decode_state_key_history_code(key_a) == decode_state_key_history_code(key_b)
    assert state_index.lookup_or_insert(key_d) == 1
    assert tables.regret.shape == (8, 3)
    assert tables.regret.dtype == np.float32
    tables.regret[0, 0] = 1.25
    assert tables.regret[0, 0] == pytest.approx(1.25)


def test_service_status_and_probe_compatibility_with_flat_runtime(tmp_path):
    service = SolverService(min_iterations=0, checkpoint_every=1, range_samples=1)
    service.runtime.iteration = 5
    service.runtime.ready_for_queries = True
    service.runtime.stable = True
    service.runtime.state = SolverState.AVAILABLE
    service._selected_specs = [{"name": "first_to_act", "display_name": "first_to_act", "history": []}]
    service._flat_state_index_by_player[0] = FlatStateIndex(str(tmp_path / "flat_status_p0"), max_states=8)
    service._flat_tables_by_player[0] = FlatMCCFRTables(str(tmp_path / "flat_status_p0"), max_states=8, max_actions=3)
    service._flat_state_index = service._flat_state_index_by_player[0]
    service._flat_tables = service._flat_tables_by_player[0]
    service._flat_state_index.lookup_or_insert(encode_state_key([], bucket=7, player=0))
    service._flat_tables.avg_strategy[0, 0] = 0.1
    service._flat_tables.avg_strategy[0, 1] = 0.3
    service._flat_tables.avg_strategy[0, 2] = 0.6
    service._flat_tables.visits[0] = 1.0

    status = service.status()
    assert status.selected_node_summary == []
    assert status.ready_for_queries is True

    probe = service.request_probe(type("Req", (), {"node": "first_to_act", "history": [], "samples": 1, "min_iteration": 0, "include_stability": True, "include_hands": True, "action_filter": None})())
    assert probe.ready is True
    assert probe.action_frequencies["bet_raise"] == pytest.approx(0.6)


def test_runtime_telemetry_snapshot_collects_memory_and_disk_usage(tmp_path):
    snapshot = runtime_telemetry_snapshot(str(tmp_path / "report.json"))

    assert "rss_mb" in snapshot
    assert "disk_bytes" in snapshot
    assert "memmap_bytes" in snapshot
    assert isinstance(snapshot["disk_bytes"], (int, float))
    assert snapshot["rss_available"] in {True, False}


def test_app_solver_accepts_checkpoint_every_alias():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [
            sys.executable,
            "app_solver.py",
            "hulh",
            "--iterations",
            "5",
            "--checkpoint-every",
            "2",
            "--range-samples",
            "1",
            "--solver",
            "outcome",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_preflop_spot_aliases_normalize_to_canonical_labels():
    service = SolverService()

    expected = {
        "first": "first_to_act",
        "first_to_act": "first_to_act",
        "open": "response_to_open",
        "response_to_open": "response_to_open",
        "limp": "response_to_limp",
        "response_to_limp": "response_to_limp",
        "response_to_limp_raise": "response_to_limp_raise",
        "limp_raise": "response_to_limp_raise",
        "response_to_open_3bet": "response_to_open_3bet",
        "3bet": "response_to_open_3bet",
        "threebet": "response_to_open_3bet",
        "opener_response_to_3bet": "response_to_open_3bet",
        "response_to_open_4bet": "response_to_open_4bet",
        "4bet": "response_to_open_4bet",
        "fourbet": "response_to_open_4bet",
        "opener_response_to_4bet": "response_to_open_4bet",
        "response_to_open_5bet": "response_to_open_5bet",
        "5bet": "response_to_open_5bet",
        "fivebet": "response_to_open_5bet",
        "opener_response_to_5bet": "response_to_open_5bet",
    }

    for alias, canonical in expected.items():
        assert service._normalize_preflop_spot(alias) == canonical


def test_first_to_act_reference_fallback_uses_sibling_aggregate_not_uniform_seed():
    service = SolverService()
    service.runtime.state = SolverState.SCORING
    service._solver = object()
    service._game = object()
    service._selected_specs = [{"name": "first_to_act", "display_name": "first_to_act", "history": []}]
    service._preflop_range_cache = {
        "first_to_act": {
            "spot": "first_to_act",
            "iteration": 2000,
            "status": "fallback_seed",
            "hands": [],
            "hand_count": 0,
            "ready": True,
            "message": "checkpoint preflop range snapshot",
            "reference_policy": {"fold": 0.2, "check_call": 0.3, "bet_raise": 0.5},
        }
    }

    response = service.get_preflop_range("first_to_act")

    assert response.ready is True
    assert response.hands == []
    assert response.metadata["reference_policy"] == {"fold": 0.2, "check_call": 0.3, "bet_raise": 0.5}
    assert "uniform action policy" not in response.message.lower()
    assert "sibling aggregate" in response.message.lower()


def test_root_selected_node_never_gets_filtered_out_by_exact_history_match():
    from app_solver import replay_history_matches_spot

    assert replay_history_matches_spot([], "first_to_act") is True
    assert replay_history_matches_spot(["bet"], "first_to_act") is True
    assert replay_history_matches_spot(["call"], "first_to_act") is True
    assert replay_history_matches_spot([], "root") is True


def test_cached_preflop_ranges_are_served_while_solver_is_still_scoring():
    service = SolverService()
    service.runtime.state = SolverState.SCORING
    service._solver = object()
    service._game = object()
    service._preflop_range_cache = {
        "first_to_act": {
            "spot": "first_to_act",
            "iteration": 2000,
            "status": "fallback_seed",
            "hands": [{"hand": "AKs", "policy": {"fold": 0.1, "check_call": 0.3, "bet_raise": 0.6}}],
            "hand_count": 1,
            "ready": True,
            "message": "checkpoint preflop range snapshot",
            "reference_policy": {"fold": 0.2, "check_call": 0.3, "bet_raise": 0.5},
        }
    }

    response = service.get_preflop_range("first_to_act")

    assert response.ready is True
    assert response.hand_count == 1
    assert response.hands[0].hand == "AKs"
    assert response.hands[0].policy["bet_raise"] == pytest.approx(0.6)


def test_solver_service_defaults_to_external_and_respects_env_override(monkeypatch):
    monkeypatch.delenv("POKERSPIEL_SOLVER", raising=False)
    service = SolverService()
    assert service.solver_name == "external"

    monkeypatch.setenv("POKERSPIEL_SOLVER", "outcome")
    service = SolverService()
    assert service.solver_name == "outcome"


def test_solver_service_respects_min_iterations_env_override(monkeypatch):
    monkeypatch.setenv("POKERSPIEL_MIN_ITERATIONS", "1000")
    service = SolverService()
    assert service.min_iterations == 1000


def test_router_uses_shared_service_singleton():
    assert router_service is app_service


def test_infer_state_context_uses_wrapped_pokerkit_state():
    game = pyspiel.load_game(
        "python_pokerkit_wrapper",
        {
            "variant": "FixedLimitTexasHoldem",
            "num_players": 2,
            "blinds": "1 2",
            "stack_sizes": "200 200",
            "antes": "0 0",
            "num_streets": 4,
            "small_bet": 2,
            "big_bet": 4,
        },
    )
    state = game.new_initial_state()

    info = infer_state_context(state)

    assert info["street"] == "preflop"
    assert info["pot_context"] == 3
    assert info["history"] == []
    assert info["legal_actions"]


def test_root_preflop_state_is_filtered_as_non_meaningful():
    game = pyspiel.load_game(
        "python_pokerkit_wrapper",
        {
            "variant": "FixedLimitTexasHoldem",
            "num_players": 2,
            "blinds": "1 2",
            "stack_sizes": "200 200",
            "antes": "0 0",
            "num_streets": 4,
            "small_bet": 2,
            "big_bet": 4,
        },
    )
    state = game.new_initial_state()

    assert is_meaningful_state(state) is False


def test_summarize_policy_profiles_counts_preflop_family_and_deeper_states():
    snapshots = [
        {
            "street": "preflop",
            "pot_context": 3,
            "legal_actions": [0, 1, 4],
            "action_probabilities": [
                {"action": 0, "probability": 0.3333333333},
                {"action": 1, "probability": 0.3333333333},
                {"action": 4, "probability": 0.3333333333},
            ],
        },
        {
            "street": "preflop",
            "pot_context": 3,
            "legal_actions": [0, 1, 4],
            "action_probabilities": [
                {"action": 0, "probability": 0.3333333333},
                {"action": 1, "probability": 0.3333333333},
                {"action": 4, "probability": 0.3333333333},
            ],
        },
        {
            "street": "flop",
            "pot_context": 6,
            "legal_actions": [0, 1, 4],
            "action_probabilities": [
                {"action": 0, "probability": 0.5},
                {"action": 1, "probability": 0.25},
                {"action": 4, "probability": 0.25},
            ],
        },
        {
            "street": "flop",
            "pot_context": 6,
            "legal_actions": [0, 1, 4],
            "action_probabilities": [
                {"action": 0, "probability": 0.5},
                {"action": 1, "probability": 0.25},
                {"action": 4, "probability": 0.25},
            ],
        },
        {
            "street": "turn",
            "pot_context": 8,
            "legal_actions": [0, 1, 4],
            "action_probabilities": [
                {"action": 0, "probability": 0.7},
                {"action": 1, "probability": 0.2},
                {"action": 4, "probability": 0.1},
            ],
        },
    ]

    summary = summarize_policy_profiles(snapshots)

    assert summary["unique_policy_profiles"] == 3
    assert summary["repeated_same_family_preflop_states"] == 1
    assert summary["deeper_non_preflop_states"] == 3


def test_sample_distinct_deal_states_reaches_multiple_hole_card_signatures():
    game = pyspiel.load_game(
        "python_pokerkit_wrapper",
        {
            "variant": "NoLimitShortDeckHoldem",
            "num_players": 2,
            "blinds": "1 2",
            "stack_sizes": "200 200",
            "antes": "0 0",
            "num_streets": 4,
        },
    )

    states = sample_distinct_deal_states(game, target_count=4, max_attempts=200)
    signatures = {exact_hole_board_signature(state) for state in states}

    assert len(states) >= 2
    assert len(signatures) >= 2


def test_prepare_selected_node_probes_keeps_unbiased_sampling_by_default():
    game = pyspiel.load_game(
        "python_pokerkit_wrapper",
        {
            "variant": "FixedLimitTexasHoldem",
            "num_players": 2,
            "blinds": "1 2",
            "stack_sizes": "200 200",
            "antes": "0 0",
            "num_streets": 4,
            "small_bet": 2,
            "big_bet": 4,
        },
    )
    specs = [{"name": "first_to_act", "history": []}]

    probes = prepare_selected_node_probes(game, specs, samples_per_node=10, max_attempts=50)

    assert len(probes) == 10


def test_replay_history_matches_spot_accepts_integer_action_ids_for_deeper_nodes():
    assert replay_history_matches_spot([4, 4], "response_to_open_3bet") is True
    assert replay_history_matches_spot([4, 4, 4], "response_to_open_4bet") is True
    assert replay_history_matches_spot([1], "response_to_limp") is True
    assert format_hulh_history_label([4, 4]) == "response_to_open_3bet"


def test_summarize_selected_node_stability_requires_nonzero_delta_before_passing():
    current_ranges = {
        "nodes": [
            {
                "name": "first_to_act",
                "display_name": "first_to_act",
                "action_frequencies": {"fold": 0.2, "check_call": 0.4, "bet_raise": 0.4},
                "hands": [],
                "sample_count": 1,
            }
        ]
    }
    previous_ranges = {
        "nodes": [
            {
                "name": "first_to_act",
                "display_name": "first_to_act",
                "action_frequencies": {"fold": 0.2, "check_call": 0.4, "bet_raise": 0.4},
                "hands": [],
                "sample_count": 1,
            }
        ]
    }

    summary = summarize_selected_node_stability(current_ranges, previous_ranges, threshold=0.98)

    assert summary["max_abs_delta"] == 0.0
    assert summary["avg_abs_delta"] == 0.0
    assert summary["passed"] is False


def test_solver_service_stays_live_after_min_iterations_when_stability_is_reached(monkeypatch):
    class FakeSolver:
        def __init__(self):
            self.iteration = 0

        def run_iteration(self):
            self.iteration += 1

        def average_policy(self):
            return {"policy": self.iteration}

    def fake_make_solver(game, solver_name):
        return FakeSolver()

    monkeypatch.setattr("api.service.pyspiel.load_game", lambda *args, **kwargs: object())
    monkeypatch.setattr("api.service.make_solver", fake_make_solver)
    monkeypatch.setattr("api.service.resolve_node_specs", lambda *args, **kwargs: [{"name": "first_to_act", "display_name": "first_to_act", "history": []}])
    monkeypatch.setattr("api.service.prepare_selected_node_probes", lambda *args, **kwargs: [{"state": "probe"}])
    monkeypatch.setattr(
        "api.service.snapshot_probe_states",
        lambda policy, probes: [{"name": "first_to_act", "action_frequencies": {"fold": 0.0, "check_call": 1.0, "bet_raise": 0.0}, "hands": []}],
    )
    monkeypatch.setattr(
        "api.service.aggregate_selected_node_ranges",
        lambda records: {"nodes": [{"name": "first_to_act", "display_name": "first_to_act", "action_frequencies": {"fold": 0.0, "check_call": 1.0, "bet_raise": 0.0}, "hands": [], "sample_count": 1}]},
    )
    monkeypatch.setattr(
        "api.service.summarize_selected_node_stability",
        lambda current_ranges, previous_ranges, threshold: {"passed": True, "max_abs_delta": 0.0, "avg_abs_delta": 0.0, "threshold": threshold, "matched_nodes": 1, "top_moving": []},
    )

    service = SolverService(max_iterations=3, checkpoint_every=1, min_iterations=1, stop_patience=1, range_samples=1)
    service._run_live_solver()

    assert service.runtime.iteration == 3
    assert service.runtime.state in {SolverState.STOPPED, SolverState.AVAILABLE}
    assert service.runtime.stable is True
    assert service.runtime.ready_for_queries is False


def test_get_preflop_range_returns_live_policy_entries_for_requested_spot():
    service = SolverService()
    service.runtime.state = SolverState.AVAILABLE
    service.runtime.ready_for_queries = True
    service.runtime.iteration = 42
    service._game = object()
    service._solver = object()
    service._selected_specs = [{"name": "response_to_open", "display_name": "response_to_open", "history": ["bet"]}]
    service._current_ranges = {
        "nodes": [
            {
                "name": "response_to_open",
                "display_name": "response_to_open",
                "history": ["bet"],
                "hands": [
                    {"hand": "TT", "policy": {"fold": 0.15, "check_call": 0.25, "bet_raise": 0.6}},
                    {"hand": "AA", "policy": {"fold": 0.05, "check_call": 0.2, "bet_raise": 0.75}},
                    {"hand": "AKs", "policy": {"fold": 0.1, "check_call": 0.3, "bet_raise": 0.6}},
                ],
            }
        ]
    }

    response = service.get_preflop_range("response_to_open")

    assert response.ready is True
    assert response.spot == "response_to_open"
    assert any(hand.hand == "TT" for hand in response.hands)
    assert any(hand.hand == "AA" for hand in response.hands)
    assert any(hand.hand == "AKs" for hand in response.hands)
    assert response.hand_count == 3


def test_get_preflop_range_uses_checkpoint_cache_when_available():
    service = SolverService()
    service.runtime.state = SolverState.AVAILABLE
    service.runtime.ready_for_queries = True
    service.runtime.iteration = 99
    service._preflop_range_cache = {
        "response_to_open": {
            "spot": "response_to_open",
            "iteration": 99,
            "ready": True,
            "hands": [
                {"hand": "TT", "policy": {"fold": 0.1, "check_call": 0.4, "bet_raise": 0.5}},
                {"hand": "AKs", "policy": {"fold": 0.08, "check_call": 0.3, "bet_raise": 0.62}},
            ],
        }
    }
    service._game = object()
    service._solver = object()
    service._selected_specs = [{"name": "response_to_open", "display_name": "response_to_open", "history": ["bet"]}]
    service._current_ranges = {"nodes": []}

    response = service.get_preflop_range("response_to_open")

    assert response.ready is True
    assert response.iteration == 99
    assert response.spot == "response_to_open"
    assert {hand.hand for hand in response.hands} == {"TT", "AKs"}


def test_get_preflop_range_rejects_sampled_probe_fallbacks():
    service = SolverService()
    service.runtime.state = SolverState.AVAILABLE
    service.runtime.ready_for_queries = True
    service.runtime.iteration = 123
    service._selected_specs = [{"name": "response_to_open", "display_name": "response_to_open", "history": ["bet"]}]
    service._game = object()
    service._solver = object()
    service._current_ranges = {"nodes": []}
    service._preflop_range_cache = {}

    response = service.get_preflop_range("response_to_open")

    assert response.ready is False
    assert "realtime sampled probes are intentionally disabled" in response.message


def test_materialize_selected_preflop_spots_creates_reference_policy_for_empty_checkpoint():
    service = SolverService()
    service.runtime.iteration = 5000
    service._selected_specs = [{"name": "first_to_act", "display_name": "first_to_act", "history": []}]
    service._game = object()
    service._solver = object()
    service._current_ranges = {"nodes": []}

    materialized = service._materialize_selected_preflop_reference()

    assert "first_to_act" in materialized
    assert materialized["first_to_act"]["status"] == "uniform_seed"
    assert materialized["first_to_act"]["ready"] is True
    assert materialized["first_to_act"]["hand_count"] == 0
    policy = materialized["first_to_act"]["reference_policy"]
    assert set(policy) == {"fold", "check_call", "bet_raise"}
    assert abs(sum(policy.values()) - 1.0) < 1e-9


def test_materialize_selected_preflop_spots_uses_populated_sibling_policy_for_first_to_act():
    service = SolverService()
    service.runtime.iteration = 5000
    service._selected_specs = [
        {"name": "first_to_act", "display_name": "first_to_act", "history": []},
        {"name": "response_to_open", "display_name": "response_to_open", "history": ["bet"]},
    ]
    service._game = object()
    service._solver = object()
    service._current_ranges = {
        "nodes": [
            {
                "name": "response_to_open",
                "display_name": "response_to_open",
                "history": ["bet"],
                "hands": [
                    {"hand": "AKs", "policy": {"fold": 0.1, "check_call": 0.3, "bet_raise": 0.6}},
                    {"hand": "QQ", "policy": {"fold": 0.2, "check_call": 0.2, "bet_raise": 0.6}},
                ],
                "sample_count": 2,
            }
        ]
    }

    materialized = service._materialize_selected_preflop_reference()

    assert "first_to_act" in materialized
    assert materialized["first_to_act"]["status"] == "fallback_seed"
    policy = materialized["first_to_act"]["reference_policy"]
    assert policy["fold"] > 0.0
    assert policy["check_call"] > 0.0
    assert policy["bet_raise"] > 0.0
    assert abs(policy["bet_raise"] - 0.6) < 0.2


def test_prepare_selected_node_probes_samples_each_node_independently(monkeypatch):
    node_specs = [
        {"name": "first_to_act", "history": []},
        {"name": "response_to_open", "history": ["bet"]},
        {"name": "response_to_limp", "history": ["call"]},
    ]

    states_by_history = {
        tuple(): object(),
        ("bet",): object(),
        ("call",): object(),
    }

    def fake_state_after_history(game, history):
        return states_by_history.get(tuple(history))

    monkeypatch.setattr("app_solver.state_after_history", fake_state_after_history)

    probes = prepare_selected_node_probes(object(), node_specs, samples_per_node=2)

    counts = {spec["name"]: 0 for spec in node_specs}
    for probe in probes:
        counts[probe["node_name"]] += 1

    assert counts == {
        "first_to_act": 2,
        "response_to_open": 2,
        "response_to_limp": 2,
    }


def test_postflop_exact_is_blocked_until_min_iterations_and_stability(monkeypatch):
    service = SolverService(min_iterations=100, checkpoint_every=10, stop_patience=1)
    service.runtime.iteration = 10
    service._last_stability = {"passed": False, "avg_abs_delta": 0.25, "max_abs_delta": 0.6, "threshold": 0.01, "matched_nodes": 1}
    service._game = object()
    service._solver = object()
    monkeypatch.setattr(service, "_sample_postflop_states", lambda **kwargs: [object()])

    response = service.request_postflop_exact(
        PostflopExactRequest(board=["Ah", "Kd", "2c"], history=["bet", "bet"], hole_cards=["As", "Qs"], samples=1)
    )

    assert response.ready is False
    assert "min_iteration" in response.message.lower() or "stability" in response.message.lower()


def test_checkpoint_every_zero_disables_checkpointing():
    service = SolverService(checkpoint_every=0, max_iterations=3, min_iterations=0, stop_patience=1)
    service._solver = type("FakeSolver", (), {"run_iteration": lambda self: None, "average_policy": lambda self: {}})()
    service._game = object()
    service._selected_specs = [{"name": "first_to_act", "display_name": "first_to_act", "history": []}]
    service._probes = []
    service._last_stability = {"passed": True, "avg_abs_delta": 0.0, "max_abs_delta": 0.0, "threshold": 0.01, "matched_nodes": 1}

    service._run_live_solver()

    assert service.runtime.iteration == 3
    assert service.runtime.state in {SolverState.TRAINING, SolverState.SCORING, SolverState.STOPPED}
    assert service.runtime.ready_for_queries is False


def test_preflop_spot_lookup_returns_single_hand_frequencies():
    service = SolverService()
    service.runtime.state = SolverState.AVAILABLE
    service.runtime.ready_for_queries = True
    service.runtime.iteration = 12345
    service._game = object()
    service._solver = object()
    service._current_ranges = {
        "nodes": [
            {
                "name": "response_to_open",
                "display_name": "response_to_open",
                "hands": [
                    {"hand": "TT", "policy": {"fold": 0.1, "check_call": 0.2, "bet_raise": 0.7}},
                    {"hand": "AKs", "policy": {"fold": 0.05, "check_call": 0.25, "bet_raise": 0.7}},
                ],
            }
        ]
    }

    response = service.get_preflop_spot("open", "TT")

    assert response.spot == "response_to_open"
    assert response.hand == "TT"
    assert response.frequencies["fold"] == 0.1
    assert response.frequencies["check_call"] == 0.2
    assert response.frequencies["bet_raise"] == 0.7
    assert response.ready is True


def test_preflop_spot_lookup_falls_back_to_live_probe_when_cache_is_empty(monkeypatch):
    service = SolverService()
    service.runtime.state = SolverState.AVAILABLE
    service.runtime.ready_for_queries = True
    service.runtime.iteration = 42
    service._game = object()
    service._solver = object()
    service._selected_specs = [{"name": "response_to_open", "display_name": "response_to_open", "history": ["bet"]}]
    service._current_ranges = {"nodes": []}

    monkeypatch.setattr(
        service,
        "request_probe",
        lambda request: type(
            "ProbeResult",
            (),
            {"ready": True, "hands": [{"hand": "TT", "policy": {"fold": 0.15, "check_call": 0.25, "bet_raise": 0.6}}]},
        )(),
    )

    response = service.get_preflop_spot("response_to_open", "TT")

    assert response.ready is True
    assert response.frequencies["fold"] == 0.15
    assert response.frequencies["check_call"] == 0.25
    assert response.frequencies["bet_raise"] == 0.6


def test_postflop_exact_lookup_returns_exact_infoset_policy(monkeypatch):
    class FakePolicy:
        def get_state_policy(self, state, player):
            return [(0, 0.25), (1, 0.25), (4, 0.5)]

    class FakeState:
        def __init__(self):
            self._wrapped_state = type("Wrapped", (), {
                "board_cards": ["Ah", "Kd", "2c"],
                "hole_cards": [["As", "Qs"], ["Kh", "Jd"]],
            })()

        def current_player(self):
            return 0

        def legal_actions(self):
            return [0, 1, 4]

        def history(self):
            return ["bet", "bet"]

    service = SolverService(min_iterations=0)
    service.runtime.state = SolverState.AVAILABLE
    service.runtime.ready_for_queries = True
    service.runtime.iteration = 42
    service.runtime.stable = True
    service._last_stability = {"passed": True, "avg_abs_delta": 0.0, "max_abs_delta": 0.0, "threshold": 0.01, "matched_nodes": 1}
    service._game = object()
    service._solver = FakePolicy()
    monkeypatch.setattr(service, "_sample_postflop_states", lambda **kwargs: [FakeState()])

    response = service.request_postflop_exact(
        PostflopExactRequest(
            board=["Ah", "Kd", "2c"],
            history=["bet", "bet"],
            hole_cards=["As", "Qs"],
            player=0,
            samples=8,
        )
    )

    assert response.ready is True
    assert response.exact_infoset_key
    assert response.action_probabilities["fold"] == 0.25
    assert response.action_probabilities["check_call"] == 0.25
    assert response.action_probabilities["bet_raise"] == 0.5


def test_postflop_range_estimate_aggregates_selected_hand_subset(monkeypatch):
    class FakePolicy:
        def get_state_policy(self, state, player):
            return [(0, 0.3), (1, 0.3), (4, 0.4)]

    class FakeState:
        def __init__(self, hole_cards):
            self._wrapped_state = type("Wrapped", (), {
                "board_cards": ["Ah", "Kd", "2c"],
                "hole_cards": [hole_cards, ["Kh", "Jd"]],
            })()

        def current_player(self):
            return 0

        def legal_actions(self):
            return [0, 1, 4]

        def history(self):
            return ["bet", "bet"]

    service = SolverService(min_iterations=0)
    service.runtime.state = SolverState.AVAILABLE
    service.runtime.ready_for_queries = True
    service.runtime.iteration = 71
    service.runtime.stable = True
    service._last_stability = {"passed": True, "avg_abs_delta": 0.0, "max_abs_delta": 0.0, "threshold": 0.01, "matched_nodes": 1}
    service._game = object()
    service._solver = FakePolicy()
    monkeypatch.setattr(
        service,
        "_sample_postflop_states",
        lambda **kwargs: [FakeState(["As", "Qs"]), FakeState(["Ac", "Kc"])],
    )

    response = service.request_postflop_range(
        PostflopRangeRequest(
            board=["Ah", "Kd", "2c"],
            history=["bet", "bet"],
            hands=["AsQs", "AcKc"],
            player=0,
            samples=8,
        )
    )

    assert response.ready is True
    assert response.hand_count == 2
    assert response.action_frequencies["fold"] == 0.3
    assert response.action_frequencies["check_call"] == 0.3
    assert response.action_frequencies["bet_raise"] == 0.4


def test_flatten_preflop_range_uses_actual_rank_set_for_short_deck():
    standard = [
        {"street": "preflop", "variant": "FixedLimitTexasHoldem", "hole_cards": ["ACE OF CLUBS (Ac)", "KING OF SPADES (Ks)"]},
        {"street": "preflop", "variant": "FixedLimitTexasHoldem", "hole_cards": ["ACE OF HEARTS (Ah)", "ACE OF DIAMONDS (Ad)"]},
        {"street": "preflop", "variant": "FixedLimitTexasHoldem", "hole_cards": ["KING OF CLUBS (Kc)", "QUEEN OF DIAMONDS (Qd)"]},
    ]
    standard_flat = aggregate_flattened_preflop_ranges(standard)
    assert standard_flat["matrix_size"] == 13
    assert len(standard_flat["cells"]) >= 3

    short_deck = [
        {"street": "preflop", "variant": "NoLimitShortDeckHoldem", "hole_cards": ["ACE OF CLUBS (Ac)", "NINE OF SPADES (9s)"]},
        {"street": "preflop", "variant": "NoLimitShortDeckHoldem", "hole_cards": ["KING OF CLUBS (Kc)", "KING OF DIAMONDS (Kd)"]},
        {"street": "preflop", "variant": "NoLimitShortDeckHoldem", "hole_cards": ["SEVEN OF HEARTS (7h)", "SIX OF SPADES (6s)"]},
    ]
    short_flat = aggregate_flattened_preflop_ranges(short_deck)
    assert short_flat["matrix_size"] == 9
    assert short_flat["deck_ranks"] == [6, 7, 8, 9, 10, 11, 12, 13, 14]
    assert flatten_preflop_bucket(["ACE OF CLUBS (Ac)", "NINE OF SPADES (9s)"], short_flat["deck_ranks"]) is not None


def test_canonical_preflop_label_uses_compact_range_tokens():
    assert canonical_preflop_label(["ACE OF CLUBS (Ac)", "KING OF SPADES (Ks)"]) == "AKo"
    assert canonical_preflop_label(["ACE OF SPADES (As)", "KING OF SPADES (Ks)"]) == "AKs"
    assert canonical_preflop_label(["QUEEN OF HEARTS (Qh)", "QUEEN OF CLUBS (Qc)"]) == "QQ"


def test_canonical_preflop_label_handles_raw_pokerkit_hole_card_strings():
    assert canonical_preflop_label("ACE OF CLUBS (Ac)|KING OF SPADES (Ks)") == "AKo"
    assert canonical_preflop_label("As|Ks") == "AKs"
    assert canonical_preflop_label("Qc|Qh") == "QQ"


def test_flatten_preflop_range_accepts_compact_pokerkit_strings():
    short = [
        {"street": "preflop", "variant": "NoLimitShortDeckHoldem", "hole_cards": "Ac|9s"},
        {"street": "preflop", "variant": "NoLimitShortDeckHoldem", "hole_cards": "Kc|Kd"},
        {"street": "preflop", "variant": "NoLimitShortDeckHoldem", "hole_cards": "7h|6s"},
    ]
    flat = aggregate_flattened_preflop_ranges(short)
    assert flat["matrix_size"] == 9
    assert flatten_preflop_bucket("Ac|9s", flat["deck_ranks"]) is not None


def test_aggregate_range_profiles_collapses_raw_pokerkit_action_ids_to_compact_families():
    snapshots = [
        {
            "street": "preflop",
            "pot_context": 3,
            "player": 1,
            "hole_cards": ["ACE OF CLUBS (Ac)", "KING OF SPADES (Ks)"],
            "legal_actions": [0, 1, 2, 4, 8, 16],
            "action_probabilities": [
                {"action": 0, "probability": 0.2},
                {"action": 1, "probability": 0.3},
                {"action": 2, "probability": 0.1},
                {"action": 4, "probability": 0.2},
                {"action": 8, "probability": 0.1},
                {"action": 16, "probability": 0.1},
            ],
        }
    ]

    rows = aggregate_range_profiles(snapshots)
    assert rows[0]["compact_label"] == "AKo"
    assert rows[0]["legal_actions"] == [0, 1, 4]
    assert rows[0]["policy"]["0"] == 0.2
    assert rows[0]["policy"]["1"] == 0.3
    assert rows[0]["policy"]["4"] == 0.5


def test_aggregate_selected_node_ranges_collapses_selected_nodes_across_hands():
    snapshots = [
        {
            "label": "first_to_act",
            "node_name": "first_to_act",
            "normalized_name": "first_to_act",
            "street": "preflop",
            "history": [],
            "selected_history": [],
            "player": 0,
            "hole_cards": ["ACE OF CLUBS (Ac)", "KING OF SPADES (Ks)"],
            "exact_infoset_key": "infoset=exact:player=0|hole=AcKs|board=|hist=",
            "action_probabilities": [
                {"action": 0, "probability": 0.2},
                {"action": 1, "probability": 0.3},
                {"action": 4, "probability": 0.5},
            ],
        },
        {
            "label": "first_to_act",
            "node_name": "first_to_act",
            "normalized_name": "first_to_act",
            "street": "preflop",
            "history": [],
            "selected_history": [],
            "player": 0,
            "hole_cards": ["QUEEN OF HEARTS (Qh)", "JACK OF CLUBS (Jc)"],
            "exact_infoset_key": "infoset=exact:player=0|hole=QhJc|board=|hist=",
            "action_probabilities": [
                {"action": 0, "probability": 0.8},
                {"action": 1, "probability": 0.1},
                {"action": 4, "probability": 0.1},
            ],
        },
    ]

    ranges = aggregate_selected_node_ranges(snapshots)
    assert len(ranges["nodes"]) == 1
    assert ranges["nodes"][0]["name"] == "first_to_act"
    assert ranges["nodes"][0]["action_frequencies"]["fold"] == 0.5
    assert ranges["nodes"][0]["action_frequencies"]["check_call"] == 0.2
    assert ranges["nodes"][0]["action_frequencies"]["bet_raise"] == 0.3
    assert ranges["nodes"][0]["sample_count"] == 2


def test_aggregate_selected_node_ranges_groups_by_exact_infoset_when_available():
    snapshots = [
        {
            "label": "first_to_act",
            "node_name": "first_to_act",
            "normalized_name": "first_to_act",
            "street": "preflop",
            "history": [],
            "selected_history": [],
            "player": 0,
            "hole_cards": ["ACE OF CLUBS (Ac)", "KING OF SPADES (Ks)"],
            "exact_infoset_key": "infoset=exact:player=0|hole=AcKs|board=|hist=",
            "action_probabilities": [
                {"action": 0, "probability": 0.2},
                {"action": 1, "probability": 0.3},
                {"action": 4, "probability": 0.5},
            ],
        },
        {
            "label": "response_to_open",
            "node_name": "response_to_open",
            "normalized_name": "response_to_open",
            "street": "preflop",
            "history": ["bet"],
            "selected_history": ["bet"],
            "player": 1,
            "hole_cards": ["ACE OF CLUBS (Ac)", "KING OF SPADES (Ks)"],
            "exact_infoset_key": "infoset=exact:player=0|hole=AcKs|board=|hist=",
            "action_probabilities": [
                {"action": 0, "probability": 0.3},
                {"action": 1, "probability": 0.2},
                {"action": 4, "probability": 0.5},
            ],
        },
    ]

    ranges = aggregate_selected_node_ranges(snapshots)
    assert len(ranges["nodes"]) == 1
    assert ranges["nodes"][0]["sample_count"] == 2
    assert ranges["nodes"][0]["action_frequencies"]["bet_raise"] == 0.5
    assert ranges["nodes"][0]["hands"][0]["policy"]["fold"] == 0.25


def test_aggregate_selected_node_ranges_groups_by_selected_node_and_hand():
    snapshots = [
        {
            "label": "first_to_act",
            "node_name": "first_to_act",
            "street": "preflop",
            "history": [],
            "selected_history": [],
            "player": 0,
            "hole_cards": ["ACE OF CLUBS (Ac)", "KING OF SPADES (Ks)"],
            "action_probabilities": [
                {"action": 0, "probability": 0.2},
                {"action": 1, "probability": 0.3},
                {"action": 4, "probability": 0.5},
            ],
        },
        {
            "label": "first_to_act",
            "node_name": "first_to_act",
            "street": "preflop",
            "history": [],
            "selected_history": [],
            "player": 0,
            "hole_cards": ["ACE OF CLUBS (Ac)", "KING OF SPADES (Ks)"],
            "action_probabilities": [
                {"action": 0, "probability": 0.3},
                {"action": 1, "probability": 0.2},
                {"action": 4, "probability": 0.5},
            ],
        },
    ]

    ranges = aggregate_selected_node_ranges(snapshots)
    assert ranges["nodes"][0]["name"] == "first_to_act"
    assert ranges["nodes"][0]["action_frequencies"]["bet_raise"] == 0.5
    assert ranges["nodes"][0]["hands"][0]["hand"] == "AKo"
    assert ranges["nodes"][0]["hands"][0]["policy"]["fold"] == 0.25


def test_filter_recent_iteration_records_keeps_only_recent_window():
    snapshots = [
        {"iteration": 10, "hole_cards": ["Ac", "Ks"], "action_probabilities": [{"action": 0, "probability": 0.9}, {"action": 4, "probability": 0.1}]},
        {"iteration": 20, "hole_cards": ["As", "Qs"], "action_probabilities": [{"action": 0, "probability": 0.2}, {"action": 4, "probability": 0.8}]},
        {"iteration": 30, "hole_cards": ["Kc", "Qd"], "action_probabilities": [{"action": 0, "probability": 0.4}, {"action": 4, "probability": 0.6}]},
        {"iteration": 40, "hole_cards": ["2d", "3d"], "action_probabilities": [{"action": 0, "probability": 0.1}, {"action": 4, "probability": 0.9}]},
    ]

    recent = filter_recent_iteration_records(snapshots, 2)

    assert [item["iteration"] for item in recent] == [30, 40]


def test_profile_variant_writes_checkpoint_stability_file(tmp_path):
    report_path = tmp_path / "hulh_run.json"
    report = profile_variant(
        "hulh",
        GAME_CONFIGS["hulh"],
        iterations=6,
        checkpoint_every=2,
        solver_name="outcome",
        history_samples=0,
        street_samples=0,
        report_mode="all",
        output_json_path=str(report_path),
    )

    checkpoint_files = sorted(tmp_path.glob("**/*checkpoint*.json"))
    assert checkpoint_files
    assert any("stability" in path.name.lower() for path in checkpoint_files)
    assert report["schema_version"] == 2
    assert report["performance"]["training"]["count"] == 6
    assert report["performance"]["training"]["measurement_scope"] == "solver.run_iteration only"
    assert report["performance"]["checkpoint_overhead"]["count"] == 3
    assert report["final_policy_records"]
    assert report["stop_policy"]["average_strategy_exported"] is True
    assert report["selected_node_ranges"]["action_families"] == ["fold", "check_call", "bet_raise"]


def test_profile_variant_keeps_all_run_artifacts_in_one_directory(tmp_path):
    report_path = tmp_path / "hulh_run.json"
    report = profile_variant(
        "hulh",
        GAME_CONFIGS["hulh"],
        iterations=6,
        checkpoint_every=2,
        solver_name="outcome",
        history_samples=0,
        street_samples=0,
        report_mode="all",
        output_json_path=str(report_path),
    )

    run_dir = report_path.parent
    assert (run_dir / "hulh_run.json").exists()
    assert (run_dir / "hulh_run_ranges.json").exists()
    assert (run_dir / "hulh_run_checkpoint_stability.json").exists()
    assert any(path.name.startswith("hulh_run_checkpoint_") for path in run_dir.iterdir())
    assert (run_dir / "hulh_run_manifest.json").exists()


def test_profile_variant_lightweight_mode_skips_per_checkpoint_json(tmp_path):
    report_path = tmp_path / "lightweight_run.json"
    report = profile_variant(
        "hulh",
        GAME_CONFIGS["hulh"],
        iterations=6,
        checkpoint_every=2,
        solver_name="outcome",
        history_samples=0,
        street_samples=0,
        report_mode="summary",
        output_json_path=str(report_path),
        artifact_mode="lightweight",
        checkpoint_history_limit=2,
    )

    run_dir = report_path.parent
    assert report["checkpoint_history"]
    assert len(report["checkpoint_history"]) <= 2
    assert not any(path.name.startswith("lightweight_run_checkpoint_") for path in run_dir.iterdir())
    assert (run_dir / "lightweight_run_ranges.json").exists()
    assert (run_dir / "lightweight_run_selected_node_summary.txt").exists()
    assert report["stop_policy"]["average_strategy_exported"] is True


def test_profile_variant_reports_memory_usage(tmp_path):
    report = profile_variant(
        "hulh",
        GAME_CONFIGS["hulh"],
        iterations=4,
        checkpoint_every=2,
        solver_name="outcome",
        history_samples=0,
        street_samples=0,
        report_mode="summary",
        output_json_path=str(tmp_path / "memory_run.json"),
    )

    memory = report["performance"]["memory"]
    assert "max_rss_mb" in memory
    assert memory["max_rss_mb"] is None or memory["max_rss_mb"] >= 0.0


def test_profile_variant_separates_stability_and_memory_thresholds(tmp_path):
    report = profile_variant(
        "hulh",
        GAME_CONFIGS["hulh"],
        iterations=4,
        checkpoint_every=2,
        solver_name="outcome",
        history_samples=0,
        street_samples=0,
        report_mode="summary",
        output_json_path=str(tmp_path / "explicit_thresholds_run.json"),
        stability_threshold=0.05,
        stop_threshold=0.85,
        memory_threshold=0.75,
    )

    stop_policy = report["stop_policy"]
    assert stop_policy["stability_threshold"] == pytest.approx(0.05)
    assert stop_policy["memory_threshold"] == pytest.approx(0.75)
    assert "memory_stop_recommended" in stop_policy
    assert report["sampling_policy"] == {"preflop": "exact_only", "postflop": "diagnostic_only"}


def test_service_status_exposes_sampling_policy(tmp_path):
    service = SolverService(min_iterations=0, checkpoint_every=1, range_samples=1)
    service.runtime.ready_for_queries = True
    service.runtime.state = SolverState.AVAILABLE
    status = service.status()
    assert status.sampling_policy == {"preflop": "exact_only", "postflop": "diagnostic_only"}


def test_hulh_history_labels_include_4bet_and_5bet_names():
    assert format_hulh_history_label(["bet", "bet", "raise"]) == "response_to_open_4bet"
    assert format_hulh_history_label(["bet", "bet", "raise", "raise"]) == "response_to_open_5bet"

    preset_names = [spec["display_name"] for spec in resolve_node_specs("hulh-preflop")]
    assert "response_to_open_4bet" in preset_names
    assert "response_to_open_5bet" in preset_names


def test_profile_variant_reports_runtime_state_machine():
    report = profile_variant(
        "hulh",
        GAME_CONFIGS["hulh"],
        iterations=6,
        checkpoint_every=2,
        solver_name="outcome",
        history_samples=0,
        street_samples=0,
        report_mode="summary",
        output_json_path=str("/tmp/runtime_state_run.json"),
    )

    runtime = report["runtime_state"]
    assert runtime["state"] in {"stable", "queryable", "running"}
    assert runtime["current_policy_in_memory"] is True
    assert runtime["latest_stable_snapshot"] is not None or runtime["state"] in {"running", "queryable"}
    assert runtime["checkpoint_on_disk"] is not None


def test_profile_variant_rejects_non_positive_iteration_count():
    with pytest.raises(ValueError, match="iterations must be greater than zero"):
        profile_variant("hulh", GAME_CONFIGS["hulh"], iterations=0)
