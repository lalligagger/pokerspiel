import os
import subprocess
import sys

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
    exact_hole_board_signature,
    filter_recent_iteration_records,
    format_hulh_history_label,
    infer_state_context,
    is_meaningful_state,
    prepare_selected_node_probes,
    profile_variant,
    resolve_node_specs,
    sample_distinct_deal_states,
    summarize_policy_profiles,
)


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


def test_solver_service_defaults_to_external_and_respects_env_override(monkeypatch):
    monkeypatch.delenv("POKERSPIEL_SOLVER", raising=False)
    service = SolverService()
    assert service.solver_name == "external"

    monkeypatch.setenv("POKERSPIEL_SOLVER", "outcome")
    service = SolverService()
    assert service.solver_name == "outcome"


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


def test_get_preflop_range_returns_full_canonical_range_from_live_policy():
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
                "hands": [{"hand": "TT", "policy": {"fold": 0.15, "check_call": 0.25, "bet_raise": 0.6}}],
            }
        ]
    }

    response = service.get_preflop_range("response_to_open")

    assert response.ready is True
    assert response.spot == "response_to_open"
    assert any(hand.hand == "TT" for hand in response.hands)
    assert any(hand.hand == "AA" for hand in response.hands)
    assert any(hand.hand == "AKs" for hand in response.hands)
    assert response.hand_count >= 169


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
