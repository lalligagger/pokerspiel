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
from profile_wrapper_solver import (
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


def test_prepare_selected_node_probes_warns_and_keeps_available_states():
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

    with pytest.warns(UserWarning, match="only sampled .* distinct deals for selected node 'first_to_act'"):
        probes = prepare_selected_node_probes(game, specs, samples_per_node=5000, max_attempts=2000)

    assert 0 < len(probes) < 5000


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


def test_profile_variant_rejects_non_positive_iteration_count():
    with pytest.raises(ValueError, match="iterations must be greater than zero"):
        profile_variant("hulh", GAME_CONFIGS["hulh"], iterations=0)
