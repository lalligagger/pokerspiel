from preflop_subgame_explorer_v2 import BOARD_EXAMPLE_1, _range_from_history, train_board_conditioned_subgame


def test_policy_moves_across_iterations():
    history = ["bet", "call"]
    rec1 = train_board_conditioned_subgame(
        board=BOARD_EXAMPLE_1,
        history=history,
        ranges={"p1": _range_from_history(history, "p1"), "p2": _range_from_history(history, "p2")},
        iterations=1,
        player_to_act=1,
    )
    rec200 = train_board_conditioned_subgame(
        board=BOARD_EXAMPLE_1,
        history=history,
        ranges={"p1": _range_from_history(history, "p1"), "p2": _range_from_history(history, "p2")},
        iterations=200,
        player_to_act=1,
    )

    delta = sum(abs(rec200["aggregate_policy"][k] - rec1["aggregate_policy"][k]) for k in rec200["aggregate_policy"])
    assert delta > 0.02, f"expected meaningful policy movement, saw delta={delta}"
