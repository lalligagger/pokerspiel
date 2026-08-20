from solver_live_dashboard import build_grid_labels, category_to_cell, hand_to_category
from solver_live_dashboard_old import category_to_cell as legacy_category_to_cell


def test_build_grid_labels_covers_full_13x13_preflop_matrix():
    labels = build_grid_labels()

    assert len(labels) == 169
    assert labels[0] == {"i": 0, "j": 0, "label": "AA"}
    assert {"i": 0, "j": 1, "label": "AKs"} in labels
    assert {"i": 1, "j": 0, "label": "AKo"} in labels
    assert {"i": 0, "j": 12, "label": "A2s"} in labels
    assert {"i": 12, "j": 0, "label": "A2o"} in labels
    assert {"i": 12, "j": 12, "label": "22"} in labels


def test_hand_to_category_requires_highest_rank_first_indexing():
    assert hand_to_category("KAs") is None
    assert hand_to_category("Q2s") == "Q2s"
    assert hand_to_category("2Qs") is None
    assert category_to_cell("AKs") == (0, 1)
    assert category_to_cell("AKo") == (1, 0)
    assert category_to_cell("Q2s") == (2, 12)
    assert category_to_cell("KAs") is None


def test_legacy_category_to_cell_preserves_high_rank_first_non_pairs():
    assert legacy_category_to_cell("A2s") == (0, 12)
    assert legacy_category_to_cell("Q2s") == (2, 12)
    assert legacy_category_to_cell("A2o") == (12, 0)
    assert legacy_category_to_cell("KAs") is None
    assert len(legacy_category_to_cell.__globals__["canonical_hand_classes"]()) == 169
