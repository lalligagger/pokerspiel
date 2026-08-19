from solver_live_dashboard import build_grid_labels


def test_build_grid_labels_covers_full_13x13_preflop_matrix():
    labels = build_grid_labels()

    assert len(labels) == 169
    assert labels[0] == {"i": 0, "j": 0, "label": "AA"}
    assert {"i": 0, "j": 1, "label": "AKs"} in labels
    assert {"i": 1, "j": 0, "label": "AKo"} in labels
    assert {"i": 0, "j": 12, "label": "A2s"} in labels
    assert {"i": 12, "j": 0, "label": "A2o"} in labels
    assert {"i": 12, "j": 12, "label": "22"} in labels
