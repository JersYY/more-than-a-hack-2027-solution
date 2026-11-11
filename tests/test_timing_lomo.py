"""Changing a held-out score must never affect that fold's selection."""
from scripts.omb_timing_lomo import select_without


def test_selection_excludes_held_out_match():
    training = {"0:4.5": {"train": 0.5, "held": 1.0},
                "8:4.5": {"train": 0.51, "held": 0.0}}
    assert select_without(training, "held") == "8:4.5"
    training["8:4.5"]["held"] = 100.0
    assert select_without(training, "held") == "8:4.5"


def test_tiny_gain_keeps_reference():
    training = {"0:4.5": {"train": 0.5, "held": 0.0},
                "8:4.5": {"train": 0.501, "held": 1.0}}
    assert select_without(training, "held") == "0:4.5"
