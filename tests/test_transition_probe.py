"""Timing experiments must not borrow evidence across halves or wrap array ends."""
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.omb_transition_probe import shift_evidence


@pytest.mark.parametrize("shift, expected", [
    (0, list(range(9))),
    (1, [0, 0, 1, 2, 0, 0, 5, 6, 0]),
    (-1, [0, 2, 3, 0, 0, 6, 7, 0, 0]),
    (3, [0] * 9),
    (-3, [0] * 9),
])
def test_shift_stays_in_half_and_zero_pads(shift, expected):
    team = np.arange(9, dtype=float)
    original = team.copy()
    context = SimpleNamespace(blocks=[(1, 1, 3), (2, 5, 7)])
    assert np.array_equal(shift_evidence(team, context, shift), expected)
    assert np.array_equal(team, original)
