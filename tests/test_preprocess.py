"""Ticket T1 -- src/preprocess.py frame_context(). Acceptance tests, written in Phase 2 before the
code (D-018); implemented in Phase 3.
"""

import numpy as np
import pytest

import games
from src.data_loader import AWAY, HOME
from src.preprocess import frame_context
from src.synthetic import SCENARIOS


N = 199319
DEFAULT_SIGNS = {(1, HOME): 1, (1, AWAY): -1, (2, HOME): -1, (2, AWAY): 1}


def test_default_blocks_are_the_published_halves():
    context = frame_context(N)
    assert context.blocks == tuple(games.half_video_blocks(games.GAME))
    assert [int(np.sum(context.half == half)) for half in (0, 1, 2)] == [40462, 79467, 79390]
    assert [int(context.half[f]) for f in (11184, 11185, 90651, 90652, 110807, 110808, 190197,
                                           190198)] == [0, 1, 1, 0, 0, 2, 2, 0]
    assert int(context.graded.sum()) == 158857


def test_custom_blocks():
    context = frame_context(100, blocks=[(1, 10, 39), (2, 60, 89)])
    assert [int(context.half[f]) for f in (9, 10, 39, 40, 59, 60, 89, 90)] == [0, 1, 1, 0, 0, 2, 2, 0]
    assert int(context.graded.sum()) == 60


@pytest.mark.parametrize("blocks", [
    [(1, 10, 50), (2, 40, 80)],     # overlapping
    [(2, 60, 89), (1, 10, 39)],     # out of order
    [(1, -1, 10)],                  # before the first frame
    [(1, 90, 100)],                 # past the last frame
    [(1, 30, 20)],                  # empty
])
def test_bad_blocks_are_refused(blocks):
    with pytest.raises(ValueError):
        frame_context(100, blocks=blocks)


def test_attack_direction_is_measured_from_positions():
    match = SCENARIOS["second_half"]()
    context = frame_context(match.tracking.n_frames, match.blocks, match.tracking)
    assert context.attack_sign == DEFAULT_SIGNS          # home attacks +x first, then -x


def test_attack_direction_follows_the_teams_not_the_default():
    # The same match with the home and away slots swapped: now "home" attacks -x first.
    match = SCENARIOS["second_half"]()
    swapped = match.tracking
    swapped.layout.team_of_slot[:] = 1 - swapped.layout.team_of_slot
    context = frame_context(swapped.n_frames, match.blocks, swapped)
    assert context.attack_sign == {(1, HOME): -1, (1, AWAY): 1, (2, HOME): 1, (2, AWAY): -1}


def test_attack_direction_defaults_without_tracking():
    context = frame_context(100, blocks=[(1, 10, 39), (2, 60, 89)])
    assert context.attack_sign == DEFAULT_SIGNS
