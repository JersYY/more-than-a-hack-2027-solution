"""Ticket T5 -- src/temporal.py decode() and majority(). Acceptance tests, written before the code.

Evidence arrays are hand-built and the FrameContext constructed directly, so T5 waits for neither
T1 nor T2-T4. With the hybrid_v1 defaults, +/-3 team evidence is worth 0.6 a frame between H and A
against a turnover cost of 5.7, and a dead score of 4 is worth 0.8 a frame against 6.9 + 5.9.
xfail(strict) until T5 lands.
"""

import numpy as np
import pytest

import baseline  # official starter/baseline.py, import only
import possession_metrics as pm
from src.config import TemporalConfig
from src.preprocess import FrameContext
from src.temporal import decode, majority

pytestmark = pytest.mark.xfail(raises=NotImplementedError, strict=True,
                               reason="ticket T5 not implemented yet (Phase 3)")

CONFIG = TemporalConfig()
H, A, D, X = pm.H, pm.A, pm.D, pm.X


def context_of(n, blocks=None):
    blocks = tuple(blocks or ((1, 0, n - 1),))
    half = np.zeros(n, dtype=np.int8)
    for number, first, last in blocks:
        half[first:last + 1] = number
    return FrameContext(n, blocks, half, {(1, 0): 1, (1, 1): -1, (2, 0): -1, (2, 1): 1})


def run(*pieces, blocks=None):
    """pieces: (frames, team, dead) runs, concatenated."""
    team = np.concatenate([np.full(n, float(t)) for n, t, _ in pieces])
    dead = np.concatenate([np.full(n, float(d)) for n, _, d in pieces])
    return decode(team, dead, context_of(len(team), blocks), CONFIG)


def expect(*runs):
    return np.concatenate([np.full(n, code, dtype=np.int8) for n, code in runs])


def test_steady_evidence_is_one_state():
    assert np.array_equal(run((200, 3, 0)), expect((200, H)))


def test_a_short_opponent_burst_is_not_a_change_of_possession():
    assert np.array_equal(run((100, 3, 0), (5, -3, 0), (100, 3, 0)), expect((205, H)))


def test_sustained_opponent_evidence_changes_possession_where_it_starts():
    assert np.array_equal(run((100, 3, 0), (60, -3, 0)), expect((100, H), (60, A)))


def test_across_a_silent_gap_the_change_comes_where_evidence_resumes():
    assert np.array_equal(run((50, 3, 0), (40, 0, 0), (50, -3, 0)), expect((90, H), (50, A)))


def test_a_stoppage_splits_two_possessions_of_the_same_team():
    states = run((100, 3, 0), (75, 0, 4), (100, 3, 0))
    assert np.array_equal(states, expect((100, H), (75, D), (100, H)))
    assert len(pm.possession_spells(states, [(0, 275)])) == 2


def test_a_player_next_to_a_dead_ball_does_not_keep_it_alive():
    # The taker stands at the ball during the stoppage, so team evidence stays high throughout.
    # Dead-ball evidence of 2 must still win: team evidence decides H vs A, not alive vs D. (With
    # team evidence counted as alive evidence in full, this stoppage would vanish -- the scripted
    # free-kick scenario showed it, docs/experiments.md.)
    assert np.array_equal(run((100, 3, 0), (75, 3, 2), (100, 3, 0)),
                          expect((100, H), (75, D), (100, H)))


def test_the_away_mirror_image():
    assert np.array_equal(run((100, -3, 0), (75, 0, 4), (100, -3, 0)),
                          expect((100, A), (75, D), (100, A)))


def test_blocks_are_decoded_independently_and_x_is_outside():
    states = run((10, 0, 0), (130, 3, 0), (20, 0, 0), (130, -3, 0), (10, 0, 0),
                 blocks=((1, 10, 139), (2, 160, 289)))
    assert np.array_equal(states, expect((10, X), (130, H), (20, X), (130, A), (10, X)))


def test_noise_below_the_fixed_point_resolution_changes_nothing():
    team = np.concatenate([np.full(100, 3.0), np.full(60, -3.0)])
    noisy = team + np.random.RandomState(0).uniform(-1e-12, 1e-12, len(team))
    context = context_of(len(team))
    assert np.array_equal(decode(team, np.zeros(160), context, CONFIG),
                          decode(noisy, np.zeros(160), context, CONFIG))


def test_decoding_is_deterministic():
    rng = np.random.RandomState(1)
    team, dead = rng.normal(0, 2, 3000), rng.normal(0, 2, 3000)
    context = context_of(3000)
    assert np.array_equal(decode(team, dead, context, CONFIG), decode(team, dead, context, CONFIG))


@pytest.mark.parametrize("window", [0, 25, 50, 125])
def test_majority_is_the_official_vote_exactly(window):
    codes = np.random.RandomState(window).randint(-1, 2, 5000).astype(np.int8)
    expected = pm.label_codes(baseline.smooth(codes, window))
    assert np.array_equal(majority(codes, window), expected)
