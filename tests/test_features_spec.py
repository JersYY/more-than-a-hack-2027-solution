"""Ticket T2 -- src/features.py build_features() against docs/feature_spec.md.

Acceptance tests written before the code: xfail(strict) until T2 lands, then they must pass and the
marker comes off. Hand-placed frames, positions in metres on the games.py pitch.
"""

import numpy as np
import pytest

import games
from src.config import Config
from src.features import FEATURES, build_features
from src.preprocess import FrameContext
from src.synthetic import SCENARIOS, make_tracking

pytestmark = pytest.mark.xfail(raises=NotImplementedError, strict=True,
                               reason="ticket T2 not implemented yet (Phase 3)")

LENGTH, WIDTH = games.GAMES[games.GAME]["pitch"]
CONFIG = Config()


def context_of(n, blocks=None):
    blocks = tuple(blocks or ((1, 0, n - 1),))
    half = np.zeros(n, dtype=np.int8)
    for number, first, last in blocks:
        half[first:last + 1] = number
    return FrameContext(n, blocks, half, {(1, 0): 1, (1, 1): -1, (2, 0): -1, (2, 1): 1})


def tracking_of(ball_m, home_m=None, away_m=None, home_speed=1.5, away_speed=1.5):
    """n frames, one home slot and one away slot; positions in metres, NaN = not tracked."""
    ball_m = np.asarray(ball_m, dtype=float)
    n = len(ball_m)
    players = np.full((n, 2, 3), np.nan)
    for slot, positions, speed in ((0, home_m, home_speed), (1, away_m, away_speed)):
        if positions is None:
            continue
        positions = np.asarray(positions, dtype=float)
        players[:, slot, 0], players[:, slot, 1] = positions[:, 0] / LENGTH, positions[:, 1] / WIDTH
        players[:, slot, 2] = np.where(np.isfinite(positions[:, 0]), speed, np.nan)
    ball = np.full((n, 3), np.nan)
    ball[:, 0], ball[:, 1] = ball_m[:, 0] / LENGTH, ball_m[:, 1] / WIDTH
    return make_tracking(players, ball, [0, 1])


def build(tracking):
    return build_features(tracking, context_of(tracking.n_frames), CONFIG)


def test_every_p0_feature_is_present_with_one_value_per_frame():
    match = SCENARIOS["turnover"]()
    features = build_features(match.tracking, context_of(match.tracking.n_frames, match.blocks),
                              CONFIG)
    assert set(FEATURES) <= set(features)
    assert all(np.shape(features[name]) == (match.tracking.n_frames,) for name in FEATURES)


def test_distances_are_metres_to_each_teams_nearest_player():
    features = build(tracking_of([(50, 30)] * 3, home_m=[(51, 30)] * 3, away_m=[(50, 33)] * 3))
    assert features["d_home_m"] == pytest.approx([1.0] * 3, rel=1e-12)
    assert features["d_away_m"] == pytest.approx([3.0] * 3, rel=1e-12)
    assert list(features["n_home"]) == [1] * 3 and list(features["n_away"]) == [1] * 3


def test_a_missing_ball_is_nan_never_zero():
    ball = [(50, 30)] * 8
    ball[2:5] = [(np.nan, np.nan)] * 3
    features = build(tracking_of(ball, home_m=[(51, 30)] * 8, away_m=[(50, 33)] * 8))
    assert list(features["ball_seen"]) == [True, True, False, False, False, True, True, True]
    for name in ("d_home_m", "d_away_m", "ball_out_m", "ball_x_m", "ball_speed_mps"):
        assert np.isnan(features[name][2:5]).all(), name
    assert features["ball_missing_s"][2:5] == pytest.approx([3 / 25.0] * 3)
    assert (features["ball_missing_s"][[0, 1, 5, 6, 7]] == 0).all()


def test_a_team_with_nobody_tracked_has_no_distance():
    features = build(tracking_of([(50, 30)] * 3, home_m=[(51, 30)] * 3, away_m=None))
    assert np.isnan(features["d_away_m"]).all() and list(features["n_away"]) == [0] * 3


def test_ball_speed_is_a_central_difference_in_metres_per_second():
    ball = [(10 + 0.4 * t, 30) for t in range(20)]                # 0.4 m a frame = 10 m/s
    speed = build(tracking_of(ball))["ball_speed_mps"]
    assert np.isnan(speed[:2]).all() and np.isnan(speed[-2:]).all()  # no frame t-2 / t+2
    assert speed[2:-2] == pytest.approx([10.0] * 16, abs=1e-9)


def test_a_tracker_jump_is_not_a_ball_speed():
    ball = [(10, 30)] * 10 + [(20, 30)] * 10                      # 10 m in one frame
    speed = build(tracking_of(ball))["ball_speed_mps"]
    assert np.isnan(speed[8:12]).all()                            # 62.5 m/s > MAX_BALL_SPEED


def test_ball_out_is_the_distance_beyond_the_nearest_line():
    ball = [(50, -2), (50, WIDTH + 2), (-3, 30), (LENGTH + 3, 30), (50, 30)]
    out = build(tracking_of(ball))["ball_out_m"]
    assert out == pytest.approx([2.0, 2.0, 3.0, 3.0, 0.0], abs=1e-9)


def test_still_is_the_length_of_the_still_run():
    ball = [(50, 30)] * 50 + [(50 + 0.4 * t, 30) for t in range(1, 51)]
    still = build(tracking_of(ball))["ball_still_s"]
    assert 1.6 <= still[20] <= 2.0                                # ~47 still frames
    assert still[75] == 0.0                                       # moving at 10 m/s


def test_team_speed_and_activity_come_from_the_speed_channel():
    n = 200
    tracking = tracking_of([(50, 30)] * n, home_m=[(40, 30)] * n, away_m=[(60, 30)] * n)
    tracking.players[:100, :, 2] = 1.0
    tracking.players[100:, :, 2] = 3.0
    features = build(tracking)
    assert features["team_speed_mps"][[0, 99, 100, 199]] == pytest.approx([1.0, 1.0, 3.0, 3.0])
    assert features["activity_mps"][40] == pytest.approx(1.0)       # 0..90: all in the first part
    assert features["activity_mps"][100] == pytest.approx((50 * 1.0 + 51 * 3.0) / 101)
