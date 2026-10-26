"""Ticket T4 -- src/dead_ball.py dead_evidence(). Acceptance tests, written before the code.

Inputs are hand-built feature columns, so T4 did not wait for T2 (D-018).
Every frame starts neutral -- ball seen and inside, between still and moving, players at the
activity threshold -- so each test moves exactly one cue.
"""

import numpy as np
import pytest

from src.config import DeadBallConfig
from src.dead_ball import BALL_MISSING_SCORE, dead_evidence


CUES = DeadBallConfig()           # the hybrid_v1 values


def features(n, **columns):
    neutral = {"ball_seen": np.ones(n, dtype=bool), "ball_out_m": np.zeros(n),
               "ball_speed_mps": np.full(n, 1.0), "ball_still_s": np.zeros(n),
               "ball_missing_s": np.zeros(n), "activity_mps": np.full(n, CUES.activity_threshold_mps)}
    neutral.update({name: np.asarray(value, dtype=neutral[name].dtype)
                    for name, value in columns.items()})
    return neutral


def test_ball_missing_kind_is_the_official_rule():
    config = DeadBallConfig(kind="ball_missing")
    e = dead_evidence(features(3, ball_seen=[True, False, True]), config)
    assert list(e) == [-BALL_MISSING_SCORE, BALL_MISSING_SCORE, -BALL_MISSING_SCORE]


def test_neutral_frames_carry_no_evidence():
    assert list(dead_evidence(features(4), CUES)) == [0.0] * 4


def test_a_ball_beyond_a_line_is_dead():
    e = dead_evidence(features(4, ball_out_m=[0.0, 0.4, 0.6, 3.0]), CUES)
    assert list(e) == [0.0, 0.0, 4.0, 4.0]


def test_a_moving_ball_inside_is_live_and_outside_is_not():
    e = dead_evidence(features(2, ball_speed_mps=[5.0, 5.0], ball_out_m=[0.0, 1.0]), CUES)
    assert list(e) == [-1.0, 4.0]


def test_a_ball_at_rest_long_enough_is_placed():
    e = dead_evidence(features(3, ball_still_s=[1.0, 1.5, 3.0]), CUES)
    assert list(e) == [0.0, 1.0, 1.0]


def test_walking_players_are_dead_evidence_and_running_ones_live():
    e = dead_evidence(features(5, activity_mps=[0.4, 1.2, 2.0, 5.0, np.nan]), CUES)
    assert e == pytest.approx([2.0, 0.0, -2.0, -2.0, 0.0])


def test_a_missing_ball_alone_is_not_dead():
    e = dead_evidence(features(2, ball_seen=[False, False], ball_out_m=[np.nan, np.nan],
                               ball_speed_mps=[np.nan, np.nan]), CUES)
    assert list(e) == [0.0, 0.0]


def test_a_long_gap_is_evidence_only_when_configured():
    gap = features(3, ball_seen=[False] * 3, ball_out_m=[np.nan] * 3, ball_speed_mps=[np.nan] * 3)
    gap["ball_missing_s"] = np.array([1.0, 5.0, 12.0])
    assert list(dead_evidence(gap, CUES)) == [0.0, 0.0, 0.0]                 # hybrid_v1: off
    on = DeadBallConfig(missing_score=1.5, missing_min_s=5.0)
    assert list(dead_evidence(gap, on)) == [0.0, 1.5, 1.5]


def test_the_cues_add_up():
    e = dead_evidence(features(1, ball_out_m=[1.0], ball_still_s=[2.0], activity_mps=[0.8]), CUES)
    assert e == pytest.approx([4.0 + 1.0 + 1.0])
