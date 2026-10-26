"""Ticket T3 -- src/possession_evidence.py team_evidence(). Acceptance tests, written before the code.

Inputs are hand-built feature columns, so T3 did not wait for T2 (D-018).
"""

import numpy as np
import pytest

from src.config import EvidenceConfig
from src.possession_evidence import team_evidence


SOFT = EvidenceConfig(kind="soft_control", control_m=1.0, free_m=2.5, max_score=3.0)
HARD = EvidenceConfig(kind="nearest_hard", max_score=3.0)


def features(d_home, d_away, seen=None, speed=None):
    d_home, d_away = np.asarray(d_home, dtype=float), np.asarray(d_away, dtype=float)
    seen = np.ones(len(d_home), dtype=bool) if seen is None else np.asarray(seen)
    speed = np.full(len(d_home), 2.0) if speed is None else np.asarray(speed, dtype=float)
    return {"ball_seen": seen, "d_home_m": d_home, "d_away_m": d_away, "ball_speed_mps": speed}


def weight(d):
    return 1.0 / (1.0 + (d / 1.0) ** 2) ** 2


def test_hard_evidence_is_official_baseline_step_one():
    e = team_evidence(features([0.5, 2.0, 1.0, np.nan, 1.0, np.nan],
                               [1.0, 0.5, 1.0, 2.0, np.nan, np.nan],
                               [True, True, True, True, False, True]), HARD)
    assert list(e) == [3.0, -3.0, 3.0, -3.0, 0.0, 0.0]


def test_soft_evidence_is_the_documented_formula():
    d_home, d_away = np.array([0.5, 1.0, 3.0, 0.2]), np.array([3.0, 1.0, 0.5, np.nan])
    w_home, w_away = weight(d_home), np.nan_to_num(weight(d_away), nan=0.0)
    expected = 3.0 * (w_home - w_away) / (w_home + w_away + weight(2.5))
    assert team_evidence(features(d_home, d_away), SOFT) == pytest.approx(expected, rel=1e-12)


def test_soft_evidence_properties():
    rng = np.random.RandomState(0)
    d_home, d_away = rng.uniform(0, 10, 500), rng.uniform(0, 10, 500)
    e = team_evidence(features(d_home, d_away), SOFT)
    assert (np.abs(e) <= 3.0).all()
    assert team_evidence(features(d_away, d_home), SOFT) == pytest.approx(-e)   # antisymmetric
    closer = team_evidence(features(d_home * 0.5, d_away), SOFT)
    assert (closer >= e - 1e-12).all()                                          # monotone


def test_no_ball_no_evidence():
    e = team_evidence(features([0.3, 0.3], [5.0, 5.0], [False, False]), SOFT)
    assert list(e) == [0.0, 0.0]


def test_a_ball_in_flight_is_nobodys():
    # A pass flying past an opponent at 3 m is not a change of possession in the making: the
    # change comes at the reception (found by the scripted turnover scenario, docs/experiments.md).
    e = team_evidence(features([5.0, 5.0, 5.0], [3.0, 3.0, 3.0], speed=[17.0, 8.5, np.nan]), SOFT)
    assert e[0] == 0.0 and e[1] == 0.0       # in flight
    assert e[2] < 0.0                        # speed unknown: no gate


def test_a_loose_ball_is_weak_evidence_and_a_controlled_one_strong():
    e = team_evidence(features([5.0, 0.5], [7.0, 3.0]), SOFT)
    assert abs(e[0]) < 0.1 * 3.0        # both teams far: nobody controls it
    assert e[1] > 0.9 * 3.0             # a home player at 0.5 m, the nearest opponent at 3 m
