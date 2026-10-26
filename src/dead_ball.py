"""Dead-ball evidence: per frame, how strongly the tracking says the ball is out of play.

    dead = dead_evidence(features, config.dead_ball)    # (n,) float64: > 0 dead, < 0 live, 0 none

The official baseline calls the ball dead exactly when the tracking cannot see it. That conflates
two different things -- the ball is missing on ~41% of graded frames, mostly in live play (DATA.md)
-- and it cannot see a stoppage while the ball is still visible (scripts/synthetic_bench.py). The
cues below are about the game state instead: a ball beyond a line, a ball at rest, players walking.
"""

import numpy as np

# Params
# kind "ball_missing" scores are this large so that, like the official baseline, nothing else can
# outvote them.
BALL_MISSING_SCORE = 1000.0


def dead_evidence(features, config):
    """(n,) float64 dead-ball evidence from a features dict. Ticket T4; tests/test_dead_ball.py.

    Inputs: features ball_seen, ball_out_m, ball_speed_mps, ball_still_s, activity_mps (NaN =
    unknown) and config: src.config.DeadBallConfig.

    kind "ball_missing" -- official baseline: +BALL_MISSING_SCORE where the ball is not seen,
        -BALL_MISSING_SCORE where it is.
    kind "cues" -- the sum of four terms, each 0 where its input is NaN:
        +out_score       where ball_out_m > out_margin_m                     (beyond a line)
        -moving_score    where ball_out_m == 0 and ball_speed_mps > moving_mps (moving in play)
        +still_score     where ball_still_s >= still_min_s                    (placed for a restart)
        clip((activity_threshold_mps - activity_mps) / activity_scale_mps,
             -activity_max_score, +activity_max_score)                       (players walking)

    A missing ball contributes nothing by itself: missing is not dead. Only + - * / and
    comparisons are used (D-016).
    """
    seen = np.asarray(features["ball_seen"], dtype=bool)
    if config.kind == "ball_missing":
        return np.where(seen, -BALL_MISSING_SCORE, BALL_MISSING_SCORE)

    out = np.asarray(features["ball_out_m"], dtype=np.float64)
    speed = np.asarray(features["ball_speed_mps"], dtype=np.float64)
    still = np.asarray(features["ball_still_s"], dtype=np.float64)
    activity = np.asarray(features["activity_mps"], dtype=np.float64)

    evidence = np.zeros(len(seen))
    with np.errstate(invalid="ignore"):                       # NaN comparisons are False: no term
        evidence += np.where(out > config.out_margin_m, config.out_score, 0.0)
        evidence -= np.where((out == 0.0) & (speed > config.moving_mps), config.moving_score, 0.0)
        evidence += np.where(still >= config.still_min_s, config.still_score, 0.0)
    walking = np.clip((config.activity_threshold_mps - activity) / config.activity_scale_mps,
                      -config.activity_max_score, config.activity_max_score)
    evidence += np.where(np.isfinite(activity), walking, 0.0)
    return evidence
