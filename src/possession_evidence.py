"""Team evidence: per frame, how strongly the tracking says home rather than away controls the ball.

    team = team_evidence(features, config.evidence)     # (n,) float64: > 0 home, < 0 away, 0 none

Evidence, not a decision. Nearest is not control (D-012): the decoder (src/temporal.py) weighs this
against the cost of a change of possession, and that is what turns "a player was near the ball"
into "the team gained control". A brush is a few frames of evidence; a controlled possession is
seconds of it.
"""

import numpy as np


def team_evidence(features, config):
    """(n,) float64 team evidence from a features dict. Ticket T3; tests/test_evidence.py.

    Inputs: features["ball_seen"], features["d_home_m"], features["d_away_m"] (NaN = no player of
    that team tracked, or no ball), features["ball_speed_mps"], and config:
    src.config.EvidenceConfig.

    kind "nearest_hard" -- official baseline step 1: +max_score where the nearest tracked player
        is home, -max_score where away (an exact distance tie goes to home), 0 where there is no
        ball or nobody is tracked.
    kind "soft_control" -- each team's nearest player gets the weight
            w(d) = 1 / (1 + (d / control_m)^2)^2        (w = 0 when that team has nobody tracked)
        and with w0 = w(free_m), the evidence is
            e = max_score * (w_home - w_away) / (w_home + w_away + w0),
        0 where there is no ball, and 0 where ball_speed_mps > flight_mps: a ball travelling
        faster than a player can run with it is in flight, and nobody controls a ball in flight --
        possession changes at the reception (CHALLENGE.md), not as the pass flies past someone.
        A NaN speed does not gate.

    Guarantees: |e| <= max_score; swapping the teams negates e; e increases as the nearest home
    player gets closer, other things equal. Only + - * / are used, so the values are identical on
    every CPU (D-016) -- no exp, log or pow on this path.
    """
    seen = np.asarray(features["ball_seen"], dtype=bool)
    d_home = np.asarray(features["d_home_m"], dtype=np.float64)
    d_away = np.asarray(features["d_away_m"], dtype=np.float64)
    home_tracked, away_tracked = np.isfinite(d_home), np.isfinite(d_away)

    if config.kind == "nearest_hard":
        with np.errstate(invalid="ignore"):
            home = seen & home_tracked & (~away_tracked | (d_home <= d_away))
            away = seen & away_tracked & (~home_tracked | (d_away < d_home))
        return config.max_score * (home.astype(np.float64) - away.astype(np.float64))

    w_home = np.where(home_tracked, control_weight(np.where(home_tracked, d_home, 0.0),
                                                   config.control_m), 0.0)
    w_away = np.where(away_tracked, control_weight(np.where(away_tracked, d_away, 0.0),
                                                   config.control_m), 0.0)
    w_free = control_weight(config.free_m, config.control_m)
    evidence = config.max_score * (w_home - w_away) / (w_home + w_away + w_free)

    speed = np.asarray(features["ball_speed_mps"], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        in_flight = speed > config.flight_mps                  # NaN compares False: no gate
    return np.where(seen & ~in_flight, evidence, 0.0)


def control_weight(distance, control_m):
    """1 / (1 + (d / control_m)^2)^2: 1 at the ball, 1/4 at control_m, ~d^-4 beyond."""
    ratio = np.asarray(distance, dtype=np.float64) / control_m
    base = 1.0 + ratio * ratio
    return 1.0 / (base * base)


def shift_team_evidence(team, context, frames):
    """Shift by signed frames, zero padding per half; never wrap or cross halves.

    Positive delays team attribution. Zero preserves the original decision path.
    This operates on evidence, before decoding, rather than shifting output labels.
    """
    if frames == 0:
        return team
    result = np.zeros_like(team)
    for _, first, last in context.blocks:
        if abs(frames) >= last - first + 1:
            continue
        if frames > 0:
            result[first + frames:last + 1] = team[first:last + 1 - frames]
        else:
            result[first:last + 1 + frames] = team[first - frames:last + 1]
    return result
