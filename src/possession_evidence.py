"""Team evidence: per frame, how strongly the tracking says home rather than away controls the ball.

    team = team_evidence(features, config.evidence)     # (n,) float64: > 0 home, < 0 away, 0 none

Evidence, not a decision. Nearest is not control (D-012): the decoder (src/temporal.py) weighs this
against the cost of a change of possession, and that is what turns "a player was near the ball"
into "the team gained control". A brush is a few frames of evidence; a controlled possession is
seconds of it.
"""


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
    raise NotImplementedError("Phase 3, ticket T3 -- docs/tickets.md")
