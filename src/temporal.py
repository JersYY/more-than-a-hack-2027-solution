"""Temporal inference: per-frame evidence in, one state per frame out.

    states = decode(team, dead, context, config.temporal)    # (n,) int8 pm codes H/A/D/X

The ranking metric scores possessions as whole intervals (EVALUATION.md), so the decision is made
for a whole half at once, offline: the most likely sequence of states given all its evidence, with
a price on every change. A short burst of opponent evidence cannot pay for two changes of
possession; seconds of it can. Looking at the whole half also means a change can be placed where
the evidence starts, instead of wherever a causal filter would have finally noticed it.
"""


def decode(team, dead, context, config):
    """(n,) int8 states. Ticket T5; tests/test_temporal.py.

    team, dead -- (n,) float64 evidence (src/possession_evidence.py, src/dead_ball.py).
    context    -- src.preprocess.FrameContext. Frames outside its blocks get pm.X.
    config     -- src.config.TemporalConfig with decoder "viterbi".

    Each graded block is decoded on its own, from no preferred initial state, maximising

        sum_t score_t(state_t) - sum_t cost(state_{t-1}, state_t)

    with s = evidence_scale, b_t = min(|team_t| / 2, alive_bonus) and
        score_t(D) = s * dead_t / 2
        score_t(H) = s * (-dead_t / 2 + b_t - max(-team_t, 0))
        score_t(A) = s * (-dead_t / 2 + b_t - max(team_t, 0))
        cost(H, A) = cost(A, H) = turnover_cost

    So score(H) - score(A) = s * team_t: team evidence decides between the teams in full. But it
    lifts the better team over D by at most s * alive_bonus, because a player standing next to the
    ball is about as likely at a restart as in play -- dead-ball evidence decides alive vs D.
    (alive_bonus ~ ln 2 is what a mixture model gives: P(team evidence | D) = the average of its
    likelihood under H and under A.)

        cost(H, D) = cost(A, D) = stop_cost
        cost(D, H) = cost(D, A) = restart_cost
        cost(x, x) = 0.

    Determinism (D-016): scores and costs are rounded to integers in units of 1/fixed_point and
    summed in int64, so the chosen path cannot depend on a CPU's last-bit rounding.

    Ties: of several equally good paths, the one whose changes come latest. Across a stretch with no
    evidence, a change of possession then happens where evidence resumes -- where the scorer's
    forward-fill of X would put it (EVALUATION.md).
    """
    raise NotImplementedError("Phase 3, ticket T5 -- docs/tickets.md")


def majority(codes, window):
    """(n,) int8 pm codes: official baseline step 2, exactly. Ticket T5; tests/test_temporal.py.

    codes -- (n,) nearest-team codes as src.features.nearest_team returns them (HOME, AWAY, -1).
    The result must equal starter/baseline.py smooth(codes, window) on every frame, with its
    letters mapped to pm codes; reuse that function rather than re-deriving it.
    """
    raise NotImplementedError("Phase 3, ticket T5 -- docs/tickets.md")
