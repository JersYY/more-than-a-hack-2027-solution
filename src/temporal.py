"""Temporal inference: per-frame evidence in, one state per frame out.

    states = decode(team, dead, context, config.temporal)    # (n,) int8 pm codes H/A/D/X

The ranking metric scores possessions as whole intervals (EVALUATION.md), so the decision is made
for a whole half at once, offline: the most likely sequence of states given all its evidence, with
a price on every change. A short burst of opponent evidence cannot pay for two changes of
possession; seconds of it can. Looking at the whole half also means a change can be placed where
the evidence starts, instead of wherever a causal filter would have finally noticed it.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import possession_metrics as pm  # noqa: E402

# Params
# Decoder state order. Index 0, 1, 2 -> possession_metrics codes; ties between changes go to the
# lower index, so D before H before A.
STATES = np.array([pm.D, pm.H, pm.A], dtype=np.int8)


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
        cost(H, D) = cost(A, D) = stop_cost
        cost(D, H) = cost(D, A) = restart_cost
        cost(x, x) = 0.

    So score(H) - score(A) = s * team_t: team evidence decides between the teams in full. But it
    lifts the better team over D by at most s * alive_bonus, because a player standing next to the
    ball is about as likely at a restart as in play -- dead-ball evidence decides alive vs D.
    (alive_bonus ~ ln 2 is what a mixture model gives: P(team evidence | D) = the average of its
    likelihood under H and under A.)

    Determinism (D-016): scores and costs are rounded to integers in units of 1/fixed_point and
    summed in int64, so the chosen path cannot depend on a CPU's last-bit rounding.

    Ties: of several equally good paths, the one whose changes come latest. Across a stretch with no
    evidence, a change of possession then happens where evidence resumes -- where the scorer's
    forward-fill of X would put it (EVALUATION.md).
    """
    team = np.asarray(team, dtype=np.float64)
    dead = np.asarray(dead, dtype=np.float64)
    scores = integer_scores(team, dead, config)
    unit = config.fixed_point
    stop, restart, turnover = (int(round(cost * unit)) for cost in
                               (config.stop_cost, config.restart_cost, config.turnover_cost))

    states = np.full(len(team), pm.X, dtype=np.int8)
    for _, first, last in context.blocks:
        block = slice(first, last + 1)
        path = _viterbi(scores[0, block].tolist(), scores[1, block].tolist(),
                        scores[2, block].tolist(), stop, restart, turnover)
        states[block] = STATES[path]
    return states


def integer_scores(team, dead, config):
    """(3, n) int64 per-frame scores of D, H, A in units of 1/fixed_point (see decode)."""
    scale = config.evidence_scale
    bonus = np.minimum(np.abs(team) / 2.0, config.alive_bonus)
    scores = np.vstack([scale * dead / 2.0,
                        scale * (-dead / 2.0 + bonus - np.maximum(-team, 0.0)),
                        scale * (-dead / 2.0 + bonus - np.maximum(team, 0.0))])
    return np.rint(scores * config.fixed_point).astype(np.int64)


def _viterbi(score_d, score_h, score_a, stop, restart, turnover):
    """State index path (0 D, 1 H, 2 A) of one block; Python ints throughout, so exact.

    The recursion is unrolled by hand: each state picks its best predecessor, a change of state
    beating staying put on an exact tie (the latest change), and D < H < A between two changes.
    """
    m = len(score_d)
    back = np.empty((m, 3), dtype=np.int8)
    d, h, a = score_d[0], score_h[0], score_a[0]
    for t in range(1, m):
        # into D: stay, or a stoppage from H or A
        best_d, from_d = d, 0
        value = h - stop
        if value >= best_d:
            best_d, from_d = value, 1
        value = a - stop
        if value > best_d or (value == best_d and from_d == 0):
            best_d, from_d = value, 2
        # into H: a restart from D, stay, or a turnover from A
        best_h, from_h = h, 1
        value = d - restart
        if value >= best_h:
            best_h, from_h = value, 0
        value = a - turnover
        if value > best_h or (value == best_h and from_h == 1):
            best_h, from_h = value, 2
        # into A: a restart from D, a turnover from H, or stay
        best_a, from_a = a, 2
        value = d - restart
        if value >= best_a:
            best_a, from_a = value, 0
        value = h - turnover
        if value > best_a or (value == best_a and from_a == 2):
            best_a, from_a = value, 1

        back[t, 0], back[t, 1], back[t, 2] = from_d, from_h, from_a
        d, h, a = best_d + score_d[t], best_h + score_h[t], best_a + score_a[t]

    state = 0
    if h > d:
        state = 1
    if a > (d if state == 0 else h):
        state = 2
    path = np.empty(m, dtype=np.int64)
    for t in range(m - 1, 0, -1):
        path[t] = state
        state = back[t, state]
    path[0] = state
    return path


def majority(codes, window):
    """(n,) int8 pm codes: official baseline step 2, exactly. Ticket T5; tests/test_temporal.py.

    codes -- (n,) nearest-team codes as src.features.nearest_team returns them (HOME, AWAY, -1).
    The result equals starter/baseline.py smooth(codes, window) on every frame, with its letters
    mapped to pm codes -- it calls that function rather than re-deriving it.
    """
    import baseline  # official starter/baseline.py, import only
    return pm.label_codes(baseline.smooth(np.asarray(codes), window))
