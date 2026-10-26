"""What the official scorer rewards, pinned on synthetic label sequences.

EVALUATION.md describes the metric and evaluation/possession_metrics.py defines it. These tests turn
the behaviours our design will lean on into executable facts, on made-up arrays -- the public
ground truth is never involved -- so a design argument in docs/ can point at a test instead of at
a paragraph.
"""

import numpy as np

import possession_metrics as pm


def codes(spec):
    """'H*100 D*20 A*50' -> int8 label codes."""
    out = []
    for part in spec.split():
        label, count = part.split("*")
        out += [pm.LABELS.index(label)] * int(count)
    return np.array(out, dtype=np.int8)


# One graded run framed by ungraded X, three possessions with a stoppage between the first two.
GT = codes("X*10 H*300 D*50 A*200 H*250 X*20")
# Ten alternating ten-second possessions with stoppages: long enough that a one-frame offset is
# irrelevant, many enough that no single prediction can overlap one of them by IoU >= 0.3.
MATCH = codes(" ".join("H*250 D*25 A*250 D*25" for _ in range(5)))


def test_perfect_prediction_scores_one():
    result = pm.evaluate(GT, GT.copy())
    assert result["spell_f1_mean"] == 1.0 and result["macro_f1"] == 1.0


def test_constant_guess_scores_zero_spell_f1():
    for label in (pm.H, pm.A, pm.D):
        assert pm.evaluate(MATCH, np.full_like(MATCH, label))["spell_f1_mean"] == 0.0


def test_dead_ball_splits_a_possession_in_two():
    # CHALLENGE.md: H H D D H H is two possessions with a stoppage, not one.
    spells = pm.possession_spells(codes("H*100 D*20 H*100"), [(0, 220)])
    assert len(spells) == 2


def test_abstaining_inside_a_possession_is_free_on_spell_f1_only():
    pred = GT.copy()
    pred[100:150] = pm.X                      # inside the first H possession
    result = pm.evaluate(GT, pred)
    assert result["spell_f1_mean"] == 1.0     # filled with the previous label: H
    assert result["macro_f1"] < 1.0           # but charged as a miss frame by frame


def test_abstaining_across_a_handover_carries_the_old_label_over_it():
    pred = GT.copy()
    pred[310:460] = pm.X                      # the whole stoppage and half of the A possession
    result = pm.evaluate(GT, pred)
    assert result["spell_f1_mean"] < 1.0      # H is filled through D and into A


def test_abstention_fill_never_crosses_an_ungraded_gap():
    gt = codes("H*100 X*50 A*100")            # two graded runs, like the two halves
    pred = codes("H*100 X*150")               # silent on the whole second run
    filled = pm.fill_abstentions(pred, pm.valid_runs(gt))
    assert (filled[150:] == pm.X).all()


def test_one_frame_late_is_free():
    # EVALUATION.md: "perfect but 1 frame late: 1.0000".
    gt = codes("X*5 " + " ".join("H*250 A*250" for _ in range(10)) + " X*5")
    late = np.concatenate(([pm.X], gt[:-1])).astype(np.int8)
    assert pm.evaluate(gt, late)["spell_f1_mean"] == 1.0


def test_jitter_costs_spell_f1_far_more_than_macro_f1():
    # One wrong frame in fifty barely moves macro-F1 and destroys every possession.
    pred = MATCH.copy()
    flips = np.flatnonzero(MATCH != pm.D)[::50]
    pred[flips] = np.where(MATCH[flips] == pm.H, pm.A, pm.H)
    result = pm.evaluate(MATCH, pred)
    assert result["macro_f1"] > 0.95
    assert result["spell_f1_mean"] == 0.0
