"""Score the official baseline -- and any configuration -- on the synthetic scenario catalogue.

    python scripts/synthetic_bench.py                                 # official baseline only
    python scripts/synthetic_bench.py --config configs/hybrid_v1.toml # plus a configuration

Synthetic labels are known by construction (src/synthetic.py), so these numbers come from a real
answer key -- for made-up football. They show which behaviours a method has, not how it will score
on the match: a method can pass every scenario and still lose on real tracking. No match data and
no ground truth are read.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _path in (ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import baseline  # noqa: E402  (official starter/baseline.py, import only)
import possession_metrics as pm  # noqa: E402
from src.features import nearest_team  # noqa: E402
from src.synthetic import SCENARIOS  # noqa: E402

# Params
NAIVE_WINDOWS = (0, 50, 125)


def naive(match, window):
    """The official baseline on a synthetic match: nearest team, then the official majority vote."""
    return pm.label_codes(baseline.smooth(nearest_team(match.tracking), window))


def score(match, codes):
    result = pm.evaluate(match.labels, codes)
    return result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"], result["n_gt_spells"]


def candidate(match, config):
    """A configuration's labels, or None while src.predict is not implemented (Phase 3)."""
    from src import predict, preprocess
    try:
        context = preprocess.frame_context(match.tracking.n_frames, match.blocks, match.tracking)
        return pm.label_codes(predict.predict(match.tracking, context, config))
    except NotImplementedError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="also score this configuration")
    args = parser.parse_args()
    config = None
    if args.config:
        from src.config import load
        config = load(args.config)

    columns = ["raw" if w == 0 else "smooth{}".format(w) for w in NAIVE_WINDOWS]
    if config is not None:
        columns.append(config.run.name)
    print("{:<22} {:>4}  {}".format("scenario", "gt", "  ".join(
        "{:>17}".format(c) for c in columns)))
    print("{:<22} {:>4}  {}".format("", "poss", "  ".join(
        "{:>17}".format("spellF1 macro  n") for _ in columns)))

    totals = {c: [] for c in columns}
    for name, build in SCENARIOS.items():
        match = build()
        cells = []
        for column, window in zip(columns, NAIVE_WINDOWS):
            spell, macro, n_pred, n_gt = score(match, naive(match, window))
            totals[column].append(spell)
            cells.append("{:>7.3f} {:>5.3f} {:>3}".format(spell, macro, n_pred))
        if config is not None:
            codes = candidate(match, config)
            if codes is None:
                cells.append("{:>17}".format("not implemented"))
            else:
                spell, macro, n_pred, n_gt = score(match, codes)
                totals[config.run.name].append(spell)
                cells.append("{:>7.3f} {:>5.3f} {:>3}".format(spell, macro, n_pred))
        n_gt = len(pm.possession_spells(match.labels, pm.valid_runs(match.labels)))
        print("{:<22} {:>4}  {}".format(name, n_gt, "  ".join(cells)))

    print("{:<22} {:>4}  {}".format("mean spellF1", "", "  ".join(
        "{:>7.3f}{:>10}".format(np.mean(v), "") if v else "{:>17}".format("-")
        for v in totals.values())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
