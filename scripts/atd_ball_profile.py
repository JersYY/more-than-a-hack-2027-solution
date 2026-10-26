"""Measure how the tactical ATD loses the ball, as an aggregate profile for src/degrade.py.

    $env:MTAH_DATA_DIR = "..."
    python scripts/atd_ball_profile.py          # -> configs/atd_ball_runs.json

Over the graded frames, the ball alternates between runs of frames where the tracking sees it and
runs where it does not. This writes the two run-length *histograms* -- counts per length bin, no
order, no frame numbers -- and the visible share. That is enough for src/degrade.py to delete the
ball from clean benchmark tracking in the same rhythm, and too little to reconstruct anything
about the match: a multiset of run lengths cannot say which frames they were (D-002).
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.data_loader import load_tracking  # noqa: E402
from src.preprocess import frame_context  # noqa: E402

# Params
OUTPUT = os.path.join(ROOT, "configs", "atd_ball_runs.json")
# Run lengths in frames, log-spaced bin edges: 1 frame (0.04 s) to 3,000 frames (2 min).
EDGES = [1, 2, 3, 4, 6, 8, 12, 16, 25, 35, 50, 75, 100, 150, 200, 300, 450, 700, 1000, 1500, 3000,
         100000]


def runs(mask):
    """Lengths of the maximal runs of True in mask."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def main():
    tracking = load_tracking()
    context = frame_context(tracking.n_frames, tracking=tracking)
    seen = tracking.ball_seen()
    seen_runs, missing_runs = [], []
    for _, first, last in context.blocks:
        block = seen[first:last + 1]
        seen_runs.append(runs(block))
        missing_runs.append(runs(~block))
    seen_runs, missing_runs = np.concatenate(seen_runs), np.concatenate(missing_runs)
    graded = context.graded

    profile = {
        "source": "tactical ATD, graded frames; aggregate run-length histograms (scripts/atd_ball_profile.py)",
        "fps": 25,
        "visible_share": round(float(seen[graded].mean()), 4),
        "edges_frames": EDGES,
        "seen_runs": np.histogram(seen_runs, bins=EDGES)[0].tolist(),
        "missing_runs": np.histogram(missing_runs, bins=EDGES)[0].tolist(),
    }
    with open(OUTPUT, "w", newline="\n") as handle:
        json.dump(profile, handle, indent=1)

    for name, values in (("seen", seen_runs), ("missing", missing_runs)):
        print("{:<8} runs {:>5}  median {:>6.1f} frames  p90 {:>7.1f}  mean {:>6.1f}".format(
            name, len(values), np.median(values), np.percentile(values, 90), values.mean()))
    print("visible share {:.1%} of graded frames -> wrote {}".format(
        profile["visible_share"], os.path.relpath(OUTPUT, ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
