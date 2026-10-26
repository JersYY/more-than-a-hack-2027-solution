"""Make clean benchmark tracking lose the ball the way the tactical ATD loses it. Ticket T8.

    from src.degrade import load_profile, degrade_ball
    noisy = degrade_ball(match, load_profile(), mode="renewal", seed=0)

The benchmark's tracking (DFL) sees the ball on nearly every frame; the ATD sees it on 59.3% of
graded frames, in long runs: ~150 missing runs with a median of 10.7 s, almost none of 2-5 s
(configs/atd_ball_runs.json, aggregates only -- scripts/atd_ball_profile.py). Calibrating on clean
tracking would tune for a problem we do not have, so every benchmark score is computed after the
ball is deleted in that rhythm.

Where the gaps fall matters as much as how long they are, and that part is not observable without
labels. Two modes bracket it:

- "renewal": seen and missing runs alternate, lengths drawn from the ATD histograms, regardless of
  what the ball is doing. If anything, harder than the ATD: it puts long gaps into live play.
- "dead_weighted": a share `dead_share` of the missing frames is placed in the benchmark's own
  dead-ball stretches -- each chosen stoppage hidden from up to 1 s after it starts to up to 1 s
  before it ends -- and the rest in live play, in ATD-length runs. On the ATD, players move at
  1.18 m/s on average while the ball is missing against 2.16 m/s while it is seen, which says
  most of its gaps are stoppages; docs/experiments.md derives the share from that.

Seeded and deterministic. Only the ball is degraded: players stay as the benchmark has them.
"""

import json
import os
import sys
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import possession_metrics as pm  # noqa: E402
from src.data_loader import Tracking  # noqa: E402

# Params
PROFILE = os.path.join(ROOT, "configs", "atd_ball_runs.json")
MODES = ("none", "renewal", "dead_weighted")
FRAMES_TO_LOSE = 25    # dead_weighted: up to 1 s after a stoppage starts / before it ends


def load_profile(path=PROFILE):
    with open(path) as handle:
        return json.load(handle)


def sample_lengths(histogram, edges, count, rng):
    """count run lengths (frames) from a histogram: a bin by its count, then uniform inside it."""
    weights = np.asarray(histogram, dtype=np.float64)
    bins = rng.choice(len(weights), size=count, p=weights / weights.sum())
    low, high = np.asarray(edges[:-1])[bins], np.asarray(edges[1:])[bins]
    return np.maximum(1, np.floor(low + rng.random_sample(count) * (high - low))).astype(np.int64)


def renewal_mask(n, profile, rng):
    """(n,) bool, True = ball missing: alternating seen/missing runs from the ATD histograms."""
    edges = profile["edges_frames"]
    missing = np.zeros(n, dtype=bool)
    position = 0
    hidden = rng.random_sample() > profile["visible_share"]
    while position < n:
        key = "missing_runs" if hidden else "seen_runs"
        length = int(sample_lengths(profile[key], edges, 1, rng)[0])
        if hidden:
            missing[position:position + length] = True
        position += length
        hidden = not hidden
    return missing


def dead_weighted_mask(labels, block, profile, dead_share, rng):
    """(len,) bool for one block: the ATD's missing share, dead_share of it inside D stretches."""
    first, last = block
    codes = labels[first:last + 1]
    n = len(codes)
    target = int(round((1.0 - profile["visible_share"]) * n))
    missing = np.zeros(n, dtype=bool)
    dead_runs = [(a - first, b - first) for a, b, code in pm.segments(labels, [(first, last + 1)])
                 if code == pm.D]
    edges = profile["edges_frames"]

    # Stoppages first, until the dead share is reached: the tracker loses the ball shortly after it
    # goes out of play and finds it again around the restart, so most of the stoppage is hidden.
    budget = int(round(dead_share * target))
    for index in rng.permutation(len(dead_runs)):
        hidden = int(missing.sum())
        if hidden >= budget:
            break
        start, stop = dead_runs[index]
        margin = max(1, min(FRAMES_TO_LOSE, (stop - start) // 4))
        first_hidden = start + rng.randint(0, margin)
        last_hidden = max(first_hidden, stop - rng.randint(0, margin))
        last_hidden = min(last_hidden, first_hidden + budget - hidden)
        missing[first_hidden:last_hidden] = True
    # The rest in live play, in ATD-length runs at random places.
    live = codes != pm.D
    attempts = 0
    while missing.sum() < target and attempts < 10000:
        attempts += 1
        length = int(sample_lengths(profile["missing_runs"], edges, 1, rng)[0])
        start = rng.randint(0, n)
        span = slice(start, min(n, start + length))
        if live[span].mean() > 0.5:
            missing[span] = True
    return missing


def degrade_ball(match, profile=None, mode="renewal", seed=0, dead_share=0.85):
    """A copy of an OtherMatch (src/omb.py) with the ball deleted; players untouched."""
    if mode not in MODES:
        raise ValueError("mode must be one of {}".format(MODES))
    if mode == "none":
        return match
    profile = profile or load_profile()
    rng = np.random.RandomState(seed)
    ball = match.tracking.ball.copy()
    for _, first, last in match.blocks:
        if mode == "renewal":
            hide = renewal_mask(last - first + 1, profile, rng)
        else:
            hide = dead_weighted_mask(match.labels, (first, last), profile, dead_share, rng)
        ball[first:last + 1][hide] = np.nan
    tracking = Tracking(match.tracking.source, match.tracking.layout, match.tracking.frame_id,
                        match.tracking.players, ball)
    return replace(match, tracking=tracking)
