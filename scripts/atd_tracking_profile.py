"""Measure how the tactical ATD smooths and loses what it tracks, for src/degrade.py (E7).

    $env:MTAH_DATA_DIR = "..."
    python scripts/atd_tracking_profile.py          # -> configs/atd_tracking_profile.json

Aggregates only, over the graded frames (D-002): autocorrelations and percentiles of acceleration,
per-team counts of tracked players, the share of frames the goalkeeper's slot is empty, and the
mean length of a player's gap. None of it can say which frame anything happened on.

The two smoothing widths are then fitted on the other-match benchmark: the Gaussian that makes its
raw ball (live frames) and player tracks show the ATD's acceleration statistics. The benchmark has
to be on disk for that (scripts/omb_fetch.py).
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import possession_metrics as pm  # noqa: E402
from src.data_loader import AWAY, HOME, load_tracking  # noqa: E402
from src.degrade import goalkeepers, smooth_runs  # noqa: E402
from src.omb import benchmark_matches, load_benchmark  # noqa: E402
from src.preprocess import frame_context  # noqa: E402

# Params
OUTPUT = os.path.join(ROOT, "configs", "atd_tracking_profile.json")
BALL_LAGS = (1, 2, 4, 8, 16)            # frames; acceleration autocorrelation of the ball
PLAYER_LAG = 4                          # beyond the reach of white measurement noise (lag <= 2)
PERCENTILES = (50, 90, 99)
BALL_SIGMAS = (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0)
PLAYER_SIGMAS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
FIT_MATCHES = 3                         # benchmark matches used for the fit (the first, sorted)
MIN_SLOT_FRAMES = 2000                  # slots tracked less than this in a half are skipped
GAP_RANGE = (25, 1500)                  # frames: a lost player, not a substitution or relabelling
INDIVIDUAL_COUNTS = (7, 10)             # tracked per team: a few players lost, not a broad failure
OUTFIELD = 10
ATD_PITCH = (105.0, 68.0)               # metadata FieldSize; only shapes, not the model, use it


def acceleration(x, mask):
    """Second difference (per frame^2, in x's units) and the frames where it is defined."""
    second = np.diff(x, 2)
    defined = mask[2:] & mask[1:-1] & mask[:-2] & np.isfinite(second)
    return second, defined


def autocorrelation(series, defined, lags):
    out = []
    for lag in lags:
        both = defined[lag:] & defined[:-lag]
        u, v = series[lag:][both], series[:-lag][both]
        u, v = u - u.mean(), v - v.mean()
        out.append(float(np.sum(u * v) / np.sqrt(np.sum(u * u) * np.sum(v * v))))
    return out


def ball_statistics(x, y, mask, fps=25.0):
    ax, defined = acceleration(x, mask)
    ay, _ = acceleration(y, mask)
    magnitude = np.sqrt(ax * ax + ay * ay)[defined] * fps * fps
    return {"acf": autocorrelation(ax, defined, BALL_LAGS),
            "acc_mps2": [float(np.percentile(magnitude, q)) for q in PERCENTILES]}


def player_acf(players, length, mask, sigma=0.0):
    values = []
    for slot in range(players.shape[1]):
        x = players[:, slot, 0] * length
        tracked = mask & np.isfinite(x)
        if tracked.sum() < MIN_SLOT_FRAMES:
            continue
        if sigma:
            x = smooth_runs(np.where(tracked, x, np.nan), sigma)
        second, defined = acceleration(x, tracked)
        values.append(autocorrelation(second, defined, (PLAYER_LAG,))[0])
    return float(np.median(values))


def ball_distance(stats, target):
    """How far a smoothed benchmark ball is from the ATD's: autocorrelations, plus log ratios of
    the acceleration percentiles."""
    acf = np.subtract(stats["acf"], target["acf"])
    acc = np.log(np.divide(stats["acc_mps2"], target["acc_mps2"]))
    return float(np.sum(acf * acf) + np.sum(acc * acc))


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def measure_atd():
    tracking = load_tracking()
    context = frame_context(tracking.n_frames, tracking=tracking)
    graded = context.graded
    length, width = ATD_PITCH
    ball = ball_statistics(tracking.ball[:, 0] * length, tracking.ball[:, 1] * width, graded)
    tracked = tracking.tracked()
    team_of_slot = np.asarray(tracking.layout.team_of_slot)

    counts, individual = {}, []
    for name, code in (("home", HOME), ("away", AWAY)):
        per_frame = tracked[:, team_of_slot == code].sum(axis=1)[graded]
        counts[name] = (np.bincount(per_frame, minlength=14)[:14] / float(len(per_frame))).tolist()
        low, high = INDIVIDUAL_COUNTS
        lost = np.where((per_frame >= low) & (per_frame <= high), OUTFIELD + 1 - per_frame, 0)
        individual.append(float(lost.mean()))

    keeper_missing, gaps = [], []
    for _, first, last in context.blocks:
        block_tracked = tracked[first:last + 1]
        for slot in goalkeepers(tracking.players, team_of_slot, first, last).values():
            keeper_missing.append(float(1.0 - block_tracked[:, slot].mean()))
        for slot in range(block_tracked.shape[1]):
            column = block_tracked[:, slot]
            seen = np.flatnonzero(column)
            if len(seen) < MIN_SLOT_FRAMES:
                continue
            inner = ~column[seen[0]:seen[-1] + 1]
            gaps.extend(runs(inner).tolist())
    gaps = np.array(gaps)
    gaps = gaps[(gaps >= GAP_RANGE[0]) & (gaps <= GAP_RANGE[1])]

    keeper = float(np.mean(keeper_missing))
    outfield = max(0.0, (float(np.mean(individual)) - keeper) / OUTFIELD)
    return {
        "ball": ball,
        "player_acf_lag{}".format(PLAYER_LAG): player_acf(tracking.players, length, graded),
        "tracked_per_team_share_0_to_13": counts,
        "individual_loss_per_team": float(np.mean(individual)),
        "goalkeeper_missing_share": keeper,
        "outfield_missing_share": outfield,
        "player_gap_frames_mean": float(gaps.mean()),
        "player_gaps_counted": int(len(gaps)),
    }


def fit_sigmas(atd):
    matches = benchmark_matches()[:FIT_MATCHES]
    if not matches:
        raise SystemExit("no benchmark on disk: python scripts/omb_fetch.py")
    ball_scores = {sigma: [] for sigma in BALL_SIGMAS}
    player_scores = {sigma: [] for sigma in PLAYER_SIGMAS}
    target = atd["player_acf_lag{}".format(PLAYER_LAG)]
    for name in matches:
        match = load_benchmark(name)
        length, width = match.pitch
        alive = (match.labels == pm.H) | (match.labels == pm.A)
        graded = match.labels != pm.X
        for sigma in BALL_SIGMAS:
            x = smooth_runs(match.tracking.ball[:, 0] * length, sigma)
            y = smooth_runs(match.tracking.ball[:, 1] * width, sigma)
            ball_scores[sigma].append(ball_distance(ball_statistics(x, y, alive), atd["ball"]))
        for sigma in PLAYER_SIGMAS:
            value = player_acf(match.tracking.players, length, graded, sigma)
            player_scores[sigma].append((value - target) ** 2)
    ball_sigma = min(BALL_SIGMAS, key=lambda s: np.mean(ball_scores[s]))
    player_sigma = min(PLAYER_SIGMAS, key=lambda s: np.mean(player_scores[s]))
    return ball_sigma, player_sigma, matches


def main():
    atd = measure_atd()
    ball_sigma, player_sigma, matches = fit_sigmas(atd)
    profile = {
        "source": "tactical ATD, graded frames, aggregates only; smoothing widths fitted on the "
                  "other-match benchmark (scripts/atd_tracking_profile.py, E7)",
        "ball_sigma_frames": ball_sigma,
        "player_sigma_frames": player_sigma,
        "goalkeeper_missing_share": round(atd["goalkeeper_missing_share"], 3),
        "outfield_missing_share": round(atd["outfield_missing_share"], 3),
        "missing_run_frames": int(round(atd["player_gap_frames_mean"])),
        "min_missing_frames": GAP_RANGE[0],
        "fitted_on": matches,
        "measured": json.loads(json.dumps(atd), parse_float=lambda v: round(float(v), 3)),
    }
    with open(OUTPUT, "w", newline="\n") as handle:
        json.dump(profile, handle, indent=1)
    print("ball: acceleration acf {} | p50/p90/p99 {} m/s2".format(
        [round(v, 2) for v in atd["ball"]["acf"]], [round(v, 1) for v in atd["ball"]["acc_mps2"]]))
    print("ball sigma {} frames, player sigma {} frames (fitted on {})".format(
        ball_sigma, player_sigma, ", ".join(matches)))
    print("goalkeeper missing {:.3f}, outfield missing {:.3f}, gap mean {:.0f} frames ({} gaps)".format(
        atd["goalkeeper_missing_share"], atd["outfield_missing_share"],
        atd["player_gap_frames_mean"], atd["player_gaps_counted"]))
    print("wrote", os.path.relpath(OUTPUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
