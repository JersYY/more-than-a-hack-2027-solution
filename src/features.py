"""Per-frame features computed from a Tracking.

build_features() returns the P0 feature table that docs/feature_spec.md defines -- names, units,
missing-value rules -- and that spec is its contract (ticket T2, tests/test_features_spec.py).

nearest_team() is step 1 of starter/baseline.py -- the team of the tracked object nearest the ball
-- computed from the arrays of src.data_loader instead of the official line-by-line parser. Same
pitch, same distance arithmetic in the same order, same tie rule, so on every frame it must give
the same code as the official parser does. That turns the official baseline files into an
end-to-end test of the loader: frame alignment, slot-to-team mapping, the metre scaling and the
NaN handling all have to be right for 199319 frames to agree (tests/test_alignment.py).

Neither is a possession model. Nearest is not control -- see docs/decisions.md D-012.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import GAME, GAMES  # noqa: E402
from src.data_loader import AWAY, HOME  # noqa: E402

# Params
NOBODY = -1     # starter/baseline.py's code for "no answer"; HOME and AWAY are the slot codes
CHUNK_FRAMES = 20000
# The P0 feature table, in docs/feature_spec.md order. P1 features join with their experiment.
FEATURES = ("ball_seen", "ball_x_m", "ball_y_m", "ball_speed_mps", "ball_out_m", "ball_still_s",
            "ball_missing_s", "d_home_m", "d_away_m", "n_home", "n_away", "team_speed_mps",
            "activity_mps")
SPEED_HALF_WINDOW = 2           # frames each side of the central difference behind ball_speed_mps
MAX_BALL_SPEED = 40.0           # m/s; faster is a tracker jump, not a kick, and becomes NaN


def build_features(tracking, context, config, pitch=None):
    """{name: (n,) array} for every name in FEATURES. Ticket T2; tests/test_features_spec.py.

    tracking -- src.data_loader.Tracking; context -- src.preprocess.FrameContext;
    config   -- src.config.Config (dead_ball.still_mps, dead_ball.activity_window_s);
    pitch    -- (length, width) in metres; default the games.py pitch. Other matches pass theirs.

    Definitions, units and missing-value rules are docs/feature_spec.md, the contract. In short:
    ball_speed_mps is the central difference over +/-SPEED_HALF_WINDOW frames, NaN unless the ball
    is seen at t - k, t and t + k, and NaN above MAX_BALL_SPEED. Distances
    are in metres on the games.py pitch, speeds in m/s, durations in seconds. Unobserved is NaN,
    never 0, except where the spec says otherwise. Only + - * / sqrt and cumulative sums in a fixed
    order are used (D-016).
    """
    length, width = pitch or GAMES[GAME]["pitch"]
    fps = GAMES[GAME]["fps"]
    n = tracking.n_frames

    ball_x = tracking.ball[:, 0] * length
    ball_y = tracking.ball[:, 1] * width
    seen = np.isfinite(ball_x) & np.isfinite(ball_y)
    features = {
        "ball_seen": seen,
        "ball_x_m": np.where(seen, ball_x, np.nan),
        "ball_y_m": np.where(seen, ball_y, np.nan),
        "ball_speed_mps": _ball_speed(ball_x, ball_y, seen, fps),
        "ball_out_m": np.where(seen, np.maximum.reduce(
            [np.zeros(n), -ball_x, ball_x - length, -ball_y, ball_y - width]), np.nan),
    }
    with np.errstate(invalid="ignore"):
        still = features["ball_speed_mps"] < config.dead_ball.still_mps   # NaN compares False
    features["ball_still_s"] = run_seconds(still, fps)
    features["ball_missing_s"] = run_seconds(~seen, fps)
    features.update(_team_distances(tracking, ball_x, ball_y, length, width))

    team_speed = _team_speed(tracking)
    features["team_speed_mps"] = team_speed
    features["activity_mps"] = centred_mean(
        team_speed, int(round(config.dead_ball.activity_window_s * fps)))
    return features


def _ball_speed(ball_x, ball_y, seen, fps):
    """Central-difference ball speed (m/s); NaN unless seen at t-k, t, t+k; NaN above the cap."""
    n, k = len(ball_x), SPEED_HALF_WINDOW
    speed = np.full(n, np.nan)
    if n > 2 * k:
        dx, dy = ball_x[2 * k:] - ball_x[:-2 * k], ball_y[2 * k:] - ball_y[:-2 * k]
        speed[k:n - k] = np.sqrt(dx * dx + dy * dy) * fps / (2.0 * k)
    speed[~seen] = np.nan
    with np.errstate(invalid="ignore"):
        speed[speed > MAX_BALL_SPEED] = np.nan
    return speed


def _team_distances(tracking, ball_x, ball_y, length, width, chunk=CHUNK_FRAMES):
    """d_home_m, d_away_m (nearest tracked slot to the ball, metres) and n_home, n_away."""
    n = tracking.n_frames
    team_of_slot = tracking.layout.team_of_slot
    out = {"d_home_m": np.full(n, np.nan), "d_away_m": np.full(n, np.nan),
           "n_home": np.zeros(n, dtype=np.int16), "n_away": np.zeros(n, dtype=np.int16)}
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        x = tracking.players[start:stop, :, 0] * length
        y = tracking.players[start:stop, :, 1] * width
        tracked = np.isfinite(x) & np.isfinite(y)
        dx = x - ball_x[start:stop, None]
        dy = y - ball_y[start:stop, None]
        squared = dx * dx + dy * dy                      # NaN where the slot or the ball is missing
        for name, code in (("home", HOME), ("away", AWAY)):
            mine = team_of_slot == code
            out["n_" + name][start:stop] = tracked[:, mine].sum(axis=1)
            nearest = np.where(np.isfinite(squared[:, mine]), squared[:, mine], np.inf).min(
                axis=1, initial=np.inf)
            found = np.isfinite(nearest)
            out["d_{}_m".format(name)][start:stop][found] = np.sqrt(nearest[found])
    return out


def _team_speed(tracking):
    """Mean speed channel over tracked slots, accumulated slot by slot (a fixed order, D-016)."""
    n = tracking.n_frames
    total, count = np.zeros(n), np.zeros(n)
    for slot in range(tracking.layout.n_slots):
        x, y, s = (tracking.players[:, slot, channel] for channel in range(3))
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
        total += np.where(valid, s, 0.0)
        count += valid
    speed = np.full(n, np.nan)
    np.divide(total, count, out=speed, where=count > 0)
    return speed


def run_seconds(mask, fps):
    """Each frame of a run of True gets that run's length in seconds; False frames get 0."""
    out = np.zeros(len(mask))
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    lengths = stops - starts
    out[np.asarray(mask, dtype=bool)] = np.repeat(lengths, lengths) / float(fps)
    return out


def centred_mean(values, half_width):
    """Mean of the finite values within +/-half_width frames, clipped to the array; NaN if none.

    Cumulative sums run in one fixed order, so the result does not depend on how a CPU would
    reorder a reduction (D-016).
    """
    n = len(values)
    finite = np.isfinite(values)
    sums = np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(n)
    low, high = np.maximum(0, index - half_width), np.minimum(n, index + half_width + 1)
    count = counts[high] - counts[low]
    mean = np.full(n, np.nan)
    np.divide(sums[high] - sums[low], count, out=mean, where=count > 0)
    return mean


def nearest_team(tracking, pitch=None, chunk=CHUNK_FRAMES):
    """(n,) int8: team code of the tracked slot nearest the ball, NOBODY where there is none.

    Distances are squared, in metres, as dx * dx + dy * dy with dx = x * length - ball_x * length
    -- the baseline's own arithmetic, so float64 results match it bit for bit. The first slot wins
    an exact tie, as the baseline's strict `<` does. A frame without a ball, or without any tracked
    slot, is NOBODY.
    """
    length, width = pitch or GAMES[GAME]["pitch"]
    team_of_slot = tracking.layout.team_of_slot
    out = np.full(tracking.n_frames, NOBODY, dtype=np.int8)

    for start in range(0, tracking.n_frames, chunk):
        stop = min(start + chunk, tracking.n_frames)
        ball_x = tracking.ball[start:stop, 0] * length
        ball_y = tracking.ball[start:stop, 1] * width
        dx = tracking.players[start:stop, :, 0] * length - ball_x[:, None]
        dy = tracking.players[start:stop, :, 1] * width - ball_y[:, None]
        distance = dx * dx + dy * dy                 # NaN wherever the slot or the ball is missing
        candidate = np.isfinite(distance)
        best = np.argmin(np.where(candidate, distance, np.inf), axis=1)
        out[start:stop] = np.where(candidate.any(axis=1), team_of_slot[best], NOBODY)
    return out
