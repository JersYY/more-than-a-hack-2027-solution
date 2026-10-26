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

# Params
NOBODY = -1     # starter/baseline.py's code for "no answer"; HOME and AWAY are the slot codes
CHUNK_FRAMES = 20000
# The P0 feature table, in docs/feature_spec.md order. P1 features join with their experiment.
FEATURES = ("ball_seen", "ball_x_m", "ball_y_m", "ball_speed_mps", "ball_out_m", "ball_still_s",
            "ball_missing_s", "d_home_m", "d_away_m", "n_home", "n_away", "team_speed_mps",
            "activity_mps")
SPEED_HALF_WINDOW = 2           # frames each side of the central difference behind ball_speed_mps
MAX_BALL_SPEED = 40.0           # m/s; faster is a tracker jump, not a kick, and becomes NaN


def build_features(tracking, context, config):
    """{name: (n,) array} for every name in FEATURES. Ticket T2; tests/test_features_spec.py.

    tracking -- src.data_loader.Tracking; context -- src.preprocess.FrameContext;
    config   -- src.config.Config (dead_ball.still_mps, dead_ball.activity_window_s).

    Definitions, units and missing-value rules are docs/feature_spec.md, the contract. In short:
    ball_speed_mps is the central difference over +/-SPEED_HALF_WINDOW frames, NaN unless the ball
    is seen at t - k, t and t + k, and NaN above MAX_BALL_SPEED. Distances
    are in metres on the games.py pitch, speeds in m/s, durations in seconds. Unobserved is NaN,
    never 0, except where the spec says otherwise. Only + - * / sqrt and cumulative sums in a fixed
    order are used (D-016).
    """
    raise NotImplementedError("Phase 3, ticket T2 -- docs/tickets.md")


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
