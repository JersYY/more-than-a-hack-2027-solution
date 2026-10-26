"""Per-frame features computed from a Tracking. Phase 1 has exactly one, and it is a check.

nearest_team() is step 1 of starter/baseline.py -- the team of the tracked object nearest the ball
-- computed from the arrays of src.data_loader instead of the official line-by-line parser. Same
pitch, same distance arithmetic in the same order, same tie rule, so on every frame it must give
the same code as the official parser does. That turns the official baseline files into an
end-to-end test of the loader: frame alignment, slot-to-team mapping, the metre scaling and the
NaN handling all have to be right for 199319 frames to agree (tests/test_alignment.py).

It is not a possession model. Nearest is not control -- see docs/decisions.md.
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
