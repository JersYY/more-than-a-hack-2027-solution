"""Frame context: which frames are graded, which half each belongs to, which way each team attacks.

    context = frame_context(tracking.n_frames, tracking=tracking)              # the real match
    context = frame_context(match.tracking.n_frames, match.blocks, match.tracking)   # synthetic

Everything downstream works inside one half and never across two -- possession_metrics.valid_runs
does the same -- so the half of every frame is decided once, here, from the published blocks
(games.half_video_blocks). No label, kickoff frame or other game-specific constant enters here.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import games  # noqa: E402
from src.data_loader import AWAY, HOME  # noqa: E402


@dataclass(frozen=True, eq=False)
class FrameContext:
    n_frames: int
    blocks: tuple                # ((half, first, last), ...) graded video frames, inclusive, in order
    half: np.ndarray             # (n,) int8: the half of each frame; 0 outside every block
    attack_sign: dict            # {(half, team): +1 if team attacks towards x = 1 that half, else -1}

    @property
    def graded(self):
        return self.half > 0


def frame_context(n_frames, blocks=None, tracking=None):
    """The FrameContext of an n_frames-long recording. Ticket T1; tests/test_preprocess.py.

    blocks   -- [(half, first, last)], default games.half_video_blocks(GAME). ValueError if they
                overlap, are out of order, or fall outside 0..n_frames-1.
    tracking -- optional. When given, the attack direction of each half is measured: the team whose
                tracked players have the lower median x over the half's frames defends x = 0, so it
                attacks with sign +1, and the other team -1. Without tracking, or with nobody
                tracked in a half: home +1 in the first half, and the signs swap every half.
    """
    if blocks is None:
        blocks = games.half_video_blocks(games.GAME)
    blocks = tuple((int(half), int(first), int(last)) for half, first, last in blocks)

    previous_last = -1
    for half, first, last in blocks:
        if not 0 <= first <= last < n_frames:
            raise ValueError("block {} is empty or outside frames 0..{}".format(
                (half, first, last), n_frames - 1))
        if first <= previous_last:
            raise ValueError("blocks overlap or are out of order: {}".format(blocks))
        previous_last = last

    frame_half = np.zeros(n_frames, dtype=np.int8)
    attack_sign = {}
    for half, first, last in blocks:
        frame_half[first:last + 1] = half
        home_sign = 1 if half % 2 == 1 else -1
        if tracking is not None:
            measured = _home_attack_sign(tracking, first, last)
            if measured is not None:
                home_sign = measured
        attack_sign[(half, HOME)] = home_sign
        attack_sign[(half, AWAY)] = -home_sign
    return FrameContext(n_frames, blocks, frame_half, attack_sign)


def _home_attack_sign(tracking, first, last):
    """+1 if home's tracked players sit at lower x than away's over the frames, -1 if higher.

    None when either team has no tracked position in the span or the medians are equal. A team
    spends most of a half in front of its own goal, so its median x says which goal it defends
    even when it dominates possession.
    """
    x = tracking.players[first:last + 1, :, 0]
    team = tracking.layout.team_of_slot
    home, away = x[:, team == HOME], x[:, team == AWAY]
    if not (np.isfinite(home).any() and np.isfinite(away).any()):
        return None
    home_median, away_median = np.nanmedian(home), np.nanmedian(away)
    if home_median == away_median:
        return None
    return 1 if home_median < away_median else -1
