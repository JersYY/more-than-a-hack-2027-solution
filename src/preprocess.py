"""Frame context: which frames are graded, which half each belongs to, which way each team attacks.

    context = frame_context(tracking.n_frames, tracking=tracking)              # the real match
    context = frame_context(match.tracking.n_frames, match.blocks, match.tracking)   # synthetic

Everything downstream works inside one half and never across two -- possession_metrics.valid_runs
does the same -- so the half of every frame is decided once, here, from the published blocks
(games.half_video_blocks). No label, kickoff frame or other game-specific constant enters here.
"""

from dataclasses import dataclass

import numpy as np


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
    raise NotImplementedError("Phase 3, ticket T1 -- docs/tickets.md")
