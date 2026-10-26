"""Synthetic matches with labels known by construction, for acceptance tests and experiments.

Nothing here reads the match data. A Script lists football situations in order; build() renders
them at 25 fps into the same Tracking class src.data_loader returns -- 11 + 11 track slots, the
ball, a speed channel -- together with the H/A/D/X label of every frame and the graded blocks.

The labels follow CHALLENGE.md as the script states them. Possession changes only at a controlled
reception (a pass that reaches an opponent) or at a restart; a brush or a duel never changes it;
the ball crossing a line, or a whistle, makes it dead from that frame until the restart. So a test
can name the behaviour it expects and why, and no expectation is learned from any match.

    s = Script(seed=0)
    s.kickoff("H", taker=9)          # first half: live from the kick
    s.carry(2.0)                     # the carrier dribbles towards the opponent goal
    s.pass_to(7, 1.0)                # a team-mate receives: possession continues
    s.brush(15, frames=4)            # an away player gets closer to the ball: no change
    s.pass_to(18, 1.2)               # an away player receives: A from the reception frame
    s.carry(2.0, hidden=True)        # the tracker loses the ball; the label does not change
    s.out(6.0)                       # over the touchline: D from the crossing frame
    s.restart("H", taker=3)          # throw-in: H from the throw
    s.pass_to(5, 1.0)
    match = s.build()                # .tracking  .labels  .blocks

Slots 0-10 are home and 11-21 away, numbered the same way within each team: 0 goalkeeper, 1-4
defenders, 5-8 midfielders, 9-10 forwards. Home attacks towards x = 1 in the first half.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES  # noqa: E402
from src.data_loader import AWAY, HOME, Layout, Tracking  # noqa: E402

# Params
FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]
N_SLOTS = 22
TEAM_OF_SLOT = np.array([HOME] * 11 + [AWAY] * 11, dtype=np.int8)
# Home shape in normalised coordinates, first half, attacking towards x = 1. Away mirrors it.
FORMATION = ((0.05, 0.50), (0.22, 0.15), (0.20, 0.38), (0.20, 0.62), (0.22, 0.85),
             (0.40, 0.15), (0.38, 0.38), (0.38, 0.62), (0.40, 0.85), (0.47, 0.40), (0.47, 0.60))
LIVE, RESTING = 1.0, 0.25        # how fast the drift clock runs: in play, and at a stoppage
DRIFT_M = (4.0, 3.0)             # amplitude of each player's wandering around the formation
CARRY_M = 0.7                    # the carrier runs this far behind the ball
BRUSH_M = 0.4                    # a brushing opponent gets this close, closer than the carrier
OUT_M = 2.0                      # how far beyond the touchline a ball that goes out ends up
MAX_SPEED = 9.0                  # m/s; the speed channel is clipped here, like a real tracker's
CODES = {"H": pm.H, "A": pm.A}


@dataclass(eq=False)
class SyntheticMatch:
    tracking: Tracking
    labels: np.ndarray           # (n,) int8 possession_metrics codes; X outside the blocks
    blocks: list                 # [(half, first, last)] graded frames, inclusive
    steps: list                  # the script, one string per step
    step_of_frame: np.ndarray    # (n,) int16 index into steps; -1 for padding frames

    def frames_of(self, prefix):
        """Frames rendered by steps whose description starts with prefix, e.g. 'brush'."""
        wanted = [index for index, step in enumerate(self.steps) if step.startswith(prefix)]
        return np.flatnonzero(np.isin(self.step_of_frame, wanted))


def frames(seconds):
    return int(round(seconds * FPS))


def make_tracking(players, ball, team_of_slot):
    """A Tracking from explicit arrays: players (n, slots, 3) and ball (n, 3), normalised x, y and
    speed in m/s, NaN where unobserved. For tests that need a handful of hand-placed frames."""
    # Copies, not views: a test that edits one Tracking must never edit another, or TEAM_OF_SLOT.
    players = np.array(players, dtype=np.float64)
    ball = np.array(ball, dtype=np.float64)
    n, slots = players.shape[:2]
    layout = Layout(
        home_team_id="HOME", away_team_id="AWAY", home_name="Home", away_name="Away",
        slot_player_ids=tuple("S{}".format(slot) for slot in range(slots)),
        team_of_slot=np.array(team_of_slot, dtype=np.int8),
        units={"x": "normalized", "y": "normalized", "s": "m/s"},
        n_fields=1 + 3 * slots + 3, frame_column=0,
        player_columns=np.arange(1, 1 + 3 * slots).reshape(slots, 3),
        ball_columns=np.arange(1 + 3 * slots, 4 + 3 * slots),
        frame_rate=float(FPS), pitch_size=(105.0, 68.0), spec_frames=((1, n),),
        baseline_slot_order=True, baseline_home=True)
    return Tracking("synthetic", layout, np.arange(1, n + 1, dtype=np.int64), players, ball)


class Script(object):
    """Football situations in order. Every method appends frames; build() renders them."""

    def __init__(self, seed=0, lead_in_s=1.0):
        rng = np.random.RandomState(seed)
        self._phase = rng.uniform(0.0, 2.0 * np.pi, (N_SLOTS, 2))
        self._omega = rng.uniform(0.35, 0.65, (N_SLOTS, 2))   # rad per second of drift clock
        self._frames = []        # (label, ball xy or None, drift clock, half, {slot: xy}, step)
        self._blocks = []
        self._block_start = None
        self._half = 1
        self._clock = 0.0
        self._ball = (0.5, 0.5)
        self._team = None        # pm.H or pm.A while the ball is live
        self._carrier = None
        self._stopped_at = 0     # first frame of the current stoppage
        self.steps = []
        self._ungraded(lead_in_s)

    # -- geometry ------------------------------------------------------------------------------

    def _formation(self, slot, half):
        x, y = FORMATION[slot % 11]
        if slot >= 11:
            x, y = 1.0 - x, 1.0 - y
        return (1.0 - x, 1.0 - y) if half == 2 else (x, y)

    def _drift(self, slot, clock, half):
        x, y = self._formation(slot, half)
        return (x + DRIFT_M[0] * np.sin(self._omega[slot, 0] * clock + self._phase[slot, 0]) / LENGTH,
                y + DRIFT_M[1] * np.sin(self._omega[slot, 1] * clock + self._phase[slot, 1]) / WIDTH)

    def _direction(self, team):
        """Unit vector, in metres, of the way `team` attacks in the current half."""
        sign = 1.0 if (team == pm.H) == (self._half == 1) else -1.0
        vector = np.array([sign, 0.15])
        return vector / np.hypot(*vector)

    def _behind(self, slot, distance_m):
        """Where `slot` stands to play the ball: distance_m behind it, facing its attack."""
        direction = self._direction(TEAM_OF_SLOT[slot])
        return (self._ball[0] - direction[0] * distance_m / LENGTH,
                self._ball[1] - direction[1] * distance_m / WIDTH)

    # -- frame bookkeeping ---------------------------------------------------------------------

    def _add(self, label, ball, activity, override=None, step=None):
        self._clock += activity / FPS
        self._frames.append((label, ball, self._clock, self._half, dict(override or {}),
                             len(self.steps) - 1 if step is None else step))

    def _ungraded(self, seconds):
        for _ in range(frames(seconds)):
            self._add(pm.X, None, RESTING, step=-1)

    def _open_block(self):
        self._block_start = len(self._frames)

    def _close_block(self):
        self._blocks.append((self._half, self._block_start, len(self._frames) - 1))
        self._block_start = None

    def _need_live(self, step):
        if self._team is None or self._carrier is None:
            raise ValueError("{} needs a live ball with a carrier; restart or kick off first".format(step))

    def _go_live(self, team, taker):
        if TEAM_OF_SLOT[taker] != team:
            raise ValueError("slot {} is not on team {}".format(taker, pm.LABELS[team]))
        self._team, self._carrier = team, taker

    def _stop(self):
        self._team, self._carrier = None, None
        self._stopped_at = len(self._frames)

    # -- live play -----------------------------------------------------------------------------

    def kickoff(self, team, taker, setup_s=0.0):
        """Ball at the centre spot; D for setup_s seconds, then live with `team` from the kick.

        setup_s = 0 is the first half of the real match, graded from the kick itself; the second
        half is graded from before its kickoff, so pass a setup there (FAQ.md).
        """
        self.steps.append("kickoff({}, {}, setup_s={})".format(team, taker, setup_s))
        if self._block_start is None:
            self._open_block()
        self._ball = (0.5, 0.5)
        self._stop()
        spot = self._behind(taker, 0.8)
        for _ in range(frames(setup_s)):
            self._add(pm.D, self._ball, RESTING, {taker: spot})
        self._go_live(CODES[team], taker)
        return self

    def _carry_frames(self, count, speed, hidden, extra=None):
        direction = self._direction(self._team)
        step = (direction[0] * speed / FPS / LENGTH, direction[1] * speed / FPS / WIDTH)
        for index in range(count):
            self._ball = (float(np.clip(self._ball[0] + step[0], 0.04, 0.96)),
                          float(np.clip(self._ball[1] + step[1], 0.04, 0.96)))
            override = {self._carrier: (self._ball[0] - direction[0] * CARRY_M / LENGTH,
                                        self._ball[1] - direction[1] * CARRY_M / WIDTH)}
            if extra is not None:
                override.update(extra(index, self._ball))
            self._add(self._team, None if hidden else self._ball, LIVE, override)

    def carry(self, seconds, speed=3.0, hidden=False):
        """The carrier dribbles towards the opponent goal; hidden = the tracker loses the ball."""
        self.steps.append("carry({}, hidden={})".format(seconds, hidden))
        self._need_live("carry")
        self._carry_frames(frames(seconds), speed, hidden)
        return self

    def brush(self, opponent, frames_close=4):
        """An opponent comes closer to the ball than the carrier for a few frames: no change."""
        self.steps.append("brush({}, {} frames)".format(opponent, frames_close))
        self._need_live("brush")
        if TEAM_OF_SLOT[opponent] == self._team:
            raise ValueError("brush needs an opponent")
        self._carry_frames(frames_close, 3.0, False,
                           lambda index, ball: {opponent: (ball[0], ball[1] + BRUSH_M / WIDTH)})
        return self

    def duel(self, opponent, seconds, period_s=0.8):
        """An opponent challenges for `seconds`, nearer and farther than the carrier in turn."""
        self.steps.append("duel({}, {})".format(opponent, seconds))
        self._need_live("duel")
        if TEAM_OF_SLOT[opponent] == self._team:
            raise ValueError("duel needs an opponent")

        def close(index, ball):
            distance = 0.65 + 0.45 * np.sin(2.0 * np.pi * index / (period_s * FPS))
            return {opponent: (ball[0], ball[1] + distance / WIDTH)}
        self._carry_frames(frames(seconds), 1.5, False, close)
        return self

    def pass_to(self, receiver, seconds, hidden=False):
        """The ball travels to `receiver`. If they are an opponent, possession changes on arrival.

        The ball arrives where the receiver's drift puts them on the frame after the pass, so the
        reception happens next to them. hidden = an aerial ball the tracker loses in flight.
        """
        self.steps.append("pass_to({}, {}, hidden={})".format(receiver, seconds, hidden))
        self._need_live("pass_to")
        count = frames(seconds)
        target = self._drift(receiver, self._clock + (count + 1) * LIVE / FPS, self._half)
        start = self._ball
        for index in range(count):
            fraction = (index + 1.0) / (count + 1.0)
            ball = (start[0] + fraction * (target[0] - start[0]),
                    start[1] + fraction * (target[1] - start[1]))
            self._add(self._team, None if hidden else ball, LIVE)
        self._ball = target
        self._team, self._carrier = TEAM_OF_SLOT[receiver], receiver
        return self

    # -- stoppages -----------------------------------------------------------------------------

    def out(self, seconds, hide_after_s=None, travel_s=0.6):
        """The ball is played over the nearer touchline: dead from the first frame beyond it.

        It comes to rest OUT_M outside and, if hide_after_s is given, the tracker loses it then.
        Players slow to walking pace over 1.5 s.
        """
        self.steps.append("out({}, hide_after_s={})".format(seconds, hide_after_s))
        self._need_live("out")
        start = self._ball
        target_y = -OUT_M / WIDTH if start[1] < 0.5 else 1.0 + OUT_M / WIDTH
        travel = frames(travel_s)
        if travel >= frames(seconds):
            raise ValueError("out needs more than {} s".format(travel_s))
        crossed = False
        for index in range(travel):
            ball = (start[0], start[1] + (index + 1.0) / travel * (target_y - start[1]))
            if not crossed and not 0.0 <= ball[1] <= 1.0:
                crossed = True
                self._stop()
            self._add(pm.D if crossed else self._team, ball, LIVE)
        self._ball = (start[0], target_y)
        self._rest(frames(seconds) - travel, hide_after_s)
        return self

    def whistle(self, seconds):
        """The referee stops play (a foul): dead from this frame, the ball stays where it is."""
        self.steps.append("whistle({})".format(seconds))
        self._need_live("whistle")
        self._stop()
        self._rest(frames(seconds), None)
        return self

    def _rest(self, count, hide_after_s):
        for index in range(count):
            hidden = hide_after_s is not None and index >= frames(hide_after_s)
            activity = max(RESTING, LIVE - (LIVE - RESTING) * (index + 1.0) / (1.5 * FPS))
            self._add(pm.D, None if hidden else self._ball, activity)

    def place(self, spot, seconds, hidden_s=1.0):
        """Dead ball fetched and placed at `spot` (normalised xy): hidden, then still there."""
        self.steps.append("place({}, {})".format(spot, seconds))
        if self._team is not None:
            raise ValueError("place needs a dead ball")
        self._ball = (float(spot[0]), float(spot[1]))
        for index in range(frames(seconds)):
            self._add(pm.D, None if index < frames(hidden_s) else self._ball, RESTING)
        return self

    def restart(self, team, taker, approach_s=1.5):
        """End the stoppage: `team` is in possession from the next frame, `taker` on the ball.

        The taker is moved next to the ball for the last approach_s seconds of the stoppage.
        """
        self.steps.append("restart({}, {})".format(team, taker))
        if self._team is not None:
            raise ValueError("restart needs a dead ball")
        spot = self._behind(taker, 0.8)
        first = max(self._stopped_at, len(self._frames) - frames(approach_s))
        for index in range(first, len(self._frames)):
            self._frames[index][4][taker] = spot
        self._go_live(CODES[team], taker)
        return self

    def new_half(self, break_s=4.0):
        """Close the graded block, add ungraded frames, open the next half (direction flips)."""
        self.steps.append("new_half({})".format(break_s))
        self._close_block()
        self._stop()
        self._ungraded(break_s)
        self._half += 1
        self._open_block()
        return self

    def __len__(self):
        return len(self._frames)

    # -- rendering -----------------------------------------------------------------------------

    def build(self, lead_out_s=1.0):
        if self._block_start is not None:
            self._close_block()
        self._ungraded(lead_out_s)
        n = len(self._frames)

        players = np.full((n, N_SLOTS, 3), np.nan)
        ball = np.full((n, 3), np.nan)
        labels = np.empty(n, dtype=np.int8)
        step_of_frame = np.empty(n, dtype=np.int16)
        for row, (label, ball_xy, clock, half, override, step) in enumerate(self._frames):
            for slot in range(N_SLOTS):
                players[row, slot, :2] = override.get(slot) or self._drift(slot, clock, half)
            if ball_xy is not None:
                ball[row, :2] = ball_xy
            labels[row] = label
            step_of_frame[row] = step

        # The speed channel, as a tracker reports it: displacement speed, smoothed over five
        # frames and clipped, so a carrier snapping back into the formation is not a 100 m/s spike.
        step = np.hypot(np.diff(players[:, :, 0], axis=0) * LENGTH,
                        np.diff(players[:, :, 1], axis=0) * WIDTH) * FPS
        speed = np.vstack([step[:1], step])
        kernel = np.ones(5) / 5.0
        for slot in range(N_SLOTS):
            speed[:, slot] = np.convolve(np.pad(speed[:, slot], 2, mode="edge"), kernel, "valid")
        players[:, :, 2] = np.minimum(speed, MAX_SPEED)

        return SyntheticMatch(make_tracking(players, ball, TEAM_OF_SLOT), labels, list(self._blocks),
                              list(self.steps), step_of_frame)


# -- the scenario catalogue -----------------------------------------------------------------------
#
# Each one isolates a behaviour CHALLENGE.md defines. docs/experiments.md lists what the official
# baseline scores on each (scripts/synthetic_bench.py) and what the MVP must score.

def stable_home(seed=0):
    """One home possession of passes and carries: one spell."""
    return (Script(seed).kickoff("H", 9).carry(2.0).pass_to(6, 1.0).carry(2.0).pass_to(2, 1.2)
            .carry(2.0).pass_to(7, 1.2).carry(2.0).build())


def turnover(seed=0):
    """Two interceptions: H, A, H with the changes at the receptions."""
    return (Script(seed).kickoff("H", 9).carry(2.0).pass_to(6, 1.0).carry(2.0).pass_to(17, 1.2)
            .carry(3.0).pass_to(16, 1.0).carry(2.0).pass_to(5, 1.2).carry(3.0).build())


def deflection(seed=0):
    """Away players brush the ball twice without control: still one home spell."""
    return (Script(seed).kickoff("H", 9).carry(2.0).brush(15, 4).carry(1.5).pass_to(6, 1.0)
            .carry(1.5).brush(16, 3).carry(2.0).pass_to(7, 1.0).carry(2.0).build())


def duel(seed=0):
    """A three-second challenge the carrier survives: still one home spell."""
    return (Script(seed).kickoff("H", 9).carry(2.0).pass_to(6, 1.0).duel(15, 3.0).carry(1.5)
            .pass_to(7, 1.0).carry(2.0).build())


def occlusion_same_team(seed=0):
    """The tracker loses the ball for 2.5 s mid-possession: still one home spell."""
    return (Script(seed).kickoff("H", 9).carry(2.0).pass_to(6, 1.0).carry(2.5, hidden=True)
            .carry(1.5).pass_to(7, 1.0).carry(2.0).build())


def occlusion_long(seed=0):
    """The ball is lost for 6 s mid-possession -- longer than a +/-2 s vote can bridge: one spell."""
    return (Script(seed).kickoff("H", 9).carry(2.0).pass_to(6, 1.0)
            .carry(6.0, speed=1.5, hidden=True).carry(1.5).pass_to(7, 1.0).carry(2.0).build())


def occlusion_turnover(seed=0):
    """An aerial ball lost in flight and won by the away side: H then A, changing on arrival."""
    return (Script(seed).kickoff("H", 9).carry(2.0).pass_to(6, 1.0).carry(1.5)
            .pass_to(17, 2.0, hidden=True).carry(3.0).pass_to(16, 1.0).carry(2.0).build())


def home_dead_home(seed=0):
    """Out for a throw-in that the same team takes: H, D, H -- two possessions, not one."""
    return (Script(seed).kickoff("H", 9).carry(2.0).pass_to(5, 1.0).carry(1.0).out(6.0)
            .restart("H", 5).pass_to(6, 1.0).carry(2.0).pass_to(2, 1.2).carry(2.0).build())


def away_dead_away(seed=0):
    """The mirror image for the away side, with the ball visible the whole stoppage: A, D, A."""
    return (Script(seed).kickoff("A", 20).carry(2.0).pass_to(17, 1.0).carry(1.0).out(6.0)
            .restart("A", 17).pass_to(16, 1.0).carry(2.0).pass_to(13, 1.2).carry(2.0).build())


def out_and_lost(seed=0):
    """Out, the tracker loses the ball, the other side throws in: H, D, A."""
    return (Script(seed).kickoff("H", 9).carry(2.0).pass_to(5, 1.0).carry(1.0)
            .out(7.0, hide_after_s=1.0).restart("A", 13).pass_to(16, 1.2).carry(2.0)
            .pass_to(17, 1.0).carry(2.0).build())


def foul_free_kick(seed=0):
    """A whistle in open play; the other team takes the free kick: H, D, A."""
    return (Script(seed).kickoff("H", 9).carry(2.0).pass_to(6, 1.0).carry(1.5).whistle(8.0)
            .restart("A", 16).pass_to(17, 1.2).carry(2.0).pass_to(13, 1.2).carry(2.0).build())


def second_half(seed=0):
    """A short first half, then the second, graded from before its kickoff: D first."""
    return (Script(seed).kickoff("H", 9).carry(2.0).pass_to(6, 1.0).carry(2.0).new_half(3.0)
            .kickoff("A", 20, setup_s=4.0).carry(2.0).pass_to(17, 1.0).carry(2.0).build())


SCENARIOS = {function.__name__: function for function in (
    stable_home, turnover, deflection, duel, occlusion_same_team, occlusion_long,
    occlusion_turnover, home_dead_home, away_dead_away, out_and_lost, foul_free_kick, second_half)}

# The bar: scenarios the official baseline (window +/-50) does not pass. Its D comes only from a
# missing ball, so it cannot represent a stoppage while the ball is visible, nor bridge an occlusion
# longer than its vote. tests/test_synthetic.py checks this list against the baseline itself.
BAR = ("occlusion_long", "home_dead_home", "away_dead_away", "out_and_lost", "foul_free_kick",
       "second_half")
