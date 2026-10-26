"""Read the provided Automatic Tracking Data (ATD) into numpy arrays, one row per video frame.

    from src.data_loader import load_tracking
    tracking = load_tracking()          # tactical -- the timeline a submission is indexed to
    tracking.players                    # (199319, 46, 3): x, y, s per track slot
    tracking.ball                       # (199319, 3):     x, y, s
    tracking.layout.team_of_slot        # (46,): 0 home, 1 away

Row i of every array is video frame i, and therefore submission row i. That is the one alignment
rule DATA.md gives for the ATD -- video_frame = frame_id - 1 -- and the loader proves it on the
file rather than assuming it: the frame ids must be exactly 1..n, in order, or loading fails.

Why not kloppy. kloppy reads this format and is the reference reader (DATA.md), but it builds a
Python object per slot per frame, and 199319 frames x 46 slots costs minutes and gigabytes
(starter/load_atd.py says the same). Here every EPTS separator becomes a comma and pandas' C
tokenizer parses the lines in seconds, with its exact float converter so every value is
bit-identical to Python's float(). tests/test_alignment.py holds the result to kloppy on frames
sampled across the match, and to the official baseline on every frame.

The column layout is read, not assumed. It comes from the DataFormatSpecification in the metadata
XML -- the element kloppy builds its regular expression from -- so the slot order is the order the
raw lines actually use. starter/baseline.py assumes instead that it is the order of the <Player>
elements and that the first <Team> is home; Layout records whether a file agrees with both, and
scripts/data_audit.py reports it.

Units are the file's own and are recorded, not converted: x and y are normalised 0-1, and the
metadata declares speed in "kph" although DATA.md says m/s -- see docs/data_audit.md for which one
the numbers support. Missing observations stay NaN, for players and ball alike. Nothing here
fills, interpolates or smooths: what a gap means is the model's question, not the loader's.
"""

import io
import itertools
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from games import DEFAULT_SOURCE, GAME, require, source_frames  # noqa: E402

# Params
HOME, AWAY = 0, 1
# Channels are stored in this order whatever order the file uses, so players[..., 0] is always x.
# Both groups carry a speed channel "s"; the ball's is the one kloppy drops (its EPTS reader keeps
# only the ball channels of the position sensor).
CHANNELS = ("x", "y", "s")
# Every separator a Metrica EPTS DataFormatSpecification uses. Mapping them all to one character
# turns a raw line into a flat CSV row whose fields are in the specification's depth-first order.
SEPARATORS = ":;"
# Lines handed to the tokenizer per call. Bounds peak memory to tens of megabytes on a ~200 MB file
# and has no effect on the result.
CHUNK_LINES = 20000
# Exact float parsing on every pandas version: from 3.1 one exact converter serves every mode and
# float_precision is deprecated; before 3.1 only "round_trip" matches Python's float() bit for bit.
_PANDAS = tuple(int(part) for part in pd.__version__.split(".")[:2] if part.isdigit())
EXACT_FLOATS = {} if _PANDAS >= (3, 1) else {"float_precision": "round_trip"}


@dataclass(frozen=True, eq=False)
class Layout:
    """Everything the metadata XML says about how to read the raw file."""
    home_team_id: str
    away_team_id: str
    home_name: str
    away_name: str
    slot_player_ids: tuple       # player id of each slot, in raw-line order
    team_of_slot: np.ndarray     # int8 per slot, HOME or AWAY
    units: dict                  # channel id -> unit, from the metadata's Sensors
    n_fields: int                # fields per raw line once every separator is a comma
    frame_column: int
    player_columns: np.ndarray   # (slots, 3): field index of x, y, s for each slot
    ball_columns: np.ndarray     # (3,): field index of the ball's x, y, s
    frame_rate: float
    pitch_size: tuple            # (length, width) in metres, from the metadata's FieldSize
    spec_frames: tuple           # (startFrame, endFrame) of each DataFormatSpecification
    # The two conventions starter/baseline.py relies on, checked against the specification.
    baseline_slot_order: bool    # raw slot order == order of the <Player> elements
    baseline_home: bool          # first <Team> element == the home team

    @property
    def n_slots(self):
        return len(self.slot_player_ids)


@dataclass(eq=False)
class Tracking:
    """One recording's ATD. Row i of every array is video frame i."""
    source: str
    layout: Layout
    frame_id: np.ndarray         # (n,) int64, exactly 1..n -- kept as the proof of alignment
    players: np.ndarray          # (n, slots, 3) float64: x, y, s; NaN where not tracked
    ball: np.ndarray             # (n, 3) float64: x, y, s; NaN where not tracked

    @property
    def n_frames(self):
        return len(self.frame_id)

    def ball_seen(self):
        """(n,) bool: frames with a ball position."""
        return np.isfinite(self.ball[:, 0]) & np.isfinite(self.ball[:, 1])

    def tracked(self):
        """(n, slots) bool: slots with a position on that frame."""
        return np.isfinite(self.players[:, :, 0]) & np.isfinite(self.players[:, :, 1])


def _flatten(element, out):
    """Depth-first ("S"|"P"|"B", id) fields of a DataFormatSpecification -- the raw-line order."""
    for child in element:
        if child.tag == "StringRegister":
            out.append(("S", child.get("name")))
        elif child.tag == "PlayerChannelRef":
            out.append(("P", child.get("playerChannelId")))
        elif child.tag == "BallChannelRef":
            out.append(("B", child.get("channelId")))
        elif child.tag == "SplitRegister":
            _flatten(child, out)
        else:
            raise ValueError("unknown element <{}> in DataFormatSpecification".format(child.tag))
    return out


def read_layout(meta_path):
    """Layout of a FIFA/EPTS metadata XML, or a ValueError naming what does not fit."""
    root = ET.parse(meta_path).getroot()
    metadata = root.find("Metadata")
    if metadata is None:
        raise ValueError("{}: no <Metadata> element".format(meta_path))

    teams = [(team.get("id"), team.findtext("Name") or "")
             for team in metadata.find("Teams").findall("Team")]
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    names = dict(teams)

    # Home is decided the way kloppy decides it: MatchParameters/Score names the local team when
    # present, otherwise the first <Team> is home. starter/baseline.py always takes the first.
    score = metadata.find("Sessions/Session/MatchParameters/Score")
    if score is not None and score.get("idLocalTeam") in names:
        home_id, away_id = score.get("idLocalTeam"), score.get("idVisitingTeam")
    else:
        home_id, away_id = teams[0][0], teams[1][0]
    if away_id not in names or away_id == home_id:
        raise ValueError("{}: cannot tell home from away".format(meta_path))

    team_of_player = {player.get("id"): player.get("teamId")
                      for player in metadata.find("Players").findall("Player")}
    player_order = [player.get("id") for player in root.iter("Player")]
    channel_of = {channel.get("id"): (channel.get("playerId"), channel.get("channelId"))
                  for channel in metadata.find("PlayerChannels").findall("PlayerChannel")}
    units = {channel.get("id"): channel.findtext("Unit")
             for sensor in metadata.iter("Sensor") for channel in sensor.iter("Channel")}

    specs = root.find("DataFormatSpecifications").findall("DataFormatSpecification")
    if not specs:
        raise ValueError("{}: no DataFormatSpecification".format(meta_path))
    fields = _flatten(specs[0], [])
    if any(_flatten(spec, []) != fields for spec in specs[1:]):
        # kloppy switches regular expression at each spec's endFrame. One layout throughout is
        # all this loader supports, and it refuses rather than misreads anything else.
        raise ValueError("{}: {} DataFormatSpecifications with different layouts".format(
            meta_path, len(specs)))
    if not fields or fields[0][0] != "S":
        raise ValueError("{}: the first field is not the frame counter".format(meta_path))
    if [kind for kind, _ in fields].count("S") != 1:
        raise ValueError("{}: expected exactly one StringRegister".format(meta_path))

    slots, columns, ball_columns = [], {}, {}
    for index, (kind, ref) in enumerate(fields):
        if kind == "P":
            player_id, channel = channel_of[ref]
            if player_id not in columns:
                slots.append(player_id)
                columns[player_id] = {}
            columns[player_id][channel] = index
        elif kind == "B":
            ball_columns[ref] = index

    missing = [(player_id, channel) for player_id in slots for channel in CHANNELS
               if channel not in columns[player_id]]
    if missing or any(channel not in ball_columns for channel in CHANNELS):
        raise ValueError("{}: channels {} expected for every slot and the ball".format(
            meta_path, list(CHANNELS)))

    field_size = metadata.find("Sessions/Session/FieldSize")
    return Layout(
        home_team_id=home_id,
        away_team_id=away_id,
        home_name=names[home_id],
        away_name=names[away_id],
        slot_player_ids=tuple(slots),
        team_of_slot=np.array([HOME if team_of_player[player_id] == home_id else AWAY
                               for player_id in slots], dtype=np.int8),
        units=units,
        n_fields=len(fields),
        frame_column=0,
        player_columns=np.array([[columns[player_id][channel] for channel in CHANNELS]
                                 for player_id in slots], dtype=np.int64),
        ball_columns=np.array([ball_columns[channel] for channel in CHANNELS], dtype=np.int64),
        frame_rate=float(metadata.findtext("GlobalConfig/FrameRate")),
        pitch_size=(float(field_size.findtext("Width")), float(field_size.findtext("Height")))
        if field_size is not None else (float("nan"), float("nan")),
        spec_frames=tuple((int(spec.get("startFrame")), int(spec.get("endFrame")))
                          for spec in specs),
        baseline_slot_order=player_order == slots,
        baseline_home=teams[0][0] == home_id,
    )


def _count_lines(path):
    """Lines in a text file, counted in binary so it costs one fast pass and no decoding."""
    count, last = 0, b"\n"
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            count += block.count(b"\n")
            last = block[-1:]
    return count + (last != b"\n")


def read_raw(raw_path, layout, chunk_lines=CHUNK_LINES):
    """(frame_ids, players, ball) for every line of a raw ATD file, in file order.

    Refuses rather than guesses. A line with a missing or extra field would be padded with NaN by
    the tokenizer instead of rejected, so the separators are counted first: every line must split
    into exactly layout.n_fields fields. Blank lines are an error for the same reason.
    """
    n_lines = _count_lines(raw_path)
    frame_ids = np.empty(n_lines, dtype=np.int64)
    players = np.empty((n_lines, layout.n_slots, len(CHANNELS)), dtype=np.float64)
    ball = np.empty((n_lines, len(CHANNELS)), dtype=np.float64)
    to_comma = str.maketrans(SEPARATORS, "," * len(SEPARATORS))

    position = 0
    with open(raw_path, "r") as handle:
        while True:
            lines = list(itertools.islice(handle, chunk_lines))
            if not lines:
                break
            first_line = position + 1
            if any(not line.strip() for line in lines):
                raise ValueError("{}: blank line between lines {} and {}".format(
                    raw_path, first_line, position + len(lines)))
            text = "".join(lines).translate(to_comma)
            separators = text.count(",")
            if separators != len(lines) * (layout.n_fields - 1):
                raise ValueError("{}: lines {}..{} hold {} separators, expected {} ({} fields a "
                                 "line)".format(raw_path, first_line, position + len(lines),
                                                separators, len(lines) * (layout.n_fields - 1),
                                                layout.n_fields))
            block = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                                **EXACT_FLOATS).to_numpy()
            frames = block[:, layout.frame_column]
            if not (np.isfinite(frames).all() and np.array_equal(frames, np.floor(frames))):
                raise ValueError("{}: non-integer frame counter in lines {}..{}".format(
                    raw_path, first_line, position + len(lines)))

            stop = position + len(block)
            frame_ids[position:stop] = frames.astype(np.int64)
            players[position:stop] = block[:, layout.player_columns]
            ball[position:stop] = block[:, layout.ball_columns]
            position = stop

    if position != n_lines:
        raise ValueError("{}: parsed {} lines of {}".format(raw_path, position, n_lines))
    return frame_ids, players, ball


def check_frame_ids(frame_ids, n_expected):
    """None if the ids are exactly 1..n_expected in order, else a one-line reason.

    This is the whole proof of video_frame = frame_id - 1: with ids 1..n in file order, line i of
    the file is frame_id i + 1, i.e. video frame i, with nothing missing and nothing repeated.
    """
    if len(frame_ids) != n_expected:
        return "{} records, expected {}".format(len(frame_ids), n_expected)
    if not np.array_equal(frame_ids, np.arange(1, n_expected + 1)):
        breaks = int(np.sum(np.diff(frame_ids) != 1))
        return "frame ids are not 1..{} in order (first {}, last {}, {} breaks)".format(
            n_expected, int(frame_ids[0]), int(frame_ids[-1]), breaks)
    return None


def load_tracking(source=DEFAULT_SOURCE, game=GAME, data_dir=None):
    """The ATD of one recording, aligned to its video, or an error saying why it cannot be."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    layout = read_layout(meta_path)
    frame_ids, players, ball = read_raw(raw_path, layout)
    reason = check_frame_ids(frame_ids, source_frames(source, game))
    if reason:
        raise ValueError("{} ATD is not aligned to its video: {}".format(source, reason))
    return Tracking(source=source, layout=layout, frame_id=frame_ids, players=players, ball=ball)
