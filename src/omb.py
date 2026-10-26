"""Other-match benchmark: real football with real labels, in this pipeline's own types.

    from src.omb import load_dfl
    match = load_dfl("outputs/omb/dfl/J03WMX")    # .tracking .labels .blocks .pitch .name

IDSSE / DFL open data (docs/experiments.md §5, CC BY 4.0) becomes a Tracking with the same
conventions as the tactical ATD -- normalised 0-1 coordinates from a corner, one slot per tracked
person, speed channel in m/s -- and its per-frame ball status and possession become
possession_metrics codes by the same recipe as this challenge's ground truth (CHALLENGE.md): D where
the ball is dead, otherwise the team the data names as owning it.

These are labels of *other* matches. RULES.md allows them in any amount, and the assistant may look
at them freely (D-002 is about the competition match). What they may not do is stand in for the
competition's own definitions unchecked: whether DFL "possession" means control or last touch is
measured in docs/experiments.md before any calibration leans on it.

The positions XML is ~419 MB per match, so it is streamed (iterparse) and each person's frames
become float32 arrays as soon as their FrameSet closes; the result is cached beside the XML as
parsed.npz, which loads in a second.
"""

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import possession_metrics as pm  # noqa: E402
from src.data_loader import AWAY, HOME  # noqa: E402
from src.synthetic import make_tracking  # noqa: E402

# Params
HALVES = (("firstHalf", 1), ("secondHalf", 2))
# Ungraded frames inserted between the halves, so nothing can run across them -- the same shape as
# the competition file, where halftime is X.
HALFTIME_FRAMES = 250
CACHE = "parsed.npz"
CACHE_VERSION = 1


@dataclass(eq=False)
class OtherMatch:
    name: str
    tracking: object             # src.data_loader.Tracking
    labels: np.ndarray           # (n,) int8 pm codes; X at halftime and wherever the data has none
    blocks: list                 # [(half, first, last)]
    pitch: tuple                 # (length, width) in metres
    speed_unit_ratio: float      # median raw S / displacement speed, before conversion to m/s


def load_dfl(folder, use_cache=True):
    """One IDSSE/DFL match from folder/meta.xml and folder/tracking.xml (scripts/omb_fetch.py)."""
    cache = os.path.join(folder, CACHE)
    if use_cache and os.path.isfile(cache):
        arrays = dict(np.load(cache, allow_pickle=False))
        if int(arrays.get("version", -1)) == CACHE_VERSION:
            return _assemble(os.path.basename(os.path.normpath(folder)), arrays)
    arrays = _parse(os.path.join(folder, "meta.xml"), os.path.join(folder, "tracking.xml"))
    np.savez(cache, version=CACHE_VERSION, **arrays)
    return _assemble(os.path.basename(os.path.normpath(folder)), arrays)


def _parse(meta_path, tracking_path):
    """Stream the positions XML into per-half arrays. Returns a dict of numpy arrays."""
    roles = {team.get("TeamId"): team.get("Role")
             for team in ET.parse(meta_path).getroot().iter("Team")}
    home = [team for team, role in roles.items() if role == "home"]
    guest = [team for team, role in roles.items() if role == "guest"]
    if len(home) != 1 or len(guest) != 1:
        raise ValueError("{}: expected one home and one guest team, got {}".format(meta_path, roles))
    team_code = {home[0]: HOME, guest[0]: AWAY}

    pitch = None
    sets = []                          # (section, team, person, N, X, Y, S, possession, status)
    current = None
    for event, element in ET.iterparse(tracking_path, events=("start", "end")):
        tag = element.tag
        if event == "start" and tag == "FrameSet":
            current = {"section": element.get("GameSection"), "team": element.get("TeamId"),
                       "person": element.get("PersonId"), "n": [], "x": [], "y": [], "s": [],
                       "possession": [], "status": []}
        elif event == "end" and tag == "Frame" and current is not None:
            get = element.get
            current["n"].append(int(get("N")))
            current["x"].append(float(get("X")))
            current["y"].append(float(get("Y")))
            current["s"].append(float(get("S") or "nan"))
            if get("BallStatus") is not None:
                current["status"].append(int(get("BallStatus")))
                current["possession"].append(int(get("BallPossession") or 0))
            element.clear()
        elif event == "end" and tag == "FrameSet":
            sets.append(current)
            current = None
            element.clear()
        elif event == "end" and tag == "PitchSize":
            pitch = (float(element.get("X")), float(element.get("Y")))

    if pitch is None:
        raise ValueError("{}: no PitchSize".format(tracking_path))
    arrays = {"pitch": np.array(pitch)}
    slots = sorted({(s["team"], s["person"]) for s in sets if s["team"] in team_code})
    arrays["team_of_slot"] = np.array([team_code[team] for team, _ in slots], dtype=np.int8)
    slot_index = {key: index for index, key in enumerate(slots)}

    for section, half in HALVES:
        balls = [s for s in sets if s["section"] == section and s["status"]]
        if len(balls) != 1:
            raise ValueError("{}: {} ball FrameSets in {}".format(tracking_path, len(balls), section))
        ball = balls[0]
        first, last = min(ball["n"]), max(ball["n"])
        n = last - first + 1
        players = np.full((n, len(slots), 3), np.nan, dtype=np.float32)
        for s in sets:
            if s["section"] != section or s["team"] not in team_code:
                continue
            rows = np.asarray(s["n"]) - first
            keep = (rows >= 0) & (rows < n)
            slot = slot_index[(s["team"], s["person"])]
            for channel, key in enumerate(("x", "y", "s")):
                players[rows[keep], slot, channel] = np.asarray(s[key], dtype=np.float32)[keep]
        ball_xy = np.full((n, 2), np.nan, dtype=np.float32)
        status = np.full(n, -1, dtype=np.int8)
        possession = np.zeros(n, dtype=np.int8)
        rows = np.asarray(ball["n"]) - first
        ball_xy[rows, 0], ball_xy[rows, 1] = ball["x"], ball["y"]
        status[rows], possession[rows] = ball["status"], ball["possession"]
        prefix = "half{}_".format(half)
        arrays[prefix + "players"], arrays[prefix + "ball"] = players, ball_xy
        arrays[prefix + "status"], arrays[prefix + "possession"] = status, possession
    return arrays


def _assemble(name, arrays):
    """OtherMatch from parsed arrays: normalise coordinates, fix the speed unit, build labels."""
    length, width = (float(v) for v in arrays["pitch"])
    parts, labels, blocks = [], [], []
    position = 0
    for _, half in HALVES:
        prefix = "half{}_".format(half)
        players = arrays[prefix + "players"].astype(np.float64)
        ball = arrays[prefix + "ball"].astype(np.float64)
        n = len(ball)
        if position:
            parts.append((np.full((HALFTIME_FRAMES,) + players.shape[1:], np.nan),
                          np.full((HALFTIME_FRAMES, 2), np.nan)))
            labels.append(np.full(HALFTIME_FRAMES, pm.X, dtype=np.int8))
            position += HALFTIME_FRAMES
        parts.append((players, ball))
        status, possession = arrays[prefix + "status"], arrays[prefix + "possession"]
        half_labels = np.full(n, pm.X, dtype=np.int8)
        half_labels[status == 0] = pm.D
        half_labels[(status == 1) & (possession == 1)] = pm.H
        half_labels[(status == 1) & (possession == 2)] = pm.A
        labels.append(half_labels)
        blocks.append((half, position, position + n - 1))
        position += n

    players = np.concatenate([p for p, _ in parts])
    ball_xy = np.concatenate([b for _, b in parts])
    # DFL: metres from the centre spot. Ours: 0-1 from a corner, y growing the other way.
    players[:, :, 0] = (players[:, :, 0] + length / 2.0) / length
    players[:, :, 1] = (width / 2.0 - players[:, :, 1]) / width
    ball = np.full((len(ball_xy), 3), np.nan)
    ball[:, 0] = (ball_xy[:, 0] + length / 2.0) / length
    ball[:, 1] = (width / 2.0 - ball_xy[:, 1]) / width

    ratio = _speed_ratio(players, length, width)
    if abs(ratio - 3.6) < 0.4:                 # km/h, as the probe suggested: convert to m/s
        players[:, :, 2] /= 3.6
    elif abs(ratio - 1.0) > 0.15:
        raise ValueError("{}: speed channel is neither m/s nor km/h (ratio {:.2f})".format(name, ratio))

    tracking = make_tracking(players, ball, arrays["team_of_slot"])
    return OtherMatch(name, tracking, np.concatenate(labels), blocks, (length, width), ratio)


def _speed_ratio(players, length, width):
    """median(S / displacement speed) over moving players: ~1 for m/s, ~3.6 for km/h (D-008)."""
    x, y, s = players[:, :, 0] * length, players[:, :, 1] * width, players[:, :, 2]
    step = np.sqrt(np.diff(x, axis=0) ** 2 + np.diff(y, axis=0) ** 2) * 25.0
    keep = np.isfinite(step) & np.isfinite(s[1:]) & (step > 2.0) & (step < 12.0)
    return float(np.median(s[1:][keep] / step[keep]))


def load_metrica_sample(folder):
    """Metrica Sample Game 3 (metadata.xml, tracking.txt, events.json) as an OtherMatch.

    The tracking is in this challenge's own EPTS format, but split into 11 DataFormatSpecifications
    as substitutions change who is tracked; each spec's columns are mapped onto one slot per player.
    There is no speed channel, so speed is the central difference of positions over +/-2 frames.

    Labels come from Metrica's event annotations by a rule fixed before any score was computed
    (docs/experiments.md §5): possession changes at a RECOVERY (Metrica's mark for a team gaining
    control) or a SET PIECE (the restarting team), and falls back to the team of a PASS, CARRY or
    SHOT that contradicts it; the ball is dead from the end of a BALL OUT, of a goal or of a shot
    that goes out, from a FAULT RECEIVED and from an offside or handball BALL LOST, until the next
    SET PIECE -- and a SET PIECE that arrives while the ball is still live makes it dead from the
    end of the previous event. A CHALLENGE changes nothing. Frames of a period before its kickoff
    or after its last event are X.
    """
    import json
    from src.data_loader import _flatten

    meta = ET.parse(os.path.join(folder, "metadata.xml")).getroot()
    md = meta.find("Metadata")
    local = md.find("Sessions/Session/MatchParameters/Score").get("idLocalTeam")
    team_of_player = {p.get("id"): (HOME if p.get("teamId") == local else AWAY)
                      for p in md.find("Players").findall("Player")}
    channel_of = {c.get("id"): (c.get("playerId"), c.get("channelId"))
                  for c in md.find("PlayerChannels").findall("PlayerChannel")}
    field = md.find("Sessions/Session/FieldSize")
    pitch = (float(field.findtext("Width")), float(field.findtext("Height")))
    slots = sorted(team_of_player)
    slot_of = {player: index for index, player in enumerate(slots)}

    specs = []
    for spec in meta.find("DataFormatSpecifications").findall("DataFormatSpecification"):
        fields = _flatten(spec, [])
        columns = []                                    # (field index, slot or -1 for ball, channel)
        for index, (kind, ref) in enumerate(fields):
            if kind == "P":
                player, channel = channel_of[ref]
                columns.append((index, slot_of[player], "xy".index(channel)))
            elif kind == "B":
                columns.append((index, -1, "xy".index(ref)))
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), len(fields), columns))

    n = specs[-1][1]
    players = np.full((n, len(slots), 3), np.nan)
    ball = np.full((n, 3), np.nan)
    to_comma = str.maketrans(":;", ",,")
    import io
    import pandas as pd
    with open(os.path.join(folder, "tracking.txt")) as handle:
        for first, last, n_fields, columns in specs:
            lines = [handle.readline() for _ in range(last - first + 1)]
            block = pd.read_csv(io.StringIO("".join(lines).translate(to_comma)), header=None,
                                dtype=np.float64).to_numpy()
            if block.shape[1] != n_fields or int(block[0, 0]) != first or int(block[-1, 0]) != last:
                raise ValueError("{}: spec {}-{} does not match the raw lines".format(folder, first, last))
            rows = slice(first - 1, last)
            for index, slot, channel in columns:
                if slot < 0:
                    ball[rows, channel] = block[:, index]
                else:
                    players[rows, slot, channel] = block[:, index]

    length, width = pitch
    x, y = players[:, :, 0] * length, players[:, :, 1] * width
    speed = np.full(x.shape, np.nan)
    speed[2:-2] = np.sqrt((x[4:] - x[:-4]) ** 2 + (y[4:] - y[:-4]) ** 2) * 25.0 / 4.0
    players[:, :, 2] = np.minimum(speed, 12.0)

    with open(os.path.join(folder, "events.json"), encoding="utf-8") as handle:
        events = json.load(handle)["data"]
    team_code = {}
    for team in md.find("Teams").findall("Team"):
        team_code[team.findtext("Name")] = HOME if team.get("id") == local else AWAY
    params = {p.findtext("Name"): p.findtext("Value")
              for p in md.find("GlobalConfig/ProviderGlobalParameters").findall("ProviderParameter")}
    halves = [(1, int(params["first_half_start"]), int(params["first_half_end"])),
              (2, int(params["second_half_start"]), int(params["second_half_end"]))]
    labels = np.full(n, pm.X, dtype=np.int8)
    blocks = []
    for half, start, end in halves:
        own = sorted((e for e in events if e["period"] == half),
                     key=lambda e: (e["start"]["frame"], e["index"]))
        kickoffs = [e for e in own if e["type"]["name"] == "SET PIECE"]
        if not kickoffs:
            continue
        first_frame = kickoffs[0]["start"]["frame"]
        last_frame = max(e["end"]["frame"] or e["start"]["frame"] for e in own)
        changes = []                                  # (frame, code), applied in order
        owner, dead, last_play = None, True, first_frame
        for event in own:
            kind, team = event["type"]["name"], team_code[event["team"]["name"]]
            begin = event["start"]["frame"]
            finish = event["end"]["frame"] or begin
            tokens = set(_subtype_tokens(event))
            if kind == "SET PIECE":
                if not dead:
                    # A restart proves the ball was dead; the last sign of play is the end of the
                    # previous event. (Fouls recorded as BALL LOST FORCED, mostly.)
                    changes.append((last_play, pm.D))
                owner, dead = team, False
                changes.append((begin, team))
                last_play = finish
                continue
            if dead:
                continue
            if kind == "RECOVERY" or (kind in ("PASS", "CARRY", "SHOT") and team != owner):
                owner = team
                changes.append((begin, team))
            last_play = max(last_play, finish)
            if kind == "BALL OUT" or (kind == "SHOT" and tokens & {"GOAL", "OUT"}):
                dead = True
                changes.append((finish, pm.D))
            elif kind == "FAULT RECEIVED" or (kind == "BALL LOST" and tokens & {"OFFSIDE", "HAND BALL"}):
                dead = True
                changes.append((begin, pm.D))
        state = pm.X
        pointer = 0
        changes.sort(key=lambda item: item[0])
        for frame in range(first_frame, last_frame + 1):
            while pointer < len(changes) and changes[pointer][0] <= frame:
                state = changes[pointer][1]
                pointer += 1
            labels[frame - 1] = state
        blocks.append((half, start - 1, end - 1))

    tracking = make_tracking(players, ball, np.array([team_of_player[p] for p in slots]))
    return OtherMatch("SG3", tracking, labels, blocks, pitch, float("nan"))


def _subtype_tokens(event):
    subtypes = event.get("subtypes")
    if subtypes is None:
        return []
    if isinstance(subtypes, dict):
        subtypes = [subtypes]
    return [item["name"] for item in subtypes]


BENCHMARK_DIR = os.path.join(ROOT, "outputs", "omb")


def benchmark_matches():
    """Names of the benchmark matches on disk: the DFL match ids, then "SG3" if present."""
    names = []
    dfl = os.path.join(BENCHMARK_DIR, "dfl")
    if os.path.isdir(dfl):
        names += sorted(m for m in os.listdir(dfl)
                        if os.path.isfile(os.path.join(dfl, m, "tracking.xml"))
                        or os.path.isfile(os.path.join(dfl, m, CACHE)))
    if os.path.isfile(os.path.join(BENCHMARK_DIR, "metrica", "SG3", "tracking.txt")):
        names.append("SG3")
    return names


def load_benchmark(name, degrade="none", seed=0, dead_share=0.85, profile=None):
    """One benchmark match, ready to score. DFL matches get the ATD-like ball loss (src/degrade.py);
    SG3 never does -- its gaps are its tracker's own."""
    from src.degrade import degrade_ball
    if name == "SG3":
        return load_metrica_sample(os.path.join(BENCHMARK_DIR, "metrica", "SG3"))
    match = load_dfl(os.path.join(BENCHMARK_DIR, "dfl", name))
    return degrade_ball(match, profile, degrade, seed, dead_share)


def label_stats(match):
    """Aggregate description of the labels: shares, possessions, durations (seconds)."""
    runs = pm.valid_runs(match.labels)
    spells = pm.possession_spells(match.labels, runs)
    durations = np.array([stop - start for start, stop, _ in spells]) / 25.0
    graded = match.labels != pm.X
    return {
        "graded_frames": int(graded.sum()),
        "share_D": round(float(np.mean(match.labels[graded] == pm.D)), 3),
        "possessions": len(spells),
        "possession_s_p10_p50_p90": [round(float(np.percentile(durations, q)), 1)
                                     for q in (10, 50, 90)],
        "possessions_under_1s": int(np.sum(durations < 1.0)),
        "stoppages": sum(1 for _, _, code in pm.segments(match.labels, runs) if code == pm.D),
    }
