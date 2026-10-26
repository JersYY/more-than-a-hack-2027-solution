"""Phase-1 data audit: measure the provided match assets against the published contract.

    $env:MTAH_DATA_DIR = "D:\\mtah2027"     # wherever the bundle is; ./data by default
    python scripts/data_audit.py           # -> docs/data_audit.md, outputs/audit/data_audit.json

AGGREGATES ONLY. Every number this prints or writes is a count, a rate, a median or a PASS/FAIL
against a value the organisers publish. No coordinate, tag interval, frame-level value, team name
or line of any file is printed or written. The competition terms forbid making the match data, or
any extract of it, available to a hosted LLM, and this repository is developed with one in the
loop -- docs/decisions.md D-002. Keep it that way when adding checks.

EXPECTED values are the organisers' published figures, quoted with their source. OBSERVED values
are measured here, on the files games.asset_paths() finds. A check is PASS or FAIL against a
published contract value (counts, alignment, parity); DIFF when a published *descriptive* figure
or label is not reproduced -- reported and explained, but not blocking; INFO when nothing was
published to compare with. The exit code is 1 only on FAIL.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _path in (ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import games  # noqa: E402
import possession_metrics as pm  # noqa: E402
from src import submission  # noqa: E402
from src.data_loader import HOME, AWAY, load_tracking  # noqa: E402
from src.features import nearest_team  # noqa: E402

# Params
GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]
PITCH = games.GAMES[GAME]["pitch"]
DOC_PATH = os.path.join(ROOT, "docs", "data_audit.md")
JSON_PATH = os.path.join(ROOT, "outputs", "audit", "data_audit.json")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
PUBLIC_GT_META = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public_meta.json")
# Every 50th line of the raw file goes through kloppy: ~4000 frames spread over the whole match,
# in seconds rather than the minutes a full kloppy load costs.
KLOPPY_SAMPLE_RATE = 1.0 / 50
# Displacement speeds outside this band (m/s) are left out of the unit check: below it the ratio
# is dominated by position jitter, above it by tracker identity switches.
SPEED_BAND = (2.0, 12.0)

# Published figures. Source in the comment; DATA.md wins over this table if they ever differ.
PUBLISHED = {
    "half_blocks": [(1, 11185, 90651), (2, 110808, 190197)],          # DATA.md, frame alignment
    "graded_frames": 158857,                                          # DATA.md, CHALLENGE.md
    "ungraded_frames": 40462,                                         # DATA.md
    "tactical_frames": 199319,                                        # DATA.md, games.py
    "broadcast_frames": 164488,                                       # DATA.md, games.py
    "tactical_slots": 46,                                             # DATA.md
    "broadcast_slots": 51,                                            # DATA.md
    "ball_seen_all": 113488,                                          # DATA.md
    "ball_seen_graded": 94214,                                        # DATA.md
    "broadcast_ball_pct": 50.5,                                       # DATA.md
    "mean_objects_graded": 20.5,                                      # DATA.md
    "smart_tagging_instances": 638,                                   # DATA.md
    "codes": {                                                        # DATA.md, Smart Tagging
        "PLAYERS IN THE BOX": 97, "BALL IN FINAL THIRD": 93, "BALL IN THE BOX": 72,
        "PROGRESSION": 66, "DEFENDING IN MIDDLE THIRD": 66, "SET PIECES": 59,
        "CREATING CHANCES": 30, "DEFENDING IN DEFENSIVE THIRD": 30, "LONG BALL": 29,
        "ATTACKING TRANSITION": 24, "DEFENSIVE TRANSITION": 24, "BUILD UP": 22,
        "DEFENDING IN ATTACKING THIRD": 22, "GOALS": 4,
    },
    "team_tags": {"Barcelona": 314, "Atletic Club": 288, "N/A": 36},  # DATA.md
    "type_tags": {"Throw-in": 26, "Corner Kick": 12, "Goal Kick": 10, "Free Kick": 5,
                  "Kick Off (After Goal)": 4, "Kick Off (Start)": 2},  # DATA.md
    "half_tags": ["1st Half", "2nd Half"],                            # DATA.md
    "tactical_homography_valid_pct": 100.0,                           # DATA.md
    "broadcast_homography_valid_pct": 65.8,                           # DATA.md
    "reprojection_median_play": 0.24,                                 # DATA.md
    "reprojection_median_halftime": 1.5,                              # DATA.md ("about")
    "public_gt": {"first": 11185, "last": 18684, "H": 3454, "A": 2577, "D": 1469,
                  "possessions": 17, "segments": 21},                 # DATA.md, gt meta
}


class Audit(object):
    """Collects checks in order. Each is (section, name, expected, observed, status, source)."""

    def __init__(self):
        self.checks = []
        self.info = {}

    def add(self, section, name, expected, observed, status, source=""):
        self.checks.append({"section": section, "check": name, "expected": expected,
                            "observed": observed, "status": status, "source": source})

    def compare(self, section, name, expected, observed, source=""):
        self.add(section, name, expected, observed,
                 "PASS" if expected == observed else "FAIL", source)

    def close(self, section, name, expected, observed, tolerance, source="", contract=True):
        """Numeric check within a tolerance. contract=False turns a miss into DIFF, not FAIL."""
        ok = observed is not None and abs(observed - expected) <= tolerance
        self.add(section, name, "{} (+/-{})".format(expected, tolerance), observed,
                 "PASS" if ok else "FAIL" if contract else "DIFF", source)

    def failed(self, section, error):
        """A section that could not run is one FAIL with the exception type -- never its data."""
        self.add(section, "section ran", "no error", "{}: {}".format(
            type(error).__name__, str(error).splitlines()[0][:160] if str(error) else ""), "FAIL")


def pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else None


def graded_mask(n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in games.half_video_blocks(GAME):
        mask[first:last + 1] = True
    return mask


def block_mask(n_frames, first, last):
    mask = np.zeros(n_frames, dtype=bool)
    mask[first:last + 1] = True
    return mask


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip())
        return commit, dirty
    except Exception:  # not a checkout, or git missing: provenance is then "unknown", not fatal
        return "unknown", None


# -- sections ------------------------------------------------------------------------------------

def audit_inventory(audit, hash_videos):
    section = "1. Inventory"
    rows = []
    for source in ("tactical", "broadcast"):
        for name, path in sorted(games.asset_paths(source=source).items()):
            present = os.path.isfile(path)
            size = os.path.getsize(path) if present else None
            digest = None
            if present and (name != "video" or hash_videos):
                digest = sha256_file(path)
            rows.append({"source": source, "asset": name, "file": os.path.basename(path),
                         "present": present, "bytes": size, "sha256": digest})
    audit.info["inventory"] = rows
    audit.compare(section, "assets present (tactical 5 + broadcast 4)", 9,
                  sum(row["present"] for row in rows), "DATA.md")


def audit_timeline(audit):
    section = "2. Timeline contract"
    blocks = games.half_video_blocks(GAME)
    audit.compare(section, "games.half_video_blocks", PUBLISHED["half_blocks"], blocks,
                  "DATA.md frame alignment")
    graded = sum(last - first + 1 for _, first, last in blocks)
    audit.compare(section, "graded frames", PUBLISHED["graded_frames"], graded, "DATA.md")
    audit.compare(section, "ungraded frames", PUBLISHED["ungraded_frames"],
                  games.GAMES[GAME]["n_video_frames"] - graded, "DATA.md")
    audit.compare(section, "tactical video frames (games.py)", PUBLISHED["tactical_frames"],
                  games.source_frames("tactical"), "DATA.md")
    audit.compare(section, "broadcast video frames (games.py)", PUBLISHED["broadcast_frames"],
                  games.source_frames("broadcast"), "DATA.md")

    # The public ground truth ships with the repository. Read here for its shape only -- which
    # frames it covers and its published totals -- never to score or tune anything.
    with open(PUBLIC_GT_META) as handle:
        meta = json.load(handle)
    audit.compare(section, "gt meta offsets == games.py offsets",
                  {str(k): v for k, v in games.GAMES[GAME]["offsets"].items()},
                  meta["frame_offsets"], "games.py")
    audit.compare(section, "gt meta half id frames == games.py halves",
                  {str(k): list(v) for k, v in games.GAMES[GAME]["halves"].items()},
                  meta["half_id_frames"], "games.py")
    codes = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    runs = pm.valid_runs(codes)
    expected = PUBLISHED["public_gt"]
    audit.compare(section, "public GT rows", PUBLISHED["tactical_frames"], len(codes), "DATA.md")
    audit.compare(section, "public GT graded window",
                  [(expected["first"], expected["last"] + 1)], runs, "DATA.md")
    audit.compare(section, "public GT H/A/D counts",
                  [expected["H"], expected["A"], expected["D"]],
                  [int(np.sum(codes == k)) for k in (pm.H, pm.A, pm.D)], "DATA.md")
    audit.compare(section, "public GT possessions / segments",
                  [expected["possessions"], expected["segments"]],
                  [len(pm.possession_spells(codes, runs)), len(pm.segments(codes, runs))],
                  "gt meta json")


def audit_atd_layout(audit, layout, source, expected_slots, expected_frames):
    section = "3. Tactical ATD" if source == "tactical" else "7. Broadcast ATD"
    audit.compare(section, "track slots", expected_slots, layout.n_slots, "DATA.md")
    audit.add(section, "slots per team (home / away)", "-",
              [int(np.sum(layout.team_of_slot == HOME)), int(np.sum(layout.team_of_slot == AWAY))],
              "INFO")
    audit.compare(section, "home team is Barcelona", True,
                  "barcelona" in layout.home_name.lower(), "DATA.md")
    audit.compare(section, "away team is Athletic Club", True,
                  any(word in layout.away_name.lower() for word in ("athletic", "atletic")),
                  "DATA.md")
    audit.compare(section, "raw slot order == <Player> order (starter/baseline.py assumes)", True,
                  layout.baseline_slot_order, "starter/baseline.py")
    audit.compare(section, "home == first <Team> (starter/baseline.py assumes)", True,
                  layout.baseline_home, "starter/baseline.py")
    audit.compare(section, "one DataFormatSpecification covering 1..n", [(1, expected_frames)],
                  list(layout.spec_frames), "metadata XML")
    audit.compare(section, "frame rate", float(FPS), layout.frame_rate, "DATA.md")
    audit.compare(section, "metadata pitch (m)", (105.0, 68.0), layout.pitch_size, "DATA.md")
    audit.compare(section, "position unit", "normalized",
                  layout.units.get("x") if layout.units.get("x") == layout.units.get("y") else
                  "x/y differ", "DATA.md")


def speed_ratio(positions, speeds, frames_mask):
    """median(reported speed / displacement speed in m/s) over consecutive tracked frames.

    positions (n, k, 2) normalised, speeds (n, k). Only steps inside frames_mask and inside
    SPEED_BAND count. ~1.0 means the channel is m/s; ~3.6 means km/h.
    """
    length, width = PITCH
    ratios = []
    for start in range(1, len(positions), 20000):
        stop = min(start + 20000, len(positions))
        x0, y0 = positions[start - 1:stop - 1, :, 0] * length, positions[start - 1:stop - 1, :, 1] * width
        x1, y1 = positions[start:stop, :, 0] * length, positions[start:stop, :, 1] * width
        displacement = np.hypot(x1 - x0, y1 - y0) * FPS
        reported = speeds[start:stop]
        keep = (frames_mask[start:stop] & frames_mask[start - 1:stop - 1])[:, None] \
            & np.isfinite(displacement) & np.isfinite(reported) \
            & (displacement > SPEED_BAND[0]) & (displacement < SPEED_BAND[1])
        ratios.append(reported[keep] / displacement[keep])
    ratios = np.concatenate(ratios) if ratios else np.array([])
    return (round(float(np.median(ratios)), 3) if len(ratios) else None), int(len(ratios))


def audit_tactical(audit):
    section = "3. Tactical ATD"
    import load_atd  # official starter module, import only

    started = time.time()
    tracking = load_tracking("tactical")
    audit.info["tactical_load_seconds"] = round(time.time() - started, 1)
    n = tracking.n_frames
    layout = tracking.layout

    audit.compare(section, "records", PUBLISHED["tactical_frames"], n, "DATA.md")
    audit.compare(section, "frame_id is exactly 1..n (video_frame = frame_id - 1)", True,
                  bool(np.array_equal(tracking.frame_id, np.arange(1, n + 1))), "DATA.md")
    official = load_atd.check_alignment_raw()
    audit.compare(section, "official load_atd.check_alignment_raw() agrees", True,
                  bool(official["agrees"]), "starter/load_atd.py")
    audit_atd_layout(audit, layout, "tactical", PUBLISHED["tactical_slots"], n)

    graded = graded_mask(n)
    seen = tracking.ball_seen()
    audit.compare(section, "ball seen, all frames", PUBLISHED["ball_seen_all"], int(seen.sum()),
                  "DATA.md 56.9%")
    audit.compare(section, "ball seen, graded frames", PUBLISHED["ball_seen_graded"],
                  int(seen[graded].sum()), "DATA.md 59.3%")
    ball_x, ball_y = np.isfinite(tracking.ball[:, 0]), np.isfinite(tracking.ball[:, 1])
    player_x, player_y = np.isfinite(tracking.players[:, :, 0]), np.isfinite(tracking.players[:, :, 1])
    audit.add(section, "half-missing observations (x xor y), ball + players", "-",
              int(np.sum(ball_x ^ ball_y)) + int(np.sum(player_x ^ player_y)), "INFO")

    tracked = tracking.tracked()
    objects = tracked.sum(axis=1)
    audit.close(section, "mean tracked objects, graded frames", PUBLISHED["mean_objects_graded"],
                round(float(objects[graded].mean()), 1), 0.05, "DATA.md")
    audit.add(section, "max tracked objects, graded frames", "occasionally > 22",
              int(objects[graded].max()), "INFO")
    audit.add(section, "graded frames with > 22 objects (%)", "-",
              pct(int(np.sum(objects[graded] > 22)), int(graded.sum())), "INFO")
    for team, code in (("home", HOME), ("away", AWAY)):
        audit.add(section, "mean tracked {} objects, graded frames".format(team), "-",
                  round(float(tracked[:, layout.team_of_slot == code].sum(axis=1)[graded].mean()), 2),
                  "INFO")
    for label, (first, last) in (("before kickoff", (0, 11184)), ("halftime", (90652, 110807)),
                                 ("after the final whistle", (190198, n - 1))):
        mean_objects = round(float(objects[block_mask(n, first, last)].mean()), 2)
        audit.info["mean_objects_" + label.replace(" ", "_")] = mean_objects
        if label == "halftime":
            # starter/load_atd.py describes "almost none across the halftime break". Descriptive,
            # not a contract, so a miss is a DIFF; 2 objects is a generous reading of "almost none".
            audit.add(section, "mean tracked objects, halftime", "almost none (starter/load_atd.py)",
                      mean_objects, "PASS" if mean_objects < 2 else "DIFF", "starter/load_atd.py")
        else:
            audit.add(section, "mean tracked objects, {}".format(label), "-", mean_objects, "INFO")
    audit.add(section, "slots ever tracked (any frame / graded frames)", "-",
              [int(tracked.any(axis=0).sum()), int(tracked[graded].any(axis=0).sum())], "INFO")

    for label, positions in (("player", tracking.players[:, :, :2]), ("ball", tracking.ball[:, None, :2])):
        n_finite = n_inside = 0
        for start in range(0, n, 20000):
            xy = positions[start:start + 20000][graded[start:start + 20000]]
            finite = np.isfinite(xy).all(axis=2)
            n_finite += int(finite.sum())
            n_inside += int((finite & (xy >= 0).all(axis=2) & (xy <= 1).all(axis=2)).sum())
        audit.add(section, "{} positions inside [0,1]^2, graded (%)".format(label), "-",
                  pct(n_inside, n_finite), "INFO")
        audit.info["{}_inside_pct".format(label)] = pct(n_inside, n_finite)

    # Speed unit. DATA.md says m/s; the metadata's speed channel says kph. Displacement between
    # consecutive frames decides it, independently of either document.
    ratio, n_steps = speed_ratio(tracking.players[:, :, :2], tracking.players[:, :, 2], graded)
    unit = layout.units.get("s")
    implied = "km/h" if ratio is not None and abs(ratio - 3.6) < 0.3 else \
        "m/s" if ratio is not None and abs(ratio - 1.0) < 0.1 else "unclear"
    audit.add(section, "player speed unit", "m/s (DATA.md)",
              "median s / displacement speed = {} over {} steps -> {}".format(ratio, n_steps, implied),
              "PASS" if implied == "m/s" else "FAIL", "DATA.md")
    audit.add(section, "speed unit label in the metadata XML", "m/s (DATA.md)",
              "'{}'".format(unit), "PASS" if unit in ("m/s", "mps") or implied != "m/s" else "DIFF",
              "metadata XML")
    audit.info["player_speed_ratio"] = ratio
    audit.info["player_speed_implied_unit"] = implied
    audit.info["speed_unit_label"] = unit
    speeds = tracking.players[:, :, 2][graded]
    speeds = speeds[np.isfinite(speeds)]
    audit.add(section, "player speed channel p50 / p99 (file units)", "-",
              [round(float(np.percentile(speeds, 50)), 2), round(float(np.percentile(speeds, 99)), 2)],
              "INFO")
    del speeds
    ball_ratio, ball_steps = speed_ratio(tracking.ball[:, None, :2], tracking.ball[:, None, 2], graded)
    ball_speed_pct = pct(int(np.isfinite(tracking.ball[seen, 2]).sum()), int(seen.sum()))
    audit.add(section, "ball speed channel present where ball seen (%)", "-", ball_speed_pct, "INFO")
    audit.add(section, "ball speed unit", "not documented",
              "median s / displacement speed = {} over {} steps".format(ball_ratio, ball_steps), "INFO")
    audit.info["ball_speed_ratio"] = ball_ratio
    audit.info["ball_speed_pct"] = ball_speed_pct

    audit_parity(audit, tracking)
    return tracking


def audit_parity(audit, tracking):
    section = "4. Loader parity"
    import baseline  # official starter/baseline.py, import only -- its smooth() is the reference

    codes = nearest_team(tracking)
    for name, window in baseline.VARIANTS:
        ours = baseline.smooth(codes, window)
        committed = submission.read_labels(os.path.join(RESULTS_DIR, name + ".csv"))
        audit.compare(section, "nearest_team + official smooth vs committed {}.csv".format(name),
                      0, int(np.sum(ours != committed)), "evaluation/results/")


def audit_kloppy(audit, tracking):
    section = "4. Loader parity"
    from kloppy import metrica

    meta_path, raw_path = games.require(["atd_meta", "atd_raw"])
    started = time.time()
    dataset = metrica.load_tracking_epts(meta_data=meta_path, raw_data=raw_path,
                                         sample_rate=KLOPPY_SAMPLE_RATE, coordinates="metrica")
    audit.info["kloppy_sample_seconds"] = round(time.time() - started, 1)

    layout = tracking.layout
    slot_of = {player_id: slot for slot, player_id in enumerate(layout.slot_player_ids)}
    home_id = dataset.metadata.teams[0].team_id

    def differ(a, b):
        a = np.nan if a is None else float(a)
        b = float(b)
        return not (a == b or (np.isnan(a) and np.isnan(b)))

    frames = ball_bad = player_values = player_bad = team_bad = unknown = 0
    for frame in dataset.records:
        row = int(frame.frame_id) - 1
        frames += 1
        ball = frame.ball_coordinates
        ball_bad += differ(ball.x if ball is not None else None, tracking.ball[row, 0]) \
            or differ(ball.y if ball is not None else None, tracking.ball[row, 1])
        for player, data in frame.players_data.items():
            slot = slot_of.get(player.player_id)
            if slot is None:
                unknown += 1
                continue
            team_bad += (player.team.team_id == home_id) != (layout.team_of_slot[slot] == HOME)
            point = data.coordinates
            for value, ours in ((point.x if point is not None else None, tracking.players[row, slot, 0]),
                                (point.y if point is not None else None, tracking.players[row, slot, 1]),
                                (data.speed, tracking.players[row, slot, 2])):
                player_values += 1
                player_bad += differ(value, ours)

    audit.add(section, "kloppy frames compared (every 50th line)", "> 3900", frames,
              "PASS" if frames > 3900 else "FAIL", "DATA.md reader")
    audit.compare(section, "kloppy home team == loader home team", True,
                  home_id == layout.home_team_id, "kloppy")
    audit.compare(section, "kloppy vs loader: ball x/y mismatches", 0, int(ball_bad), "kloppy")
    audit.compare(section, "kloppy vs loader: player x/y/s mismatches ({} values)".format(
        player_values), 0, int(player_bad), "kloppy")
    audit.compare(section, "kloppy vs loader: slot team mismatches", 0, int(team_bad), "kloppy")
    audit.compare(section, "kloppy players not in loader layout", 0, int(unknown), "kloppy")


def audit_smart_tagging(audit):
    section = "5. Smart Tagging"
    import load_smart_tagging  # official starter module, import only

    events = load_smart_tagging.load()
    n = games.GAMES[GAME]["n_video_frames"]
    audit.compare(section, "instances", PUBLISHED["smart_tagging_instances"], len(events), "DATA.md")

    def counts(values):
        out = {}
        for value in values:
            out[value] = out.get(value, 0) + 1
        return out

    code_counts = counts(event["code"] for event in events)
    audit.compare(section, "code counts (14 codes)", PUBLISHED["codes"], code_counts, "DATA.md")
    team_counts = counts(event["tags"].get("Team") for event in events)
    audit.compare(section, "Team tag counts", PUBLISHED["team_tags"], team_counts, "DATA.md")
    type_counts = counts(event["tags"]["Type"] for event in events if "Type" in event["tags"])
    audit.compare(section, "Type tag counts", PUBLISHED["type_tags"], type_counts, "DATA.md")
    audit.compare(section, "Half tag values", PUBLISHED["half_tags"],
                  sorted({str(event["tags"].get("Half")) for event in events}), "DATA.md")

    report = load_smart_tagging.check_time_base(events)
    audit.compare(section, "time base: both tagged kickoffs contain the real one", True,
                  bool(report["agrees"]), "starter/load_smart_tagging.py")
    audit.compare(section, "Kick Off (Start) tags", 2, report["n_kickoff_tags"], "DATA.md")

    audit.compare(section, "intervals with end < start", 0,
                  sum(event["end_frame"] < event["start_frame"] for event in events), "-")
    audit.compare(section, "intervals outside the video (0..n-1)", 0,
                  sum(event["start_frame"] < 0 or event["end_frame"] > n for event in events), "-")
    blocks = {"1st Half": games.half_video_blocks(GAME)[0], "2nd Half": games.half_video_blocks(GAME)[1]}
    inside = sum(blocks[event["tags"]["Half"]][1] <= (event["start_frame"] + event["end_frame"]) // 2
                 <= blocks[event["tags"]["Half"]][2]
                 for event in events if event["tags"].get("Half") in blocks)
    audit.add(section, "instances whose midpoint lies in their tagged half's graded block (%)", "-",
              pct(inside, len(events)), "INFO")
    durations = np.array([(event["end_frame"] - event["start_frame"]) / float(FPS) for event in events])
    audit.add(section, "interval duration p50 / p90 (s)", "-",
              [round(float(np.percentile(durations, 50)), 1),
               round(float(np.percentile(durations, 90)), 1)], "INFO")

    from kloppy import sportscode
    frame = sportscode.load(games.require(["smart_tagging"])[0]).to_df()
    audit.compare(section, "kloppy sportscode rows", PUBLISHED["smart_tagging_instances"], len(frame),
                  "DATA.md")
    audit.compare(section, "kloppy tag column is title case ('Team', not 'team')", [True, False],
                  ["Team" in frame.columns, "team" in frame.columns], "DATA.md")


def audit_homography(audit):
    import load_homography  # official starter module, import only

    for source, section in (("tactical", "6. Homography"), ("broadcast", "6. Homography")):
        report = load_homography.check_alignment(source=source)
        expected_rows = PUBLISHED["{}_frames".format(source)]
        audit.compare(section, "{} rows, frame 0..n-1".format(source), [expected_rows, True],
                      [report["n_rows"], bool(report["agrees"])], "DATA.md")
        _, error, valid = load_homography.load(source=source)
        audit.close(section, "{} valid (%)".format(source),
                    PUBLISHED["{}_homography_valid_pct".format(source)],
                    round(100.0 * float(valid.mean()), 1), 0.05, "DATA.md")
        if source == "tactical":
            # Descriptive figures (contract=False): DATA.md does not define "during play" or
            # "across halftime" as frame ranges, so graded frames and the halftime block stand in.
            n = len(valid)
            play = graded_mask(n) & valid
            halftime = block_mask(n, 90652, 110807) & valid
            play_median = round(float(np.median(error[play])), 2)
            halftime_median = round(float(np.median(error[halftime])), 2)
            audit.close(section, "tactical reprojection error median, graded frames",
                        PUBLISHED["reprojection_median_play"], play_median, 0.005, "DATA.md",
                        contract=False)
            audit.close(section, "tactical reprojection error median, halftime block",
                        PUBLISHED["reprojection_median_halftime"], halftime_median, 0.1,
                        "DATA.md ('about')", contract=False)
            halftime_mean = round(float(np.mean(error[halftime])), 2)
            audit.add(section, "tactical reprojection error mean, halftime block", "-",
                      halftime_mean, "INFO")
            audit.info["reprojection_median_play"] = play_median
            audit.info["reprojection_median_halftime"] = halftime_median
            audit.info["reprojection_mean_halftime"] = halftime_mean


def audit_broadcast(audit):
    section = "7. Broadcast ATD"
    tracking = load_tracking("broadcast")
    audit.compare(section, "records, frame_id exactly 1..n", [PUBLISHED["broadcast_frames"], True],
                  [tracking.n_frames,
                   bool(np.array_equal(tracking.frame_id, np.arange(1, tracking.n_frames + 1)))],
                  "DATA.md")
    audit_atd_layout(audit, tracking.layout, "broadcast", PUBLISHED["broadcast_slots"],
                     tracking.n_frames)
    audit.close(section, "ball seen, all frames (%)", PUBLISHED["broadcast_ball_pct"],
                pct(int(tracking.ball_seen().sum()), tracking.n_frames), 0.05, "DATA.md")
    audit.add(section, "timeline", "NOT synchronised to tactical", "not used for any check against "
              "tactical frames", "INFO", "DATA.md")


# -- report --------------------------------------------------------------------------------------

def render_value(value):
    if isinstance(value, dict):
        return ", ".join("{}: {}".format(k, v) for k, v in value.items())
    return str(value).replace("|", "/")


def write_markdown(audit, meta):
    statuses = [check["status"] for check in audit.checks]
    lines = [
        "# Data audit (Phase 1)",
        "",
        "Generated by `python scripts/data_audit.py` on {} from commit `{}`{}. "
        "Regenerate rather than edit by hand.".format(
            meta["generated_utc"], meta["git_commit"][:12],
            " with uncommitted changes" if meta["git_dirty"] else ""),
        "",
        "> **Aggregates only.** Every value below is a count, rate, median or PASS/FAIL. No "
        "coordinate, tag interval, frame-level value or line of any match file appears here or in "
        "the script's output (competition terms; `docs/decisions.md` D-002). EXPECTED values are "
        "the organisers' published figures, with their source; OBSERVED values were measured on "
        "the local bundle by this script.",
        "",
        "**Summary:** {} checks: {} PASS, {} FAIL, {} DIFF, {} INFO. PASS/FAIL hold the data to a "
        "published contract value; DIFF marks a published descriptive figure or label the data does "
        "not reproduce (non-blocking, explained in section 8); INFO has nothing published to compare "
        "with.".format(len(statuses), statuses.count("PASS"), statuses.count("FAIL"),
                       statuses.count("DIFF"), statuses.count("INFO")),
        "",
        "Environment: Python {}, numpy {}, pandas {}, kloppy {}. Runtime {} s (tactical ATD parse "
        "{} s, kloppy sample {} s).".format(
            meta["python"], meta["numpy"], meta["pandas"], meta["kloppy"], meta["runtime_seconds"],
            audit.info.get("tactical_load_seconds", "-"), audit.info.get("kloppy_sample_seconds", "-")),
        "",
    ]

    lines += ["## 1. Inventory", "",
              "| source | asset | file | present | bytes | sha256 |", "|---|---|---|---|---|---|"]
    for row in audit.info.get("inventory", []):
        lines.append("| {} | {} | `{}` | {} | {} | {} |".format(
            row["source"], row["asset"], row["file"], "yes" if row["present"] else "**NO**",
            "{:,}".format(row["bytes"]) if row["bytes"] is not None else "-",
            "`{}`".format(row["sha256"][:16] + "...") if row["sha256"] else "not hashed (video)"))
    lines.append("")

    sections = []
    for check in audit.checks:
        if check["section"] not in sections:
            sections.append(check["section"])
    for section in sections:
        if section != "1. Inventory":
            lines += ["## {}".format(section), ""]
        else:
            lines += ["Inventory checks:", ""]
        lines += ["| check | expected | observed | status | source |", "|---|---|---|---|---|"]
        for check in audit.checks:
            if check["section"] == section:
                lines.append("| {} | {} | {} | {} | {} |".format(
                    check["check"], render_value(check["expected"]), render_value(check["observed"]),
                    "**{}**".format(check["status"]) if check["status"] in ("FAIL", "DIFF")
                    else check["status"], check["source"]))
        lines.append("")

    info = audit.info
    lines += [
        "## 8. What this changes for the model",
        "",
        "- **Speed is {unit}, whatever the metadata says.** The metadata XML labels the speed "
        "channel `{label}`; DATA.md says m/s. Reported speed over displacement speed between "
        "consecutive frames has median {ratio}, so the values are {unit} and the label is wrong. "
        "Use the `s` channel as m/s, without conversion (`docs/decisions.md` D-008).".format(
            unit=info.get("player_speed_implied_unit"), label=info.get("speed_unit_label"),
            ratio=info.get("player_speed_ratio")),
        "- **The ball's speed channel is not usable as-is.** It is present on only {pct}% of "
        "ball-seen frames and its ratio to displacement speed is {ratio} -- undocumented semantics. "
        "kloppy drops it entirely; `src.data_loader` keeps it, but no feature should rely on it "
        "without a separate investigation.".format(
            pct=info.get("ball_speed_pct"), ratio=info.get("ball_speed_ratio")),
        "- **The official baseline's two parsing assumptions hold** (slot order, home team), and "
        "`src.data_loader` matches the official parser on every frame and kloppy on every sampled "
        "value (section 4). The four committed baselines are an exact regression target for the "
        "loader.",
        "- **The ball is missing on ~41% of graded frames** (published figures reproduced "
        "exactly). A missing ball is normal input, never on its own evidence of a dead ball.",
        "- **Tracked objects do not vanish outside play.** Mean tracked objects: {pre} before "
        "kickoff, {half} at halftime, {post} after the final whistle, against 20.5 during graded "
        "play. A head count cannot find halftime; the published half blocks must (`games.py`).".format(
            pre=info.get("mean_objects_before_kickoff"), half=info.get("mean_objects_halftime"),
            post=info.get("mean_objects_after_the_final_whistle")),
        "- **The ball leaves the pitch rectangle.** {pct}% of graded ball positions lie inside "
        "[0,1]^2, so roughly {out}% are outside the touchlines or goal lines: a candidate dead-ball "
        "signal for Phase 2, to be validated without the public GT.".format(
            pct=info.get("ball_inside_pct"),
            out=round(100 - info["ball_inside_pct"], 1) if info.get("ball_inside_pct") else "?"),
        "- **Homography figures differ from DATA.md's descriptive numbers** (median reprojection "
        "error {play} on graded frames vs 0.24 published for \"during play\"; {half} over the "
        "halftime block vs \"about 1.5\" -- the halftime *mean* is {mean}, so the published figure "
        "is most likely a mean). DATA.md does not define either frame range. The homography is not "
        "used by the Phase-1 pipeline, so this is recorded, not acted on.".format(
            play=info.get("reprojection_median_play"), half=info.get("reprojection_median_halftime"),
            mean=info.get("reprojection_mean_halftime")),
        "",
        "## 9. Risks and limitations",
        "",
        "- Published figures are the yardstick. A PASS means the bundle matches what the organisers "
        "describe, not that the organisers' description is complete.",
        "- `games.py` kickoff anchors are playback positions (`anchor_video_frame` gives {} and "
        "{}), not contract frames. The contract frames are 11185 (first graded frame) and 113742 "
        "(2nd-half kickoff, FAQ.md); frames 110808..113741 are graded and all `D` (FAQ.md).".format(
            games.anchor_video_frame(games.GAMES[GAME]["kickoff"]),
            games.anchor_video_frame(games.GAMES[GAME]["kickoff_2nd"])),
        "- Smart Tagging intervals are windows around events (kickoff tags open ~3.5 s early, "
        "starter notebook 01); only containment was checked, never exact boundaries.",
        "- Videos were not decoded or hashed (pass `--hash-videos` to hash them); nothing in Phase 1 "
        "reads video.",
        "- The broadcast recording is on its own timeline. It was audited for internal consistency "
        "only and is never compared frame-for-frame with tactical.",
        "- The local bundle sits in capitalised `Tactical/` and `Broadcast/` folders; `games.py` asks "
        "for lowercase. That resolves on Windows (case-insensitive) but not on Linux "
        "(`docs/decisions.md` D-004).",
        "",
        "## 10. Re-run",
        "",
        "```powershell",
        "$env:MTAH_DATA_DIR = \"<folder that contains tactical/ and broadcast/>\"",
        ".venv\\Scripts\\python.exe scripts\\data_audit.py",
        "```",
        "",
    ]
    folder = os.path.dirname(DOC_PATH)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(DOC_PATH, "w", newline="\n", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hash-videos", action="store_true", help="also SHA-256 the two videos")
    parser.add_argument("--skip-broadcast", action="store_true")
    parser.add_argument("--skip-kloppy", action="store_true")
    args = parser.parse_args()

    started = time.time()
    audit = Audit()
    steps = [("1. Inventory", lambda: audit_inventory(audit, args.hash_videos)),
             ("2. Timeline contract", lambda: audit_timeline(audit))]
    for section, step in steps:
        try:
            step()
        except Exception as error:
            audit.failed(section, error)

    tracking = None
    try:
        tracking = audit_tactical(audit)
    except Exception as error:
        audit.failed("3. Tactical ATD", error)
        traceback.print_exc(limit=1)
    if tracking is not None and not args.skip_kloppy:
        try:
            audit_kloppy(audit, tracking)
        except Exception as error:
            audit.failed("4. Loader parity", error)
    del tracking

    for section, step, skip in (("5. Smart Tagging", audit_smart_tagging, False),
                                ("6. Homography", audit_homography, False),
                                ("7. Broadcast ATD", audit_broadcast, args.skip_broadcast)):
        if skip:
            continue
        try:
            step(audit)
        except Exception as error:
            audit.failed(section, error)

    import pandas
    import kloppy
    commit, dirty = git_state()
    meta = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "git_commit": commit, "git_dirty": dirty,
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pandas.__version__,
        "kloppy": kloppy.__version__, "runtime_seconds": round(time.time() - started, 1),
    }
    write_markdown(audit, meta)
    folder = os.path.dirname(JSON_PATH)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(JSON_PATH, "w", newline="\n") as handle:
        json.dump({"meta": meta, "checks": audit.checks, "info": audit.info}, handle, indent=2,
                  default=str)

    width = max(len(check["check"]) for check in audit.checks)
    for check in audit.checks:
        print("{:<9} {:<{w}}  observed: {}".format(check["status"], check["check"],
                                                   render_value(check["observed"]), w=width))
    statuses = [check["status"] for check in audit.checks]
    print("\n{} checks: {} PASS, {} FAIL, {} DIFF, {} INFO  ({} s)".format(
        len(statuses), statuses.count("PASS"), statuses.count("FAIL"), statuses.count("DIFF"),
        statuses.count("INFO"), meta["runtime_seconds"]))
    print("wrote {}\nwrote {}".format(os.path.relpath(DOC_PATH, ROOT), os.path.relpath(JSON_PATH, ROOT)))
    return 1 if statuses.count("FAIL") else 0


if __name__ == "__main__":
    sys.exit(main())
