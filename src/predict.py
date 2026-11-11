"""The whole pipeline -- tracking in, one label per frame out -- and its command line.

    python -m src.predict --config configs/hybrid_v1.toml --output outputs/candidates/hybrid_v1.csv

predict() is pure: arrays in, labels out, no I/O, so the synthetic acceptance tests and the real
match run exactly the same code. run() adds the I/O. It loads the match through games.py (set
MTAH_DATA_DIR), writes the CSV with src.submission.write (LF, refused if invalid), and writes
<output>.manifest.json beside it: config digest, git commit, input and output SHA-256,
environment, runtime and aggregate diagnostics -- never frame-level values (D-002).
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import games  # noqa: E402
import possession_metrics as pm  # noqa: E402
from src import submission  # noqa: E402
from src.config import load  # noqa: E402
from src.dead_ball import dead_evidence  # noqa: E402
from src.features import build_features, nearest_team  # noqa: E402
from src.possession_evidence import team_evidence, shift_team_evidence  # noqa: E402
from src.preprocess import frame_context  # noqa: E402
from src.temporal import decode, majority  # noqa: E402

# Params
FPS = games.GAMES[games.GAME]["fps"]
LABELS = np.array(pm.LABELS)


def predict(tracking, context, config, tags=None, pitch=None):
    """(n,) '<U1' labels H/A/D/X for every frame. Ticket T6; tests/test_predict.py.

    tracking -- src.data_loader.Tracking;  context -- src.preprocess.FrameContext;
    config   -- src.config.Config;          tags -- Smart Tagging events, used only if
                config.smart_tagging.enabled (P1, ticket T9);
    pitch    -- (length, width) in metres for matches other than this one; default games.py's.

    decoder "majority": src.features.nearest_team -> src.temporal.majority. With
        output.ungraded = "keep" this is the official baseline file, frame for frame.
    decoder "viterbi": src.features.build_features -> team_evidence, dead_evidence (+ tag evidence)
        -> src.temporal.decode -> letters.
    output.ungraded = "X": X on every frame outside context.blocks; "keep": decoder output there.
    Deterministic: the same inputs give the same labels, bit for bit.
    """
    if config.smart_tagging.enabled:
        raise NotImplementedError("Smart Tagging evidence is P1 (ticket T9, experiment E5)")

    if config.temporal.decoder == "majority":
        codes = majority(nearest_team(tracking, pitch), config.temporal.window)
    else:
        features = build_features(tracking, context, config, pitch)
        team = shift_team_evidence(team_evidence(features, config.evidence), context,
                                   config.temporal.team_shift_frames)
        codes = decode(team,
                       dead_evidence(features, config.dead_ball), context, config.temporal)

    labels = LABELS[codes]
    if config.output.ungraded == "X":
        labels[~context.graded] = "X"
    return labels


def diagnostics(labels, context):
    """Aggregate, label-free description of a prediction, per half (docs/experiments.md §4).

    Counts, shares and durations only -- what a coach could read off the file, nothing that is a
    frame-level value (D-002), and nothing that needs a ground truth.
    """
    codes = pm.label_codes(labels)
    halves = []
    for half, first, last in context.blocks:
        runs = [(first, last + 1)]
        segments = pm.segments(codes, runs)
        spells = [stop - start for start, stop, code in segments if code in (pm.H, pm.A)]
        stoppages = [stop - start for start, stop, code in segments if code == pm.D]
        n = last - first + 1
        minutes = n / (FPS * 60.0)
        spell_s = np.array(spells, dtype=np.float64) / FPS
        halves.append({
            "half": half,
            "frames": n,
            "share": {name: round(float(np.sum(codes[first:last + 1] == code)) / n, 4)
                      for name, code in (("H", pm.H), ("A", pm.A), ("D", pm.D), ("X", pm.X))},
            "possessions": len(spells),
            "possession_s": {"p10": _percentile(spell_s, 10), "p50": _percentile(spell_s, 50),
                             "p90": _percentile(spell_s, 90),
                             "mean": round(float(spell_s.mean()), 2) if len(spell_s) else None},
            "possessions_under_1s": int(np.sum(spell_s < 1.0)),
            "stoppages": len(stoppages),
            "stoppage_s_p50": _percentile(np.array(stoppages, dtype=np.float64) / FPS, 50),
            "changes_per_minute": round((len(segments) - 1) / minutes, 2) if minutes else None,
        })
    total = sum(half["possessions"] for half in halves)
    return {"label_counts": {name: int(np.sum(codes == code)) for code, name in enumerate(pm.LABELS)},
            "possessions": total, "halves": halves}


def _percentile(values, q):
    return round(float(np.percentile(values, q)), 2) if len(values) else None


def run(config_path, output_path, data_dir=None):
    """Predict the tactical match and write the CSV and its manifest. Returns the CSV path.

    The manifest is <output_path>.manifest.json: run name, config path and digest, git commit and
    dirty flag, SHA-256 of the ATD files read and of the CSV written, Python and package versions,
    runtime, and aggregate diagnostics (label counts, possessions per half, mean duration).
    """
    from src.data_loader import load_tracking  # imported here: the tests import predict() alone

    started = time.time()
    config = load(config_path)
    tracking = load_tracking(data_dir=data_dir)
    context = frame_context(tracking.n_frames, tracking=tracking)
    labels = predict(tracking, context, config)
    submission.write(output_path, labels)

    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], data_dir=data_dir)
    import pandas
    commit, dirty = _git_state()
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run": {"name": config.run.name, "description": config.run.description},
        "config_path": os.path.relpath(config_path, ROOT).replace(os.sep, "/"),
        "config_digest": config.digest(),
        "config": config.to_dict(),
        "git_commit": commit,
        "git_dirty": dirty,
        "command": "python -m src.predict --config {} --output {}".format(
            os.path.relpath(config_path, ROOT).replace(os.sep, "/"),
            os.path.relpath(output_path, ROOT).replace(os.sep, "/")),
        "inputs_sha256": {"atd_meta": submission.sha256(meta_path),
                          "atd_raw": submission.sha256(raw_path)},
        "output_sha256": submission.sha256(output_path),
        "official_check": submission.check(output_path) or "OK",
        "strict_check": submission.check_strict(output_path) or "OK",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pandas.__version__},
        "runtime_seconds": round(time.time() - started, 1),
        "diagnostics": diagnostics(labels, context),
    }
    with open(output_path + ".manifest.json", "w", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
    return output_path


def _git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip())
        return commit, dirty
    except Exception:  # not a checkout: provenance is "unknown", not fatal
        return "unknown", None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="a configs/*.toml file")
    parser.add_argument("--output", required=True, help="CSV to write; the manifest goes beside it")
    parser.add_argument("--data-dir", help="the data bundle, if not in MTAH_DATA_DIR or ./data")
    args = parser.parse_args(argv)
    path = run(args.config, args.output, args.data_dir)

    with open(path + ".manifest.json") as handle:
        manifest = json.load(handle)
    print("wrote {}  ({} s)".format(os.path.relpath(path), manifest["runtime_seconds"]))
    print("  sha256 {}  official check {}  strict check {}".format(
        manifest["output_sha256"][:16], manifest["official_check"], manifest["strict_check"]))
    for half in manifest["diagnostics"]["halves"]:
        print("  half {half}: H {H:.1%} A {A:.1%} D {D:.1%} | {p} possessions, median {m} s, "
              "{u} under 1 s | {s} stoppages, median {sm} s | {c} changes/min".format(
                  half=half["half"], H=half["share"]["H"], A=half["share"]["A"],
                  D=half["share"]["D"], p=half["possessions"], m=half["possession_s"]["p50"],
                  u=half["possessions_under_1s"], s=half["stoppages"],
                  sm=half["stoppage_s_p50"], c=half["changes_per_minute"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
