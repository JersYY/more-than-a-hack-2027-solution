"""Experiment E6: calibrate configuration parameters on the other-match benchmark.

    python scripts/omb_calibrate.py --base configs/hybrid_v1.toml --out configs/hybrid_v2.toml
    python scripts/omb_calibrate.py --base configs/hybrid_v1.toml --lomo     # leave-one-match-out

Coordinate descent over GRID -- a small grid per parameter, declared here before any score was
seen -- maximising the mean official spellF1 over every benchmark match and degradation seed.
A value replaces the current one only if it gains more than MIN_GAIN; passes repeat until nothing
moves. --lomo repeats the whole descent with each match held out and scores the held-out match,
which is the honest estimate of what calibration buys on a match it never saw (docs/experiments.md
§5 selection rule).

Only the benchmark is used: other matches' labels (D-017). The competition match, its public
window and its private ground truth play no part.

Features are computed once per (match, seed) and cached: the parameters tuned here act only on
evidence and decoding. dead_ball.still_mps and dead_ball.activity_window_s shape the features
themselves, so they stay at the base configuration's values.
"""

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import possession_metrics as pm  # noqa: E402
from src.config import from_dict, load  # noqa: E402
from src.dead_ball import dead_evidence  # noqa: E402
from src.degrade import load_profile  # noqa: E402
from src.features import build_features  # noqa: E402
from src.omb import benchmark_matches, load_benchmark  # noqa: E402
from src.possession_evidence import team_evidence  # noqa: E402
from src.preprocess import frame_context  # noqa: E402
from src.temporal import decode  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "outputs", "omb", "results")
MIN_GAIN = 0.002            # spellF1; smaller moves are not worth a change of configuration
MAX_PASSES = 3
# Declared before any benchmark score existed. Each list contains the hybrid_v1 default.
GRID = (
    ("temporal", "evidence_scale", [0.05, 0.1, 0.2, 0.4, 0.8]),
    ("temporal", "turnover_cost", [3.0, 4.5, 5.7, 7.0, 9.0, 12.0]),
    ("temporal", "stop_cost", [4.0, 5.5, 6.9, 8.5, 10.0]),
    ("temporal", "restart_cost", [4.0, 5.0, 5.9, 7.0, 8.5]),
    ("temporal", "alive_bonus", [0.3, 0.7, 1.2]),
    ("dead_ball", "activity_threshold_mps", [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]),
    ("dead_ball", "activity_scale_mps", [0.2, 0.4, 0.8]),
    ("dead_ball", "activity_max_score", [1.0, 2.0, 3.0, 4.0]),
    ("dead_ball", "out_score", [2.0, 4.0, 6.0]),
    ("dead_ball", "still_score", [0.0, 1.0, 2.0]),
    ("dead_ball", "moving_score", [0.0, 1.0, 2.0]),
    # Added after the first benchmark comparison (SG3: low dead-ball recall), D-023.
    ("dead_ball", "missing_score", [0.0, 0.5, 1.0, 2.0, 3.0]),
    ("dead_ball", "missing_min_s", [2.0, 5.0, 10.0]),
    ("evidence", "control_m", [0.7, 1.0, 1.5]),
    ("evidence", "free_m", [1.5, 2.5, 4.0]),
    ("evidence", "flight_mps", [6.0, 8.0, 12.0]),
)


def prepare(matches, degrade, seeds, dead_share, base):
    """[(name, labels, context, features)] -- the expensive part, once."""
    profile = load_profile()
    cases = []
    for name in matches:
        for seed in seeds:
            match = load_benchmark(name, degrade, seed, dead_share, profile)
            context = frame_context(match.tracking.n_frames, match.blocks, match.tracking)
            features = build_features(match.tracking, context, base, match.pitch)
            cases.append((name, match.labels, context, features))
    return cases


def spell_f1(case, config):
    _, labels, context, features = case
    states = decode(team_evidence(features, config.evidence),
                    dead_evidence(features, config.dead_ball), context, config.temporal)
    return pm.evaluate(labels, states)["spell_f1_mean"]


def with_value(config, section, key, value):
    data = config.to_dict()
    data[section][key] = value
    return from_dict(data, source="calibration")


def mean_score(cases, config):
    return float(np.mean([spell_f1(case, config) for case in cases]))


def descend(cases, config, log=None):
    """Coordinate descent over GRID from config; returns (best config, best mean spellF1)."""
    best = mean_score(cases, config)
    for passes in range(MAX_PASSES):
        moved = False
        for section, key, values in GRID:
            current = getattr(getattr(config, section), key)
            scored = []
            for value in values:
                candidate = with_value(config, section, key, value)
                scored.append((mean_score(cases, candidate) if value != current else best, value))
            score, value = max(scored, key=lambda item: (item[0], -abs(item[1] - current)))
            if value != current and score > best + MIN_GAIN:
                config, best, moved = with_value(config, section, key, value), score, True
                if log is not None:
                    log.append({"pass": passes, "param": "{}.{}".format(section, key),
                                "from": current, "to": value, "spell_f1": round(score, 4)})
                    print("  pass {} {}.{}: {} -> {}   mean spellF1 {:.4f}".format(
                        passes, section, key, current, value, score), flush=True)
        if not moved:
            break
    return config, best


def to_toml(config, header):
    lines = ["# " + line for line in header.splitlines()] + [""]
    for section, values in config.to_dict().items():
        lines.append("[{}]".format(section))
        for key, value in values.items():
            if isinstance(value, bool):
                text = "true" if value else "false"
            elif isinstance(value, str):
                text = '"{}"'.format(value)
            else:
                text = repr(value)
            lines.append("{} = {}".format(key, text))
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", help="write the calibrated configuration here")
    parser.add_argument("--name", default="hybrid_v2")
    parser.add_argument("--degrade", default="dead_weighted")
    parser.add_argument("--dead-share", type=float, default=0.85)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--lomo", action="store_true", help="leave-one-match-out estimate")
    args = parser.parse_args()

    base = load(args.base)
    matches = benchmark_matches()
    started = time.time()
    cases = prepare(matches, args.degrade, args.seeds, args.dead_share, base)
    print("prepared {} cases from {} matches ({:.0f} s)".format(len(cases), len(matches),
                                                                time.time() - started), flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.lomo:
        rows = []
        for held_out in matches:
            train = [case for case in cases if case[0] != held_out]
            test = [case for case in cases if case[0] == held_out]
            tuned, _ = descend(train, base)
            rows.append({"held_out": held_out, "base": mean_score(test, base),
                         "calibrated": mean_score(test, tuned)})
            print("held out {}: base {:.4f} -> calibrated {:.4f}".format(
                held_out, rows[-1]["base"], rows[-1]["calibrated"]), flush=True)
        gains = [row["calibrated"] - row["base"] for row in rows]
        print("LOMO mean gain {:+.4f} (min {:+.4f}, max {:+.4f}) over {} held-out matches".format(
            np.mean(gains), np.min(gains), np.max(gains), len(gains)))
        with open(os.path.join(RESULTS_DIR, "lomo_{}.json".format(time.strftime("%Y%m%d-%H%M%S"))),
                  "w", newline="\n") as handle:
            json.dump({"args": vars(args), "rows": rows}, handle, indent=1)
        return 0

    log = []
    print("base {}: mean spellF1 {:.4f}".format(base.run.name, mean_score(cases, base)), flush=True)
    tuned, score = descend(cases, base, log)
    tuned = dataclasses.replace(tuned, run=dataclasses.replace(
        tuned.run, name=args.name,
        description="{} calibrated on the IDSSE/DFL benchmark (E6), degrade={} dead_share={}".format(
            base.run.name, args.degrade, args.dead_share)))
    print("calibrated: mean spellF1 {:.4f} ({:.0f} s)".format(score, time.time() - started))
    with open(os.path.join(RESULTS_DIR, "calibration_{}.json".format(time.strftime("%Y%m%d-%H%M%S"))),
              "w", newline="\n") as handle:
        json.dump({"args": vars(args), "log": log, "score": score, "config": tuned.to_dict()},
                  handle, indent=1)
    if args.out:
        header = ("Experiment E6: {} calibrated by scripts/omb_calibrate.py on the IDSSE/DFL\n"
                  "benchmark ({} matches, degrade={}, dead_share={}, seeds={}).\n"
                  "Mean benchmark spellF1 {:.4f}. Other matches' labels only (D-017); nothing\n"
                  "here was fitted to the competition match or its public window.").format(
                      base.run.name, len(matches), args.degrade, args.dead_share, args.seeds, score)
        with open(args.out, "w", newline="\n") as handle:
            handle.write(to_toml(tuned, header))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
