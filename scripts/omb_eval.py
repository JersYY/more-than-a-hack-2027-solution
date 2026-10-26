"""Score configurations on the other-match benchmark with the official metric.

    python scripts/omb_eval.py configs/baseline.toml configs/hybrid_v1.toml
    python scripts/omb_eval.py configs/hybrid_v1.toml --degrade dead_weighted --seeds 0 1 2
    python scripts/omb_eval.py --stats            # the benchmark's own label statistics
    python scripts/omb_eval.py configs/final.toml --degrade dead_weighted --realism   # E7

Real football, real labels -- of other matches (D-017). Every number comes from the official
possession_metrics.evaluate, on IDSSE/DFL matches parsed by src/omb.py and, unless --degrade none,
with the ball deleted in the ATD's rhythm by src/degrade.py -- and with --realism, smoothed and
losing players as the ATD does (E7). These are the only accuracy numbers
this project may use to choose between versions; the public 5-minute window is not one of them
(D-020).
"""

import argparse
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
from src.config import load  # noqa: E402
from src.degrade import MODES, load_profile, load_realism  # noqa: E402
from src.omb import benchmark_matches, label_stats, load_benchmark  # noqa: E402
from src.predict import predict  # noqa: E402
from src.preprocess import frame_context  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "outputs", "omb", "results")
METRICS = ("spell_f1_mean", "macro_f1", "alive_dead_f1", "team_acc_both_alive")


def score(match, config):
    context = frame_context(match.tracking.n_frames, match.blocks, match.tracking)
    codes = pm.label_codes(predict(match.tracking, context, config, pitch=match.pitch))
    result = pm.evaluate(match.labels, codes)
    row = {metric: result[metric] for metric in METRICS}
    row["n_pred_spells"], row["n_gt_spells"] = result["n_pred_spells"], result["n_gt_spells"]
    return row


def evaluate(configs, matches, degrade, seeds, dead_share, profile, realism=None):
    """{config name: {match: [row per seed]}}."""
    out = {config.run.name: {} for config in configs}
    for name in matches:
        for seed in seeds:
            match = load_benchmark(name, degrade, seed, dead_share, profile, realism)
            for config in configs:
                out[config.run.name].setdefault(name, []).append(score(match, config))
    return out


def summarise(results):
    """Mean over seeds per match, then mean and spread over matches, per config."""
    summary = {}
    for config_name, per_match in results.items():
        per_match_means = {match: {metric: float(np.mean([row[metric] for row in rows]))
                                   for metric in METRICS + ("n_pred_spells", "n_gt_spells")}
                           for match, rows in per_match.items()}
        spells = [values["spell_f1_mean"] for values in per_match_means.values()]
        summary[config_name] = {
            "per_match": per_match_means,
            "mean": {metric: float(np.mean([v[metric] for v in per_match_means.values()]))
                     for metric in METRICS},
            "spell_f1_min": float(np.min(spells)), "spell_f1_max": float(np.max(spells)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="*", help="configs/*.toml to score")
    parser.add_argument("--degrade", choices=MODES, default="renewal")
    parser.add_argument("--dead-share", type=float, default=0.85)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--realism", action="store_true",
                        help="also smooth and lose players as the ATD does (E7)")
    parser.add_argument("--matches", nargs="+", help="default: every downloaded match")
    parser.add_argument("--stats", action="store_true", help="print the benchmark's label statistics")
    parser.add_argument("--tag", default="", help="name for the results file")
    args = parser.parse_args()

    matches = args.matches or benchmark_matches()
    if not matches:
        print("no benchmark data: python scripts/omb_fetch.py")
        return 1
    if args.stats:
        for name in matches:
            match = load_benchmark(name)
            print(name, json.dumps(label_stats(match)), "speed ratio {:.2f}".format(
                match.speed_unit_ratio))
        if not args.configs:
            return 0

    configs = [load(path) for path in args.configs]
    started = time.time()
    realism = load_realism() if args.realism else None
    results = evaluate(configs, matches, args.degrade, args.seeds, args.dead_share, load_profile(),
                       realism)
    summary = summarise(results)

    print("degrade={} dead_share={} realism={} seeds={} matches={}  ({:.0f} s)".format(
        args.degrade, args.dead_share, args.realism, args.seeds, len(matches),
        time.time() - started))
    print("{:<22} {:>8} {:>8} {:>8} {:>8}   {}".format(
        "config", "spellF1", "macroF1", "aliveF1", "teamAcc", "per match spellF1 (pred/gt spells)"))
    for name, values in summary.items():
        print("{:<22} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}   {}".format(
            name, values["mean"]["spell_f1_mean"], values["mean"]["macro_f1"],
            values["mean"]["alive_dead_f1"], values["mean"]["team_acc_both_alive"],
            "  ".join("{} {:.3f} ({:.0f}/{:.0f})".format(m[-3:], v["spell_f1_mean"],
                                                         v["n_pred_spells"], v["n_gt_spells"])
                      for m, v in values["per_match"].items())))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "{}{}{}_{}.json".format(
        args.tag + "_" if args.tag else "", args.degrade, "_realism" if args.realism else "",
        time.strftime("%Y%m%d-%H%M%S")))
    with open(path, "w", newline="\n") as handle:
        json.dump({"args": vars(args), "summary": summary, "raw": results}, handle, indent=1)
    print("wrote", os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
