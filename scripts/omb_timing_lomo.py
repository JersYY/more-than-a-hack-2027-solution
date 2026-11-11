"""Validate E8's unchanged timing grid by leave-one-match-out selection.

Each fold selects only on the other matches, seeds 0/1; its held-out match
uses seeds 2/3. No competition labels are loaded. DFL and SG3 were previously
used by the project: this checks stability, not independent external accuracy.
"""
import json
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from omb_transition_probe import GRID, MIN_GAIN, guard, prepare, score
from src.config import load
from src.degrade import load_realism
from src.omb import benchmark_matches

REFERENCE = "0:4.5"
CANDIDATE = "8:4.5"


def select_without(training, held_out):
    """Select a variant using means across every match except held_out."""
    averages = {variant: float(np.mean([value for name, value in values.items()
                                        if name != held_out]))
                for variant, values in training.items()}
    best = max(averages, key=averages.get)
    return best if averages[best] > averages[REFERENCE] + MIN_GAIN else REFERENCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="configs/hybrid_v2.toml",
                        help="External-only calibration base; excludes D-026 target diagnostics")
    parser.add_argument("--tag", default="external_only")
    args = parser.parse_args()
    base = load(str(ROOT / args.base))
    variants = {"{}:{}".format(*variant): variant for variant in GRID if guard(base, variant)}
    matches = benchmark_matches()
    if len(matches) != 8 or "SG3" not in matches:
        raise ValueError("This validation requires all seven DFL matches and SG3")
    training = {key: {} for key in variants}
    testing = {key: {} for key in variants}
    for name in matches:
        for destination, seeds in ((training, [0, 1]), (testing, [2, 3])):
            cases = prepare([name], "dead_weighted", seeds, 0.9, base, load_realism())
            for key, variant in variants.items():
                destination[key][name] = score(cases, base, variant)["spellF1"]
            del cases
        print("scored", name, flush=True)
    rows = []
    for name in matches:
        selected = select_without(training, name)
        reference = testing[REFERENCE][name]
        value = testing[selected][name]
        rows.append({"held_out": name, "selected": selected, "reference": reference,
                     "selected_score": value, "gain": value - reference,
                     "fixed_timing_v1": testing[CANDIDATE][name]})
        print("held out {}: {} {:.4f} -> {:.4f}".format(name, selected, reference, value), flush=True)
    gains = [row["gain"] for row in rows]
    metrica_gain = testing[CANDIDATE]["SG3"] - testing[REFERENCE]["SG3"]
    report = {"experiment": "E8 LOMO", "base": args.base, "grid": GRID,
              "train_seeds": [0, 1], "test_seeds": [2, 3],
              "training": training, "testing": testing, "folds": rows,
              "mean_lomo_gain": float(np.mean(gains)),
              "min_lomo_gain": min(gains), "metrica_fixed_candidate_gain": metrica_gain,
              "mean_fixed_reference": float(np.mean(list(testing[REFERENCE].values()))),
              "mean_fixed_candidate": float(np.mean(list(testing[CANDIDATE].values()))),
              "acceptance_gate": bool(np.mean(gains) > MIN_GAIN and min(gains) >= 0 and metrica_gain >= 0),
              "limitations": "Existing calibration datasets; timing grid only, other parameters remain fixed."}
    suffix = "_" + args.tag if args.tag else ""
    path = ROOT / ("outputs/omb/results/e8_timing_lomo" + suffix + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("LOMO mean gain {:+.4f}; minimum {:+.4f}; Metrica gain {:+.4f}; gate={}".format(
        np.mean(gains), min(gains), metrica_gain, report["acceptance_gate"]), flush=True)


if __name__ == "__main__":
    main()
