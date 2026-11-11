"""E8: bounded timing/turnover experiment, using other matches only.

Run: python scripts/omb_transition_probe.py
Positive shifts delay team evidence; shifts never cross half boundaries.
Grid and split below are fixed before scoring. Select on four DFL matches,
seeds 0/1; evaluate the selected candidate once on three held-out matches,
seeds 2/3. This is a new split of an existing calibration benchmark, not a
claim of validation on previously unseen datasets. Never alters final.toml.
"""

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "evaluation", ROOT / "scripts"):
    sys.path.insert(0, str(folder))

import possession_metrics as pm
from omb_calibrate import prepare
from src.config import load
from src.dead_ball import dead_evidence
from src.degrade import load_realism
from src.possession_evidence import team_evidence, shift_team_evidence
from src.preprocess import frame_context
from src.features import build_features
from src.synthetic import SCENARIOS
from src.temporal import decode

TRAIN = ["J03WMX", "J03WN1", "J03WOH", "J03WOY"]
HOLDOUT = ["J03WPY", "J03WQQ", "J03WR9"]
GRID = [(0, 4.5)] + [(shift, 4.5) for shift in (-12, -8, -4, 4, 8, 12)] + [(0, cost) for cost in (3.0, 5.7)]
MIN_GAIN = 0.002


def shift_evidence(team, context, frames):
    """Shift with zero padding inside each inclusive half block."""
    return shift_team_evidence(team, context, frames)


def prediction(case, config, variant):
    _, truth, context, features = case
    shift, cost = variant
    temporal = dataclasses.replace(config.temporal, turnover_cost=cost)
    team = shift_evidence(team_evidence(features, config.evidence), context, shift)
    return decode(team, dead_evidence(features, config.dead_ball), context, temporal)


def guard(config, variant):
    for build in SCENARIOS.values():
        match = build()
        context = frame_context(match.tracking.n_frames, match.blocks, match.tracking)
        case = ("synthetic", match.labels, context,
                build_features(match.tracking, context, config))
        result = pm.evaluate(match.labels, prediction(case, config, variant))
        if result["spell_f1_mean"] < 1 - 1e-12 or result["accuracy"] < 0.9:
            return False
    return True


def score(cases, config, variant, diagnostics=False):
    per_match = {}
    short = {"total": 0, "no_same_team_overlap": 0, "best_iou_below_0_5": 0}
    for case in cases:
        name, truth, _, _ = case
        pred = prediction(case, config, variant)
        runs = pm.valid_runs(truth)
        pred = pm.fill_abstentions(pred, runs)
        value = pm.spell_scores(truth, pred, runs)["mean_f1"]
        per_match.setdefault(name, []).append(value)
        if diagnostics:
            spells = pm.possession_spells(pred, runs)
            for start, stop, team in pm.possession_spells(truth, runs):
                if stop - start >= 50:
                    continue
                short["total"] += 1
                intersections = [(max(0, min(stop, b) - max(start, a)), a, b)
                                 for a, b, code in spells if code == team]
                best = max((overlap / (stop - start + b - a - overlap)
                            for overlap, a, b in intersections), default=0)
                short["no_same_team_overlap"] += int(best == 0)
                short["best_iou_below_0_5"] += int(best < 0.5)
    means = {name: float(np.mean(values)) for name, values in per_match.items()}
    return {"spellF1": float(np.mean(list(means.values()))), "per_match": means,
            "short_spells_under_2s": short if diagnostics else None}


def main():
    config = load(str(ROOT / "configs/final.toml"))
    accepted = [variant for variant in GRID if guard(config, variant)]
    print("Synthetic guard: {}/{} candidates pass".format(len(accepted), len(GRID)), flush=True)
    train = prepare(TRAIN, "dead_weighted", [0, 1], 0.9, config, load_realism())
    rows = []
    for variant in accepted:
        result = score(train, config, variant)
        rows.append({"variant": variant, **result})
        print("train shift={} cost={}: {:.4f}".format(*variant, result["spellF1"]), flush=True)
    baseline = next(row for row in rows if row["variant"] == (0, 4.5))
    best = max(rows, key=lambda row: row["spellF1"])
    chosen = best["variant"] if best["spellF1"] > baseline["spellF1"] + MIN_GAIN else (0, 4.5)
    del train
    holdout = prepare(HOLDOUT, "dead_weighted", [2, 3], 0.9, config, load_realism())
    reference = score(holdout, config, (0, 4.5), diagnostics=True)
    candidate = score(holdout, config, chosen, diagnostics=True) if chosen != (0, 4.5) else reference
    gains = {name: candidate["per_match"][name] - reference["per_match"][name] for name in HOLDOUT}
    promote = chosen != (0, 4.5) and candidate["spellF1"] > reference["spellF1"] + MIN_GAIN and min(gains.values()) >= 0
    report = {"experiment": "E8", "train": TRAIN, "holdout": HOLDOUT,
              "grid": GRID, "guard_passed": accepted, "train_results": rows,
              "selected": chosen, "holdout_reference": reference,
              "holdout_candidate": candidate, "holdout_gains": gains,
              "promotion_gate_passed": promote,
              "note": "DFL previously used in calibration; split evidence is not external validation."}
    folder = ROOT / "outputs/omb/results"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "e8_transition_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("holdout baseline={:.4f} candidate={:.4f} promotion_gate={}".format(
        reference["spellF1"], candidate["spellF1"], promote), flush=True)
    print("short-spell diagnostics:", json.dumps(reference["short_spells_under_2s"]), flush=True)


if __name__ == "__main__":
    main()
