"""Verify the regenerated official baseline against the committed files, then promote the fallback.

    python starter/baseline.py --out outputs/baseline_regen     # the official generator, unchanged
    python scripts/verify_baseline.py                           # this

Experiment E0 of the playbook: the official baseline is reproducible. For each of the four
variants, with no match data read here at all:

  - the official validator, run as its own process, accepts the regenerated file;
  - its labels equal the committed file's on every one of the 199319 frames;
  - its bytes equal the committed file's once line endings are normalised. The committed copies
    reach a Windows checkout as CRLF through git's autocrlf, and starter/baseline.py writes the
    platform's own line ending, so raw bytes are not the comparison that means anything.

Then baseline_smooth50 -- the published floor, spellF1 0.2433 on the full match (EVALUATION.md),
which this repository cannot measure -- is rewritten by src.submission as an LF file in
outputs/fallback/, validated again, and recorded in a manifest with the commit, the commands, the
environment and the SHA-256 of every file involved. That is the always-valid submission every later
phase must beat and may fall back to. outputs/ is gitignored: it is a set of labels for the match.
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import games  # noqa: E402
from src import submission  # noqa: E402

# Params
VARIANTS = ("baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125")
FALLBACK = "baseline_smooth50"
COMMITTED_DIR = os.path.join(ROOT, "evaluation", "results")
REGEN_DIR = os.path.join(ROOT, "outputs", "baseline_regen")
FALLBACK_DIR = os.path.join(ROOT, "outputs", "fallback")
VALIDATOR = os.path.join(ROOT, "evaluation", "validate_submission.py")
# Published by the organisers for the full match; quoted, never measured here (private GT).
PUBLISHED_FULL_MATCH = {"spellF1": 0.2433, "macroF1": 0.6724, "source": "EVALUATION.md"}


def run_validator(path):
    """(exit code, output lines) of the official validator, run exactly as a participant would."""
    completed = subprocess.run([sys.executable, VALIDATOR, path], capture_output=True, text=True)
    return completed.returncode, (completed.stdout + completed.stderr).strip().splitlines()


def git_state():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip())
        return commit, dirty
    except Exception:  # provenance is then "unknown", not fatal
        return "unknown", None


def relative(path):
    return os.path.relpath(path, ROOT).replace(os.sep, "/")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--regen", default=REGEN_DIR, help="where starter/baseline.py wrote to")
    args = parser.parse_args()

    rows, ok = [], True
    for name in VARIANTS:
        regen = os.path.join(args.regen, name + ".csv")
        committed = os.path.join(COMMITTED_DIR, name + ".csv")
        if not os.path.isfile(regen):
            print("FAIL  {} missing -- run: python starter/baseline.py --out {}".format(
                relative(regen), relative(args.regen)))
            return 1
        code, lines = run_validator(regen)
        mismatches = int(np.sum(submission.read_labels(regen) != submission.read_labels(committed)))
        same_bytes = (submission.sha256(regen, normalise_newlines=True)
                      == submission.sha256(committed, normalise_newlines=True))
        row = {"variant": name, "validator_exit": code, "validator": lines[0] if lines else "",
               "label_mismatches": mismatches, "same_bytes_lf_normalised": same_bytes,
               "sha256_regen": submission.sha256(regen),
               "sha256_regen_lf": submission.sha256(regen, normalise_newlines=True),
               "sha256_committed_lf": submission.sha256(committed, normalise_newlines=True)}
        rows.append(row)
        ok = ok and code == 0 and mismatches == 0 and same_bytes
        print("{}  {:<20} validator exit {}  mismatching frames {}  identical bytes (LF) {}".format(
            "PASS" if code == 0 and mismatches == 0 and same_bytes else "FAIL", name, code,
            mismatches, same_bytes))

    # Promote the fallback: same labels, rewritten LF so its hash does not depend on the platform.
    labels = submission.read_labels(os.path.join(args.regen, FALLBACK + ".csv"))
    fallback = submission.write(os.path.join(FALLBACK_DIR, FALLBACK + ".csv"), labels)
    code, lines = run_validator(fallback)
    strict = submission.check_strict(fallback)
    matches_committed = (submission.sha256(fallback)
                         == submission.sha256(os.path.join(COMMITTED_DIR, FALLBACK + ".csv"),
                                              normalise_newlines=True))
    ok = ok and code == 0 and strict is None and matches_committed
    print("{}  fallback {}  validator exit {}  strict check {}  == committed (LF) {}".format(
        "PASS" if code == 0 and strict is None and matches_committed else "FAIL", relative(fallback),
        code, "OK" if strict is None else strict, matches_committed))

    import pandas
    commit, dirty = git_state()
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"])
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "purpose": "Phase-1 fallback: official baseline_smooth50 regenerated from the ATD",
        "status": "PASS" if ok else "FAIL",
        "git_commit": commit,
        "git_dirty": dirty,
        "commands": ["python starter/baseline.py --out {}".format(relative(args.regen)),
                     "python scripts/verify_baseline.py"],
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pandas.__version__},
        "inputs": {"atd_meta_sha256": submission.sha256(meta_path),
                   "atd_raw_sha256": submission.sha256(raw_path)},
        "variants": rows,
        "fallback": {
            "path": relative(fallback),
            "sha256": submission.sha256(fallback),
            "line_endings": "LF",
            "validator_exit": code,
            "validator_output": lines,
            "strict_check": strict or "OK",
            "identical_to_committed_lf": matches_committed,
            "label_counts": {label: int(np.sum(labels == label)) for label in ("H", "A", "D", "X")},
            "published_full_match_score": PUBLISHED_FULL_MATCH,
            "measured_full_match_score": None,
        },
    }
    path = os.path.join(FALLBACK_DIR, "manifest.json")
    with open(path, "w", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
    print("wrote {}  (fallback sha256 {})".format(relative(path), manifest["fallback"]["sha256"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
