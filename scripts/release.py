"""Build the submission package from the frozen configuration, and check the exact files.

    $env:MTAH_DATA_DIR = "<folder containing tactical/ and broadcast/>"
    python scripts/release.py                         # -> outputs/submission/

SUBMISSION.md asks for the files to be named after the team. For "Aziz goes to barcelona" this
writes:

    outputs/submission/Aziz goes to barcelona.csv    the possession file, from configs/final.toml
    outputs/submission/Aziz goes to barcelona.md     the technical document (docs/technical_report.md)
    outputs/submission/release_report.json           every check below, with hashes and versions

Every check runs on the exact file that would be uploaded, not on an earlier copy:
  - the official validator, evaluation/validate_submission.py, run as its own process;
  - the playbook's independent structure check: header exactly frame,label, 199,319 rows, frames
    0..199318 in order, no blanks, labels in H/A/D/X;
  - a second, independent generation must give the same bytes;
  - SHA-256, commit, environment, runtime and peak memory are recorded.

outputs/ is gitignored: a submission is a full set of labels for the match and does not belong in
a public repository. Nothing here publishes or submits anything.
"""

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from src import submission  # noqa: E402
from src.config import load  # noqa: E402
from src.predict import run  # noqa: E402

# Params
TEAM = "Aziz goes to barcelona"
CONFIG = os.path.join(ROOT, "configs", "final.toml")
REPORT = os.path.join(ROOT, "docs", "technical_report.md")
OUT_DIR = os.path.join(ROOT, "outputs", "submission")
VALIDATOR = os.path.join(ROOT, "evaluation", "validate_submission.py")
N_FRAMES = 199319


def peak_memory_mb():
    """Peak resident memory of this process so far, in MB (Windows or POSIX); None if unknown."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters),
                                                         wintypes.DWORD]
            kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
            if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                                    ctypes.byref(counters), counters.cb):
                return None
            return round(counters.PeakWorkingSetSize / 1e6, 1)
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3, 1)   # Linux: KB
    except Exception:  # memory is a diagnostic; its absence must not stop a release
        return None


def structure_check(path):
    """The playbook's independent check (Phase 4, step 2). Returns a list of failures."""
    table = pd.read_csv(path, dtype={"label": "object"}, keep_default_na=False)
    failures = []
    if list(table.columns) != ["frame", "label"]:
        failures.append("columns {}".format(list(table.columns)))
    if len(table) != N_FRAMES:
        failures.append("{} rows".format(len(table)))
    elif table["frame"].tolist() != list(range(N_FRAMES)):
        failures.append("frame column is not 0..{} in order".format(N_FRAMES - 1))
    if (table["label"] == "").any():
        failures.append("blank labels")
    if not table["label"].isin({"H", "A", "D", "X"}).all():
        failures.append("labels outside H/A/D/X")
    return failures


def git(*args):
    try:
        return subprocess.check_output(["git"] + list(args), cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", default=TEAM)
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, args.team + ".csv")
    md_path = os.path.join(OUT_DIR, args.team + ".md")

    started = time.time()
    generated = run(args.config, os.path.join(ROOT, "outputs", "candidates", "final.csv"))
    runtime = round(time.time() - started, 1)
    shutil.copyfile(generated, csv_path)
    shutil.copyfile(generated + ".manifest.json", os.path.join(OUT_DIR, "possession_manifest.json"))
    shutil.copyfile(REPORT, md_path)

    # Every check below is on csv_path itself -- the file that would be uploaded.
    validator = subprocess.run([sys.executable, VALIDATOR, csv_path], capture_output=True, text=True)
    with tempfile.TemporaryDirectory() as folder:
        again = run(args.config, os.path.join(folder, "again.csv"))
        second_hash = submission.sha256(again)
    sha = submission.sha256(csv_path)
    with open(generated + ".manifest.json") as handle:
        manifest = json.load(handle)

    checks = {
        "official_validator": {"exit": validator.returncode,
                               "output": validator.stdout.strip().splitlines()},
        "official_check_in_process": submission.check(csv_path) or "OK",
        "strict_check": submission.check_strict(csv_path) or "OK",
        "playbook_structure_check": structure_check(csv_path) or "OK",
        "regenerated_identical": second_hash == sha,
        "line_endings": "LF" if b"\r" not in open(csv_path, "rb").read() else "CRLF",
    }
    ok = (validator.returncode == 0 and checks["official_check_in_process"] == "OK"
          and checks["strict_check"] == "OK" and checks["playbook_structure_check"] == "OK"
          and checks["regenerated_identical"] and checks["line_endings"] == "LF")
    report = {
        "status": "READY_FOR_SUBMISSION" if ok else "CHECKS_FAILED",
        "submitted": False,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "team": args.team,
        "files": {
            "possession_csv": {"path": os.path.relpath(csv_path, ROOT).replace(os.sep, "/"),
                               "sha256": sha, "bytes": os.path.getsize(csv_path)},
            "technical_document": {"path": os.path.relpath(md_path, ROOT).replace(os.sep, "/"),
                                   "sha256": submission.sha256(md_path)},
        },
        "repository": {"remote": git("remote", "get-url", "mine"), "commit": git("rev-parse", "HEAD"),
                       "dirty": bool(git("status", "--porcelain"))},
        "config": {"path": os.path.relpath(args.config, ROOT).replace(os.sep, "/"),
                   "digest": load(args.config).digest()},
        "checks": checks,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__},
        "runtime_seconds": runtime,
        "peak_memory_mb": peak_memory_mb(),
        "diagnostics": manifest["diagnostics"],
    }
    with open(os.path.join(OUT_DIR, "release_report.json"), "w", newline="\n") as handle:
        json.dump(report, handle, indent=1)

    print("{}  ({})".format(report["status"], args.team))
    print("  {}  sha256 {}".format(report["files"]["possession_csv"]["path"], sha))
    print("  official validator exit {} | strict {} | structure {} | regenerated identical {} | {}".format(
        validator.returncode, checks["strict_check"], checks["playbook_structure_check"],
        checks["regenerated_identical"], checks["line_endings"]))
    print("  commit {} dirty {} | runtime {} s | peak memory {} MB".format(
        (report["repository"]["commit"] or "?")[:7], report["repository"]["dirty"], runtime,
        report["peak_memory_mb"]))
    print("  {}".format(report["files"]["technical_document"]["path"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
