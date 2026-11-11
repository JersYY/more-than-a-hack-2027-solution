"""Build the submission package from the frozen configuration, and check the exact files.

    $env:MTAH_DATA_DIR = "<folder containing tactical/ and broadcast/>"
    python scripts/release.py                         # -> outputs/submission_external/

SUBMISSION.md asks for the files to be named after the team. For "Aziz goes to barcelona" this
writes:

    outputs/submission_external/Aziz goes to barcelona.csv
    outputs/submission_external/Aziz goes to barcelona.md
    outputs/submission_external/release_report.json

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
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

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
CONFIG = os.path.join(ROOT, "configs", "final_external.toml")
REPORT = os.path.join(ROOT, "docs", "submission_report.md")
OUT_DIR = os.path.join(ROOT, "outputs", "submission_external")
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
    parser.add_argument("--report", default=REPORT)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    if not args.team or any(char in args.team for char in '/\\:') or args.team in (".", ".."):
        parser.error("team must be a filename, not a path")
    out_dir = os.path.abspath(args.out_dir)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, args.team + ".csv")
    md_path = os.path.join(out_dir, args.team + ".md")
    config_path = os.path.join(out_dir, "config.toml")
    archive_path = os.path.join(out_dir, "submission.zip")
    if os.path.isfile(archive_path):
        os.replace(archive_path, archive_path + ".previous-" + str(time.time_ns()))
    shutil.copyfile(args.config, config_path)

    started = time.time()
    generated = run(config_path, csv_path)
    runtime = round(time.time() - started, 1)
    shutil.copyfile(args.report, md_path)

    # Every check below is on csv_path itself -- the file that would be uploaded.
    validator = subprocess.run([sys.executable, VALIDATOR, csv_path], capture_output=True, text=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as folder:
        again = run(config_path, os.path.join(folder, "again.csv"))
        second_hash = submission.sha256(again)
    sha = submission.sha256(csv_path)
    with open(generated + ".manifest.json") as handle:
        manifest = json.load(handle)

    with open(md_path, encoding="utf-8") as handle:
        document = handle.read()
    documented_hash = re.search(r"<!-- submission-sha256: ([0-9a-f]{64}) -->", document)
    documented_config = re.search(r"<!-- config-digest: ([0-9a-f]{64}) -->", document)
    config_digest = load(config_path).digest()
    checks = {
        "official_validator": {"exit": validator.returncode,
                               "output": validator.stdout.strip().splitlines()},
        "official_check_in_process": submission.check(csv_path) or "OK",
        "strict_check": submission.check_strict(csv_path) or "OK",
        "playbook_structure_check": structure_check(csv_path) or "OK",
        "regenerated_identical": second_hash == sha,
        "line_endings": "LF" if b"\r" not in open(csv_path, "rb").read() else "CRLF",
        "document_matches_csv": bool(documented_hash and documented_hash[1] == sha),
        "document_matches_config": bool(documented_config and documented_config[1] == config_digest),
        "config_copy_matches_source": submission.sha256(config_path) == submission.sha256(args.config),
    }
    ok = (validator.returncode == 0 and checks["official_check_in_process"] == "OK"
          and checks["strict_check"] == "OK" and checks["playbook_structure_check"] == "OK"
          and checks["regenerated_identical"] and checks["line_endings"] == "LF"
          and checks["document_matches_csv"] and checks["document_matches_config"]
          and checks["config_copy_matches_source"])
    report = {
        "status": "READY_FOR_TEAM_REVIEW" if ok else "CHECKS_FAILED",
        "submitted": False,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "team": args.team,
        "files": {
            "possession_csv": {"path": os.path.relpath(csv_path, ROOT).replace(os.sep, "/"),
                               "sha256": sha, "bytes": os.path.getsize(csv_path)},
            "technical_document": {"path": os.path.relpath(md_path, ROOT).replace(os.sep, "/"),
                                   "sha256": submission.sha256(md_path)},
            "config": {"path": os.path.relpath(config_path, ROOT).replace(os.sep, "/"),
                       "sha256": submission.sha256(config_path)},
        },
        "repository": {"remote": git("remote", "get-url", "mine"), "commit": git("rev-parse", "HEAD"),
                       "dirty": bool(git("status", "--porcelain"))},
        "config": {"path": os.path.relpath(args.config, ROOT).replace(os.sep, "/"),
                   "digest": config_digest},
        "checks": checks,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__},
        "runtime_seconds": runtime,
        "peak_memory_mb": peak_memory_mb(),
        "diagnostics": manifest["diagnostics"],
    }
    with open(os.path.join(out_dir, "release_report.json"), "w", newline="\n") as handle:
        json.dump(report, handle, indent=1)

    # Explicit allowlist: never zip the workspace, data, benchmark caches or old outputs.
    names = [args.team + ".csv", args.team + ".csv.manifest.json", args.team + ".md",
             "config.toml", "release_report.json"]
    if ok:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                archive.write(os.path.join(out_dir, name), arcname=name)
        print("  archive {}".format(os.path.relpath(archive_path, ROOT)))

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
