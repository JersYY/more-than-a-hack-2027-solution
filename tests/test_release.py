"""The freeze (D-027): configs/final.toml is release candidate rc1, value for value."""

import os
import json
import subprocess
import sys
import zipfile

import pytest

from src.config import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_final_is_rc1_in_every_model_value():
    final = load(os.path.join(ROOT, "configs", "final.toml")).to_dict()
    rc1 = load(os.path.join(ROOT, "configs", "rc1.toml")).to_dict()
    assert final.pop("run")["name"] == "final"
    rc1.pop("run")
    assert final == rc1


def test_final_v2_freezes_the_validated_timing_candidate():
    revision = load(os.path.join(ROOT, "configs/final_v2.toml")).to_dict()
    candidate = load(os.path.join(ROOT, "configs/timing_v1.toml")).to_dict()
    assert revision.pop("run")["name"] == "final_v2"
    candidate.pop("run")
    assert revision == candidate


def test_external_final_preserves_external_base_except_timing():
    revision = load(os.path.join(ROOT, "configs/final_external.toml")).to_dict()
    base = load(os.path.join(ROOT, "configs/hybrid_v2.toml")).to_dict()
    assert revision.pop("run")["name"] == "final_external"
    base.pop("run")
    assert revision["temporal"]["team_shift_frames"] == 8
    revision["temporal"]["team_shift_frames"] = 0
    assert revision == base


@pytest.mark.data
def test_release_packages_current_model_and_rejects_mismatched_document(tmp_path):
    from src.submission import sha256

    folder = tmp_path / "package"
    command = [sys.executable, os.path.join(ROOT, "scripts", "release.py"),
               "--out-dir", str(folder)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True)
    assert result.returncode == 0
    report = json.loads((folder / "release_report.json").read_text())
    assert report["status"] == "READY_FOR_TEAM_REVIEW"
    assert report["submitted"] is False
    assert report["config"]["digest"] == load(os.path.join(ROOT, "configs", "final_external.toml")).digest()
    assert sha256(str(folder / "Aziz goes to barcelona.csv")) == "f6f78ee82869ba63044548e7884d544455396e872e349834747560200998852b"
    with zipfile.ZipFile(folder / "submission.zip") as archive:
        assert set(archive.namelist()) == {
            "Aziz goes to barcelona.csv", "Aziz goes to barcelona.csv.manifest.json",
            "Aziz goes to barcelona.md", "config.toml", "release_report.json"}
        assert archive.testzip() is None
    bad_report = tmp_path / "wrong_report.md"
    bad_report.write_text("This document does not describe this CSV or config.", encoding="utf-8")
    failed = subprocess.run(command + ["--report", str(bad_report)], cwd=ROOT, capture_output=True)
    assert failed.returncode == 1
    assert not (folder / "submission.zip").exists()
    assert len(list(folder.glob("submission.zip.previous-*"))) == 1
    checks = json.loads((folder / "release_report.json").read_text())["checks"]
    assert checks["document_matches_csv"] is False
    assert checks["document_matches_config"] is False
