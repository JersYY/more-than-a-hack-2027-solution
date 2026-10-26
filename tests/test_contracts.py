"""The submission contract (SUBMISSION.md) and the competition constants (games.py, DATA.md).

No match data needed: these run on a fresh clone and in public CI.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

import games
from src import submission

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 199319
BASELINES = ("baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125")


def write_raw(path, header, rows):
    """A CSV written verbatim, for files src.submission.write() would refuse to produce."""
    with open(path, "w", newline="\n") as handle:
        handle.write(header + "\n")
        handle.writelines(row + "\n" for row in rows)
    return path


def test_published_constants():
    assert games.GAMES[games.GAME]["n_video_frames"] == N
    assert games.GAMES[games.GAME]["fps"] == 25
    assert games.source_frames("broadcast") == 164488
    blocks = games.half_video_blocks(games.GAME)
    assert blocks == [(1, 11185, 90651), (2, 110808, 190197)]
    assert sum(last - first + 1 for _, first, last in blocks) == 158857


@pytest.mark.parametrize("frame, half", [
    (0, None), (11184, None), (11185, 1), (90651, 1), (90652, None),
    (110807, None), (110808, 2), (113742, 2), (190197, 2), (190198, None), (N - 1, None)])
def test_half_boundaries(frame, half):
    assert games.half_of_video_frame(games.GAME, frame) == half


def test_asset_paths_follow_the_data_dir(tmp_path):
    paths = games.asset_paths(data_dir=str(tmp_path))
    assert set(paths) == {"video", "atd_meta", "atd_raw", "homography", "smart_tagging"}
    assert all(path.startswith(str(tmp_path)) for path in paths.values())
    assert os.path.basename(paths["atd_raw"]) == "BAR-ATH_tactical_atd.txt"
    assert "smart_tagging" not in games.asset_paths(data_dir=str(tmp_path), source="broadcast")


@pytest.mark.parametrize("name", BASELINES)
def test_committed_baselines_validate(name):
    path = os.path.join(ROOT, "evaluation", "results", name + ".csv")
    assert submission.check(path) is None
    assert submission.check_strict(path) is None


def test_writer_output_is_valid_lf_and_round_trips(tmp_path):
    labels = np.full(N, "X", dtype="<U1")
    labels[11185:90652] = "H"
    labels[110808:190198] = "D"
    path = submission.write(str(tmp_path / "s.csv"), labels)
    assert submission.check(path) is None
    assert submission.check_strict(path) is None
    with open(path, "rb") as handle:
        data = handle.read()
    assert b"\r" not in data
    assert data.startswith(b"frame,label\n0,X\n1,X\n")
    assert data.endswith(b"\n199318,X\n")
    assert (submission.read_labels(path) == labels).all()


def test_writer_is_byte_deterministic(tmp_path):
    labels = np.array(list("HADX"))[np.random.RandomState(0).randint(0, 4, N)]
    first = submission.write(str(tmp_path / "a.csv"), labels)
    second = submission.write(str(tmp_path / "b.csv"), labels)
    assert submission.sha256(first) == submission.sha256(second)


def test_writer_refuses_invalid_labels(tmp_path):
    with pytest.raises(ValueError):
        submission.write(str(tmp_path / "short.csv"), ["H"] * (N - 1))
    with pytest.raises(ValueError):
        submission.write(str(tmp_path / "alphabet.csv"), ["Q"] * N)
    assert not os.listdir(str(tmp_path))  # nothing invalid reached the disk


@pytest.mark.parametrize("mutate, reason", [
    (lambda rows: rows[:-1], "row count"),
    (lambda rows: ["{},X".format(i + 1) for i in range(N)], "not 0.."),
    (lambda rows: rows[:5] + ["5,"] + rows[6:], "empty label"),
    (lambda rows: rows[:5] + ["5,Q"] + rows[6:], "labels outside"),
])
def test_official_validation_rejects_broken_files(tmp_path, mutate, reason):
    path = write_raw(str(tmp_path / "broken.csv"), "frame,label",
                     mutate(["{},X".format(i) for i in range(N)]))
    found = submission.check(path)
    assert found is not None and reason in found


def test_strict_check_is_stricter_than_the_official_one(tmp_path):
    # The official reader tolerates header case and padded lower-case labels; the strict one not.
    path = write_raw(str(tmp_path / "loose.csv"), "Frame,Label", ["{}, h ".format(i) for i in range(N)])
    assert submission.check(path) is None
    assert submission.check_strict(path) is not None


def test_validator_cli_exit_codes(tmp_path):
    validator = os.path.join(ROOT, "evaluation", "validate_submission.py")
    good = submission.write(str(tmp_path / "good.csv"), ["X"] * N)
    bad = write_raw(str(tmp_path / "bad.csv"), "frame,label", ["0,X"])
    assert subprocess.run([sys.executable, validator, good], capture_output=True).returncode == 0
    assert subprocess.run([sys.executable, validator, bad], capture_output=True).returncode == 1


def test_sha256_normalisation_only_touches_line_endings(tmp_path):
    lf = tmp_path / "lf.csv"
    crlf = tmp_path / "crlf.csv"
    lf.write_bytes(b"frame,label\n0,X\n")
    crlf.write_bytes(b"frame,label\r\n0,X\r\n")
    assert submission.sha256(str(lf)) != submission.sha256(str(crlf))
    assert submission.sha256(str(lf)) == submission.sha256(str(crlf), normalise_newlines=True)
