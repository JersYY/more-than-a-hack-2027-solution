"""Write and check submission files.

The contract is SUBMISSION.md and the authority is evaluation/validate_submission.py, which runs
exactly the official scorer's checks; check() below calls the same functions, it does not
re-implement them. What this module adds is determinism.

starter/baseline.py writes with open(path, "w"), so its bytes depend on the platform -- CRLF on
Windows, LF elsewhere -- and git's autocrlf can change them again on checkout. RULES.md asks for
the same file when the organisers re-run the repository, "not approximately", so write() always
emits LF and ASCII, and sha256() hashes the bytes exactly as written.
"""

import hashlib
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES  # noqa: E402

# Params
HEADER = "frame,label"
N_FRAMES = GAMES[GAME]["n_video_frames"]


def write(path, labels, n_frames=N_FRAMES):
    """Write labels as a frame,label CSV with LF line endings. Refuses a file that would not validate.

    Refusing here rather than at validation time means an invalid file never exists on disk to be
    uploaded by mistake.
    """
    labels = np.asarray(labels).astype(str)
    if len(labels) != n_frames:
        raise ValueError("{} labels, expected {}".format(len(labels), n_frames))
    bad = sorted(set(labels.tolist()) - set(pm.LABELS))
    if bad:
        raise ValueError("labels outside {}: {}".format(list(pm.LABELS), bad[:5]))

    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", newline="\n", encoding="ascii") as handle:
        handle.write(HEADER + "\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def check(path, n_frames=N_FRAMES):
    """None if the official validation accepts the file, else its one-line reason."""
    try:
        table = pm.read_label_csv(path)
    except Exception as error:  # same catch as validate_submission.py
        return "could not read: {}".format(error)
    return pm.validate_submission(table, n_frames)


def check_strict(path, n_frames=N_FRAMES):
    """None if the file meets the stricter structure Phase 4 asks for, else a reason.

    Supplementary to check(), never a replacement: the header must be exactly frame,label (the
    official reader tolerates case and whitespace), labels exactly H/A/D/X with no padding.
    """
    table = pd.read_csv(path, dtype={"label": "object"}, keep_default_na=False)
    if list(table.columns) != ["frame", "label"]:
        return "columns {} != ['frame', 'label']".format(list(table.columns))
    if len(table) != n_frames:
        return "{} rows, expected {}".format(len(table), n_frames)
    if not np.array_equal(table["frame"].to_numpy(), np.arange(n_frames)):
        return "frame column is not 0..{} in order".format(n_frames - 1)
    bad = sorted(set(table["label"].tolist()) - set(pm.LABELS))
    if bad:
        return "labels outside {}: {}".format(list(pm.LABELS), bad[:5])
    return None


def read_labels(path):
    """Label column as a numpy array of one-letter strings, normalised the way the scorer reads it."""
    return np.array(pm.LABELS)[pm.label_codes(pm.read_label_csv(path))]


def sha256(path, normalise_newlines=False):
    """Hex SHA-256 of a file's bytes; with normalise_newlines, of its bytes with CRLF read as LF.

    The normalised hash is for comparing against files whose line endings git or Windows chose,
    such as the committed baselines. A submission is identified by the plain one.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        data = handle.read()
    digest.update(data.replace(b"\r\n", b"\n") if normalise_newlines else data)
    return digest.hexdigest()
