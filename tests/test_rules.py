"""The competition rules a machine can check (RULES.md), checked on every test run.

- No frame number of this match in the model code or configurations: the graded halves reach
  the model only through games.half_video_blocks (RULES.md "rules and constants tied to specific
  frames"; D-015).
- No model code or configuration refers to a ground-truth file. The public window verifies the
  pipeline and nothing else (RULES.md §2; D-009, D-020).
- No match data, and nothing reconstructing it, in the repository or anywhere in its history.
  The only CSVs are the organisers' own public files (RULES.md "The data").

The rest needs a person, not a test: that nobody labelled or corrected anything by hand, and how
the public window was used. docs/release_checklist.md §4 records the evidence for those.

Needs git; skipped in a copy of the repository without its history.
"""

import io
import os
import subprocess
import tokenize
import tomllib

import pytest

import games

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Params
# The organisers' public files: four official baselines and the 5-minute ground truth. Nothing
# else in the repository may be a set of labels.
ORGANISER_CSVS = {
    "evaluation/gt/BAR-ATH_possession_gt_public.csv",
    "evaluation/results/baseline_raw.csv",
    "evaluation/results/baseline_smooth25.csv",
    "evaluation/results/baseline_smooth50.csv",
    "evaluation/results/baseline_smooth125.csv",
}
MAX_FILE_BYTES = 2_000_000          # the largest legitimate file is a 1.68 MB organiser CSV
DATA_SUFFIXES = (".mp4", ".npz", ".npy", ".pkl", ".parquet", ".feather", ".h5", ".gzip",
                 "_atd.txt", "_atd.xml", "_homography.csv", "_smart_tagging.xml")
GT_REFERENCES = ("possession_gt", "evaluation/gt", "evaluation\\gt")
CONFIG_INT_LIMIT = 1000             # a frame index would be far above it; fixed_point is a scale


def match_frames():
    """Frame numbers this match's documents publish: half edges, kickoffs, file length, +/- 1."""
    frames = set()
    for _, first, last in games.half_video_blocks(games.GAME):
        frames.update({first, last})
    meta = games.GAMES[games.GAME]
    for key in ("kickoff", "kickoff_2nd"):
        frames.add(games.anchor_video_frame(meta[key]))
    frames.update({meta["n_video_frames"], meta["n_video_frames"] - 1})
    return {f + d for f in frames for d in (-1, 0, 1)}


def git(*args):
    try:
        out = subprocess.check_output(["git"] + list(args), cwd=ROOT, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return out.decode("utf-8", "replace").splitlines()


def python_files(folder):
    path = os.path.join(ROOT, folder)
    return sorted(os.path.join(path, name) for name in os.listdir(path) if name.endswith(".py"))


def number_literals(path):
    """Numbers written in the code itself -- not in comments or strings."""
    with open(path, encoding="utf-8") as handle:
        tokens = tokenize.generate_tokens(io.StringIO(handle.read()).readline)
        values = set()
        for token in tokens:
            if token.type == tokenize.NUMBER:
                value = complex(token.string.replace("_", "")).real
                if value.is_integer():
                    values.add(int(value))
    return values


def config_numbers(node, path=()):
    """(dotted key, value) for every number in a parsed TOML document."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from config_numbers(value, path + (key,))
    elif isinstance(node, list):
        for value in node:
            yield from config_numbers(value, path)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield ".".join(path), node


def path_problem(path):
    """Why a tracked path may not be in the repository, or None."""
    lower = path.lower()
    if lower.endswith(".csv") and path not in ORGANISER_CSVS:
        return "a CSV that is not one of the organisers' public files"
    if lower.startswith("data/") and lower != "data/readme.md":
        return "inside data/, where the gated bundle goes"
    if lower.startswith("outputs/"):
        return "inside outputs/, which holds match-derived files"
    if lower.endswith(DATA_SUFFIXES):
        return "a match-data file type"
    return None


def test_match_frames_are_the_published_ones():
    frames = match_frames()
    assert {11185, 90651, 110808, 190197, 199318} <= frames


@pytest.mark.parametrize("path", python_files("src"), ids=os.path.basename)
def test_model_code_has_no_frame_number_of_this_match(path):
    assert not number_literals(path) & match_frames()


@pytest.mark.parametrize("path", python_files("src") + [
    os.path.join(ROOT, "configs", name) for name in sorted(os.listdir(os.path.join(ROOT, "configs")))],
    ids=os.path.basename)
def test_model_and_configs_never_refer_to_a_ground_truth_file(path):
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    assert not [ref for ref in GT_REFERENCES if ref in text]


@pytest.mark.parametrize("name", sorted(n for n in os.listdir(os.path.join(ROOT, "configs"))
                                        if n.endswith(".toml")))
def test_configs_hold_no_frame_numbers(name):
    with open(os.path.join(ROOT, "configs", name), "rb") as handle:
        values = list(config_numbers(tomllib.load(handle)))
    large = [key for key, value in values
             if abs(value) >= CONFIG_INT_LIMIT and not key.endswith("fixed_point")]
    assert large == []
    assert not {int(v) for _, v in values if float(v).is_integer()} & match_frames()


def test_no_match_data_is_tracked():
    tracked = git("ls-files")
    assert [(p, path_problem(p)) for p in tracked if path_problem(p)] == []
    too_big = [p for p in tracked
               if os.path.isfile(os.path.join(ROOT, p))
               and os.path.getsize(os.path.join(ROOT, p)) > MAX_FILE_BYTES]
    assert too_big == []


def test_no_match_data_was_ever_committed():
    """The history becomes public with the repository: every path and blob of HEAD's history."""
    paths = {p for p in git("log", "HEAD", "--pretty=format:", "--name-only") if p}
    assert [(p, path_problem(p)) for p in sorted(paths) if path_problem(p)] == []
    objects = subprocess.run(["git", "rev-list", "--objects", "HEAD"], cwd=ROOT,
                             capture_output=True, check=True).stdout
    # %(rest) makes cat-file split each "<sha> <path>" input line at the sha.
    sizes = subprocess.run(["git", "cat-file", "--batch-check=%(objecttype) %(objectsize) %(rest)"],
                           cwd=ROOT, input=objects, capture_output=True, check=True).stdout
    blob_sizes = [int(fields[1]) for fields in (line.split() for line in sizes.decode().splitlines())
                  if fields[0] == "blob"]
    assert max(blob_sizes) <= MAX_FILE_BYTES


def test_the_organiser_csvs_are_unchanged():
    """Ours never overwrite theirs: each matches its blob in the organisers' last commit."""
    upstream = "f8f2629"
    if not git("cat-file", "-t", upstream):
        pytest.skip("organisers' commit not in this history")
    changed = [p for p in sorted(ORGANISER_CSVS)
               if git("rev-parse", "HEAD:" + p) != git("rev-parse", upstream + ":" + p)]
    assert changed == []
