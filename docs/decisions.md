# Decisions

One entry per decision that is not obvious from the code. Newest last. Each says what was decided,
why, what was rejected, and the evidence. Source hierarchy, from the playbook: signed data-use
terms > official repository docs and scorer code > files in this repository > the playbook.

---

## D-001 — Work in a fresh clone of the official repository

- **Decision.** The workspace (`More Than A Hack/`) held the playbook and the data bundle but no
  repository, so the official repository was cloned into `More Than A Hack/more-than-a-hack-2027/`
  at `f8f2629` (`main`, "Clarifying the guidelines for annotating the provided video."). Organiser
  files — `games.py`, `evaluation/`, `starter/`, the `*.md` contracts — are used and never edited.
  The only change to a tracked file is `.gitignore` (adds `outputs/` and `.pytest_cache/`).
- **Rejected.** Cloning into the workspace root: it would put the 19 GB data bundle inside the git
  tree, one `git add -A` away from a public commit.
- **Consequence.** `origin` is still the organisers' repository. The public submission repository
  (Phase 4) needs a new remote, and publishing needs the user's approval.

## D-002 — The AI assistant sees aggregates only, never match data

- **Conflict.** The playbook (§0) says there is "no blanket ban on AI processing" and that the
  agent "may work with match data". FAQ.md and OVERVIEW.md, quoting the Terms and Conditions,
  say: *"participants shall not upload, disclose or otherwise make available to any third-party
  hosted or cloud-based LLM any videos, datasets, confidential information or other restricted
  materials provided in connection with the Competition, or any extracts thereof."* This
  repository is developed with a cloud-hosted coding assistant, and the terms outrank the
  playbook.
- **Decision.** The assistant never opens a file from the bundle, and no script prints a row,
  coordinate, tag interval, frame-level value or line of a match file. Code that reads the data
  runs locally and reports counts, rates, medians and PASS/FAIL only (`scripts/data_audit.py`).
  Data tests assert on scalars computed beforehand, so a failure message cannot print arrays
  (`tests/conftest.py`). Anything finer-grained goes to gitignored `outputs/` for people to read.
- **What the assistant has seen.** The public repository (docs, scorer, the public 5-minute
  ground truth, the four baseline files); schema-level names from the ATD metadata (element names,
  channel ids `x`/`y`/`s`, separators, the `kph` unit label); the team names the official
  `starter/baseline.py` prints; aggregate statistics; and once, before the check was changed to a
  boolean, one opaque internal team-id string.
- **Open, for the user.** Whether the signed terms allow even aggregate statistics to reach a
  hosted LLM. If they do not, switch every script to PASS/FAIL-only output. A local LLM would not be
  "third-party hosted" and is the route if anyone ever needs model help *looking* at the data.

## D-003 — Python environment

- **Decision.** A venv in `.venv/` (gitignored) on Python 3.12.10, built from the organisers'
  `requirements.txt` plus `kloppy==3.18.0` (the pin in `requirements-starter.txt`) and `pytest`:
  `requirements-dev.txt`. Exact versions in `requirements-lock.txt` — numpy 2.5.4, pandas 3.1.0,
  kloppy 3.18.0, pytest 9.1.1. matplotlib and Jupyter are not installed; nothing needs them yet.
- **Evidence it is compatible.** The organisers test the scorer under Python 3.8 and 3.11. Under
  this environment `evaluation/run_evaluation.py` reproduces the public-window table in README.md
  to four decimals for all four baselines.
- **Open.** Phase 4 fixes the final pins and proves a clean install.

## D-004 — Leave the data where it is; point `MTAH_DATA_DIR` at it

- **Decision.** The bundle stays in `More Than A Hack/Tactical/` and `More Than A Hack/Broadcast/`,
  outside the repository. Every data command sets
  `MTAH_DATA_DIR=C:\Users\USER\Documents\Kuliah\Lomba\More Than A Hack`. `games.py` asks for
  lowercase `tactical/` and `broadcast/`; that resolves here because NTFS is case-insensitive.
  On Linux the folders must be lowercase, as the bundle's README lays them out.
- **Rejected.** Moving or copying 19 GB of the user's data. Directory junctions under `data/`:
  some delete tools follow junctions into their target, and the target is the only copy.

## D-005 — Our own ATD loader, held to both reference readers

- **Decision.** `src/data_loader.py` reads the raw ATD with pandas' C tokenizer after mapping every
  EPTS separator to a comma, taking the column layout from the metadata's
  `DataFormatSpecification` (what kloppy builds its regex from) rather than assuming it. 199,319
  frames parse in ~3 s into ~225 MB of float64 arrays. A full kloppy load costs "minutes and
  gigabytes" (`starter/load_atd.py`), and this machine has ~1.2 GB of free RAM.
- **Evidence.** 0 mismatching frames against the official parser on all 199,319 frames (through
  the four official baseline files), and 0 mismatching values against kloppy on 3,987 sampled
  frames (550,206 player values) — `docs/data_audit.md` §4, `tests/test_alignment.py`.
- **Kept on purpose.** File units, NaN as NaN, the ball's speed channel (kloppy drops it). No
  filling or smoothing in the loader.

## D-006 — Submission files are written LF/ASCII

- **Decision.** `src/submission.py` writes `frame,label` with `\n` line endings and ASCII, always.
  `starter/baseline.py` writes the platform's line ending (CRLF here, LF on Linux), and RULES.md
  requires that re-running the repository produce *the same file*. Hashes are of exact bytes; a
  newline-normalised hash exists only for comparing with git-checked-out files (autocrlf).
- **Evidence.** The LF fallback's git blob id equals the organisers' committed blob for
  `baseline_smooth50.csv` (`1e1e4675…`): byte-identical to their file.

## D-007 — The fallback submission

- **Decision.** `outputs/fallback/baseline_smooth50.csv`: the official `baseline_smooth50`,
  regenerated from the ATD by the unmodified `starter/baseline.py` (run with
  `--out outputs/baseline_regen` so the committed baselines are never overwritten) and rewritten LF.
  SHA-256 `6d4e04b80b9016cc16ada99697bf219d3a74b58750dd6d5ec5a4cd5fc32ab4b9`; manifest in
  `outputs/fallback/manifest.json`. Its full-match score, **spellF1 0.2433 / macroF1 0.6724**, is
  the organisers' published figure (EVALUATION.md) — it cannot be measured here.

## D-008 — The speed channel is m/s; the metadata's `kph` label is wrong

- **Evidence.** Reported speed over displacement speed between consecutive frames: median
  0.989 over 951,939 player steps in graded frames — m/s, as DATA.md says. Player speed p50
  1.41, p99 6.44.
- **Decision.** Use the player `s` channel as m/s without conversion. The ball's `s` channel is
  present on only 8.7% of ball-seen frames, with a ratio of 0.308 to displacement speed:
  undocumented, not used until investigated.

## D-009 — The public ground truth checks the pipeline, never chooses between versions

- **Decision.** The 5-minute public ground truth is used for exactly two things: the scorer smoke
  test (`run_evaluation.py` reproducing README.md), and shape checks in the audit (covered window,
  label counts, possession count — all published). No module in `src/` reads it. Metric behaviour
  is pinned on synthetic sequences instead (`tests/test_scorer_semantics.py`). Rules: RULES.md §2,
  EVALUATION.md "The public window is not a leaderboard".

## D-010 — Windows specifics

- **Console encoding.** The console is cp1252; run Python with `-X utf8`, which, unlike
  `PYTHONUTF8`, also works under `-I`.
- **Smart App Control.** Right after `pip install`, Windows blocked pandas' compiled extensions
  on first import ("An Application Control policy has blocked this file"). Re-running worked and
  imports have been stable since. If it comes back after installing new binary packages, re-run
  once; if it persists, the user has to allow it in Windows Security. It is not to be bypassed.

## D-011 — Audit statuses; documentation mismatches are DIFF, not FAIL

- **Decision.** In `scripts/data_audit.py`, PASS/FAIL holds the data to a published *contract*
  value (counts, alignment, parity). DIFF marks a published *descriptive* figure the data does
  not reproduce, which is reported but does not block. The four DIFFs:
  - tracked objects at halftime: 12.5 on average, where `starter/load_atd.py` says "almost none";
  - the metadata's `kph` label (D-008);
  - tactical reprojection error median: 0.23 on graded frames vs 0.24 "during play";
  - the halftime median, 0.39, vs "about 1.5". The halftime *mean* is 1.5, so the published figure
    is most likely a mean.

  The homography is not used by anything yet.

## D-012 — `nearest_team` is a loader check, not the model

- **Decision.** `src/features.py:nearest_team` reproduces step 1 of the official baseline
  bit-for-bit so the baseline files can test the loader. The nearest player is not the player in
  control (CHALLENGE.md, "What does not count as a change of possession"); the possession model is
  Phase 2's to design.
