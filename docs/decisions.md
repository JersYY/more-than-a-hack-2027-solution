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

---

# Phase 2 — architecture and experiment contracts

## D-013 — The model: per-frame evidence, then an offline per-half Viterbi over {D, H, A}

- **Decision.** Two evidence streams per frame, team (H vs A) and dead ball (D vs live), decoded
  together by a three-state Viterbi with a cost on every change, run separately for each half
  (`docs/architecture.md` §4).
- **Why.** The ranking metric scores whole possessions. A change cost is the direct way to say "a
  brush is not a change of possession, seconds of control are". Offline decoding places changes
  where the evidence starts; a causal filter places them where it finally noticed.
- **Rejected for now:**
  - a causal hysteresis state machine (the playbook's first suggestion): late boundaries by
    construction;
  - EM-fitted HMMs: tuning to this match's own statistics with no check;
  - a learned sequence model first: nothing to train on until the other-match benchmark exists.
    That is now E6b.

## D-014 — Configuration in TOML, not YAML

- **Decision.** `configs/*.toml`, read by the standard library's `tomllib`, with the schema in
  `src/config.py`. The playbook names `configs/*.yaml`.
- **Why.** No new dependency; comments are allowed; unknown keys are errors.
- **Cost.** The pipeline needs Python ≥ 3.11 (the lock is on 3.12). The official scorer still runs
  on 3.8.

## D-015 — Output policy, and no game-specific frame constants in the model

- **Output.** Graded frames always get H/A/D. `X` goes only outside the graded blocks. On
  spellF1 an `X` equals repeating the last label, because the scorer forward-fills it. On macroF1
  it is a guaranteed miss. So writing the decoder's label is never worse.
- **Constants.** The model may use the published half blocks (`games.half_video_blocks`), because
  they are the grading definition. It must not use the kickoff frames or any other frame number,
  even published ones. The second half's 2,934 pre-kickoff `D` frames must come out of the
  dead-ball evidence. RULES.md forbids "rules and constants tied to specific frames", and a
  system that works only because it knows this match's frame numbers would not work on the next
  match.

## D-016 — Determinism contract

- **Decision.** No randomness at inference.
- **Decision path.** Only `+ − × ÷ sqrt`, comparisons, `min`/`max`, and sums in a fixed order.
  No `exp`, `log`, `pow` or `hypot`: their vectorised versions differ between CPUs in the last
  bit.
- **Decoder.** Scores and costs are rounded to fixed point (1e-6) and accumulated in int64.
- **Output.** LF/ASCII (D-006).
- **Why.** RULES.md §3 requires the same file when the organisers re-run, on their machine. Ticket
  T10 checks it; Phase 4 repeats it on a second machine.

## D-017 — Choose between versions on other matches' labels

- **Decision.** Parameters are calibrated and versions compared only on an other-match benchmark,
  always with the official `possession_metrics`:
  - IDSSE/DFL: 7 matches, native per-frame BallStatus and BallPossession, CC BY 4.0;
  - Metrica Sample Game 3: our format, labels derived from events.

  Synthetic scenarios check behaviours. On-match diagnostics only veto.
- **Access, verified 2026-10-26.** `ndownloader.figshare.com` serves the files (~419 MB per DFL
  tracking file). kloppy's `springernature.figshare.com` URLs answer HTTP 202 to scripts. Metrica
  is on raw.githubusercontent.com (53 MB).
- **Open.** Whether DFL "possession" means control or last touch. T7 measures it before any
  calibration (`docs/experiments.md` §5).

## D-018 — Acceptance tests before code

- **Decision.** Every P0 module has its acceptance tests written in Phase 2, marked
  `xfail(raises=NotImplementedError, strict=True)`. The stub's NotImplementedError is the only
  accepted failure. Any other error, or a pass, fails the suite. So the marker must come off when
  the code lands.
- **The behavioural specification** is the synthetic scenario catalogue (`src/synthetic.py`). Its
  labels follow CHALLENGE.md by construction and are themselves tested
  (`tests/test_synthetic.py`).

## D-019 — Two design changes found by the feasibility spike

A throwaway implementation of T1–T6 (`docs/experiments.md` §7) passed 62 of 65 acceptance tests
at first. Two of the failures were design flaws, not threshold issues:

- **The flight gate.** A pass flying past an opponent at 3–4 m produced weak opponent evidence.
  That moved the change of possession ~0.7 s before the reception, but CHALLENGE.md puts it at
  control. Now there is no team evidence while the ball moves faster than 8 m/s, faster than a
  player can run with it (`evidence.flight_mps`).
- **The bounded alive bonus.** Team evidence entered the H and A scores in full, so a free-kick
  taker standing at the ball kept it "alive" ~0.8 s into the stoppage. Now team evidence decides
  H vs A in full, but alive vs D only up to `temporal.alive_bonus` = 0.7 ≈ ln 2. That is the bound
  a mixture model gives (`docs/architecture.md` §4.3).

Both have discriminating tests: the old formulation fails them and the new one passes. With them
the spike passed 69 of 69. The third failure was a spec gap: a central-difference ball speed
across an occlusion. The ball must now be seen at t−2, t and t+2.

## D-020 — The public window, pre-registered

The protocol is in `docs/experiments.md` §2:

- one gross-error check per frozen release candidate (teams swapped; alive and dead confused);
- never during development, never compared between candidates, never quoted as performance.

It is written down before any candidate exists, so it cannot be bent around one.

## D-021 — Smart Tagging semantics (P1)

- **Code families.** Offensive codes mean the tagged team has the ball. Defensive codes mean the
  *other* team has it. SET PIECES mark a restart. GOALS are followed by a dead ball. These follow
  OVERVIEW.md's definitions.
- **Team mapping.** Team values map to sides through the ATD's own team names (`Layout.home_name`,
  `Layout.away_name`, which are "Barcelona" and "Atletic Club"), not through a hard-coded string.
- **Use.** Off in every P0 configuration (E5).

## D-022 — The spike stays out of `src/`

The feasibility spike's modules are kept in `outputs/spike_phase2/`, which is gitignored and
local. Phase 3 implements the tickets in `src/` with the tests as the specification. The spike is
a comparison point, not code to paste. It was written against the tests in an hour.
