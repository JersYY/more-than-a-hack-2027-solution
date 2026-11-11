# Aziz goes to barcelona — frame-level possession for More than a Hack 2027

Our solution to [More than a Hack 2027](README_ORGANISERS.md), Phase 1: for each of the 199,319
frames of the tactical video of FC Barcelona vs Athletic Club, decide whether home has the ball
(H), away has it (A), or the ball is dead (D).

The tracking gives two kinds of evidence per frame:

- **team control:** whose player can play the ball;
- **dead ball:** the ball beyond a line, at rest or long missing, or players walking.

An offline Viterbi decoder then labels each half as a whole, charging a price for every change of
possession, stoppage and restart. A touch or a duel cannot pay for two changes of possession; a
controlled possession can. Every parameter was chosen on **other matches' labels**: 7 public
Bundesliga matches and a Metrica sample game. None was chosen on the competition match's ground
truth.

**Write-up:** [docs/submission_report.md](docs/submission_report.md) ·
**Design:** [docs/architecture.md](docs/architecture.md) ·
**Evidence:** [docs/experiments.md](docs/experiments.md) ·
**Every decision, with its reason:** [docs/decisions.md](docs/decisions.md)

## Results we can state

**Current recommendation after the team's rule clarification:** use
`configs/final_external.toml` (external E6 base plus E8 timing). This excludes
the old D-026 target-match plausibility adjustment inherited by final/final_v2.
Eight-match realistic spellF1: external base 0.4618 -> 0.4692; all eight timing
LOMO folds improve. [Rules and parameter provenance](docs/rules_update.md).
Generate it with `--config configs/final_external.toml`; its CSV SHA-256 is
`f6f78ee82869ba63044548e7884d544455396e872e349834747560200998852b`.
Historical configurations and scores below remain for audit purposes.
`scripts/release.py` now packages this recommendation into `outputs/submission_external/`.

**Historical revision:** `configs/final_v2.toml` adds an eight-frame team-evidence
delay. Eight-match realistic spellF1 improves from 0.4625 to 0.4698; all eight
timing-grid leave-one-match-out folds improve. These datasets were used in prior
calibration. [Protocol, limitations and reproduction](docs/e8_results.md).
To generate this revision, use `--config configs/final_v2.toml` in the prediction
command below. Its SHA-256 is
`b1afaa1e9b35467d83f696f0b28b8d7054483358f1c52e943a41062c92069c55`.
The original `final.toml` and historical results below remain available for audit.

Mean official spellF1 (`evaluation/possession_metrics.py`) on 8 other matches. The ball is
deleted the way this match's tracking loses it, and the random seeds were not used in
calibration:

| | official baseline (smooth50) | first-principles MVP | **final** |
|---|---|---|---|
| primary setting | 0.320 | 0.365 | **0.515** |
| random ball loss (harshest) | 0.244 | 0.348 | **0.434** |
| clean tracking | 0.316 | 0.462 | **0.562** |

Leave-one-match-out: calibrating without a match and then scoring that match improves it every
time (8 of 8; mean +0.156).

**The score on the competition's ground truth is unknown to us.** It is private. The organisers
publish 0.2433 for their baseline.

## Reproduce the submission

You need Python 3.11+ (`tomllib`) and the competition's data bundle. The latest
revision was tested on Python 3.13, Windows, with requirements-lock.txt.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt          # Linux/macOS: .venv/bin/python
# the gated bundle, as the organisers ship it: data/tactical/..., data/broadcast/...
# or anywhere else, with:  export MTAH_DATA_DIR=/path/to/bundle   (PowerShell: $env:MTAH_DATA_DIR = "...")
.venv/Scripts/python -m src.predict --config configs/final_external.toml --output outputs/final_external.csv
.venv/Scripts/python evaluation/validate_submission.py outputs/final_external.csv
```

The file you get should have this SHA-256:

```
f6f78ee82869ba63044548e7884d544455396e872e349834747560200998852b
```

Prediction takes about 4 s and under 1 GB of memory. It has no randomness, and the decoder works
in integer fixed point, so reruns are byte-identical (`tests/test_determinism.py`). The CSV is
written with LF line endings on every operating system.

Other commands:

```bash
python -m pytest -q          # all tests; the ones needing the gated data skip without it
python scripts/release.py    # builds outputs/submission_external/ for team review; never submits
python scripts/data_audit.py # measures the bundle against the published contract (aggregates)
```

## Reproduce the benchmark (other matches)

```bash
python scripts/omb_fetch.py                                    # IDSSE/DFL, 7 matches, ~2.6 GB
# Metrica Sample Game 3 -> outputs/omb/metrica/SG3/{metadata.xml,tracking.txt,events.json}
#   from https://github.com/metrica-sports/sample-data (data/Sample_Game_3/)
python scripts/omb_eval.py configs/hybrid_v2.toml configs/final_external.toml --degrade dead_weighted --dead-share 0.9 --seeds 2 3 --realism
python scripts/omb_timing_lomo.py --base configs/hybrid_v2.toml
python scripts/omb_calibrate.py --base configs/hybrid_v1.toml --guard   # the E6 calibration
```

## Rules we followed

- **No hand labels or corrections** on the provided match, and no constants tied to its frames. The
  second half's pre-kickoff stoppage is found from the evidence.
- **No ground truth for any choice.** The public 5 minutes ran once, as a pre-registered check
  that teams are not swapped. The private ground truth was never seen.
- **Labelled data only from other matches,** with their sources below.
- **No match data in this repository.** `data/` and `outputs/` are gitignored. The files under
  `configs/` and `docs/` that describe the match hold aggregate statistics only.
- **The checks a machine can make run with the tests** (`tests/test_rules.py`):
  - no frame number of this match in `src/` or `configs/`;
  - no model code or configuration refers to a ground-truth file;
  - no match data in the repository or anywhere in its history.

  Rule by rule, with evidence: [docs/release_checklist.md §4](docs/release_checklist.md).

## Layout

```
src/          the pipeline: data_loader, preprocess, features, possession_evidence,
              dead_ball, temporal (decoder), predict (CLI), submission (writer/checks),
              config; benchmark tooling: omb, degrade, synthetic (test scenarios)
configs/      final.toml (frozen), rc1/hybrid_v2/hybrid_v1/baseline.toml, atd_ball_runs.json
scripts/      release, data_audit, verify_baseline, synthetic_bench, omb_fetch/eval/calibrate
tests/        contracts, alignment, scorer semantics, acceptance tests per module, determinism
docs/         technical report, architecture, feature spec, experiments, decisions, audit
evaluation/   the organisers' scorer, validator and public 5-minute ground truth (unchanged)
starter/      the organisers' starter code (unchanged)
games.py      the organisers' match constants and asset paths (unchanged)
```

## Credits and licences

- **The competition repository** (scorer, starter code, documentation in `README_ORGANISERS.md`,
  `CHALLENGE.md`, `DATA.md`, …): Metrica Sports and FC Barcelona Innovation Hub, under the MIT
  licence in `LICENSE`. The match data is licensed separately and is not here.
- **IDSSE / DFL open data:** Bassek, M., Rein, R., Weber, H. et al. "An integrated dataset of
  spatiotemporal and event data in elite soccer." *Scientific Data* 12, 195 (2025). CC BY 4.0.
  Used for calibration and validation only.
- **Metrica Sports sample data,** Sample Game 3: https://github.com/metrica-sports/sample-data. Used
  for validation and calibration only.
- **Solution code** (`src/`, `scripts/`, `tests/`, `configs/`, `docs/`): team Aziz goes to
  barcelona.
