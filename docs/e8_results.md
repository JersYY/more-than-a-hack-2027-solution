# E8: team evidence timing (2026-11-11)

Release packaging is now updated: `scripts/release.py` defaults to
`final_external.toml` and builds `outputs/submission_external/`. Earlier notes
below about the old release script describe the state before this packaging
round. Current checklist: [release_external.md](release_external.md).

**Current recommendation: `configs/final_external.toml`.** The team supplied
stricter external-only tuning rules after the initial experiment. The historical
final/final_v2 results below remain traceable, but inherit D-026 target-match
plausibility tuning. The final section records the rerun excluding that choice.

The first improvement round found a small, consistent gain by delaying team
evidence eight frames (0.32 seconds at 25 fps) before Viterbi decoding. This is
initially an experimental candidate, `configs/timing_v1.toml`, now frozen as
`configs/final_v2.toml` after the follow-up validation below. The original submission
`configs/final.toml` has not been replaced. No competition ground truth was read.

## Protocol

`scripts/omb_transition_probe.py` fixes its grid and split before scoring:

- Training: J03WMX, J03WN1, J03WOH, J03WOY, seeds 0/1.
- Held-out check: J03WPY, J03WQQ, J03WR9, seeds 2/3.
- Realistic tracking, dead-weighted ball loss with dead share 0.9.
- Shifts: -12, -8, -4, 0, 4, 8, 12 frames at turnover cost 4.5;
  also turnover costs 3.0 and 5.7 at zero shift. Nine candidates total.
- Every candidate must preserve spellF1 1.0 and frame accuracy >=0.90 on
  every synthetic scenario; six pass. Selection requires training gain >0.002.
- Held-out acceptance requires mean gain >0.002 with no match regressing.

The DFL datasets were used by earlier project calibration. This split isolates
this round's candidate selection; it is not independent external validation.
Eight matches are a small sample. Follow-up Metrica and leave-one-match-out
checks are recorded below; these do not replace independent external validation.

## Results

| Setting | Frozen final spellF1 | timing_v1 spellF1 |
|---|---:|---:|
| Training, 4 DFL matches, seeds 0/1 | 0.4601 | 0.4706 |
| Held-out check, 3 DFL matches, seeds 2/3 | 0.4596 | 0.4664 |
| Full realistic benchmark, 7 DFL matches, seeds 2/3 | 0.4548 | 0.4626 |
| Clean tracking, 7 DFL matches | 0.5612 | 0.5777 |

Full realistic macroF1: 0.8189 -> 0.8217. Each of the seven matches improves
on realistic spellF1. The full benchmark is a descriptive follow-up, not an
additional untouched selection set. Clean tracking was checked after selection.

Short possessions remain the major limitation. On the held-out check, across
both seeds, 638 ground-truth possessions last under two seconds. The number with
no same-team prediction overlap only falls from 341 to 340. Those whose best
same-team interval IoU is below 0.5 fall from 571 to 570. These diagnostics use
best overlap per interval, not the scorer's one-to-one interval assignment.
Timing helps overall segmentation; it does not solve short-possession recall.

## Implementation and reproduction

`temporal.team_shift_frames` defaults to zero, retaining existing predictions.
Nonzero shifts zero-pad within each half, never wrapping or crossing halves.
The same helper is used by inference, calibration, and the timing experiment.

```powershell
.\.venv\Scripts\python.exe scripts/omb_transition_probe.py
.\.venv\Scripts\python.exe scripts/omb_eval.py configs/final.toml configs/timing_v1.toml --degrade dead_weighted --dead-share 0.9 --seeds 2 3 --realism --tag e8_pipeline
$env:MTAH_DATA_DIR = "C:\Users\Brahm\Documents\barcelona"
.\.venv\Scripts\python.exe -m src.predict --config configs/timing_v1.toml --output outputs/local_verification/timing_v1.csv
```

Local reports: `outputs/omb/results/e8_transition_probe.json`,
`e8_pipeline_dead_weighted_realism_20261111-181357.json`, and
`e8_clean_none_20261111-181454.json`. Candidate CSV passes official and strict
checks. The frozen final CSV still matches its documented SHA-256. No claim is
made about full-match Barcelona-Athletic accuracy.

Validation: 301 tests passed with the competition bundle available, including
candidate determinism and shift boundary checks. An initial test attempt hit
Windows permissions on the default temporary directory; using a new directory
under outputs/local_verification resolved it without changing test assertions.

## Follow-up: Metrica and timing-grid LOMO

Downloaded Metrica's official Sample Game 3 metadata, tracking and event files
from `https://github.com/metrica-sports/sample-data/tree/master/data/Sample_Game_3`.
Local paths are `outputs/omb/metrica/SG3/{metadata.xml,tracking.txt,events.json}`.
Metrica realistic spellF1: 0.5165 -> 0.5204 (seeds 2/3).

`scripts/omb_timing_lomo.py` uses the unchanged, synthetically guarded E8 grid.
Each fold selects from the other seven matches using seeds 0/1, then scores its
excluded match on seeds 2/3. Other model parameters remain fixed; this validates
the timing adjustment, not the entire historical calibration procedure.
Selection excludes the held-out match by construction and by a unit test.

- Eight of eight folds improve. Mean held-out gain: +0.0066; minimum: +0.0012.
- Seven folds select eight frames; the fold excluding J03WN1 selects four.
- Fixed eight-frame revision, eight-match realistic spellF1: 0.4625 -> 0.4698.
- Acceptance gate declared in the validator: LOMO mean gain >0.002,
  no regressing fold, and no Metrica regression. All conditions pass.
- These remain previously used calibration datasets; results indicate stability,
  not an unbiased estimate on a wholly new dataset.

The revision is frozen separately as `configs/final_v2.toml`; the original
`final.toml`, original release packaging and old submission hash remain available.
Nothing has been published or submitted. The release script still packages the
original final, so use the explicit command below for the new revision:

```powershell
.\.venv\Scripts\python.exe scripts/omb_timing_lomo.py
.\.venv\Scripts\python.exe -m src.predict --config configs/final_v2.toml --output outputs/local_verification/final_v2.csv
```

The new CSV passes official and strict checks, with SHA-256
`b1afaa1e9b35467d83f696f0b28b8d7054483358f1c52e943a41062c92069c55`.
Results: `outputs/omb/results/e8_timing_lomo.json` and
`e8_metrica_dead_weighted_realism_20261111-181716.json`.

## Rules clarification: external-only rerun

See [rules_update.md](rules_update.md). The rerun uses hybrid_v2's externally
calibrated stop/restart costs 6.9/5.9, excluding D-026's 8.9/7.9. The unchanged
timing grid and synthetic guard are applied again; no target-video comparison,
target GT scoring or target-plausibility parameter selection occurs.

- Mean training score (all 8 external matches, seeds 0/1): 0.47122 baseline,
  0.47821 at eight-frame delay; eight frames wins the declared grid.
- Eight-match realistic test-seed mean: 0.461766 -> 0.469152.
- Metrica: 0.5116 -> 0.5186.
- Timing-grid LOMO: all 8 improve, mean +0.006372, minimum +0.002227.
  Six folds choose eight frames, two choose four. Other model parameters remain
  fixed; historical E6 dataset reuse still limits independence of the estimate.

Recommended config: `configs/final_external.toml`, tested against an assertion
that it matches hybrid_v2 in every model value except the eight-frame shift.
CSV: `outputs/local_verification/final_external.csv`, official and strict OK.
SHA-256: `f6f78ee82869ba63044548e7884d544455396e872e349834747560200998852b`.
Results: `outputs/omb/results/e8_timing_lomo_external_only.json`.

```powershell
.\.venv\Scripts\python.exe scripts/omb_timing_lomo.py --base configs/hybrid_v2.toml --tag external_only
$env:MTAH_DATA_DIR = "C:\Users\Brahm\Documents\barcelona"
.\.venv\Scripts\python.exe -m src.predict --config configs/final_external.toml --output outputs/local_verification/final_external.csv
```

The original release script still packages the historical final; it must not be
treated as packaging the external-only revision. No publication or submission
has occurred. This does not establish full-match Barcelona-Athletic accuracy.

Final external-only verification: 311 tests passed, including deterministic
reruns and a check that only timing differs from the externally calibrated base.
Pipeline report: `outputs/omb/results/external_pipeline_dead_weighted_realism_20261111-182540.json`.
