# Frame-level possession from automatic tracking

Team: **Aziz goes to barcelona**. Revision: **final_external**, 11 November 2026.

<!-- submission-sha256: f6f78ee82869ba63044548e7884d544455396e872e349834747560200998852b -->
<!-- config-digest: 46ba1e590fb9298c41d2f4f7f33211d8978ebc750361adde96d7867fffcd86a7 -->

## Task, input and output

For each of the 199,319 tactical-video frames of Barcelona versus Athletic Club,
the system outputs H (home possession), A (away possession), D (dead ball), or X
outside the two graded halves. Output is an algorithm-generated `frame,label`
CSV with frame numbers 0 through 199318. No manual corrections are applied.

The only model input is tactical ATD: the `_atd.xml` metadata and `_atd.txt`
tracking files, together with the organiser's pitch, frame rate and half blocks
in `games.py`. Frame alignment is `video_frame = frame_id - 1`, checked by the
loader. Positions are converted to metres; missing observations remain missing.
Player speed is treated as m/s. Ball speed is derived from positions because
the supplied ball-speed channel is sparse and has uncertain semantics.

Tactical video, Broadcast, homography and Smart Tagging are not used. Broadcast
has a different timeline; Smart Tagging has not been validated on tagged
external matches. No prediction is checked against the provided video.

## Algorithm and possession rules

Nearest-player distances provide soft team-control evidence. For each team,
`w(d) = 1 / (1 + (d / 1 m)^2)^2`. The signed evidence is
`3 * (w_home - w_away) / (w_home + w_away + w(1.5 m))`. Unseen balls provide
no team evidence; ball speed above 8 m/s suppresses evidence during flight.
This favours control at reception instead of changing possession as a pass
flies past an opponent. Loose balls preserve possession unless sustained
opponent evidence justifies a change. Short touches and duels are discouraged
from producing extra possessions by the temporal transition cost.

Dead-ball evidence combines a ball beyond the pitch margin (0.5 m), stillness
for at least 1.5 seconds, missing-ball runs of at least 2 seconds, and low
collective player activity. Motion inside the pitch provides live-ball evidence.
A missing ball alone is not automatically a stoppage.

An offline three-state Viterbi decoder operates independently per half.
Turnover, stop and restart costs are 4.5, 6.9 and 5.9; evidence scale is 0.8
and the live bonus is capped at 1.2. Team evidence is delayed eight frames
(0.32 seconds) before decoding, zero-padded within each half. No evidence crosses
halftime. D/H/A scores use integer fixed-point arithmetic for deterministic
decisions. X is written outside graded halves, never as a graded-frame abstention.
The restarting team is inferred from control evidence; no manual restart labels
or target-frame constants are added.

## External calibration data, rights and provenance

- **IDSSE/DFL:** seven German first/second-division matches, approximately one
  million graded tracking frames. Provider labels `BallStatus` and
  `BallPossession` yield D/H/A. Bassek et al., *Scientific Data* 12, 195 (2025),
  [dataset paper](https://www.nature.com/articles/s41597-025-04505-y),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Metrica Sports Sample Game 3:** one match, 142,698 graded frames. Labels
  are derived algorithmically from Metrica's event annotations: RECOVERY,
  controlling PASS/CARRY/SHOT and SET PIECE establish a team; BALL OUT, foul,
  offside, handball and goal-related events establish dead-ball intervals until
  restart. The recipe predates this experiment. Source:
  [Metrica sample data](https://github.com/metrica-sports/sample-data).
  Its README offers community sample use and requests source acknowledgement
  for public use; no standalone standard license is declared in the root listing.
  Credit belongs to Metrica Sports. Source data is not redistributed here.
- **Synthetic checks:** 12 code-generated scenarios describing general task
  behaviours, not manually labelled observations of the provided game.

The base is `hybrid_v2`, calibrated on external matches (E6). This revision
excludes historical D-026 stop/restart adjustments selected through target-match
plausibility diagnostics. All model tuning in this revision uses external
matches only. The provided video's ground truth and the video itself are not
used for selection. The provided data is processed locally, not uploaded to
hosted AI services.

## Validation and limitations

External benchmark tracking is degraded by ball loss, position smoothing and
player loss to test robustness to tracking quality. The unchanged E8 timing grid
tests shifts -12, -8, -4, 0, 4, 8 and 12 frames, plus two alternative turnover
costs at zero shift. Candidates must retain perfect synthetic spellF1 and
frame accuracy of at least 0.90 on every scenario.

The timing shift is selected using seeds 0/1. Evaluation uses seeds 2/3 and the
organiser's scorer. In each leave-one-match-out fold, timing is selected without
that match, then scored on the excluded match. Other E6 parameters stay fixed.

| Realistic benchmark, eight external matches | External base | This revision |
|---|---:|---:|
| Mean spellF1 | 0.4618 | **0.4692** |
| Mean macroF1 | 0.8200 | **0.8222** |
| Metrica spellF1 | 0.5116 | **0.5186** |

All eight timing-grid LOMO folds improve: mean gain +0.0064, minimum +0.0022.
Six folds choose eight frames, two choose four. A final eight-frame setting is
selected on the external training seeds. These are previously used calibration
datasets, so the results are stability evidence, not an independent estimate on
wholly unseen data. F1 is not percentage accuracy. Full-match competition
accuracy is unknown because the full ground truth is private.

Short possessions remain poorly recovered. Tracking occlusions, identity errors,
restart handling, and domain differences between event-derived and provider
possession labels remain limitations. A timing shift helps segmentation but
does not solve short-possession recall. Relative velocity and touch detection
are possible future work, requiring external-only validation.

## Reproduction and deliverables

Inference needs Python 3.11+ (`tomllib`), numpy and pandas. The revision was
verified locally on Python 3.13 with exact package versions from
`requirements-lock.txt`; the original development environment used Python 3.12.
There is no pretrained model, machine-learning framework or external inference API.

From the repository root in Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
$env:MTAH_DATA_DIR = "<folder containing Tactical and Broadcast>"
.\.venv\Scripts\python.exe -m src.predict --config configs/final_external.toml --output outputs/final_external.csv
.\.venv\Scripts\python.exe evaluation/validate_submission.py outputs/final_external.csv
.\.venv\Scripts\python.exe scripts/release.py
```

The CSV's expected SHA-256 is
`f6f78ee82869ba63044548e7884d544455396e872e349834747560200998852b`.
The release script builds `outputs/submission_external/` with the team-named
CSV and technical document, an exact configuration copy, prediction manifest,
release report, and a ZIP of only those files. It checks the exact packaged CSV,
regenerates independently, and binds this document to the CSV hash and config
digest. It never publishes or submits the package. The source repository must
also be reviewed and made available for organiser reproduction.

Benchmark reproduction uses `scripts/omb_fetch.py`, Sample Game 3 from Metrica,
`scripts/omb_eval.py` and `scripts/omb_timing_lomo.py --base configs/hybrid_v2.toml`.
Sources and full experiment details remain in `docs/e8_results.md` and
`docs/rules_update.md` in the repository.
