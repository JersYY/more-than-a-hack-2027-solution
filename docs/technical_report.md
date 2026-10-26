# Frame-level possession from automatic tracking: evidence, dead-ball cues and a priced decoder

*More than a Hack 2027, Phase 1: technical document of team **Aziz goes to barcelona**.*
*Submission: `configs/final.toml` (release candidate rc1, frozen), possession file SHA-256
`db7538d3c9fae9285cc9c40ea6b1c62426c22f1a916cf3ac31c3fdaad90eea61`.*

## 1. The problem, and the approach in one paragraph

Each of the 199,319 tactical-video frames gets H (home in possession), A (away), D (dead ball) or
X (outside the two graded halves). The ranking metric, spellF1, scores possessions as whole
intervals. So we treat the task as segmentation, not per-frame classification.

Each frame yields two pieces of evidence from the tracking:

- **team evidence:** whose player can play the ball;
- **dead-ball evidence:** is the ball beyond a line, at rest, missing for long, or are the
  players walking.

A Viterbi decoder then labels each half as a whole, offline. It charges a price for every change
of possession, every stoppage and every restart. A brush or a duel is a few frames of evidence and
cannot pay for two changes of possession. A controlled possession is seconds of evidence and can.
The parameters were chosen on **other matches' labels**: 7 Bundesliga matches and a Metrica sample
game. None was chosen on the competition match's ground truth.

## 2. Assets and timeline

| asset | used | how |
|---|---|---|
| tactical ATD (tracking) | **yes** | the only model input. Our own loader reads the layout from the metadata's DataFormatSpecification. It matches the official parser on all 199,319 frames and kloppy on 550,206 sampled values. `video_frame = frame_id - 1`, verified on the file |
| tactical video | no | not needed by the model; possible future work: ball detection in occlusions |
| Smart Tagging | no (implemented interface only) | intervals are phases, not possessions, and no benchmark has tags to validate a prior against |
| homography | no | only needed for video work |
| broadcast video / ATD / homography | no | a different timeline with no synchronisation |
| half blocks (`games.py`) | yes | the grading definition: decoding never crosses a half, and X goes outside them |

Two data facts shaped the model:

- **The speed channel is m/s,** despite the metadata's `kph` label. We checked it against
  displacement (median ratio 0.989).
- **The ATD loses the ball in long blocks,** not short flickers: about 150 runs over the graded
  frames, median 10.7 s. Players move at 1.18 m/s while it is missing against 2.16 m/s while it
  is seen. Most of the ball's absence therefore coincides with stoppages.

## 3. Features (per frame, `docs/feature_spec.md`)

- **Ball:**
  - position in metres;
  - speed, as a central difference over ±2 frames, NaN across gaps or above 40 m/s;
  - distance beyond the pitch rectangle;
  - length of the still run and of the missing run it belongs to.
- **Players:** the distance from the ball to each team's nearest tracked player, and tracked
  counts per team.
- **Activity:** the mean speed of all tracked players, averaged over ±2 s.

Missing observations stay missing (NaN), never zero. Only `+ − × ÷ √` and fixed-order sums are
used, so the output does not depend on the CPU (§7).

## 4. Possession evidence, and the rules it encodes

**Control weight.** Each team's nearest player gets `w(d) = 1/(1 + (d/1 m)²)²`. Team evidence
compares the two weights against a "nobody controls it" reference `w(1.5 m)`:

```
team = 3 · (w_home − w_away) / (w_home + w_away + w(1.5 m))
```

The possession rules we drew the line on:

- **Loose ball:** nobody within ~1.5 m means almost no evidence, so the decoder keeps the last
  team. CHALLENGE.md: "possession stays with the last team that had it."
- **Ball in flight:** above 8 m/s, faster than a player can run with it, team evidence is zero.
  Possession changes at the **reception**, not as a pass flies past an opponent.
- **Touch, deflection, duel:** short bursts of opponent evidence cannot pay for two changes of
  possession (§6), so they do not change it.
- **Restarts:** the restarting team is whoever controls the ball when dead-ball evidence ends.
  There are no hand-coded restart rules and no frame numbers. The second half's pre-kickoff
  stoppage is found from the evidence, like any other.

## 5. Dead-ball evidence

The terms are summed:

| cue | effect |
|---|---|
| ball more than 0.5 m beyond a line | +4 |
| ball moving faster than 3 m/s inside the pitch | −1 |
| ball still for at least 1.5 s | +1 |
| ball missing for at least 2 s | +1 |
| players' activity vs 1.6 m/s | clip((1.6 − v)/0.8, ±1) |

A short gap alone is never "dead". A long gap is evidence, weighed against how fast the players
are moving.

## 6. The decoder

Three states, D, H and A, decoded per half by Viterbi:

- **Scores.** With evidence scale s = 0.8 and an alive bonus of 1.2:
  - D scores `s·dead/2`;
  - H scores `s·(−dead/2 + min(|team|/2, 1.2) − max(−team, 0))`, and A symmetrically.

  Team evidence decides H vs A in full. It can only weakly favour "alive" over D, because a taker
  standing next to a placed ball is normal at a restart.
- **Prices.** Turnover 4.5, stoppage 8.9, restart 7.9.
- **Ties.** Across a stretch with no evidence, the change goes where evidence resumes. That is
  where the scorer's own forward-fill of X would put it.
- **Integer arithmetic.** Scores are rounded to integers (units of 10⁻⁶) and summed in int64.
- **Output.** No X is written on graded frames. On spellF1, an X only repeats the previous label;
  on macroF1 it is a guaranteed miss.

## 7. Training and validation data (no labels of the provided game)

| dataset | what | labelled by | amount | licence |
|---|---|---|---|---|
| IDSSE / DFL open data (Bassek et al., *Sci Data* 12, 195, 2025) | 7 German league matches, 25 fps positions with the ball | the data provider: per-frame BallStatus and BallPossession | 7 matches, ~1.0M graded frames | CC BY 4.0 |
| Metrica Sports Sample Game 3 | 1 match, same tracking format as ours, plus events | Metrica's event annotations; per-frame labels derived by a rule fixed before scoring (§7.1) | 1 match, 142,698 graded frames | Metrica sample-data repository |
| synthetic scenarios | 12 scripted situations (interception, brush, duel, occlusion, throw-in, foul, second-half kickoff, …) | labels by construction, following CHALLENGE.md | 12 scenarios | ours |

### 7.1 How the labelled data was used

- **Labels.** H/A/D follow the same recipe as the competition's ground truth: D where the ball is
  dead, otherwise the owning team. For Sample Game 3, possession changes at a RECOVERY or a
  restart; the ball is dead from a BALL OUT, foul, offside, handball or goal until the next
  restart.
- **Realistic ball loss.** The benchmark's clean ball tracks are degraded to the competition
  tracking's rhythm: ball-missing run lengths are drawn from an *aggregate* histogram of the
  ATD's gaps, and 90% of hidden frames fall in stoppages. Robustness was checked at 70%, 100%
  and fully random placement.
- **Selection.** Parameters were chosen by coordinate descent over a declared grid. The objective
  was the mean official spellF1 over the 8 matches. A guard rejected any setting that broke a
  synthetic behaviour (it rejected, for example, a setting that split possessions in duels).
- **Generalisation.** Leave-one-match-out: the held-out match gains +0.156 on average, and all 8
  gain.
- **The competition match** contributed only aggregate plausibility checks: possession count,
  dead share, stoppage count. One choice was made on them: higher stoppage costs, which were
  neutral on the benchmark.
- **The public 5-minute ground truth was not used for any choice.** It ran once as a format and
  alignment check.

## 8. Results we can state honestly

Benchmark mean spellF1 (8 matches, ball degraded, seeds never used in calibration):

| | official baseline | first-principles MVP | **final (rc1)** |
|---|---|---|---|
| primary setting | 0.320 | 0.365 | **0.515** |
| random gaps | 0.244 | 0.348 | **0.434** |

On the competition match, the final configuration predicts:

- 425 possessions with a median of 4.8 s;
- 45.7% dead ball;
- 149 stoppages.

All of this lies within or near the range of real matches. **We do not know its score on the
competition's ground truth.** That ground truth is private, and the benchmark is other leagues,
other tracking and other annotators.

## 9. Limitations and what we would do next

- **Domain shift.** The benchmark's tracking is cleaner than the ATD. Only ball loss is
  simulated, not identity switches, duplicates or position noise.
- **Dead-ball handling.** Players carrying the ball back for a restart can still look like brief
  live play: 13 short blips remain.
- **Short possessions.** Possessions under 2 s are frequent in real football and still mostly
  missed. Better touch detection would help: ball-direction changes, and ball–player relative
  velocity.
- **No restart prior.** The "last touch" rule, where the other team restarts after the ball goes
  out, is not used yet.
- **Smart Tagging** could add a validated team prior. It needs labelled matches that come with
  tags.

## 10. Reproduction

```powershell
# Python 3.12 (3.11+ for tomllib)
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
$env:MTAH_DATA_DIR = "<folder containing tactical/ and broadcast/>"
.venv\Scripts\python.exe -m src.predict --config configs/final.toml --output outputs/final.csv
.venv\Scripts\python.exe evaluation/validate_submission.py outputs/final.csv
# expected SHA-256: db7538d3c9fae9285cc9c40ea6b1c62426c22f1a916cf3ac31c3fdaad90eea61
.venv\Scripts\python.exe -m pytest -q          # all tests; data tests need MTAH_DATA_DIR
```

- **Determinism.** Two runs write byte-identical files; this is tested. The CSV's SHA-256 is in
  its manifest.
- **Benchmark.** To rebuild it:
  - `python scripts/omb_fetch.py` downloads the DFL data;
  - Sample Game 3 comes from Metrica's sample-data repository;
  - `python scripts/omb_eval.py configs/final.toml` scores it;
  - `python scripts/omb_calibrate.py --base configs/hybrid_v1.toml --guard` reruns the
    calibration.
