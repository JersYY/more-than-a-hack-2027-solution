# Experiments

How this project decides what works without a ground truth to tune on, and what each decision
rests on. Every experiment adds an entry to the log (§6), including the ones that fail.

---

## 1. The rules every experiment follows

| | allowed | not allowed |
|---|---|---|
| **The provided match** | running the pipeline; **aggregate** diagnostics (§4); unsupervised or algorithmic pseudo-labels with no human decision (RULES.md) | hand labels or corrections in any amount, for any purpose; rules or constants tied to specific frames; any human-produced data about this game |
| **The public 5-minute GT** | one pre-registered sanity check of the release candidate (§2) | training, tuning, thresholds, choosing between two versions, any comparison of candidates |
| **The private GT** | — | anything |
| **Other matches** | labelled data of any kind, with provenance documented (RULES.md, "Hand-labelled data from other games is fine") | — |
| **Synthetic matches** | anything: labels are known by construction (`src/synthetic.py`) | claiming synthetic scores predict the match score |
| **The AI assistant** | sees code, synthetic data, public other-match data and aggregates of the match | sees no row, coordinate, tag interval or frame-level value of the match (D-002) |

A known baseline number is never a measured one. The organisers' figures (spellF1 0.2433 for
`baseline_smooth50` on the full match) are quoted as theirs. No score of ours on the private ground
truth is claimed anywhere, because nobody can measure it.

## 2. The public window: what it is for, and the protocol

RULES.md, EVALUATION.md and FAQ.md agree: the 5 public minutes check that a pipeline is
**correct**, meaning frames aligned, file well-formed and labels meaning what you think. They are
not a leaderboard. With 17 possessions, one missed possession moves spellF1 by about six points,
and the window ranks the four official baselines in the wrong order (EVALUATION.md).

**Protocol (pre-registered here, before any candidate exists):**

1. It is run once per release candidate, after the candidate is frozen, and never during
   development.
2. It only detects gross errors. The candidate fails the check if
   `team_acc_both_alive < 0.5` (teams swapped), or `alive_dead_f1 < 0.3` (dead and alive
   confused), or the run errors.
3. A failure means a bug to find, not a parameter to move. The fix is justified on its own terms,
   and the check runs again.
4. Its numbers are never compared between candidates and never quoted as performance.

Phase 1 used the window only for the scorer smoke test (`run_evaluation.py` reproducing the
organisers' README, now `README_ORGANISERS.md`)
and for shape checks in `docs/data_audit.md` (covered frames, published counts).

## 3. Evaluation routes

| route | what it measures | strength | weakness |
|---|---|---|---|
| **S: synthetic scenarios** (`src/synthetic.py`, `scripts/synthetic_bench.py`) | whether a method has a behaviour CHALLENGE.md defines: holds through a brush, splits at a stoppage, changes at a reception | exact labels, as many as we want, inspectable frame by frame | made-up football: no tracker noise, no identity switches |
| **OMB: other-match benchmark** (§5) | spellF1 / macroF1 with the **official scorer** on real football with real labels | the only real accuracy signal we may use | other league and tracking system; label definitions may differ |
| **U: unsupervised diagnostics** (§4) | plausibility on *our* match | the actual target data | no accuracy: a plausible file can still be wrong |
| **R: reproducibility** | byte-identical regeneration, cross-machine | binary and decisive | — |

**Decisions use S and OMB. U only vetoes:** an implausible output is a bug signal. A plausible
output does not count as evidence of quality.

## 4. Unsupervised diagnostics on the match (aggregates only, D-002)

Every candidate run writes these to its manifest. None of them is an accuracy:

- label shares per half (H / A / D)
- possessions per half, and spell duration p10 / p50 / p90
- share of spells shorter than 1 s, and changes of possession per minute (a flicker detector)
- stoppages per half and their median duration
- agreement with Smart Tagging, when tagging is **not** an input: the share of offensive-phase
  frames labelled with the tagged team, and the share of SET PIECES intervals containing a D→live
  boundary
- runtime and peak memory

A diagnostic far outside football's plausible range is a veto. Examples: more than 1,000
possessions, D above 70% of graded frames, a median spell under 2 s. Inside the range, it says
nothing.

## 5. The other-match benchmark (OMB): Phase 3 builds it

| dataset | content | labels → H/A/D | access (checked 2026-10-26) | licence |
|---|---|---|---|---|
| **IDSSE / DFL** (Bassek et al., *Sci Data* 12, 195, 2025) | 7 Bundesliga matches, 25 fps tracking with ball | native, every frame: `BallStatus` (0 dead → D) and `BallPossession` (1 → H, 2 → A). The same recipe as our ground truth (CHALLENGE.md) | `https://ndownloader.figshare.com/files/<id>` works. kloppy's `springernature.figshare.com` URLs return HTTP 202 to scripts. ~419 MB of XML per match, so a streaming parser is needed (~1.2 GB of free RAM here) | CC BY 4.0 |
| **Metrica Sample Game 3** | 1 match, the **same EPTS format as ours** (`src/data_loader.py` reads it unchanged), plus events | derived from Metrica's event annotations by a documented rule: the team of the last controlling event; D from BALL OUT or foul to the next SET PIECE | `raw.githubusercontent.com/metrica-sports/sample-data`: tracking 53 MB, events 1.7 MB | Metrica sample-data licence; check it before use |

**Open questions Phase 3 must answer from the data itself (public, so the assistant may inspect
it):**

1. Do DFL possessions mean "control" or "last touch"? Compare their spell-duration distribution
   with Metrica Sample Game 3's event-derived spells. If they disagree, DFL calibrates the dead-ball
   model only, not the turnover cost.
2. How clean is each source's ball track? DFL's is near-complete. The degradation model
   (ticket T8) deletes ball observations to match our ATD's **aggregate** gap statistics:
   visibility 59.3% on graded frames and the gap-length distribution. It does that before any
   calibration.

**Selection rule (pre-registered):** a change is kept if it raises the OMB mean spellF1 by more
than the leave-one-match-out spread and does not break any synthetic acceptance test. Every
comparison runs with the official `possession_metrics.evaluate`.

## 6. Experiment catalogue

| id | hypothesis | P | route | acceptance | status |
|---|---|---|---|---|---|
| E0 | the official baseline is reproducible | P0 | R | 4/4 variants identical, 0 frames differ | **PASS** (Phase 1, D-007) |
| E1 | soft control evidence with a flight gate beats nearest-player codes | P0 | S, OMB | synthetic acceptance (`tests/test_evidence.py`, `test_predict.py`); OMB ≥ baseline | spike PASS on S |
| E2 | gaps carry no evidence and the decoder holds state through them | P0 | S, OMB | `occlusion_*` scenarios at 1.0 | spike PASS on S |
| E3 | an offline Viterbi with change costs beats the majority vote | P0 | S, OMB | the bar scenarios beat the baseline (`tests/test_predict.py`) | spike PASS on S |
| E4 | dead-ball cues (out, still, walking) with missing ≠ dead | P0 | S, OMB | the H-D-H, A-D-A, foul and second-half scenarios at 1.0; OMB alive/dead F1 | spike PASS on S |
| E4b | the last-touch restart prior (second-order states) | P1 | OMB | OMB spellF1 gain above the noise | planned |
| E5 | Smart Tagging soft priors | P1 | S, U | does not break S; agreement diagnostics; no OMB (no tagging there) | planned |
| E6 | calibrated costs, scale and cue weights | P1 | OMB | leave-one-match-out gain; robust across matches | planned |
| E6b | learned evidence (numpy logistic) from OMB labels | P2 | OMB | beats E6 leave-one-match-out | planned |
| E7 | ball detection in the tactical video for gaps | P3 | OMB, S | only after P0–P2 are solid | parked |
| E8 | broadcast fusion | P3 | — | needs a synchronisation we would build | parked |
| E9 | abstention on low-confidence continuations (macroF1 only) | P2 | OMB | macroF1 gain, spellF1 unchanged by construction | planned |
| E10 | byte-identical regeneration on another machine | P0 (release) | R | same SHA-256 | Phase 4 |

## 7. Results so far

### E0 — official baseline reproducible (Phase 1)

Four out of four variants regenerated byte-identical. The fallback's git blob equals the
organisers' committed blob. See `docs/agent_progress.md`, Run 1.

### Synthetic bar and feasibility spike (Phase 2, 2026-10-26)

`python scripts/synthetic_bench.py`. These are labels known by construction. Each column shows
spellF1, macroF1 and the number of predicted possessions. The `hybrid_v1` column comes from the
throwaway spike (kept locally in `outputs/spike_phase2/`, not in `src/`). It will be re-measured
when Phase 3 implements the tickets.

```text
scenario                 gt                raw           smooth50          smooth125          hybrid_v1
                       poss   spellF1 macro  n   spellF1 macro  n   spellF1 macro  n   spellF1 macro  n
stable_home               1    0.500 0.327   3    1.000 0.333   1    1.000 0.333   1    1.000 0.333   1
turnover                  3    0.600 0.590   5    1.000 0.606   3    1.000 0.597   3    1.000 0.635   3
deflection                1    0.100 0.324   7    1.000 0.333   1    1.000 0.333   1    1.000 0.333   1
duel                      1    0.057 0.299  13    1.000 0.333   1    1.000 0.333   1    1.000 0.333   1
occlusion_same_team       1    0.160 0.277   4    1.000 0.333   1    1.000 0.333   1    1.000 0.333   1
occlusion_long            1    0.080 0.230   4    0.267 0.307   2    1.000 0.333   1    1.000 0.333   1
occlusion_turnover        2    0.400 0.581   4    1.000 0.613   2    1.000 0.592   2    1.000 0.667   2
home_dead_home            2    0.320 0.259   3    0.133 0.264   1    0.133 0.264   1    1.000 0.663   2
away_dead_away            2    0.320 0.264   3    0.133 0.266   1    0.133 0.266   1    1.000 0.659   2
out_and_lost              2    0.400 0.869   5    0.640 0.656   3    0.700 0.507   2    1.000 0.981   2
foul_free_kick            2    0.350 0.450   6    0.600 0.497   2    0.600 0.497   2    1.000 0.994   2
second_half               2    0.300 0.541   6    0.800 0.571   2    0.800 0.571   2    1.000 0.994   2
mean spellF1                   0.299              0.714              0.781              1.000
```

With 5 more seeds (the players' movement changes, the labels do not), the spike scored a mean of
1.000 on every seed. The official smooth50 scored 0.699 to 0.723.

**What this shows.** The official baseline fails every stoppage where the ball stays visible, and
every occlusion longer than its vote. The MVP design has all twelve behaviours. The full
acceptance suite passed against the spike with `--runxfail`: 69 tests, 67 synthetic and 2 on the
real data. That includes the baseline-config reproduction of `baseline_smooth50.csv` on all
199,319 real frames. The contract is therefore implementable as written.

**What it does not show.** Anything about the real match. Synthetic tracking has no noise, no
identity switches and no duplicated tracks. That is OMB's job (E1–E4, E6).

**Two design flaws the spike found and fixed (D-019).** Each now has a discriminating acceptance
test:

1. *Mid-pass evidence.* A ball flying past an opponent at 3–4 m gave weak opponent evidence. That
   pulled the change of possession ~0.7 s before the reception. Fix: no team evidence while the
   ball moves faster than 8 m/s (`tests/test_evidence.py::test_a_ball_in_flight_is_nobodys`).
2. *Team evidence counted as "alive" evidence.* The free-kick taker standing next to the ball
   delayed the detected stoppage by ~0.8 s. Fix: team evidence decides H vs A in full, but alive
   vs D only up to a bounded bonus (`tests/test_temporal.py::
   test_a_player_next_to_a_dead_ball_does_not_keep_it_alive`).

## 8. Log template (one entry per experiment, appended below)

```yaml
id: E3
hypothesis: "..."
git_commit: "<commit or UNCOMMITTED>"
config: "<configs/....toml> (digest <first 12 hex>)"
assets: "<SYNTHETIC_ONLY | OMB: <matches> | MATCH (aggregates only)>"
tests: "<suites and results>"
omb_scores: "<official scorer on OMB, per match + mean, or n/a>"
local_diagnostics: "<the §4 aggregates; NOT an accuracy>"
public_gt_used_for_selection: false
status: KEEP | REJECT | INVESTIGATE
notes: "<why>"
```

## 9. Log

```yaml
id: SPIKE-P2
hypothesis: "The P0 contract (T1-T6) is implementable and solves every synthetic scenario."
git_commit: "UNCOMMITTED (scratch copy; reference kept in outputs/spike_phase2/)"
config: "configs/hybrid_v1.toml with flight_mps 8.0, alive_bonus 0.7"
assets: "SYNTHETIC_ONLY, plus the real ATD for the baseline-config reproduction test"
tests: "69/69 acceptance tests pass with --runxfail (67 synthetic + 2 data); baseline config == baseline_smooth50.csv on 199,319 frames"
omb_scores: "n/a (Phase 3)"
local_diagnostics: "n/a (no candidate was run on the match in Phase 2)"
public_gt_used_for_selection: false
status: KEEP
notes: "Found and fixed mid-pass evidence and team-evidence-as-alive (D-019). Synthetic only."
```

---

## 10. Phase 3 results (2026-10-26/27)

All benchmark numbers below come from the official `possession_metrics` on other matches' labels.
Competition-match numbers are aggregates of our own predictions, never scores: no ground truth of
that match was read.

### 10.1 The benchmark

| match | source | graded frames | D share | possessions | median | stoppages |
|---|---|---|---|---|---|---|
| J03WMX | DFL Köln–Bayern | 145,967 | 41.5% | 511 | 3.0 s | 103 |
| J03WN1 | DFL Bochum–Leverkusen | 141,561 | 51.7% | 500 | 3.1 s | 107 |
| J03WOH | DFL Düsseldorf–Regensburg | 137,214 | 47.6% | 402 | 3.9 s | 102 |
| J03WOY | DFL Düsseldorf–Rostock | 142,536 | 46.5% | 456 | 3.6 s | 113 |
| J03WPY | DFL Düsseldorf–Nürnberg | 146,211 | 44.3% | 421 | 3.8 s | 97 |
| J03WQQ | DFL Düsseldorf–St. Pauli | 142,345 | 40.7% | 401 | 4.7 s | 99 |
| J03WR9 | DFL Düsseldorf–Kaiserslautern | 146,810 | 41.9% | 370 | 4.2 s | 92 |
| SG3 | Metrica Sample Game 3 (events) | 142,698 | 42.0% | 331 | 4.8 s | 89 |

**Open question 1 of §5, answered.** DFL possessions behave like Metrica's event-derived ones: a
median of 3–5 s, 7–17% shorter than 1 s, 89–113 stoppages. So DFL is used for every parameter,
not only the dead-ball ones.

### 10.2 How the ATD loses the ball, and what that means (aggregates only)

- **Run lengths.** ~150 missing runs over graded frames, median 10.7 s, p90 34 s; only 18 are
  shorter than 5 s (`configs/atd_ball_runs.json`).
- **Player pace.** Players move at 1.18 m/s on average while the ball is missing, and at 2.16 m/s
  while it is seen. On labelled matches, live play runs at 2.20–2.43 m/s and stoppages at
  1.13–1.38.
- **Mixture estimate.** Solving the mixture: ~96–100% of the ATD's missing frames are probably
  stoppages, and the ATD's dead share is ~44–52% of graded frames.

This is why the long-gap cue exists (D-023), and why the benchmark's primary degradation places
90% of the hidden frames in stoppages (D-024).

### 10.3 Scores on the benchmark (mean over 8 matches; DFL degraded, seeds 2–3 never used in calibration)

| ball-loss assumption | official baseline | hybrid_v1 | **rc1** |
|---|---|---|---|
| clean tracking (no ball deleted) | 0.3161 | 0.4621 | **0.5624** |
| 70% of gaps in stoppages | 0.2890 | 0.3489 | **0.4836** |
| 90% of gaps in stoppages (primary) | 0.3195 | 0.3652 | **0.5151** |
| 100% of gaps in stoppages | 0.3397 | 0.3812 | **0.5424** |
| random gaps (renewal) | 0.2437 | 0.3484 | **0.4337** |

rc1 also has the best macroF1 (0.84 under the primary setting, against 0.71 for the baseline) and
the best team accuracy. **Leave-one-match-out** (guarded calibration without the match, seed 0):
the held-out match gains +0.156 on average, every one of the 8 gains (min +0.068 SG3, max +0.192),
and SG3 gains even though its format and labels differ from every training match.

### 10.4 The competition match: rc1's aggregates (plausibility only)

| | hybrid_v1 | hybrid_v2 | **rc1** | benchmark truth range |
|---|---|---|---|---|
| D share of graded frames | ~30% | 45.7% | **45.7%** | 41–52% (ATD estimate 44–52%) |
| possessions | 242 | 438 | **425** | 331–511 |
| median possession | 12–14 s | 4.7 s | **4.8 s** | 3.0–4.8 s |
| stoppages | 84 | 161 | **149** | 89–113 |
| live blips < 3 s between stoppages | — | 21 | **13** | 3–8 |

### 10.5 Log

```yaml
id: E1-E4 (P0 MVP, real match)
hypothesis: "Soft control + flight gate + dead-ball cues + Viterbi run end to end on the match."
git_commit: "c1d8368"
config: "configs/hybrid_v1.toml"
assets: "MATCH (aggregates only)"
tests: "all 69 acceptance tests pass; run twice -> identical bytes"
omb_scores: "mean 0.3681 vs baseline 0.3255 (degraded, seeds 0-1); wins 8/8 matches"
local_diagnostics: "242 possessions, median 12-14 s, D ~30%: under-segmenting, too little D"
public_gt_used_for_selection: false
status: KEEP (as the guarded starting point)
notes: "Under-segmentation: catches 5% of possessions < 2 s on clean DFL."
---
id: E6-unguarded
hypothesis: "Calibrating on the benchmark improves spellF1."
config: "hybrid_v1 -> coordinate descent, dead_weighted 0.9, seeds 0-1"
omb_scores: "0.5355"
public_gt_used_for_selection: false
status: REJECT
notes: "Broke synthetic duel and occlusion behaviours (pre-registered rule). D-025."
---
id: E6-guarded (hybrid_v2)
hypothesis: "The same calibration under the synthetic guard keeps most of the gain."
git_commit: "this commit"
config: "configs/hybrid_v2.toml"
omb_scores: "0.5268 (seeds 0-1), 0.5150 (fresh seeds 2-3); LOMO +0.156, 8/8 matches gain"
public_gt_used_for_selection: false
status: KEEP
notes: "Guard cost 0.009. Grid entries for the long-gap cue were added after the SG3 comparison (D-023)."
---
id: D-026 (rc1)
hypothesis: "Raising stop/restart costs removes mid-stoppage blips on the match at no benchmark cost."
config: "configs/rc1.toml"
omb_scores: "0.5151 (vs 0.5150), renewal 0.4337 (vs 0.4332)"
local_diagnostics: "blips 21 -> 13, stoppages 161 -> 149, possessions 425"
public_gt_used_for_selection: false
status: KEEP -> release candidate rc1
notes: "Chosen on a label-free plausibility diagnostic; benchmark-neutral."
---
id: E5 (Smart Tagging)
status: NOT RUN
notes: "No benchmark match has Smart Tagging, so a gain could not be validated. Deferred."
```
