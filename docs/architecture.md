# Architecture

**Status:** Phase 2 design. The interfaces exist as documented stubs in `src/`. Acceptance tests
for every P0 module exist in `tests/` and are `xfail(strict)` until Phase 3 implements them. A
throwaway feasibility spike passed all of them; the results are in `docs/experiments.md`.

Read with `docs/feature_spec.md` (every feature's definition), `docs/experiments.md` (how the
design is validated without ground truth), `docs/tickets.md` (the Phase 3 work packages) and
`docs/decisions.md` (D-013 onward).

---

## 1. The problem, as the metric sees it

Every frame gets `H`, `A` or `D`. The ranking metric, spellF1, scores **possessions**: maximal runs
of one team with the ball alive, matched one-to-one against the truth by temporal IoU
(EVALUATION.md). So the real task is segmentation:

- cut each half into **live** and **dead** intervals;
- cut each live interval into **team spells** that alternate between H and A;
- start each live interval with the team that restarts.

`H H D D H H` is two possessions, so a missed stoppage merges two possessions into one. A spell
that flickers to the other team for half a second is split into three. The three hard parts:

| difficulty | why it is hard | where the design answers it |
|---|---|---|
| the ball is missing on 40.7% of graded frames | no ball, no nearest player | §4.3, §4.5: no evidence means no change |
| "controlled" possession vs touches, duels, deflections | nearest player ≠ player in control (D-012) | §4.1 soft control and the flight gate; §4.3 change costs |
| stoppages with the ball still visible | the baseline's only dead signal is "ball missing" | §4.2 dead-ball cues |

## 2. Pipeline

```text
 games.py ─ half blocks, pitch, asset paths (MTAH_DATA_DIR)
    │
    ▼
 src/data_loader.py  load_tracking() ─► Tracking            players (n,S,3) ball (n,3) layout   [done, Phase 1]
    │
    ▼
 src/preprocess.py   frame_context() ─► FrameContext        half (n,) blocks attack_sign        [T1]
    │
    ▼
 src/features.py     build_features() ─► {name: (n,)}       docs/feature_spec.md                [T2]
    │                                                       (+ src/smart_tagging.py, P1)        [T9]
    ├──────────────────────────────┐
    ▼                              ▼
 src/possession_evidence.py     src/dead_ball.py
 team_evidence() ─► team (n,)   dead_evidence() ─► dead (n,)                                    [T3, T4]
    │  > 0 home, < 0 away          │  > 0 dead, < 0 live
    └──────────────┬───────────────┘
                   ▼
 src/temporal.py   decode() ─► states (n,) pm codes          per-half Viterbi over {D, H, A}    [T5]
                   ▼
 src/predict.py    predict() ─► labels (n,) H/A/D/X          output policy (X outside halves)   [T6]
                   run()     ─► CSV (src/submission.py, LF) + <csv>.manifest.json
```

**Baseline path.** With `configs/baseline.toml` the same `predict()` takes a second route:
`nearest_team` (Phase 1) and then `temporal.majority`, which is the official vote. The spike
showed this route reproduces `baseline_smooth50.csv` on all 199,319 frames, which makes it a
standing regression test of the orchestration.

## 3. Data contracts

| object | shape / type | meaning | missing |
|---|---|---|---|
| `Tracking.players` | (n, S, 3) float64 | x, y normalised 0–1; s in **m/s** (D-008) | NaN |
| `Tracking.ball` | (n, 3) float64 | x, y normalised; s undocumented, unused | NaN |
| `FrameContext.half` | (n,) int8 | half of each frame, 0 outside the graded blocks | — |
| `FrameContext.attack_sign` | {(half, team): ±1} | +1 = attacks towards x = 1 | defaults (§5) |
| features | {name: (n,)} | `docs/feature_spec.md` | NaN, never 0 |
| `team` | (n,) float64 | team evidence, `|team| ≤ max_score` | 0 = no information |
| `dead` | (n,) float64 | dead-ball evidence | 0 = no information |
| states | (n,) int8 | `possession_metrics` codes H=0 A=1 D=2 X=3 | X outside blocks |
| labels | (n,) `<U1` | the submission column | — |

One code system runs through the whole pipeline: `possession_metrics`' H/A/D/X, with
`data_loader.HOME == pm.H == 0` and `AWAY == pm.A == 1`. That way the official scorer's functions
apply to any intermediate result.

## 4. The model

### 4.1 Team evidence — `src/possession_evidence.py`

Each team's nearest tracked player to the ball gets a control weight
`w(d) = 1 / (1 + (d / control_m)²)²`. It is 1 at the ball, ¼ at `control_m` = 1 m, and falls fast
beyond that. A third reference weight `w0 = w(free_m)` stands for "nobody controls it". Then:

```text
team = max_score · (w_home − w_away) / (w_home + w_away + w0)        ∈ [−max_score, +max_score]
```

- Close home player, distant opponent: ≈ +max_score.
- Ball loose and far from everyone: ≈ 0. The ball is nobody's, and the decoder keeps the last team.
- **No ball, no evidence:** 0.
- **The flight gate:** 0 where the ball moves faster than `flight_mps` (8 m/s). That is faster
  than a sprinter can carry it, so the ball is in flight and nobody controls it. Possession
  changes at the reception (CHALLENGE.md), not as a pass flies past someone. The spike found this
  (D-019).
- Only `+ − × ÷` are used (§7).

This is evidence, not a decision. A brush is a few frames of it. A controlled possession is
seconds of it. The difference is priced in the decoder, not here.

### 4.2 Dead-ball evidence — `src/dead_ball.py`

Four cues, summed. Each contributes 0 where its input is unknown.

| cue | evidence | why |
|---|---|---|
| ball beyond a line by more than 0.5 m | +4 | out of play by definition |
| ball inside and moving faster than 3 m/s | −1 | a ball in play usually moves |
| ball still (< 0.5 m/s) for a run of ≥ 1.5 s | +1 | placed for a restart, not dribbled |
| mean player speed over ±2 s, relative to 1.2 m/s | ±2 max | players walk at stoppages and run in play |

**A missing ball contributes nothing by itself.** The ball is missing on ~41% of graded frames,
mostly in live play (DATA.md). The official baseline equates missing with dead, and that is the
main reason it fails every visible-ball stoppage in the synthetic bench (`docs/experiments.md`).

### 4.3 The decoder — `src/temporal.py`

The decoder is offline and runs **per graded block** (half). It is a Viterbi over three states
{D, H, A}, with no preferred initial state:

```text
maximise  Σ_t score_t(state_t) − Σ_t cost(state_{t−1}, state_t)

  b_t        = min(|team_t| / 2, alive_bonus)
  score_t(D) = s · dead_t / 2
  score_t(H) = s · (−dead_t / 2 + b_t − max(−team_t, 0))
  score_t(A) = s · (−dead_t / 2 + b_t − max(team_t, 0))

  cost(H↔A) = turnover_cost   cost(H/A→D) = stop_cost   cost(D→H/A) = restart_cost
```

- **Why offline.** The whole file is produced at once, so each half can be decoded with all its
  evidence. A causal hysteresis filter notices a change only after the evidence has
  accumulated, then places it late. Offline decoding places it where the evidence starts. A
  one-second offset already costs spellF1 heavily (0.6539 for "perfect but 1 s late",
  EVALUATION.md).
- **Why change costs.** A burst of opponent evidence shorter than about
  `2 · turnover_cost / (s · Δscore)` cannot pay for two changes of possession. That is a
  deflection (CHALLENGE.md). Seconds of evidence can pay for one, and that is control.
- **Why the bounded alive bonus.** `score(H) − score(A) = s · team`, so team evidence decides
  between the teams in full. But it can lift the better team over D by at most
  `s · alive_bonus`. A taker standing next to a placed ball is as normal at a restart as a
  dribbler is in play, so alive vs D must be decided by dead-ball evidence. A mixture model,
  P(team evidence | D) = ½·P(·|H) + ½·P(·|A), gives exactly this kind of bound, ln 2. The spike
  found the problem (D-019).
- **Ties go to the latest change.** Across a stretch with no evidence, any switch point scores
  the same. The decoder switches where evidence resumes. That is where the scorer's own forward
  fill of `X` would put the change (EVALUATION.md), and it matches the rule that possession stays
  with the last team until someone else controls the ball.
- **Integers.** Scores and costs are rounded to multiples of `1/fixed_point` and summed as int64.
  The chosen path therefore cannot depend on a CPU's last-bit rounding (§7).

The default costs are first-principles priors: `cost ≈ ln(mean duration in frames)`. That gives
5.7 for possessions of ~12 s, 6.9 for ~40 s of live play between stoppages, and 5.9 for stoppages
of ~15 s. `evidence_scale` = 0.2 discounts per-frame evidence for autocorrelation (~5 correlated
frames per independent observation). **None of these values is fitted to this match.**
Experiment E6 calibrates them on other matches.

### 4.4 Restarts and kickoffs

There are no restart rules and no kickoff frames in the code. A stoppage ends where the dead-ball
evidence turns: the ball starts moving and players speed up. The restarting team is whichever
team's evidence follows: the taker is next to the ball, then the first reception. The second half
is graded from before its kickoff, and those 2,934 frames are all `D` (FAQ.md). The model must
find that from the evidence: a still ball at the centre spot and walking players. It must not
come from a published frame number (D-015). The synthetic `second_half` scenario tests exactly
this.

P1 upgrade (E4b): the "last touch" rule of football, where the team that did *not* put the ball
out restarts. It needs a second-order state (D-after-H, D-after-A) and would be validated on
other-match labels.

### 4.5 Missing ball

There is no special mode. Frames without a ball carry zero team evidence and only the
player-speed dead-ball cue, so the decoder holds its state through the gap. Where the evidence
resumes:

- **Same team:** no change, one possession. Synthetic `occlusion_same_team` (2.5 s) and
  `occlusion_long` (6 s, longer than the baseline's ±2 s vote can bridge).
- **Other team:** the change goes at the gap's end, by the tie rule. That is where the ball
  reappears at the receiver's feet. Synthetic `occlusion_turnover`.
- **A stoppage inside the gap:** found by the player-speed cue. Synthetic `out_and_lost`.

### 4.6 Smart Tagging (P1, E5) — `src/smart_tagging.py`

A small additive prior, off in every P0 configuration:

- **Offensive codes:** the tagged team is in possession.
- **Defensive codes:** the *other* team is in possession.
- **Set pieces:** dead before the restart inside the interval.
- **N/A:** no team.

Team values are matched to the ATD's own team names (`Layout.home_name`, `Layout.away_name`),
never to a hard-coded string. Intervals are phases, not possessions, so the weight stays small
enough that tracking evidence always wins (DATA.md).

## 5. Boundary handling

| boundary | rule |
|---|---|
| halves | decoded independently; nothing carries over; no switch cost at a half start |
| outside the graded blocks | `X` (`output.ungraded = "X"`); the decoder never runs there |
| first half | graded from the kick itself; the first spell starts on frame 1 of the block |
| second half | graded from 2,934 frames before its kickoff; `D` must come from the evidence |
| recording edges | features that need t ± k are NaN there; the decoder is unaffected |
| attack direction | measured per half from team median x (T1); default home +1 in half 1, swapping each half |
| block definitions | `games.half_video_blocks` only; the kickoff anchors in `games.py` are playback positions (`docs/data_audit.md` §9) |

## 6. Output policy

Graded frames always get `H`, `A` or `D`. Never `X` (D-015). The scorer fills an `X` with the
previous label in the same half before computing spellF1, so on the ranking metric an `X` equals
repeating the last answer. Under macroF1 it is a guaranteed miss. Writing the decoder's label is
therefore never worse on spellF1 and usually better on macroF1, the tie-break.

`X` goes only outside the graded blocks, where nothing is scored. P2 experiment E9: abstain on
low-confidence frames that repeat the previous label. That leaves spellF1 unchanged by
construction and could gain macroF1. It needs posteriors and is not P0.

## 7. Determinism contract

RULES.md: re-running the repository must reproduce the submitted file, "not approximately — the
same file". The organisers' machine is not ours.

1. Inference has no randomness: no seeds, no sampling, no multithreaded reductions.
2. On the decision path, only operations IEEE-754 rounds correctly: `+ − × ÷ sqrt`, comparisons,
   `min`/`max`. Sums run in a fixed order: sequential `cumsum`, and slot-by-slot accumulation.
   No `exp`, `log`, `pow` or `hypot`. Vectorised SIMD versions of those differ between CPUs in
   the last bit.
3. The decoder rounds to fixed point and accumulates in int64. A last-bit difference upstream
   could then flip a decision only at an exact rounding boundary.
4. The output is written LF and ASCII by `src/submission.write` (D-006), with its SHA-256 in the
   manifest.
5. Phase 4 regenerates the file from a clean environment and compares hashes. If possible, a
   teammate's machine reruns it too.

## 8. Configuration

TOML, read with the standard library's `tomllib` (D-014). The schema is `src/config.py`: typed,
defaulted and range-checked. Unknown keys are errors. `Config.digest()` identifies a run in its
manifest.

- `configs/baseline.toml`: the official baseline as a configuration (regression anchor).
- `configs/hybrid_v1.toml`: the P0 MVP. Every value carries a comment with its first-principles
  reason.

New behaviour enters behind a configuration switch with its default off. Every candidate is
therefore a configuration plus a commit, and the fallback (`outputs/fallback/`) stays one command
away.

## 9. Upgrade path (each one reversible, each with its experiment)

| priority | upgrade | experiment | validation route |
|---|---|---|---|
| P0 | the MVP above | E1–E4 | synthetic acceptance + the other-match benchmark |
| P1 | calibrated parameters | E6 | other-match benchmark, leave-one-match-out |
| P1 | last-touch restart prior (second-order states) | E4b | other-match benchmark |
| P1 | ball–player relative velocity in team evidence | E1b | synthetic + other-match benchmark |
| P1 | Smart Tagging priors | E5 | synthetic + on-match agreement diagnostics |
| P2 | learned evidence (logistic, numpy) from other-match labels | E6b | other-match benchmark, leave-one-match-out |
| P2 | abstention on low-confidence continuations (macroF1) | E9 | other-match benchmark |
| P3 | ball detection in the tactical video for gaps | E7 | other-match benchmark; only if P0–P2 are solid |
| P3 | broadcast fusion | E8 | needs a synchronisation we would have to build |
