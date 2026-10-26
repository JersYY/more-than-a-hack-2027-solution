# Engineering tickets (Phase 3)

Each ticket says what goes in and out, how to build it, what can go wrong, and the test that
proves it is done. The specification is the stub docstring in `src/` and the matching test file;
these tickets add the how.

**Done means:**

- the ticket's test file passes with its `pytestmark = pytest.mark.xfail(...)` line deleted;
- the whole suite passes;
- `docs/experiments.md` has a log entry if the ticket is an experiment.

A strict xfail that starts passing fails the suite. That forces the marker off at the moment the
code lands. A reference spike that passed every acceptance test sits in `outputs/spike_phase2/`
(local, not committed). Compare against it; do not paste it.

## Dependency graph

```text
T1 preprocess ──┐
T2 features ────┼──► T6 predict + CLI ──► T10 determinism ──► first real-match candidate
T3 evidence ────┤         ▲
T4 dead ball ───┤         │
T5 temporal ────┘         │
T7 OMB data ──► T8 degradation ──► E6 calibration ──┘ (re-tunes the configs, not the code)
T9 smart tagging (P1) ──► E5
```

T1–T5 do not depend on each other: each test file builds its own inputs. T7/T8 can run in
parallel with all of them.

---

## T1 — `src/preprocess.py::frame_context` (P0)

- **In:** `n_frames`, optional `blocks` (default `games.half_video_blocks`), optional `Tracking`.
- **Out:** `FrameContext(n_frames, blocks, half, attack_sign)` (the dataclass already exists).
- **Steps:**
  1. Validate the blocks: inside the recording, non-empty, strictly increasing.
  2. Fill `half`.
  3. For each half, compare the nan-median x of each team's tracked slots: the lower median
     attacks +1.
  4. Fall back to the default signs.
- **Failure modes:**
  - a half with nobody tracked (fall back);
  - median ties (fall back);
  - swapped teams, which is why there is a test with the slots swapped.
- **Done:** `tests/test_preprocess.py` (10 tests).

## T2 — `src/features.py::build_features` (P0)

- **In:** `Tracking`, `FrameContext`, `Config`.
- **Out:** a dict with every name in `features.FEATURES`, each `(n,)`, as `docs/feature_spec.md`
  defines it.
- **Steps:**
  1. Ball columns in metres.
  2. Central-difference speed with the seen-at-t−2/t/t+2 rule and the 40 m/s guard.
  3. `ball_out_m`.
  4. Run lengths (a vectorised run-length helper) for still and missing.
  5. Per-team nearest distances in chunks of 20,000 frames: about 70 MB per temporary on the real
     match, with 1.2 GB free.
  6. Slot-by-slot speed sums.
  7. A `cumsum` rolling mean for activity.
- **Failure modes:**
  - NaN leaking as 0;
  - a central difference spanning an occlusion;
  - an off-by-one in the window;
  - memory on the real match.
- **Done:** `tests/test_features_spec.py` (9 tests), and a run on the real tactical ATD under
  10 s that prints aggregates only.

## T3 — `src/possession_evidence.py::team_evidence` (P0)

- **In:** features (`ball_seen`, `d_home_m`, `d_away_m`, `ball_speed_mps`) and an `EvidenceConfig`.
- **Out:** `(n,)` float64 in `[−max_score, max_score]`.
- **Steps:** follow the docstring formulas exactly, using `+ − × ÷` only.
- **Failure modes:**
  - NaN distances must mean weight 0, not NaN evidence;
  - the flight gate must not fire on a NaN speed.
- **Done:** `tests/test_evidence.py` (6 tests).

## T4 — `src/dead_ball.py::dead_evidence` (P0)

- **In:** features (`ball_seen`, `ball_out_m`, `ball_speed_mps`, `ball_still_s`, `activity_mps`)
  and a `DeadBallConfig`.
- **Out:** `(n,)` float64.
- **Steps:** four additive terms, each 0 where its input is NaN, plus the official `ball_missing`
  kind.
- **Failure modes:**
  - a missing ball scoring as dead;
  - `NaN > x` comparisons warning (wrap them in `np.errstate`).
- **Done:** `tests/test_dead_ball.py` (8 tests).

## T5 — `src/temporal.py::decode`, `majority` (P0)

- **In:** `team`, `dead`, `FrameContext`, `TemporalConfig`. For `majority`: nearest-team codes
  and a window.
- **Out:** `(n,)` int8 `possession_metrics` codes; X outside the blocks.
- **Steps:**
  1. Integer scores: `np.rint(score × fixed_point)` as int64. Integer costs.
  2. A forward pass per block with back-pointers. Ties prefer the predecessor that changes state,
     then D < H < A, which gives the latest change.
  3. Backtrack.
  4. `majority` wraps the official `starter/baseline.py:smooth`. Do not re-derive it.
- **Failure modes:**
  - a tie rule that yields the earliest change, not the latest (the silent-gap test catches it);
  - float accumulation;
  - state carried across halves.
- **Performance:** a Python loop is ~1–2 s over 158,857 graded frames. Vectorise only if a
  profile says so.
- **Done:** `tests/test_temporal.py` (14 tests).

## T6 — `src/predict.py::predict`, `run`, CLI (P0, integration)

- **In:** `Tracking`, `FrameContext`, `Config` (`predict`); config path and output path (`run`).
- **Out:**
  - labels `(n,)` `<U1`;
  - a CSV written by `src.submission.write`;
  - `<csv>.manifest.json`: config digest, git commit and dirty flag, input and output SHA-256,
    versions, runtime, and the §4 diagnostics of `docs/experiments.md`, aggregates only.
- **Steps:**
  1. Route by decoder.
  2. Apply the output policy.
  3. Write the file, then the manifest.
- **Failure modes:**
  - the majority route accidentally applying X (the baseline must say "keep");
  - a manifest leaking frame-level values (D-002).
- **Done:** `tests/test_predict.py` (22 tests, incl. 2 data tests). Then the first candidate:
  `python -m src.predict --config configs/hybrid_v1.toml --output outputs/candidates/hybrid_v1.csv`,
  with the validator and diagnostics recorded in `docs/experiments.md`.

## T7 — other-match benchmark data (P1, can start any time)

- **In:** the public datasets in `docs/experiments.md` §5.
- **Out:** `scripts/omb_fetch.py` (downloads to `outputs/omb/`, gitignored) and
  `src/omb.py` (match → `Tracking` + H/A/D/X labels + blocks).
- **Steps:**
  1. DFL: a streaming XML parser (`iterparse`), never a full load, because each file is ~419 MB.
     Labels from `BallStatus` and `BallPossession`.
  2. Metrica Sample Game 3: tracking via `src/data_loader.read_layout/read_raw`; labels derived
     from the events by the rule in §5, written down before looking at any score.
- **Failure modes:**
  - pitch orientation and coordinate conventions differing from ours (normalise to ours);
  - frame rates;
  - half boundaries;
  - a label definition mismatch. Measure possession durations first; see §5 question 1.
- **Done:**
  - per match, the official scorer runs `labels` vs `labels` at 1.0;
  - the official baseline is scored on every OMB match;
  - provenance is written into `docs/experiments.md`.

## T8 — degradation model (P1, after T7)

- **In:** OMB tracking, and *aggregate* statistics of our ATD: graded ball visibility 59.3%, the
  gap-length histogram, the half-missing count 0.
- **Out:** `src/degrade.py`: a deterministic, seeded deletion of ball and player observations, so
  OMB tracking looks like our ATD in those aggregates.
- **Done:** the degraded OMB matches our ATD's visibility within 1 point, and its gap-length
  histogram within a stated tolerance.

## T9 — Smart Tagging evidence (P1)

- **In:** `starter/load_smart_tagging.load()` events, the ATD `Layout`, a `SmartTaggingConfig`.
- **Out:** `(team, dead)` priors as the stub docstring defines them.
- **Done:**
  - new acceptance tests on hand-built events;
  - the E5 log entry, including the on-match agreement diagnostics *with tagging off*.

## T10 — determinism check (P0, release gate)

- **Out:** `tests/test_determinism.py`:
  - two `run()` calls give identical bytes;
  - the decoder output is invariant to ±1e-12 noise in the evidence (already in T5);
  - a static check that no `exp`, `log`, `pow` or `hypot` appears on the decision path. The
    synthetic generator is exempt.
- **Done:** the tests pass. The Phase 4 cross-machine run is planned with a teammate.
