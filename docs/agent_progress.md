# Agent progress

The resume point for any session. Read this, then `docs/decisions.md`, then check the workspace —
never assume a claim below without re-running its command.

## Before you touch anything

1. **Do not open or print match data.** The competition terms forbid making the data, or any
   extract of it, available to a hosted LLM (`docs/decisions.md` D-002). Code that reads the
   bundle runs locally and reports aggregates or PASS/FAIL only.
2. Data commands need the bundle location (D-004). In PowerShell:

   ```powershell
   cd "C:\Users\USER\Documents\Kuliah\Lomba\More Than A Hack\more-than-a-hack-2027"
   $env:MTAH_DATA_DIR = "C:\Users\USER\Documents\Kuliah\Lomba\More Than A Hack"
   .venv\Scripts\python.exe -X utf8 -m pytest -q      # after Phase 2: 120 passed, 69 xfailed
   ```

3. Do not overwrite `evaluation/results/baseline_*.csv`; regenerate into `outputs/` (D-007).
4. Work happens on branch `solution`. `main` is the organisers' history. Never push to `origin`
   (the organisers' repository).
5. An `xfailed` test is an acceptance test whose code does not exist yet (D-018). Implementing a
   ticket means deleting its file's `pytestmark` line and making the tests pass.

---

## Run 1 — Phase 1: reconnaissance, data contracts and baseline

**2026-10-26 15:25 UTC** · base commit `f8f2629` (organisers' `main`) · **all work uncommitted**:
nothing has been committed or pushed. The user did not ask for it, and `origin` is the organisers'
repository (D-001).

```text
PHASE: 1
STATUS: PASS
FILES CREATED/CHANGED:
  changed  .gitignore                  (+ outputs/, .pytest_cache/)
  new      src/__init__.py, src/data_loader.py, src/features.py, src/submission.py
  new      scripts/data_audit.py, scripts/verify_baseline.py
  new      tests/conftest.py, tests/test_contracts.py, tests/test_scorer_semantics.py,
           tests/test_alignment.py, pytest.ini
  new      requirements-dev.txt, requirements-lock.txt
  new      docs/data_audit.md (generated), docs/decisions.md, docs/agent_progress.md
  local    outputs/ (gitignored): baseline_regen/, fallback/, audit/, logs/
COMMANDS EXECUTED (all exit 0 unless noted):
  git clone https://github.com/metrica-sports/more-than-a-hack-2027 more-than-a-hack-2027
  python -m venv .venv ; pip install -r requirements.txt kloppy==3.18.0 pytest
  python evaluation/run_evaluation.py --no-detail        (first try: exit 1, Windows Smart App
                                                          Control blocked a pandas DLL; re-run OK, D-010)
  python evaluation/validate_submission.py evaluation/results/baseline_{raw,smooth25,smooth50,smooth125}.csv
  python starter/baseline.py --out outputs/baseline_regen
  python scripts/verify_baseline.py
  python scripts/data_audit.py                           (an earlier run exited 1 on two homography
                                                          figures; reclassified as DIFF, D-011)
  python -m pytest -q                                    (with and without MTAH_DATA_DIR)
EVIDENCE:
  scorer      public-window board reproduces README.md exactly: smooth125 0.5812/0.4242,
              smooth50 0.6207/0.3256, smooth25 0.6200/0.2868, raw 0.5972/0.0378 (macroF1/spellF1)
  baseline    official generator re-run: 4/4 variants validate, 0 mismatching frames, identical
              bytes after newline normalisation (E0)
  fallback    outputs/fallback/baseline_smooth50.csv, LF; official validator OK; strict check OK;
              git blob 1e1e4675... == organisers' committed blob; sha256 6d4e04b80b9016cc...
              (full-match spellF1 0.2433 is the organisers' published number, not measured here)
  data audit  docs/data_audit.md: 87 checks, 64 PASS, 0 FAIL, 4 DIFF, 19 INFO, 8 s
  alignment   ATD frame_id exactly 1..199319 -> video_frame = frame_id - 1 proven on the file;
              Smart Tagging time base agrees (both kickoffs); homography 199319 rows, frame 0..n-1;
              half blocks 11185..90651 / 110808..190197, 158857 graded frames
  loader      0 mismatching frames vs official parser (all 199319 frames, 4 variants);
              0 mismatching values vs kloppy (3987 sampled frames, 550206 player values)
  tests       60 passed with data (7 s); 49 passed + 11 skipped without data (3 s)
  git         git add -A --dry-run stages nothing from outputs/, data, videos or scorer reports
BLOCKERS: none.
  One decision for the user (D-002): confirm the signed terms allow aggregate statistics to reach
  a hosted LLM; if not, scripts switch to PASS/FAIL-only output.
NEXT COMMAND:
  Resume the MORE THAN A HACK 2027 playbook from docs/agent_progress.md and EXECUTE PHASE 2:
  ARCHITECTURE & EXPERIMENT CONTRACTS (playbook section "PHASE 2"), keeping D-002.
```

### Gate 1 checklist

| Gate 1 requirement | Evidence | Status |
|---|---|---|
| Repository checked | clone at `f8f2629`; all contracts read: README, CHALLENGE, DATA, RULES, EVALUATION, SUBMISSION, FAQ, OVERVIEW; scorer, validator, `games.py`, starter scripts and notebooks | PASS |
| Rules checked | conflict with the playbook found and resolved toward the terms (D-002); public-GT rule (D-009) | PASS |
| Commands checked | scorer, validator, official baseline generator all run; outputs as documented | PASS |
| Baseline | regenerated, byte-identical to the committed files; fallback promoted (D-007) | PASS |
| Paths | `games.asset_paths()` via `MTAH_DATA_DIR` resolves all 9 assets (D-004) | PASS |
| Frame contracts | ATD `frame_id - 1`, Smart Tagging `round(s*25)`, homography 0-based, half blocks: all measured | PASS |
| Tests | 60 pass with data; data tests skip cleanly without it | PASS |

### What Phase 2 should know (from the audit; details in `docs/data_audit.md` §8)

- The ball is missing on 40.7% of graded frames. Missing is normal input, not evidence of a dead
  ball.
- Player speed is m/s despite the metadata's `kph` label. The ball's speed channel is sparse
  (8.7%) and its semantics are undocumented.
- 4.2% of graded ball positions fall outside `[0,1]^2`: a dead-ball candidate signal.
- Tracked objects do not vanish at halftime (12.5 on average), so stoppages and halves cannot be
  found by head count. The half blocks come from `games.py`.
- 46 slots (24 home / 22 away), ~20.5 tracked per graded frame, up to 25. Slots are tracklets,
  not players.
- Smart Tagging instances run 22.8 s at the median. They are phases, not possessions, and 36 have
  `Team = N/A`, including every throw-in.
- The ranking metric fills `X` forward within a half, so abstaining is free inside a possession
  and costly across a handover (pinned in `tests/test_scorer_semantics.py`).

---

## Run 2 — Phase 2: architecture and experiment contracts

**2026-10-26 16:06 UTC** · branch `solution`, on top of Phase 1 commit `f281e3b` · committed at the end of
this run (see git log).

```text
PHASE: 2
STATUS: PASS
FILES CREATED/CHANGED:
  new      docs/architecture.md, docs/feature_spec.md, docs/experiments.md, docs/tickets.md
  changed  docs/decisions.md (D-013..D-022), docs/agent_progress.md
  new      src/config.py (implemented), configs/baseline.toml, configs/hybrid_v1.toml
  new      src/synthetic.py (implemented: scenario generator + 12-scenario catalogue)
  new      stubs with full contracts: src/preprocess.py, src/possession_evidence.py,
           src/dead_ball.py, src/temporal.py, src/predict.py, src/smart_tagging.py
  changed  src/features.py (+ FEATURES contract and build_features stub)
  new      scripts/synthetic_bench.py
  new      tests/test_config.py, tests/test_synthetic.py (passing);
           tests/test_preprocess.py, test_features_spec.py, test_evidence.py, test_dead_ball.py,
           test_temporal.py, test_predict.py (acceptance, xfail strict until Phase 3)
  local    outputs/spike_phase2/ (gitignored feasibility spike, D-022)
COMMANDS EXECUTED (all exit 0 unless noted):
  git switch -c solution ; git commit (Phase 1, f281e3b)        user approved committing
  curl range requests to figshare / GitHub                      OMB access check (D-017)
  python scripts/synthetic_bench.py                              official baseline bar
  pytest -q --runxfail (scratch spike copy)                      62/65, then 69/69 after D-019
  python -m pytest -q (with / without MTAH_DATA_DIR)             120 passed + 69 xfailed /
                                                                 109 passed + 13 skipped + 67 xfailed
EVIDENCE:
  bar         official smooth50 mean spellF1 on the 12 synthetic scenarios 0.714; fails all 6
              visible-ball stoppage / long-occlusion scenarios (src.synthetic.BAR)
  spike       hybrid_v1 MVP 1.000 on all 12, on 6 seeds; the full contract passed 69/69,
              including the baseline config reproducing baseline_smooth50.csv on 199,319 real frames
  design      two flaws found and fixed with discriminating tests (D-019): flight gate, bounded
              alive bonus
  contracts   69 acceptance tests: T1 10, T2 9, T3 6, T4 8, T5 14, T6 22; each module's tests
              build their own inputs, so T1-T5 are independent
  rules       public-GT protocol pre-registered (D-020); no frame constants (D-015); OMB = other
              matches only (D-017); assistant sees aggregates only (D-002)
BLOCKERS: none.
  Still open from Run 1: whether the signed terms allow aggregates to reach a hosted LLM (D-002).
NEXT COMMAND:
  EXECUTE PHASE 3: implement tickets T1-T6 (docs/tickets.md) until the acceptance suite passes
  with every xfail marker removed, then run configs/hybrid_v1.toml on the match (aggregates only),
  then build the other-match benchmark (T7, T8). Keep D-002, D-015, D-016, D-020.
```

### Gate 2 checklist

| Gate 2 requirement | Evidence | Status |
|---|---|---|
| Every P0 module has a documented interface | stubs T1–T6 with contracts in their docstrings; `docs/architecture.md` §3 data contracts | PASS |
| Synthetic acceptance cases for every P0 module | 69 tests in six files, built from hand-made inputs or `src/synthetic.py` | PASS |
| A second agent could implement each module independently | the spike did, from the contracts alone, and passed 69/69 | PASS |
| The MVP beats naive framewise switching in synthetic tests | spike 1.000 vs the official baseline 0.714 (`docs/experiments.md` §7); `tests/test_predict.py` makes it a requirement | PASS |
| No proposed validation violates the rules | `docs/experiments.md` §1–§3; D-015, D-017, D-020 | PASS |
| Deliverables | architecture.md, feature_spec.md, experiments.md, tickets.md, config schema and two configs | PASS |
