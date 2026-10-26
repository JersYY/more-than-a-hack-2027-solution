# Feature specification

This is the contract for `src/features.py:build_features` (ticket T2). It is checked by
`tests/test_features_spec.py`. When this file and the code disagree, fix the code, or change this
file and its test together in the same commit.

## Conventions

- **One value per video frame.** Every feature is an `(n,)` array; row `i` is video frame `i`,
  which is submission row `i`.
- **Units.** Metres on the `games.py` pitch (105.3 × 68.0). x runs along the length, y across.
  Speeds are in m/s, durations in seconds, FPS = 25.
- **Missing is NaN, never 0**, unless a row below says otherwise. A missing ball is not a dead
  ball, and a missing player is not a distant one.
- **Determinism (D-016).** Only `+ − × ÷ sqrt`, comparisons, `min`/`max`, and sums in a fixed
  order (sequential `cumsum`, slot-by-slot accumulation).
- **Provenance.** Everything comes from the tactical ATD (`src/data_loader.py`). No feature reads
  the video, the broadcast recording, the homography or any ground truth. Smart Tagging enters only
  through the P1 rows, behind `smart_tagging.enabled`.

## P0 features

| # | name | dtype | unit | definition | missing / edge rule | used by |
|---|---|---|---|---|---|---|
| 1 | `ball_seen` | bool | — | ball x **and** y are finite | — | evidence, dead ball |
| 2 | `ball_x_m` | float64 | m | ball x × 105.3 | NaN where not seen | diagnostics |
| 3 | `ball_y_m` | float64 | m | ball y × 68.0 | NaN where not seen | diagnostics |
| 4 | `ball_speed_mps` | float64 | m/s | central difference: distance between the ball at t−2 and t+2, × 25 / 4 | NaN unless the ball is seen at t−2, t **and** t+2; NaN at the first and last 2 frames; NaN above 40 m/s (a tracker jump, not a kick) | flight gate, dead ball |
| 5 | `ball_out_m` | float64 | m | distance beyond the pitch rectangle: `max(0, −x, x−L, −y, y−W)` in metres | NaN where not seen | dead ball |
| 6 | `ball_still_s` | float64 | s | for frames with `ball_speed_mps < still_mps` (config), the length of the maximal run of such frames containing t | **0** where not still, including NaN speed | dead ball |
| 7 | `ball_missing_s` | float64 | s | for frames without a ball, the length of the maximal run of such frames containing t | **0** where the ball is seen | diagnostics, P1 gap rules |
| 8 | `d_home_m` | float64 | m | distance from the ball to the nearest tracked home slot | NaN without a ball or a tracked home slot | team evidence |
| 9 | `d_away_m` | float64 | m | the same for away | as above | team evidence |
| 10 | `n_home` | int16 | count | tracked home slots (x and y finite) | — | diagnostics |
| 11 | `n_away` | int16 | count | tracked away slots | — | diagnostics |
| 12 | `team_speed_mps` | float64 | m/s | mean of the speed channel `s` over tracked slots with a finite `s`, both teams, summed slot by slot | NaN when no slot qualifies | `activity_mps` |
| 13 | `activity_mps` | float64 | m/s | centred mean of `team_speed_mps` over ±`round(activity_window_s × 25)` frames, clipped to the recording, averaging finite values only (cumulative sums) | NaN when the window holds no finite value | dead ball |

Notes:

- Rows 6, 7 and 13 look at both past and future frames. That is fine because inference is
  offline (`docs/architecture.md` §4.3). Rows 6 and 7 deliberately give the whole run its full
  length: a ball that will stay still for 3 s is "placed" from its first still frame.
- Row 12 uses the tracker's speed channel rather than differences of positions. The audit
  (`docs/data_audit.md`, D-008) checked that it is m/s, consistent with displacement (median
  ratio 0.989). It is already smoothed by the tracker, and track-identity switches do not show up
  in it as 100 m/s jumps.
- Rows 4 and 5 keep off-pitch values. A ball 2 m beyond the touchline is information, not an error.

## Candidate P1 features (added with their experiment, not before)

| name | definition | experiment |
|---|---|---|
| `rel_speed_home_mps`, `rel_speed_away_mps` | `|v_ball − v_nearest player|` for each team: control means moving with the ball | E1b |
| `ball_turn` | change of the ball's direction between consecutive central-difference velocities: touches | E1b |
| `home_centroid_x_m`, `away_centroid_x_m` | team centroids, in the attack direction of `FrameContext.attack_sign` | E5 / E6b |
| `tag_team`, `tag_dead` | Smart Tagging priors (`src/smart_tagging.py`) | E5 |
| `kickoff_shape` | both teams in their own halves with the ball near the centre spot | E4 |

None of these is computed until its experiment starts. Each one adds a row to the tables above
and a test to `tests/test_features_spec.py`.
