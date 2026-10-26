"""The synthetic scenario generator (src/synthetic.py): labels mean what CHALLENGE.md says.

These pin the answer key the acceptance tests rely on. If a scenario's labels were wrong, every
test built on it would be testing the wrong behaviour, so the key itself is checked here.
"""

import numpy as np
import pytest

import baseline  # official starter/baseline.py, import only
import possession_metrics as pm
from src.data_loader import AWAY
from src.features import nearest_team
from src.synthetic import BAR, SCENARIOS, TEAM_OF_SLOT, Script


def nearest_slot(match, row):
    ball = match.tracking.ball[row, :2] * np.array([105.3, 68.0])
    players = match.tracking.players[row, :, :2] * np.array([105.3, 68.0])
    return int(np.argmin(np.hypot(*(players - ball).T)))


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_is_well_formed(name):
    match = SCENARIOS[name]()
    n = match.tracking.n_frames
    assert match.labels.shape == (n,) and match.tracking.players.shape == (n, 22, 3)
    assert np.array_equal(match.tracking.frame_id, np.arange(1, n + 1))
    graded = np.zeros(n, dtype=bool)
    for _, first, last in match.blocks:
        graded[first:last + 1] = True
    assert ((match.labels == pm.X) == ~graded).all()                 # X exactly outside blocks
    assert pm.valid_runs(match.labels) == [(f, l + 1) for _, f, l in match.blocks]
    assert np.isfinite(match.tracking.players).all()                 # every slot tracked
    assert (match.tracking.players[:, :, 2] <= 9.0).all()


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_scenario_is_deterministic(name):
    first, second = SCENARIOS[name](), SCENARIOS[name]()
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.tracking.players, second.tracking.players)
    assert np.array_equal(first.tracking.ball, second.tracking.ball, equal_nan=True)


def test_the_seed_moves_players_not_labels():
    a, b = SCENARIOS["turnover"](seed=0), SCENARIOS["turnover"](seed=1)
    assert np.array_equal(a.labels, b.labels)
    assert not np.allclose(a.tracking.players, b.tracking.players)


def test_a_brush_makes_the_opponent_nearest_without_changing_possession():
    match = SCENARIOS["deflection"]()
    rows = match.frames_of("brush")
    assert len(rows) == 4 + 3
    assert all(TEAM_OF_SLOT[nearest_slot(match, row)] == AWAY for row in rows)
    assert (match.labels[rows] == pm.H).all()


def test_the_carrier_is_next_to_the_ball():
    match = SCENARIOS["stable_home"]()
    rows = match.frames_of("carry")
    distance = [np.hypot(*((match.tracking.ball[r, :2] - match.tracking.players[r, nearest_slot(match, r), :2])
                           * np.array([105.3, 68.0]))) for r in rows]
    assert max(distance) <= 0.75


def test_possession_changes_on_the_reception_frame():
    match = SCENARIOS["turnover"]()
    changes = np.flatnonzero(np.diff(match.labels.astype(int)) != 0) + 1
    graded_changes = [c for c in changes if match.labels[c - 1] != pm.X and match.labels[c] != pm.X]
    passes = [i for i, step in enumerate(match.steps) if step.startswith("pass_to")]
    # every change is the first frame after the last frame of a pass to an opponent
    pass_ends = {int(np.flatnonzero(match.step_of_frame == i)[-1]) + 1 for i in passes}
    assert set(graded_changes) <= pass_ends and len(graded_changes) == 2


def test_out_is_dead_from_the_first_frame_beyond_the_line():
    match = SCENARIOS["home_dead_home"]()
    rows = match.frames_of("out")
    ball_y = match.tracking.ball[rows, 1]
    outside = ~((ball_y >= 0) & (ball_y <= 1))
    first_dead = rows[np.argmax(match.labels[rows] == pm.D)]
    assert first_dead == rows[np.argmax(outside)]
    assert (match.labels[rows][outside] == pm.D).all() and (match.labels[rows][~outside] == pm.H).all()


def test_hidden_steps_have_no_ball():
    match = SCENARIOS["occlusion_long"]()
    rows = match.frames_of("carry(6.0")
    assert len(rows) == 150 and np.isnan(match.tracking.ball[rows, :2]).all()
    assert (match.labels[rows] == pm.H).all()


def test_players_slow_down_at_a_stoppage():
    match = SCENARIOS["away_dead_away"]()
    speed = match.tracking.players[:, :, 2].mean(axis=1)
    dead = np.flatnonzero(match.labels == pm.D)[40:]          # after the 1.5 s slow-down
    live = np.flatnonzero((match.labels == pm.H) | (match.labels == pm.A))
    assert speed[dead].mean() < 0.6 * speed[live].mean()


def test_second_half_is_graded_from_before_its_kickoff():
    match = SCENARIOS["second_half"]()
    (_, _, _), (half, first, _) = match.blocks
    assert half == 2
    assert (match.labels[first:first + 100] == pm.D).all() and match.labels[first + 100] == pm.A


def test_scripts_refuse_impossible_football():
    with pytest.raises(ValueError):
        Script().carry(1.0)                         # nothing is live before a kickoff
    with pytest.raises(ValueError):
        Script().kickoff("H", 15)                   # slot 15 is an away player
    with pytest.raises(ValueError):
        Script().kickoff("H", 9).brush(5, 3)        # a team-mate cannot brush


# The bar (src.synthetic.BAR): what the official baseline fails here is what the MVP is built to
# add (docs/experiments.md, E3/E4).
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_official_baseline_bar(name):
    match = SCENARIOS[name]()
    codes = pm.label_codes(baseline.smooth(nearest_team(match.tracking), 50))
    spell = pm.evaluate(match.labels, codes)["spell_f1_mean"]
    if name in BAR:
        assert spell < 1.0
    else:
        assert spell == 1.0
