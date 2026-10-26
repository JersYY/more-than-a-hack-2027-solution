"""Ticket T6 -- src/predict.py, end to end. Acceptance tests, written before the code.

This is where "the MVP beats naive framewise switching" (playbook, Phase 2) becomes checkable:
every synthetic scenario must be solved, including the ones the official baseline fails
(src.synthetic.BAR). Integration ticket: it needs T1-T5. xfail(strict) until T6 lands.
"""

import json
import os

import numpy as np
import pytest

import baseline  # official starter/baseline.py, import only
import possession_metrics as pm
from src import submission
from src.config import load
from src.features import nearest_team
from src.predict import predict, run
from src.preprocess import frame_context
from src.synthetic import BAR, SCENARIOS

pytestmark = pytest.mark.xfail(raises=NotImplementedError, strict=True,
                               reason="ticket T6 not implemented yet (Phase 3)")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HYBRID = load(os.path.join(ROOT, "configs", "hybrid_v1.toml"))
BASELINE = load(os.path.join(ROOT, "configs", "baseline.toml"))


def predicted(match, config):
    context = frame_context(match.tracking.n_frames, match.blocks, match.tracking)
    return predict(match.tracking, context, config)


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_mvp_solves_every_synthetic_scenario(name):
    match = SCENARIOS[name]()
    result = pm.evaluate(match.labels, pm.label_codes(predicted(match, HYBRID)))
    assert result["spell_f1_mean"] == 1.0
    assert result["accuracy"] >= 0.95


@pytest.mark.parametrize("name", BAR)
def test_the_mvp_beats_the_official_baseline_where_it_fails(name):
    match = SCENARIOS[name]()
    naive = pm.label_codes(baseline.smooth(nearest_team(match.tracking), 50))
    ours = pm.label_codes(predicted(match, HYBRID))
    assert (pm.evaluate(match.labels, ours)["spell_f1_mean"]
            > pm.evaluate(match.labels, naive)["spell_f1_mean"])


def test_ungraded_frames_follow_the_output_policy():
    match = SCENARIOS["second_half"]()
    graded = match.labels != pm.X
    hybrid = predicted(match, HYBRID)
    assert (hybrid[~graded] == "X").all() and (hybrid[graded] != "X").all()
    assert (predicted(match, BASELINE) != "X").all()        # "keep": the baseline never writes X


def test_prediction_is_deterministic():
    match = SCENARIOS["turnover"]()
    assert np.array_equal(predicted(match, HYBRID), predicted(match, HYBRID))


@pytest.mark.data
def test_the_baseline_config_reproduces_the_official_file(tactical):
    labels = predict(tactical, frame_context(tactical.n_frames, tracking=tactical), BASELINE)
    committed = submission.read_labels(os.path.join(ROOT, "evaluation", "results",
                                                    "baseline_smooth50.csv"))
    mismatching_frames = int(np.sum(labels != committed))
    assert mismatching_frames == 0


@pytest.mark.data
def test_the_cli_writes_a_valid_file_and_its_manifest(tmp_path):
    path = run(os.path.join(ROOT, "configs", "baseline.toml"), str(tmp_path / "baseline.csv"))
    assert submission.check(path) is None and submission.check_strict(path) is None
    with open(path + ".manifest.json") as handle:
        manifest = json.load(handle)
    assert manifest["config_digest"] == BASELINE.digest()
    assert manifest["output_sha256"] == submission.sha256(path)
