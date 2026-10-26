"""The configuration schema (src/config.py) and the two committed configurations."""

import os

import pytest

from src import config as cfg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(ROOT, "configs")


def write(tmp_path, text, name="c.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="ascii")
    return str(path)


@pytest.mark.parametrize("name", ["baseline.toml", "hybrid_v1.toml"])
def test_committed_configs_load(name):
    config = cfg.load(os.path.join(CONFIGS, name))
    assert config.run.name and config.run.description
    assert len(config.digest()) == 64


def test_baseline_config_is_the_official_baseline():
    config = cfg.load(os.path.join(CONFIGS, "baseline.toml"))
    assert (config.evidence.kind, config.dead_ball.kind) == ("nearest_hard", "ball_missing")
    assert (config.temporal.decoder, config.temporal.window) == ("majority", 50)
    assert config.output.ungraded == "keep"


def test_empty_file_gives_the_defaults(tmp_path):
    assert cfg.load(write(tmp_path, "")) == cfg.Config()


def test_digest_ignores_formatting_but_not_values(tmp_path):
    a = cfg.load(write(tmp_path, "[temporal]\nturnover_cost = 5.7\nstop_cost = 6.9\n", "a.toml"))
    b = cfg.load(write(tmp_path, "# comment\n[temporal]\nstop_cost = 6.9   # x\nturnover_cost = 5.7\n",
                       "b.toml"))
    c = cfg.load(write(tmp_path, "[temporal]\nturnover_cost = 5.8\n", "c.toml"))
    assert a.digest() == b.digest() == cfg.Config().digest()
    assert c.digest() != a.digest()


def test_integers_are_accepted_for_floats(tmp_path):
    config = cfg.load(write(tmp_path, "[evidence]\nmax_score = 3\n"))
    assert config.evidence.max_score == 3.0 and isinstance(config.evidence.max_score, float)


@pytest.mark.parametrize("text, message", [
    ("[evidnce]\nkind = 'soft_control'\n", "unknown section"),
    ("[evidence]\ncontrol_meters = 1.0\n", "unknown key"),
    ("[evidence]\nmax_score = 'high'\n", "expected a number"),
    ("[temporal]\nfixed_point = 1.5\n", "expected an integer"),
    ("[smart_tagging]\nenabled = 1\n", "expected true/false"),
    ("[temporal]\ndecoder = 'hmm'\n", "temporal.decoder must be one of"),
    ("[evidence]\ncontrol_m = 0\n", "evidence.control_m must be > 0"),
    ("[temporal]\nturnover_cost = -1\n", "temporal.turnover_cost must be >= 0"),
    ("[temporal]\ndecoder = 'majority'\n", "needs evidence.kind = 'nearest_hard'"),
])
def test_bad_files_are_rejected_with_a_reason(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        cfg.load(write(tmp_path, text))
