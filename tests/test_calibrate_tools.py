"""scripts/omb_calibrate.py writes configurations by code: they must load back to the same thing."""

import os
import sys

from src.config import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import omb_calibrate  # noqa: E402


def test_written_configs_load_back_identical(tmp_path):
    base = load(os.path.join(ROOT, "configs", "hybrid_v1.toml"))
    changed = omb_calibrate.with_value(base, "temporal", "turnover_cost", 9.0)
    path = tmp_path / "written.toml"
    path.write_text(omb_calibrate.to_toml(changed, "header\nline two"), encoding="ascii")
    again = load(str(path))
    assert again.digest() == changed.digest() and again.temporal.turnover_cost == 9.0


def test_the_fast_spell_f1_is_the_official_one():
    import possession_metrics as pm
    from src.predict import predict
    from src.preprocess import frame_context
    from src.synthetic import SCENARIOS
    base = load(os.path.join(ROOT, "configs", "hybrid_v1.toml"))
    for name in ("turnover", "foul_free_kick", "second_half"):
        match = SCENARIOS[name]()
        context = frame_context(match.tracking.n_frames, match.blocks, match.tracking)
        features = omb_calibrate.build_features(match.tracking, context, base)
        fast = omb_calibrate.spell_f1((name, match.labels, context, features), base)
        official = pm.evaluate(match.labels, pm.label_codes(predict(match.tracking, context, base)))
        assert fast == official["spell_f1_mean"]


def test_every_grid_contains_its_hybrid_v1_default():
    # The descent starts from hybrid_v1; a grid without its default could not keep it.
    base = load(os.path.join(ROOT, "configs", "hybrid_v1.toml"))
    for section, key, values in omb_calibrate.GRID:
        assert getattr(getattr(base, section), key) in values, (section, key)
