"""The freeze (D-027): configs/final.toml is release candidate rc1, value for value."""

import os

from src.config import load

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_final_is_rc1_in_every_model_value():
    final = load(os.path.join(ROOT, "configs", "final.toml")).to_dict()
    rc1 = load(os.path.join(ROOT, "configs", "rc1.toml")).to_dict()
    assert final.pop("run")["name"] == "final"
    rc1.pop("run")
    assert final == rc1
