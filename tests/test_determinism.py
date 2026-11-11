"""Ticket T10 -- the same inputs give the same file, on this machine and, by construction, on others.

RULES.md: re-running the repository must reproduce the submission, "not approximately -- the same
file". Two checks here. The run-twice test proves it on this machine. The static test guards the
reason it should hold on another machine too: nothing on the decision path uses a function whose
last bit varies between CPUs or libm builds (D-016).
"""

import os
import re

import numpy as np
import pytest

from src import submission
from src.predict import run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Every module a prediction runs through. src/synthetic.py is test scaffolding and exempt.
DECISION_PATH = ("data_loader.py", "preprocess.py", "features.py", "possession_evidence.py",
                 "dead_ball.py", "temporal.py", "predict.py", "submission.py", "config.py")
# Transcendental and power functions: correctly rounded in no standard, vectorised differently on
# different CPUs. Basic arithmetic and sqrt are IEEE-754 exact and stay allowed.
FORBIDDEN = re.compile(r"\b(?:np|numpy|math)\.(?:exp|exp2|expm1|log|log2|log10|log1p|power|"
                       r"float_power|hypot|sin|cos|tan|arctan2|arctan|tanh|sinh|cosh)\b"
                       r"|[\w)\]]\s*\*\*\s*[\w(.]")    # binary power; f(**kwargs) is not one


def code_lines(path):
    """Source lines with comments and docstrings removed -- the arithmetic that actually runs."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    return [line.split("#", 1)[0] for line in text.splitlines()]


@pytest.mark.parametrize("module", DECISION_PATH)
def test_no_cpu_dependent_functions_on_the_decision_path(module):
    path = os.path.join(ROOT, "src", module)
    offending = [(number, line.strip()) for number, line in enumerate(code_lines(path), start=1)
                 if FORBIDDEN.search(line)]
    assert offending == []


@pytest.mark.data
@pytest.mark.parametrize("config", ["baseline.toml", "hybrid_v1.toml", "final.toml", "timing_v1.toml", "final_v2.toml", "final_external.toml"])
def test_running_twice_writes_the_same_bytes(tmp_path, config):
    first = run(os.path.join(ROOT, "configs", config), str(tmp_path / "first.csv"))
    second = run(os.path.join(ROOT, "configs", config), str(tmp_path / "second.csv"))
    assert submission.sha256(first) == submission.sha256(second)
