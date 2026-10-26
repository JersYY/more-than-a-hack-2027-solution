"""Shared test setup: import paths, and a skip for tests that need the gated match data.

Tests marked `data` run only when games.require() finds the bundle (./data or MTAH_DATA_DIR).
Everywhere else -- a fresh clone, public CI -- they are skipped and the rest still runs on
synthetic fixtures.

Data tests assert on scalars computed beforehand, never on arrays, so that a failure message
cannot print match data (docs/decisions.md D-002). Keep it that way.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import games  # noqa: E402


def data_available():
    try:
        games.require(["atd_meta", "atd_raw", "smart_tagging", "homography"])
        return True
    except (IOError, OSError):
        return False


def pytest_collection_modifyitems(config, items):
    if data_available():
        return
    skip = pytest.mark.skip(reason="gated match data not found: set MTAH_DATA_DIR or unpack into ./data")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def tactical():
    """The tactical ATD, parsed once per test session (~3 s)."""
    from src.data_loader import load_tracking
    return load_tracking("tactical")
