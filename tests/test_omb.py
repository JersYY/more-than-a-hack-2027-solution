"""The other-match benchmark: the DFL parser (src/omb.py) and the degradation (src/degrade.py).

Tiny hand-written files in the IDSSE/DFL format, so these run without the 3 GB download: the label
recipe, the coordinate conversion, the speed-unit detection and the cache are what a wrong
benchmark score would hide.
"""

import numpy as np
import pytest

import possession_metrics as pm
from src.data_loader import AWAY, HOME
from src.degrade import degrade_ball, renewal_mask, dead_weighted_mask
from src.omb import HALFTIME_FRAMES, load_dfl

META = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><MatchInformation><Teams>
<Team TeamId="CLU-H" TeamName="Home FC" Role="home"/>
<Team TeamId="CLU-G" TeamName="Guest FC" Role="guest"/>
</Teams></MatchInformation></PutDataRequest>"""


def frames(first, xs, ys, speeds, ball=None):
    out = []
    for i, (x, y, s) in enumerate(zip(xs, ys, speeds)):
        extra = "" if ball is None else ' Z="0.00" BallPossession="{}" BallStatus="{}"'.format(*ball[i])
        out.append('<Frame N="{}" X="{:.2f}" Y="{:.2f}" D="0" S="{:.2f}" A="0" M="1"{}/>'.format(
            first + i, x, y, s, extra))
    return "\n".join(out)


def tracking_xml(n=10):
    # Each player runs at 3 m/s (0.12 m a frame) with S in km/h (10.8), as DFL reports it.
    xs = [0.12 * i for i in range(n)]
    sets = []
    for section, first in (("firstHalf", 10000), ("secondHalf", 100000)):
        for team, person, y in (("CLU-H", "P1", 10.0), ("CLU-H", "P2", -10.0),
                                ("CLU-G", "P3", 20.0), ("CLU-G", "P4", -20.0),
                                ("referee", "R1", 0.0)):
            sets.append('<FrameSet GameSection="{}" TeamId="{}" PersonId="{}">\n{}\n</FrameSet>'.format(
                section, team, person, frames(first, xs, [y] * n, [10.8] * n)))
        # status/possession: 3 frames home alive, 2 dead, 3 away alive, 2 home alive
        ball = [(1, 1)] * 3 + [(1, 0)] * 2 + [(2, 1)] * 3 + [(1, 1)] * 2
        sets.append('<FrameSet GameSection="{}" TeamId="BALL" PersonId="B">\n{}\n</FrameSet>'.format(
            section, frames(first, [-50.0] + [0.0] * (n - 1), [30.0] * n, [0.0] * n, ball)))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<PutDataRequest><Positions>\n'
            '<MetaData Type="pitch-size"><PitchSize X="100.00" Y="60.00"/></MetaData>\n'
            + "\n".join(sets) + "\n</Positions></PutDataRequest>")


@pytest.fixture()
def dfl_folder(tmp_path):
    (tmp_path / "meta.xml").write_text(META, encoding="utf-8")
    (tmp_path / "tracking.xml").write_text(tracking_xml(), encoding="utf-8")
    return str(tmp_path)


def test_labels_follow_the_ground_truth_recipe(dfl_folder):
    match = load_dfl(dfl_folder)
    half = [pm.H] * 3 + [pm.D] * 2 + [pm.A] * 3 + [pm.H] * 2
    expected = half + [pm.X] * HALFTIME_FRAMES + half
    assert match.labels.tolist() == expected
    assert match.blocks == [(1, 0, 9), (2, 10 + HALFTIME_FRAMES, 19 + HALFTIME_FRAMES)]


def test_coordinates_become_ours(dfl_folder):
    match = load_dfl(dfl_folder)
    assert match.pitch == (100.0, 60.0)
    ball = match.tracking.ball
    assert ball[0, :2].tolist() == [0.0, 0.0]          # (-50, +30): our top-left corner
    assert ball[1, :2].tolist() == [0.5, 0.0]          # the halfway line, at y = +30
    assert sorted(match.tracking.layout.team_of_slot.tolist()) == [HOME, HOME, AWAY, AWAY]


def test_speed_in_kmh_is_converted_to_mps(dfl_folder):
    match = load_dfl(dfl_folder)
    assert match.speed_unit_ratio == pytest.approx(3.6, abs=0.01)
    speeds = match.tracking.players[:10, :, 2]
    assert np.allclose(speeds[np.isfinite(speeds)], 3.0)


def test_the_cache_gives_the_same_match(dfl_folder):
    first, second = load_dfl(dfl_folder), load_dfl(dfl_folder)      # second load: parsed.npz
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.tracking.players, second.tracking.players, equal_nan=True)


PROFILE = {"visible_share": 0.6, "edges_frames": [1, 10, 100, 1000],
           "seen_runs": [0, 5, 5], "missing_runs": [0, 10, 0]}


def test_renewal_mask_matches_the_profile_and_is_seeded():
    a = renewal_mask(200000, PROFILE, np.random.RandomState(0))
    b = renewal_mask(200000, PROFILE, np.random.RandomState(0))
    assert np.array_equal(a, b)
    # seen runs average ~(55 + 550) / 2 frames, missing ~55: missing share ~ 55 / (302 + 55)
    assert 0.10 < a.mean() < 0.22


def test_dead_weighted_mask_puts_its_share_in_stoppages():
    # 50% dead; 40% of frames to hide, 80% of those (32% of all frames) inside stoppages.
    labels = np.array(([pm.H] * 500 + [pm.D] * 500) * 50, dtype=np.int8)
    mask = dead_weighted_mask(labels, (0, len(labels) - 1), PROFILE, 0.8, np.random.RandomState(0))
    assert abs(mask.mean() - 0.4) < 0.03                      # 1 - visible_share
    in_dead = mask[labels == pm.D].sum() / float(mask.sum())
    assert abs(in_dead - 0.8) < 0.05


def test_degrading_touches_only_the_ball(dfl_folder):
    match = load_dfl(dfl_folder)
    noisy = degrade_ball(match, PROFILE, "renewal", seed=3)
    assert np.array_equal(noisy.labels, match.labels)
    assert np.array_equal(noisy.tracking.players, match.tracking.players, equal_nan=True)
    assert degrade_ball(match, PROFILE, "none") is match
