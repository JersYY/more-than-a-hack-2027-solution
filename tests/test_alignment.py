"""The loader's parsing and frame alignment: synthetic files first, the real bundle second.

The synthetic half runs anywhere and pins the rules: layout from the DataFormatSpecification,
NaN kept, malformed lines refused, frame ids proven 1..n, nearest-team semantics identical to the
official parser. The `data` half runs only where the gated bundle is present and holds the loader
to the official baseline files and to kloppy, asserting on counts only.
"""

import os

import numpy as np
import pytest

import baseline  # official starter/baseline.py, import only
import games
from src import submission
from src.data_loader import AWAY, HOME, Tracking, check_frame_ids, read_layout, read_raw
from src.features import NOBODY, nearest_team

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PITCH = games.GAMES[games.GAME]["pitch"]


# -- synthetic fixtures ------------------------------------------------------------------------

def synthetic_meta(slot_order, team_of_player, team_order=("T1", "T2"), local="T1", n_frames=4,
                   extra_spec=""):
    """A minimal Metrica FIFA/EPTS metadata XML. Players are P1..Pk, channels x, y, s."""
    players = "".join('<Player id="P{0}" teamId="{1}"><Name>Track {0}</Name>'
                      '<ShirtNumber>{0}</ShirtNumber></Player>'.format(p, t)
                      for p, t in sorted(team_of_player.items()))
    channels = "".join('<PlayerChannel id="P{0}_{1}" playerId="P{0}" channelId="{1}"/>'.format(p, c)
                       for p in sorted(team_of_player) for c in "xys")
    groups = "".join('<SplitRegister separator=",">' + "".join(
        '<PlayerChannelRef playerChannelId="P{}_{}"/>'.format(p, c) for c in "xys")
        + "</SplitRegister>" for p in slot_order)
    spec = ('<DataFormatSpecification startFrame="1" endFrame="{}" separator=":">'
            '<StringRegister name="frameCount"/><SplitRegister separator=";">{}</SplitRegister>'
            '<SplitRegister separator=","><BallChannelRef channelId="x"/>'
            '<BallChannelRef channelId="y"/><BallChannelRef channelId="s"/></SplitRegister>'
            '</DataFormatSpecification>').format(n_frames, groups)
    visiting = [t for t in team_order if t != local][0]
    return (
        '<?xml version="1.0" encoding="utf-8"?><main><Metadata>'
        '<GlobalConfig><FrameRate>25</FrameRate><ProviderName>Metrica Sports</ProviderName>'
        '</GlobalConfig><Sessions><Session id="S1"><FieldSize><Width>105</Width>'
        '<Height>68</Height></FieldSize><MatchParameters><Score idLocalTeam="{}" '
        'idVisitingTeam="{}"><LocalTeamScore>0</LocalTeamScore><VisitingTeamScore>0'
        '</VisitingTeamScore></Score></MatchParameters></Session></Sessions>'
        '<Teams>{}</Teams><Players>{}</Players><Devices><Device id="D1"><Sensors>'
        '<Sensor id="position"><Name>position</Name><Channels>'
        '<Channel id="x"><Name>x</Name><Unit>normalized</Unit></Channel>'
        '<Channel id="y"><Name>y</Name><Unit>normalized</Unit></Channel></Channels></Sensor>'
        '<Sensor id="speed"><Name>speed</Name><Channels><Channel id="s"><Name>s</Name>'
        '<Unit>m/s</Unit></Channel></Channels></Sensor></Sensors></Device></Devices>'
        '<PlayerChannels>{}</PlayerChannels></Metadata>'
        '<DataFormatSpecifications>{}{}</DataFormatSpecifications></main>').format(
            local, visiting, "".join('<Team id="{0}"><Name>Team {0}</Name></Team>'.format(t)
                                     for t in team_order),
            players, channels, spec, extra_spec)


def raw_line(frame_id, slots, ball):
    """One raw ATD line. A slot or the ball given as None is written NaN,NaN,NaN."""
    def group(values):
        return "NaN,NaN,NaN" if values is None else ",".join(repr(float(v)) for v in values)
    return "{}:{}:{}".format(frame_id, ";".join(group(slot) for slot in slots), group(ball))


def write(path, text):
    with open(str(path), "w", newline="\n") as handle:
        handle.write(text)
    return str(path)


def synthetic_tracking(tmp_path, team_of_player, frames):
    """(meta path, raw path, Tracking) from frames = [(slots, ball), ...] in conventional order."""
    order = sorted(team_of_player)
    meta = write(tmp_path / "meta.xml", synthetic_meta(order, team_of_player, n_frames=len(frames)))
    raw = write(tmp_path / "raw.txt", "".join(raw_line(i + 1, slots, ball) + "\n"
                                              for i, (slots, ball) in enumerate(frames)))
    layout = read_layout(meta)
    frame_ids, players, ball = read_raw(raw, layout)
    return meta, raw, Tracking("synthetic", layout, frame_ids, players, ball)


# -- layout ------------------------------------------------------------------------------------

def test_layout_follows_the_specification_not_the_player_elements(tmp_path):
    # Raw order P3, P1, P2 and a Score naming T2 as local: both baseline conventions broken.
    meta = write(tmp_path / "m.xml", synthetic_meta((3, 1, 2), {1: "T1", 2: "T2", 3: "T2"},
                                                    local="T2"))
    layout = read_layout(meta)
    assert layout.slot_player_ids == ("P3", "P1", "P2")
    assert layout.home_team_id == "T2"
    assert list(layout.team_of_slot) == [HOME, AWAY, HOME]
    assert not layout.baseline_slot_order and not layout.baseline_home
    assert layout.n_fields == 1 + 3 * 3 + 3
    assert layout.frame_rate == 25.0 and layout.pitch_size == (105.0, 68.0)


def test_layout_flags_conventional_files(tmp_path):
    meta = write(tmp_path / "m.xml", synthetic_meta((1, 2, 3), {1: "T1", 2: "T2", 3: "T2"}))
    layout = read_layout(meta)
    assert layout.baseline_slot_order and layout.baseline_home
    assert list(layout.team_of_slot) == [HOME, AWAY, AWAY]


def test_layout_refuses_specifications_that_disagree(tmp_path):
    other = ('<DataFormatSpecification startFrame="5" endFrame="9" separator=":">'
             '<StringRegister name="frameCount"/></DataFormatSpecification>')
    meta = write(tmp_path / "m.xml", synthetic_meta((1, 2), {1: "T1", 2: "T2"}, extra_spec=other))
    with pytest.raises(ValueError):
        read_layout(meta)


# -- raw parsing -------------------------------------------------------------------------------

def test_raw_values_and_missing_observations(tmp_path):
    frames = [([(0.1, 0.2, 1.5), None, (0.9, 0.8, 7.25)], (0.11, 0.21, 9.0)),
              ([None, None, None], None),
              ([(0.0, 1.0, 0.0), (0.5, 0.5, 2.0), None], (1.02, -0.01, 0.5))]
    _, _, tracking = synthetic_tracking(tmp_path, {1: "T1", 2: "T2", 3: "T2"}, frames)
    assert list(tracking.frame_id) == [1, 2, 3]
    assert tracking.players.shape == (3, 3, 3) and tracking.ball.shape == (3, 3)
    assert tracking.players[0, 2].tolist() == [0.9, 0.8, 7.25]
    assert np.isnan(tracking.players[0, 1]).all() and np.isnan(tracking.ball[1]).all()
    assert tracking.ball[2].tolist() == [1.02, -0.01, 0.5]      # off-pitch values kept, not clipped
    assert tracking.ball_seen().tolist() == [True, False, True]
    assert tracking.tracked().sum(axis=1).tolist() == [2, 0, 2]


def test_raw_refuses_a_line_with_a_missing_field(tmp_path):
    meta = write(tmp_path / "m.xml", synthetic_meta((1, 2), {1: "T1", 2: "T2"}))
    good = raw_line(1, [(0.1, 0.1, 1.0), (0.2, 0.2, 1.0)], (0.3, 0.3, 1.0))
    raw = write(tmp_path / "r.txt", good + "\n" + good.rsplit(",", 1)[0] + "\n")
    with pytest.raises(ValueError, match="separators"):
        read_raw(raw, read_layout(meta))


def test_raw_refuses_a_blank_line(tmp_path):
    meta = write(tmp_path / "m.xml", synthetic_meta((1, 2), {1: "T1", 2: "T2"}))
    good = raw_line(1, [(0.1, 0.1, 1.0), (0.2, 0.2, 1.0)], (0.3, 0.3, 1.0))
    raw = write(tmp_path / "r.txt", good + "\n\n" + good.replace("1:", "2:", 1) + "\n")
    with pytest.raises(ValueError, match="blank"):
        read_raw(raw, read_layout(meta))


@pytest.mark.parametrize("ids, n, ok", [
    ([1, 2, 3, 4], 4, True),       # the contract: 1..n
    ([0, 1, 2, 3], 4, False),      # zero-based would shift every row by one
    ([1, 2, 4, 5], 4, False),      # a gap
    ([1, 2, 2, 3], 4, False),      # a duplicate
    ([1, 2, 3], 4, False),         # short
])
def test_frame_ids_must_be_exactly_one_to_n(ids, n, ok):
    assert (check_frame_ids(np.array(ids), n) is None) == ok


def test_smart_tagging_seconds_map_to_round_seconds_times_25(tmp_path):
    import load_smart_tagging  # official starter module
    folder = tmp_path / "tactical"
    folder.mkdir()
    write(folder / "BAR-ATH_tactical_smart_tagging.xml",
          "<file><ALL_INSTANCES><instance><ID>1</ID><start>447.4</start><end>452.44</end>"
          "<code>SET PIECES</code><label><group>Half</group><text>1st Half</text></label>"
          "</instance></ALL_INSTANCES></file>")
    event = load_smart_tagging.load(data_dir=str(tmp_path))[0]
    assert (event["start_frame"], event["end_frame"]) == (11185, 11311)


# -- nearest team -----------------------------------------------------------------------------

def test_nearest_team_rules(tmp_path):
    # A power-of-two pitch makes 0.25 and 0.75 of it exact, so frame 0 is a true tie; with the real
    # 105.3 m pitch the two distances differ in the last bit and there is no tie to test.
    frames = [([(0.25, 0.5, 0), (0.75, 0.5, 0), None], (0.5, 0.5, 0)),        # exact tie: first slot
              ([(0.25, 0.5, 0), (0.75, 0.5, 0), None], None),                 # no ball
              ([None, None, None], (0.5, 0.5, 0)),                            # nobody tracked
              ([None, (0.85, 0.12, 0), (0.1, 0.1, 0)], (0.9, 0.1, 0))]        # NaN slot skipped
    _, _, tracking = synthetic_tracking(tmp_path, {1: "T1", 2: "T2", 3: "T1"}, frames)
    assert nearest_team(tracking, (128.0, 64.0)).tolist() == [HOME, NOBODY, NOBODY, AWAY]


def test_nearest_team_matches_the_official_parser_bit_for_bit(tmp_path):
    rng = np.random.RandomState(7)
    frames = []
    for _ in range(400):
        slots = [None if rng.rand() < 0.3 else tuple(rng.rand(2)) + (rng.rand() * 8,)
                 for _ in range(8)]
        ball = None if rng.rand() < 0.4 else tuple(rng.rand(2)) + (rng.rand() * 20,)
        frames.append((slots, ball))
    team_of_player = {p: ("T1" if p % 2 else "T2") for p in range(1, 9)}
    meta, raw, tracking = synthetic_tracking(tmp_path, team_of_player, frames)

    official = baseline.nearest_player_labels(raw, baseline.team_of_each_track(meta), len(frames),
                                              PITCH)
    assert np.array_equal(nearest_team(tracking, PITCH), official)


# -- the real bundle (counts only) -------------------------------------------------------------

@pytest.mark.data
def test_tactical_atd_is_one_record_per_video_frame(tactical):
    n = tactical.n_frames
    ids_are_one_to_n = bool(np.array_equal(tactical.frame_id, np.arange(1, n + 1)))
    assert n == 199319
    assert ids_are_one_to_n
    assert tactical.layout.n_slots == 46
    assert tactical.layout.spec_frames == ((1, 199319),)


@pytest.mark.data
def test_official_baseline_assumptions_hold(tactical):
    home_is_barcelona = "barcelona" in tactical.layout.home_name.lower()
    assert tactical.layout.baseline_slot_order
    assert tactical.layout.baseline_home
    assert home_is_barcelona


@pytest.mark.data
def test_ball_visibility_matches_the_published_counts(tactical):
    graded = np.zeros(tactical.n_frames, dtype=bool)
    for _, first, last in games.half_video_blocks(games.GAME):
        graded[first:last + 1] = True
    seen = tactical.ball_seen()
    counts = (int(seen.sum()), int(seen[graded].sum()))
    assert counts == (113488, 94214)                 # DATA.md


@pytest.mark.data
@pytest.mark.parametrize("name, window", baseline.VARIANTS)
def test_loader_reproduces_the_official_baselines(tactical, name, window):
    ours = baseline.smooth(nearest_team(tactical), window)
    committed = submission.read_labels(os.path.join(ROOT, "evaluation", "results", name + ".csv"))
    mismatching_frames = int(np.sum(ours != committed))
    assert mismatching_frames == 0


@pytest.mark.data
def test_loader_agrees_with_kloppy_on_sampled_frames(tactical):
    from kloppy import metrica
    meta, raw = games.require(["atd_meta", "atd_raw"])
    dataset = metrica.load_tracking_epts(meta_data=meta, raw_data=raw, sample_rate=1.0 / 500,
                                         coordinates="metrica")
    slot_of = {player_id: slot for slot, player_id in enumerate(tactical.layout.slot_player_ids)}

    def same(a, b):
        return a == b or (np.isnan(a) and np.isnan(b))

    frames = bad = 0
    for frame in dataset.records:
        row = int(frame.frame_id) - 1
        frames += 1
        bad += not (same(frame.ball_coordinates.x, tactical.ball[row, 0])
                    and same(frame.ball_coordinates.y, tactical.ball[row, 1]))
        for player, data in frame.players_data.items():
            slot = slot_of[player.player_id]
            bad += not (same(data.coordinates.x, tactical.players[row, slot, 0])
                        and same(data.coordinates.y, tactical.players[row, slot, 1])
                        and same(data.speed, tactical.players[row, slot, 2]))
    assert frames > 300
    assert bad == 0


@pytest.mark.data
def test_speed_channel_is_metres_per_second(tactical):
    # The metadata says "kph"; DATA.md says m/s. Displacement decides (docs/decisions.md D-008).
    first, last = games.half_video_blocks(games.GAME)[0][1:]
    xy = tactical.players[first:last + 1, :, :2] * np.array(PITCH)
    step = np.hypot(*(xy[1:] - xy[:-1]).transpose(2, 0, 1)) * games.GAMES[games.GAME]["fps"]
    reported = tactical.players[first + 1:last + 1, :, 2]
    keep = np.isfinite(step) & np.isfinite(reported) & (step > 2) & (step < 12)
    ratio = float(np.median(reported[keep] / step[keep]))
    assert 0.9 < ratio < 1.1


@pytest.mark.data
def test_smart_tagging_count_and_time_base():
    import load_smart_tagging
    events = load_smart_tagging.load()
    result = (len(events), bool(load_smart_tagging.check_time_base(events)["agrees"]))
    assert result == (638, True)


@pytest.mark.data
def test_tactical_homography_is_one_row_per_video_frame():
    import load_homography
    report = load_homography.check_alignment()
    assert (report["n_rows"], bool(report["agrees"])) == (199319, True)
