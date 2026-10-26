"""Smart Tagging as soft evidence. P1 -- experiment E5, ticket T9; off in every P0 configuration.

The tagging is automatic, not ground truth, and its intervals are phases, not possessions
(DATA.md). It enters only as a small additive prior, so tracking evidence can always overrule it.

What a tag says about possession follows the code definitions in OVERVIEW.md:

- OFFENSIVE codes describe the team on the ball: the tagged team is in possession.
- DEFENSIVE codes describe the team without it: the *other* team is in possession. A Defensive
  Transition is the team that has just lost the ball.
- SET PIECES are restarts: the ball is dead before the restart inside the interval and live after
  it, and the tagged team restarts. Every throw-in is tagged Team = N/A.
- GOALS: the ball is dead after the goal until the other team's kickoff.

Team values are matched to the tracking's own team names (Layout.home_name / away_name: the ATD
metadata and the tagging both say "Barcelona" and "Atletic Club"), never to a hard-coded string.
"""

# Params
OFFENSIVE = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL",
             "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX")
DEFENSIVE = ("DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
             "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION")
RESTARTS = ("SET PIECES",)
GOALS = ("GOALS",)


def tag_evidence(events, layout, n_frames, config):
    """(team, dead): two (n,) float64 arrays added to the tracking evidence. Ticket T9.

    events -- starter/load_smart_tagging.load() records; layout -- the ATD Layout (team names);
    config -- src.config.SmartTaggingConfig.

    team: over every OFFENSIVE interval, +phase_score for a home tag and -phase_score for an away
          tag; DEFENSIVE intervals the opposite sign; N/A tags nothing. Overlaps add up.
    dead: +set_piece_score from the start of each SET PIECES interval to the first frame where the
          tracked ball moves faster than DeadBallConfig.moving_mps inside it (the restart), or over
          the first half of the interval when the ball is never seen moving.
    Codes and values match case-insensitively (DATA.md).
    """
    raise NotImplementedError("P1, ticket T9 -- docs/tickets.md")
