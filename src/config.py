"""Run configuration: one TOML file says everything a prediction run depends on.

    from src.config import load
    config = load("configs/hybrid_v1.toml")
    config.temporal.turnover_cost          # a float, typed and range-checked
    config.digest()                        # SHA-256 of the resolved configuration, for manifests

The schema is the dataclasses below: every field is typed, defaulted and documented, so a file
needs only the values it changes. Unknown sections and keys are errors rather than being ignored:
a misspelt parameter must not quietly fall back to its default and change a submission.

TOML rather than YAML: Python 3.11+ reads it with the standard library (tomllib), so the format
adds no dependency, and it still allows the comments that say why each value is what it is.
Parameter *values* and their rationale live in configs/*.toml and docs/architecture.md; this file
fixes only their names, types and ranges.
"""

import dataclasses
import hashlib
import json
import tomllib

# Params
EVIDENCE_KINDS = ("nearest_hard", "soft_control")
DEAD_BALL_KINDS = ("ball_missing", "cues")
DECODERS = ("majority", "viterbi")
UNGRADED = ("X", "keep")
ABSTAIN = ("never",)


@dataclasses.dataclass(frozen=True)
class RunConfig:
    name: str = "unnamed"
    description: str = ""


@dataclasses.dataclass(frozen=True)
class EvidenceConfig:
    """Team evidence per frame (src/possession_evidence.py): > 0 favours home control."""
    kind: str = "soft_control"     # nearest_hard: official baseline step 1 | soft_control
    control_m: float = 1.0         # distance (m) at which a player's control weight is 1/4
    free_m: float = 2.5            # reference distance (m) for "nobody controls the ball"
    max_score: float = 3.0         # cap on |evidence| per frame
    flight_mps: float = 8.0        # soft_control: a faster ball is in flight -- no team evidence


@dataclasses.dataclass(frozen=True)
class DeadBallConfig:
    """Dead-ball evidence per frame (src/dead_ball.py): > 0 favours D."""
    kind: str = "cues"             # ball_missing: official baseline (D iff no ball) | cues
    out_margin_m: float = 0.5      # ball this far beyond a line counts as out
    out_score: float = 4.0
    moving_mps: float = 3.0        # a ball faster than this, inside the pitch, is live evidence
    moving_score: float = 1.0      # subtracted
    still_mps: float = 0.5         # a ball slower than this is still...
    still_min_s: float = 1.5       # ...and still for at least this long counts as placed
    still_score: float = 1.0
    activity_window_s: float = 2.0       # half-width of the collective-speed average
    activity_threshold_mps: float = 1.2  # mean player speed below this is dead evidence
    activity_scale_mps: float = 0.4      # evidence per this much below/above the threshold
    activity_max_score: float = 2.0


@dataclasses.dataclass(frozen=True)
class TemporalConfig:
    """The decoder (src/temporal.py)."""
    decoder: str = "viterbi"       # majority: official baseline step 2 | viterbi
    window: int = 50               # majority only: half-width of the vote, frames
    evidence_scale: float = 0.2    # per-frame evidence is autocorrelated; this discounts it
    alive_bonus: float = 0.7       # most team evidence can say for "alive" over D (~ln 2)
    turnover_cost: float = 5.7     # H <-> A
    stop_cost: float = 6.9         # H/A -> D
    restart_cost: float = 5.9      # D -> H/A
    fixed_point: int = 1000000     # scores are decoded as integers in units of 1/fixed_point


@dataclasses.dataclass(frozen=True)
class SmartTaggingConfig:
    """Smart Tagging priors (src/smart_tagging.py). P1: off until experiment E5."""
    enabled: bool = False
    phase_score: float = 0.3       # team evidence per frame inside a team-tagged phase
    set_piece_score: float = 1.0   # dead-ball evidence before a tagged restart


@dataclasses.dataclass(frozen=True)
class OutputConfig:
    ungraded: str = "X"            # X: write X outside the graded halves | keep: decoder output
    abstain: str = "never"         # graded frames always get H/A/D (docs/architecture.md §6)


@dataclasses.dataclass(frozen=True)
class Config:
    run: RunConfig = RunConfig()
    evidence: EvidenceConfig = EvidenceConfig()
    dead_ball: DeadBallConfig = DeadBallConfig()
    temporal: TemporalConfig = TemporalConfig()
    smart_tagging: SmartTaggingConfig = SmartTaggingConfig()
    output: OutputConfig = OutputConfig()

    def to_dict(self):
        return dataclasses.asdict(self)

    def digest(self):
        """SHA-256 of the resolved configuration. Same values -> same digest, whatever the file
        looked like (key order, comments, defaults written out or left implicit)."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _coerce(section, key, value, field_type):
    """value as field_type, or a ValueError naming section.key. bool is never an int here."""
    where = "{}.{}".format(section, key)
    if field_type is bool:
        if not isinstance(value, bool):
            raise ValueError("{}: expected true/false, got {!r}".format(where, value))
        return value
    if field_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("{}: expected an integer, got {!r}".format(where, value))
        return value
    if field_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("{}: expected a number, got {!r}".format(where, value))
        return float(value)
    if field_type is str:
        if not isinstance(value, str):
            raise ValueError("{}: expected a string, got {!r}".format(where, value))
        return value
    raise TypeError("{}: unsupported field type {}".format(where, field_type))


def from_dict(data, source="<dict>"):
    """Config from a parsed TOML document, or a ValueError naming the first problem."""
    sections = {}
    known = {field.name: field.type for field in dataclasses.fields(Config)}
    for section, values in data.items():
        if section not in known:
            raise ValueError("{}: unknown section [{}]; known: {}".format(
                source, section, sorted(known)))
        if not isinstance(values, dict):
            raise ValueError("{}: [{}] must be a table".format(source, section))
        fields = {field.name: field.type for field in dataclasses.fields(known[section])}
        unknown = sorted(set(values) - set(fields))
        if unknown:
            raise ValueError("{}: unknown key(s) {} in [{}]; known: {}".format(
                source, unknown, section, sorted(fields)))
        sections[section] = known[section](**{
            key: _coerce(section, key, value, fields[key]) for key, value in values.items()})
    config = Config(**sections)
    validate(config, source)
    return config


def validate(config, source="<config>"):
    """Raise ValueError if any value is out of range or the combination is not runnable."""
    def require(condition, message):
        if not condition:
            raise ValueError("{}: {}".format(source, message))

    require(config.evidence.kind in EVIDENCE_KINDS,
            "evidence.kind must be one of {}".format(EVIDENCE_KINDS))
    require(config.dead_ball.kind in DEAD_BALL_KINDS,
            "dead_ball.kind must be one of {}".format(DEAD_BALL_KINDS))
    require(config.temporal.decoder in DECODERS,
            "temporal.decoder must be one of {}".format(DECODERS))
    require(config.output.ungraded in UNGRADED, "output.ungraded must be one of {}".format(UNGRADED))
    require(config.output.abstain in ABSTAIN, "output.abstain must be one of {}".format(ABSTAIN))

    for name in ("control_m", "free_m", "max_score", "flight_mps"):
        require(getattr(config.evidence, name) > 0, "evidence.{} must be > 0".format(name))
    for name in ("out_margin_m", "moving_mps", "still_mps", "still_min_s", "activity_window_s",
                 "activity_threshold_mps", "activity_scale_mps"):
        require(getattr(config.dead_ball, name) > 0, "dead_ball.{} must be > 0".format(name))
    for name in ("out_score", "moving_score", "still_score", "activity_max_score"):
        require(getattr(config.dead_ball, name) >= 0, "dead_ball.{} must be >= 0".format(name))
    require(config.temporal.window >= 0, "temporal.window must be >= 0")
    require(config.temporal.evidence_scale > 0, "temporal.evidence_scale must be > 0")
    require(config.temporal.alive_bonus >= 0, "temporal.alive_bonus must be >= 0")
    for name in ("turnover_cost", "stop_cost", "restart_cost"):
        require(getattr(config.temporal, name) >= 0, "temporal.{} must be >= 0".format(name))
    require(config.temporal.fixed_point >= 1, "temporal.fixed_point must be >= 1")
    for name in ("phase_score", "set_piece_score"):
        require(getattr(config.smart_tagging, name) >= 0,
                "smart_tagging.{} must be >= 0".format(name))

    # The majority vote counts hard per-frame codes, so it is defined only over the official
    # baseline's own evidence: nearest team where the ball is seen, nobody where it is not.
    if config.temporal.decoder == "majority":
        require(config.evidence.kind == "nearest_hard" and config.dead_ball.kind == "ball_missing",
                "temporal.decoder = 'majority' needs evidence.kind = 'nearest_hard' and "
                "dead_ball.kind = 'ball_missing'")
        require(not config.smart_tagging.enabled,
                "temporal.decoder = 'majority' cannot use smart_tagging")


def load(path):
    """Config from a TOML file."""
    with open(path, "rb") as handle:
        return from_dict(tomllib.load(handle), source=path)
