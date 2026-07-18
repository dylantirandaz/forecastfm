"""Stable richer NBA feature schema built from licensed evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from forecastfm.integrity import canonical_sha256
from forecastfm.nba_evidence import (
    EvidenceKind,
    NbaEvidenceBundle,
    NbaEvidenceError,
    local_numeric_feature_vector,
    require_canonical_float,
    tinker_numeric_feature_vector,
)

NBA_RICH_SCHEMA_VERSION = 2
NBA_RICH_MISSING_POLICY = (
    "Reject a game with any missing required source value; use a stated default only for a "
    "genuine no-history state."
)


@dataclass(frozen=True, slots=True)
class NbaFeatureSpec:
    """One versioned numeric feature definition shared by every connector."""

    name: str
    kind: EvidenceKind
    unit: str
    definition: str
    minimum: float
    maximum: float


NBA_RICH_FEATURE_SPECS = (
    NbaFeatureSpec(
        "rest_days",
        "rest",
        "days",
        "Full days since the prior game; zero for a season opener.",
        0.0,
        30.0,
    ),
    NbaFeatureSpec(
        "back_to_back",
        "back_to_back",
        "indicator",
        "One when the prior game was on the preceding local date; otherwise zero.",
        0.0,
        1.0,
    ),
    NbaFeatureSpec(
        "games_last_7",
        "team_metric",
        "games",
        "Games with tipoffs in the seven days strictly before the forecast cutoff.",
        0.0,
        7.0,
    ),
    NbaFeatureSpec(
        "road_games_last_7",
        "travel",
        "games",
        "Away games with tipoffs in the seven days strictly before the cutoff.",
        0.0,
        7.0,
    ),
    NbaFeatureSpec(
        "travel_miles",
        "travel",
        "great_circle_miles",
        "Great-circle miles from the prior-game venue to the current venue; opener zero.",
        0.0,
        15_000.0,
    ),
    NbaFeatureSpec(
        "travel_time_zones",
        "travel",
        "absolute_utc_offset_hours",
        "Absolute venue UTC-offset change at each scheduled tipoff, including DST.",
        0.0,
        26.0,
    ),
    NbaFeatureSpec(
        "roster_continuity",
        "roster",
        "fraction_0_1",
        "Prior-game rotation-minute share belonging to the current pre-cutoff roster.",
        0.0,
        1.0,
    ),
    NbaFeatureSpec(
        "expected_lineup_continuity",
        "expected_lineup",
        "fraction_0_1",
        "Share of the latest projected starting five who started the prior game.",
        0.0,
        1.0,
    ),
    NbaFeatureSpec(
        "rolling_team_net_rating",
        "team_metric",
        "points_per_100_possessions",
        "Possession-weighted net rating over the ten strictly prior games; empty history zero.",
        -100.0,
        100.0,
    ),
    NbaFeatureSpec(
        "rolling_player_value",
        "player_metric",
        "points_per_100_possessions",
        "Projected-minutes-weighted player net rating over each player's ten strictly prior "
        "games; genuine no-history players are zero and missing source rows are rejected.",
        -100.0,
        100.0,
    ),
    NbaFeatureSpec(
        "schedule_strength",
        "schedule_strength",
        "elo_points",
        "Mean pregame opponent Elo over the ten strictly prior games; empty history 1500.",
        0.0,
        3_000.0,
    ),
)
NBA_LOCAL_HEALTH_FEATURE_SPECS = (
    NbaFeatureSpec(
        "unavailable_rotation_minutes",
        "injury",
        "minutes",
        "Prior-game minutes for players unavailable in the latest pre-cutoff report.",
        0.0,
        300.0,
    ),
    NbaFeatureSpec(
        "unavailable_rotation_value",
        "injury",
        "player_value_minutes",
        "Unavailable players' prior-game minutes times their strictly pre-cutoff rolling value.",
        -30_000.0,
        30_000.0,
    ),
)
NBA_RICH_FEATURE_NAMES = tuple(spec.name for spec in NBA_RICH_FEATURE_SPECS)
NBA_LOCAL_HEALTH_FEATURE_NAMES = tuple(spec.name for spec in NBA_LOCAL_HEALTH_FEATURE_SPECS)
NBA_RICH_SCHEMA_SHA256 = canonical_sha256(
    {
        "schema_version": NBA_RICH_SCHEMA_VERSION,
        "missing_policy": NBA_RICH_MISSING_POLICY,
        "standard": [
            {
                "name": spec.name,
                "kind": spec.kind,
                "unit": spec.unit,
                "definition": spec.definition,
                "minimum": spec.minimum,
                "maximum": spec.maximum,
            }
            for spec in NBA_RICH_FEATURE_SPECS
        ],
        "local_health": [
            {
                "name": spec.name,
                "kind": spec.kind,
                "unit": spec.unit,
                "definition": spec.definition,
                "minimum": spec.minimum,
                "maximum": spec.maximum,
            }
            for spec in NBA_LOCAL_HEALTH_FEATURE_SPECS
        ],
    }
)


@dataclass(frozen=True, slots=True)
class NbaRichFeatures:
    """Team-minus-opponent values in the predeclared richer feature schema."""

    rest_days_difference: float
    back_to_back_difference: float
    games_last_7_difference: float
    road_games_last_7_difference: float
    travel_miles_difference: float
    travel_time_zones_difference: float
    roster_continuity_difference: float
    expected_lineup_continuity_difference: float
    rolling_team_net_rating_difference: float
    rolling_player_value_difference: float
    schedule_strength_difference: float

    def __post_init__(self) -> None:
        _require_difference_values(self.vector, NBA_RICH_FEATURE_SPECS)

    @property
    def vector(self) -> tuple[float, ...]:
        """Return values in stable ``NBA_RICH_FEATURE_NAMES`` order."""
        return (
            self.rest_days_difference,
            self.back_to_back_difference,
            self.games_last_7_difference,
            self.road_games_last_7_difference,
            self.travel_miles_difference,
            self.travel_time_zones_difference,
            self.roster_continuity_difference,
            self.expected_lineup_continuity_difference,
            self.rolling_team_net_rating_difference,
            self.rolling_player_value_difference,
            self.schedule_strength_difference,
        )

    def as_dict(self) -> dict[str, float]:
        """Return a readable feature mapping in stable model order."""
        return dict(zip(NBA_RICH_FEATURE_NAMES, self.vector, strict=True))

    @classmethod
    def from_vector(cls, values: tuple[float, ...]) -> NbaRichFeatures:
        """Construct features from the frozen vector order with full validation."""
        if len(values) != len(NBA_RICH_FEATURE_NAMES):
            raise NbaEvidenceError("richer NBA feature count is invalid")
        return cls(
            rest_days_difference=values[0],
            back_to_back_difference=values[1],
            games_last_7_difference=values[2],
            road_games_last_7_difference=values[3],
            travel_miles_difference=values[4],
            travel_time_zones_difference=values[5],
            roster_continuity_difference=values[6],
            expected_lineup_continuity_difference=values[7],
            rolling_team_net_rating_difference=values[8],
            rolling_player_value_difference=values[9],
            schedule_strength_difference=values[10],
        )

    def side_swap(self) -> NbaRichFeatures:
        """Exchange team and opponent by exactly negating every feature."""
        return NbaRichFeatures.from_vector(tuple(_negate(value) for value in self.vector))


def local_rich_features_from_bundle(
    bundle: NbaEvidenceBundle,
    *,
    action_at: datetime,
) -> NbaRichFeatures:
    """Build richer local-model features after checking source rights."""
    _require_feature_records(bundle, NBA_RICH_FEATURE_SPECS)
    values = local_numeric_feature_vector(
        bundle,
        NBA_RICH_FEATURE_NAMES,
        action_at=action_at,
    )
    return NbaRichFeatures.from_vector(values)


def tinker_rich_features_from_bundle(
    bundle: NbaEvidenceBundle,
    *,
    action_at: datetime,
) -> NbaRichFeatures:
    """Build richer Tinker features after rights and health checks."""
    _require_feature_records(bundle, NBA_RICH_FEATURE_SPECS)
    values = tinker_numeric_feature_vector(
        bundle,
        NBA_RICH_FEATURE_NAMES,
        action_at=action_at,
    )
    return NbaRichFeatures.from_vector(values)


def local_health_feature_vector(
    bundle: NbaEvidenceBundle,
    *,
    action_at: datetime,
) -> tuple[float, ...]:
    """Build the explicitly local-only availability vector."""
    _require_feature_records(bundle, NBA_LOCAL_HEALTH_FEATURE_SPECS)
    return local_numeric_feature_vector(
        bundle,
        NBA_LOCAL_HEALTH_FEATURE_NAMES,
        action_at=action_at,
    )


def _require_feature_records(
    bundle: NbaEvidenceBundle,
    specs: tuple[NbaFeatureSpec, ...],
) -> None:
    expected = {spec.name: spec for spec in specs}
    expected_kinds = {name: spec.kind for name, spec in expected.items()}
    if any(record.kind != expected_kinds.get(record.feature_name) for record in bundle.records):
        raise NbaEvidenceError("evidence kinds do not match the predeclared schema")
    for record in bundle.records:
        spec = expected[record.feature_name]
        _require_side_value(record.team_value, spec)
        _require_side_value(record.opponent_value, spec)


def _require_side_value(value: float, spec: NbaFeatureSpec) -> None:
    require_canonical_float(value, spec.name)
    if value < spec.minimum or value > spec.maximum:
        raise NbaEvidenceError(f"{spec.name} lies outside its declared per-team range")
    if spec.unit == "indicator" and value not in (0.0, 1.0):
        raise NbaEvidenceError(f"{spec.name} must be zero or one")


def _require_difference_values(
    values: tuple[float, ...],
    specs: tuple[NbaFeatureSpec, ...],
) -> None:
    if len(values) != len(specs):
        raise NbaEvidenceError("richer NBA feature count is invalid")
    for value, spec in zip(values, specs, strict=True):
        require_canonical_float(value, spec.name)
        maximum_difference = spec.maximum - spec.minimum
        if abs(value) > maximum_difference:
            raise NbaEvidenceError(f"{spec.name} difference lies outside its declared range")
        if spec.unit == "indicator" and value not in (-1.0, 0.0, 1.0):
            raise NbaEvidenceError(f"{spec.name} difference must be minus one, zero, or one")


def _negate(value: float) -> float:
    if value == 0.0:
        return 0.0
    return -value
