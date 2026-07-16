"""Stable richer NBA feature schema built from licensed evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite

from forecastfm.integrity import canonical_sha256
from forecastfm.nba_evidence import (
    EvidenceKind,
    NbaEvidenceBundle,
    NbaEvidenceError,
    local_numeric_feature_vector,
    tinker_numeric_feature_vector,
)

NBA_RICH_SCHEMA_VERSION = 1
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


NBA_RICH_FEATURE_SPECS = (
    NbaFeatureSpec(
        "rest_days",
        "rest",
        "days",
        "Full days since the prior game; zero for a season opener.",
    ),
    NbaFeatureSpec(
        "back_to_back",
        "back_to_back",
        "indicator",
        "One when the prior game was on the preceding local date; otherwise zero.",
    ),
    NbaFeatureSpec(
        "games_last_7",
        "team_metric",
        "games",
        "Games with tipoffs in the seven days strictly before the forecast cutoff.",
    ),
    NbaFeatureSpec(
        "road_games_last_7",
        "travel",
        "games",
        "Away games with tipoffs in the seven days strictly before the cutoff.",
    ),
    NbaFeatureSpec(
        "travel_miles",
        "travel",
        "great_circle_miles",
        "Great-circle miles from the prior-game venue to the current venue; opener zero.",
    ),
    NbaFeatureSpec(
        "travel_time_zones",
        "travel",
        "absolute_utc_offset_hours",
        "Absolute venue UTC-offset change at each scheduled tipoff, including DST.",
    ),
    NbaFeatureSpec(
        "roster_continuity",
        "roster",
        "fraction_0_1",
        "Prior-game rotation-minute share belonging to the current pre-cutoff roster.",
    ),
    NbaFeatureSpec(
        "expected_lineup_continuity",
        "expected_lineup",
        "fraction_0_1",
        "Share of the latest projected starting five who started the prior game.",
    ),
    NbaFeatureSpec(
        "rolling_team_net_rating",
        "team_metric",
        "points_per_100_possessions",
        "Possession-weighted net rating over the ten strictly prior games; empty history zero.",
    ),
    NbaFeatureSpec(
        "rolling_rotation_value",
        "player_metric",
        "prior_season_raptor_per_100",
        "Projected-minutes weighted prior-season RAPTOR; rookie or missing value zero.",
    ),
    NbaFeatureSpec(
        "schedule_strength",
        "schedule_strength",
        "elo_points",
        "Mean pregame opponent Elo over the ten strictly prior games; empty history 1500.",
    ),
)
NBA_LOCAL_HEALTH_FEATURE_SPECS = (
    NbaFeatureSpec(
        "unavailable_rotation_minutes",
        "injury",
        "minutes",
        "Prior-game minutes for players unavailable in the latest pre-cutoff report.",
    ),
    NbaFeatureSpec(
        "unavailable_rotation_value",
        "injury",
        "prior_season_raptor_minutes",
        "Unavailable players' prior-game minutes times prior-season RAPTOR.",
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
            }
            for spec in NBA_RICH_FEATURE_SPECS
        ],
        "local_health": [
            {
                "name": spec.name,
                "kind": spec.kind,
                "unit": spec.unit,
                "definition": spec.definition,
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
    rolling_rotation_value_difference: float
    schedule_strength_difference: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in self.vector):
            raise NbaEvidenceError("richer NBA features must be finite")

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
            self.rolling_rotation_value_difference,
            self.schedule_strength_difference,
        )

    def as_dict(self) -> dict[str, float]:
        """Return a readable feature mapping in stable model order."""
        return dict(zip(NBA_RICH_FEATURE_NAMES, self.vector, strict=True))

    def side_swap(self) -> NbaRichFeatures:
        """Exchange team and opponent by exactly negating every feature."""
        return _from_vector(tuple(_negate(value) for value in self.vector))


def local_rich_features_from_bundle(
    bundle: NbaEvidenceBundle,
    *,
    action_at: datetime,
) -> NbaRichFeatures:
    """Build richer local-model features after checking source rights."""
    _require_feature_kinds(bundle)
    values = local_numeric_feature_vector(
        bundle,
        NBA_RICH_FEATURE_NAMES,
        action_at=action_at,
    )
    return _from_vector(values)


def tinker_rich_features_from_bundle(
    bundle: NbaEvidenceBundle,
    *,
    action_at: datetime,
) -> NbaRichFeatures:
    """Build richer Tinker features after rights and health checks."""
    _require_feature_kinds(bundle)
    values = tinker_numeric_feature_vector(
        bundle,
        NBA_RICH_FEATURE_NAMES,
        action_at=action_at,
    )
    return _from_vector(values)


def local_health_feature_vector(
    bundle: NbaEvidenceBundle,
    *,
    action_at: datetime,
) -> tuple[float, ...]:
    """Build the explicitly local-only availability vector."""
    expected_kinds = {spec.name: spec.kind for spec in NBA_LOCAL_HEALTH_FEATURE_SPECS}
    if any(record.kind != expected_kinds.get(record.feature_name) for record in bundle.records):
        raise NbaEvidenceError("local health evidence does not match its predeclared schema")
    return local_numeric_feature_vector(
        bundle,
        NBA_LOCAL_HEALTH_FEATURE_NAMES,
        action_at=action_at,
    )


def _require_feature_kinds(bundle: NbaEvidenceBundle) -> None:
    expected_kinds: dict[str, EvidenceKind] = {
        spec.name: spec.kind for spec in NBA_RICH_FEATURE_SPECS
    }
    if any(record.kind != expected_kinds.get(record.feature_name) for record in bundle.records):
        raise NbaEvidenceError("richer evidence kinds do not match the predeclared schema")


def _from_vector(values: tuple[float, ...]) -> NbaRichFeatures:
    if len(values) != len(NBA_RICH_FEATURE_NAMES):
        raise NbaEvidenceError("richer NBA feature count is invalid")
    return NbaRichFeatures(
        rest_days_difference=values[0],
        back_to_back_difference=values[1],
        games_last_7_difference=values[2],
        road_games_last_7_difference=values[3],
        travel_miles_difference=values[4],
        travel_time_zones_difference=values[5],
        roster_continuity_difference=values[6],
        expected_lineup_continuity_difference=values[7],
        rolling_team_net_rating_difference=values[8],
        rolling_rotation_value_difference=values[9],
        schedule_strength_difference=values[10],
    )


def _negate(value: float) -> float:
    if value == 0.0:
        return 0.0
    return -value
