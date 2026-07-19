"""Build private-prototype model tables from season games, features, and Elo state.

Home-perspective rows only: the side swap is the exact negation by construction. The prototype
Elo is the disclosed carryover margin-of-victory replay in ``nba_mov_elo``; neutral-site games
carry zero home advantage. The schedule strength feature reads each prior game's pregame
opponent rating from the same replay.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from forecastfm.elo_residual import EloResidualRow
from forecastfm.nba_feature_builder import GameFeatures
from forecastfm.nba_rich import NBA_LOCAL_HEALTH_FEATURE_NAMES, NBA_RICH_FEATURE_NAMES
from forecastfm.nba_season_games import SeasonGame

NBA_PROTOTYPE_DATASET_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PrototypeGameRow:
    """One home-perspective model row with identity and answer."""

    question_id: str
    game_id: int
    season: int
    game_date: date
    elo_home_probability: float
    features_standard: tuple[float, ...]
    features_health: tuple[float, ...] | None
    home_won: bool


def question_id_for(game_id: int) -> str:
    """Return the stable cohort identity for one game."""
    return f"nba-{game_id}"


def build_prototype_rows(
    games: list[SeasonGame],
    features: list[GameFeatures],
    home_probabilities: Mapping[int, float],
) -> list[PrototypeGameRow]:
    """Assemble home-perspective model rows aligned to replayed home probabilities."""
    rows: list[PrototypeGameRow] = []
    for game, game_features in zip(games, features, strict=True):
        standard = _differences(game_features)
        health = (
            (
                game_features.health[1][0] - game_features.health[0][0],
                game_features.health[1][1] - game_features.health[0][1],
            )
            if game_features.health is not None
            else None
        )
        rows.append(
            PrototypeGameRow(
                question_id=question_id_for(game.game_id),
                game_id=game.game_id,
                season=game.season_label,
                game_date=game.game_date,
                elo_home_probability=home_probabilities[game.game_id],
                features_standard=standard,
                features_health=health,
                home_won=game.home_won,
            )
        )
    return rows


def to_residual_row(row: PrototypeGameRow, *, include_health: bool) -> EloResidualRow:
    """Convert one prototype row into an Elo-residual training row."""
    features = row.features_standard
    if include_health:
        if row.features_health is None:
            raise ValueError(f"game {row.game_id} lacks health features")
        features = features + row.features_health
    return EloResidualRow(
        question_id=row.question_id,
        elo_probability=row.elo_home_probability,
        features=features,
        outcome=1 if row.home_won else 0,
    )


def feature_names(*, include_health: bool) -> tuple[str, ...]:
    """Return the feature-name order for one model variant."""
    if include_health:
        return NBA_RICH_FEATURE_NAMES + NBA_LOCAL_HEALTH_FEATURE_NAMES
    return NBA_RICH_FEATURE_NAMES


def fit_rms_scales(rows: list[PrototypeGameRow], *, include_health: bool) -> tuple[float, ...]:
    """Compute uncentered RMS scales over original training rows only."""
    vectors = [to_residual_row(row, include_health=include_health).features for row in rows]
    names = feature_names(include_health=include_health)
    scales: list[float] = []
    for index in range(len(names)):
        mean_square = sum(vector[index] ** 2 for vector in vectors) / len(vectors)
        scales.append(mean_square**0.5 if mean_square > 0.0 else 1.0)
    return tuple(scales)


def _differences(game_features: GameFeatures) -> tuple[float, ...]:
    away, home = game_features.away, game_features.home
    return (
        home.rest_days - away.rest_days,
        home.back_to_back - away.back_to_back,
        home.games_last_7 - away.games_last_7,
        home.road_games_last_7 - away.road_games_last_7,
        home.travel_miles - away.travel_miles,
        home.travel_time_zones - away.travel_time_zones,
        home.roster_continuity - away.roster_continuity,
        home.expected_lineup_continuity - away.expected_lineup_continuity,
        home.rolling_team_net_rating - away.rolling_team_net_rating,
        home.rolling_player_value - away.rolling_player_value,
        home.schedule_strength - away.schedule_strength,
    )
