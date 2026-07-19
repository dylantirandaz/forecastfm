"""Carryover margin-of-victory Elo for the private prototype lane.

A stronger disclosed baseline than the frozen per-season-reset replay: ratings carry between
seasons with 0.75 retention (FiveThirtyEight convention), and each game updates by
K times a margin-of-victory multiplier that damps blowouts of much-weaker opponents
(ln(margin + 1) * 2.2 / (0.001 * winner_pre_game_edge + 2.2)). This is an approximation of the
public FiveThirtyEight recipe, not a replication. The prototype's model offset, schedule
strength feature, raw baseline, and training-only recalibration all read from this same replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log

NBA_MOV_ELO_SCHEMA_VERSION = 1


class NbaMovEloError(ValueError):
    """Raised when a margin-of-victory Elo input violates its contract."""


@dataclass(frozen=True, slots=True)
class MovEloRecipe:
    """Frozen numerical choices for the carryover margin-of-victory replay."""

    initial_rating: float = 1500.0
    k_factor: float = 20.0
    rating_scale: float = 400.0
    home_advantage: float = 100.0
    carryover: float = 0.75

    def __post_init__(self) -> None:
        for name, value in (
            ("initial_rating", self.initial_rating),
            ("k_factor", self.k_factor),
            ("rating_scale", self.rating_scale),
            ("home_advantage", self.home_advantage),
        ):
            if not isfinite(value) or value <= 0.0:
                raise NbaMovEloError(f"{name} must be positive and finite")
        if not 0.0 < self.carryover <= 1.0:
            raise NbaMovEloError("carryover must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class EloGameResult:
    """One completed game in chronological order for the replay."""

    game_id: int
    home_abbreviation: str
    away_abbreviation: str
    home_score: int
    away_score: int
    neutral: bool


@dataclass(frozen=True, slots=True)
class MovEloReplay:
    """Pre-game ratings and home probabilities for every replayed game."""

    ratings: dict[tuple[int, str], float]
    home_probabilities: dict[int, float]

    def home_probability(self, game_id: int) -> float:
        """Return the pre-game home win probability for one game."""
        try:
            return self.home_probabilities[game_id]
        except KeyError as exc:
            raise NbaMovEloError(f"unknown game id: {game_id}") from exc


def replay_mov_elo(
    seasons: list[list[EloGameResult]],
    recipe: MovEloRecipe | None = None,
) -> MovEloReplay:
    """Replay chronological seasons with carryover and margin-of-victory updates."""
    settings = recipe or MovEloRecipe()
    ratings: dict[str, float] = {}
    recorded_ratings: dict[tuple[int, str], float] = {}
    probabilities: dict[int, float] = {}
    for season_games in seasons:
        ratings = {
            team: settings.initial_rating + settings.carryover * (rating - settings.initial_rating)
            for team, rating in ratings.items()
        }
        for game in season_games:
            _require_game(game)
            home_rating = ratings.get(game.home_abbreviation, settings.initial_rating)
            away_rating = ratings.get(game.away_abbreviation, settings.initial_rating)
            advantage = 0.0 if game.neutral else settings.home_advantage
            difference = home_rating + advantage - away_rating
            expected_home = _win_probability(difference, settings.rating_scale)
            recorded_ratings[(game.game_id, game.home_abbreviation)] = home_rating
            recorded_ratings[(game.game_id, game.away_abbreviation)] = away_rating
            probabilities[game.game_id] = expected_home
            margin = abs(game.home_score - game.away_score)
            home_won = game.home_score > game.away_score
            if home_won:
                winner_edge = difference
            else:
                winner_edge = away_rating - home_rating - advantage
            multiplier = log(margin + 1.0) * 2.2 / (0.001 * winner_edge + 2.2)
            shift = settings.k_factor * multiplier * ((1.0 if home_won else 0.0) - expected_home)
            ratings[game.home_abbreviation] = home_rating + shift
            ratings[game.away_abbreviation] = away_rating - shift
    return MovEloReplay(ratings=recorded_ratings, home_probabilities=probabilities)


def _win_probability(difference: float, scale: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-difference / scale))


def _require_game(game: EloGameResult) -> None:
    if game.home_abbreviation == game.away_abbreviation:
        raise NbaMovEloError("home and away abbreviations must differ")
    if game.home_score < 0 or game.away_score < 0 or game.home_score == game.away_score:
        raise NbaMovEloError(f"game {game.game_id} has an invalid final score")
