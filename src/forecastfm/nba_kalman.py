"""DARKO-lite Kalman-filtered player impact ratings from per-game plus-minus.

Each player carries a scalar random-walk state theta (per-100-possession impact). One game
yields one observation per player: raw plus-minus scaled to 100 estimated possessions, where
a player's estimated possessions are the team's possessions times the player's share of team
seconds (the same approximation ``nba_team_history`` discloses). The observation weight is
the estimated possessions: the observation variance is ``r = game_noise**2 / possessions``
with the disclosed player game noise of 20 points per 100 possessions, i.e. ``r = 400 /
possessions``. The process noise is q = 1.5 per observed game. Ratings reset at every season
boundary to the prior-season causal RAPM fit (mean, variance 25) — a disclosed simplification
that discards all in-season information across the offseason; players absent from the prior
fit start at mean 0, variance 100. Updates run strictly in tipoff order within a season, so
the rating recorded for a game uses only games that already tipped off. Unlike DARKO this is
a single scalar per player: teammates and opponents are not regressed out, their contribution
lands in the disclosed observation noise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from forecastfm.nba_pbp import PlayerGameLine, normalize_player_name
from forecastfm.nba_season_games import SeasonGame

NBA_KALMAN_SCHEMA_VERSION = 1

TEAM_SECONDS_PER_GAME = 5 * 48 * 60


class NbaKalmanError(ValueError):
    """Raised when a Kalman rating input or configuration violates its contract."""


@dataclass(frozen=True, slots=True)
class KalmanConfig:
    """Frozen DARKO-lite filter constants."""

    process_noise_per_game: float = 1.5
    game_noise_per_100: float = 20.0
    prior_variance: float = 25.0
    no_prior_mean: float = 0.0
    no_prior_variance: float = 100.0

    def __post_init__(self) -> None:
        variances = (
            ("process_noise_per_game", self.process_noise_per_game),
            ("game_noise_per_100", self.game_noise_per_100),
            ("prior_variance", self.prior_variance),
            ("no_prior_variance", self.no_prior_variance),
        )
        for name, value in variances:
            if not isfinite(value) or value <= 0.0:
                raise NbaKalmanError(f"{name} must be positive and finite")
        if not isfinite(self.no_prior_mean):
            raise NbaKalmanError("no_prior_mean must be finite")


@dataclass(frozen=True, slots=True)
class KalmanRatings:
    """Filtered pregame ratings for every (game_id, team), keyed by ID and by name."""

    by_id: dict[tuple[int, str], dict[int, float]]
    by_name: dict[tuple[int, str], dict[str, float]]


def ratings_at(
    games: Mapping[int, Sequence[SeasonGame]],
    prior_rapm: Mapping[int, Mapping[int, float]] | None = None,
    config: KalmanConfig | None = None,
) -> KalmanRatings:
    """Filter each season in tipoff order; return pregame means per (game_id, team).

    ``prior_rapm`` maps each season label to that season's causal prior means keyed by player
    ID; callers build it with ``fit_season_ratings`` over strictly earlier seasons so no
    in-season information leaks. ``None`` starts every player at the no-prior state.
    """
    settings = config or KalmanConfig()
    by_id: dict[tuple[int, str], dict[int, float]] = {}
    by_name: dict[tuple[int, str], dict[str, float]] = {}
    for season in sorted(games):
        ordered = sorted(games[season], key=lambda game: (game.tipoff, game.game_id))
        means, variances = _season_priors(settings, prior_rapm, season)
        for game in ordered:
            for side in (game.away_abbreviation, game.home_abbreviation):
                key = (game.game_id, side)
                by_id[key], by_name[key] = _pregame_ratings(settings, game, side, means)
                _update_side(settings, game, side, means, variances)
    return KalmanRatings(by_id=by_id, by_name=by_name)


def _season_priors(
    settings: KalmanConfig,
    prior_rapm: Mapping[int, Mapping[int, float]] | None,
    season: int,
) -> tuple[dict[int, float], dict[int, float]]:
    """Reset every player to the season's RAPM prior, or to the no-prior state."""
    means = dict(prior_rapm.get(season, {})) if prior_rapm is not None else {}
    variances = dict.fromkeys(means, settings.prior_variance)
    return means, variances


def _pregame_ratings(
    settings: KalmanConfig,
    game: SeasonGame,
    side: str,
    means: Mapping[int, float],
) -> tuple[dict[int, float], dict[str, float]]:
    """Snapshot one side's pregame filtered means, keyed by ID and normalized name."""
    by_player: dict[int, float] = {}
    by_player_name: dict[str, float] = {}
    for line in game.pbp.player_lines:
        if line.team_abbreviation != side:
            continue
        rating = means.get(line.player_id, settings.no_prior_mean)
        by_player[line.player_id] = rating
        name = game.pbp.player_names.get(line.player_id)
        if name is not None:
            by_player_name[" ".join(normalize_player_name(name))] = rating
    return by_player, by_player_name


def _update_side(
    settings: KalmanConfig,
    game: SeasonGame,
    side: str,
    means: dict[int, float],
    variances: dict[int, float],
) -> None:
    """Apply one game's observations for one team to the shared filter state."""
    stats = next(s for s in game.pbp.team_stats if s.team_abbreviation == side)
    if stats.possessions <= 0.0:
        return
    for line in game.pbp.player_lines:
        if line.team_abbreviation == side:
            _update_player(settings, stats.possessions, line, means, variances)


def _update_player(
    settings: KalmanConfig,
    team_possessions: float,
    line: PlayerGameLine,
    means: dict[int, float],
    variances: dict[int, float],
) -> None:
    """Run one scalar predict/update step for one player's game line."""
    mean = means.get(line.player_id, settings.no_prior_mean)
    variance = variances.get(line.player_id, settings.no_prior_variance)
    predicted = variance + settings.process_noise_per_game
    if line.seconds_played <= 0:
        variances[line.player_id] = predicted
        return
    played = team_possessions * line.seconds_played / TEAM_SECONDS_PER_GAME
    observation = line.plus_minus / played * 100.0
    observation_variance = settings.game_noise_per_100**2 / played
    gain = predicted / (predicted + observation_variance)
    means[line.player_id] = mean + gain * (observation - mean)
    variances[line.player_id] = (1.0 - gain) * predicted
