"""Regularized adjusted plus-minus (RAPM) ratings from play-by-play stints.

Each stint is one observation: the ten players on court, an estimated possession count, and the
point differential. A ridge regression over all stints assigns every player a per-100-possession
impact rating plus a free home-advantage scalar. Fitting is deterministic seeded mini-batch
gradient descent on the mean weighted squared error plus a penalty on the mean squared rating,
dependency-free in the style of ``elo_residual``. Ratings for a season are fit only on stints
from strictly earlier seasons, so they are legal pregame inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from random import Random

from forecastfm.nba_pbp import StintRecord, normalize_player_name, read_pbp_games

NBA_RAPM_SCHEMA_VERSION = 1
_MIN_STINT_POSSESSIONS = 0.5


class NbaRapmError(ValueError):
    """Raised when a RAPM fit input or result violates its contract."""


@dataclass(frozen=True, slots=True)
class RapmFitConfig:
    """Frozen deterministic RAPM optimizer settings."""

    ridge_lambda: float = 0.01
    learning_rate: float = 1.0
    epochs: int = 120
    batch_size: int = 8_192
    seed: int = 20_260_718

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or self.epochs <= 0:
            raise NbaRapmError("epochs must be a positive integer")
        if isinstance(self.batch_size, bool) or self.batch_size <= 0:
            raise NbaRapmError("batch_size must be a positive integer")
        for name, value in (
            ("ridge_lambda", self.ridge_lambda),
            ("learning_rate", self.learning_rate),
        ):
            if not isfinite(value) or value <= 0.0:
                raise NbaRapmError(f"{name} must be positive and finite")


@dataclass(frozen=True, slots=True)
class RapmRatings:
    """Fitted per-100-possession player ratings with the free home advantage."""

    ratings: dict[int, float]
    home_advantage: float
    stint_count: int
    player_count: int

    def rating_for(self, player_id: int) -> float:
        """Return one player's rating, zero for genuinely unseen players."""
        return self.ratings.get(player_id, 0.0)


def fit_rapm_ratings(
    stints: Sequence[StintRecord],
    config: RapmFitConfig | None = None,
) -> RapmRatings:
    """Fit RAPM over one stint sample with a deterministic seeded optimizer."""
    settings = config or RapmFitConfig()
    observations = _prepare(stints)
    if not observations:
        raise NbaRapmError("RAPM requires at least one usable stint")
    players = sorted({player for obs in observations for player in obs.home + obs.away})
    index = {player: position for position, player in enumerate(players)}
    beta = [0.0] * len(players)
    home_advantage = 0.0
    state = _FitState(
        observations=observations,
        index=index,
        total_weight=sum(obs.weight for obs in observations),
        settings=settings,
    )
    for epoch in range(settings.epochs):
        order = list(range(len(observations)))
        Random(settings.seed + epoch).shuffle(order)
        for start in range(0, len(order), settings.batch_size):
            batch = order[start : start + settings.batch_size]
            home_advantage, beta = _batch_step(state, batch, beta, home_advantage)
    ratings = {player: beta[index[player]] for player in players}
    return RapmRatings(
        ratings=ratings,
        home_advantage=home_advantage,
        stint_count=len(observations),
        player_count=len(players),
    )


def fit_season_ratings(
    season_files: Mapping[int, Path],
    season: int,
    config: RapmFitConfig | None = None,
    failures: list[str] | None = None,
) -> RapmRatings:
    """Fit causal RAPM for one season using only the three strictly earlier seasons."""
    stints: list[StintRecord] = []
    for prior in range(season - 3, season):
        path = season_files.get(prior)
        if path is None or not path.exists():
            continue
        for game in read_pbp_games(path, failures):
            stints.extend(game.stints)
    return fit_rapm_ratings(stints, config)


def fit_season_ratings_by_name(
    season_files: Mapping[int, Path],
    season: int,
    config: RapmFitConfig | None = None,
    failures: list[str] | None = None,
) -> dict[str, float]:
    """Fit causal RAPM and key ratings by normalized player name.

    Name keying bridges data sources with different player ID spaces (stats.nba.com IDs versus
    ESPN athlete IDs); collisions merge into the more extreme rating and are rare enough to
    ignore at team level.
    """
    local_failures: list[str] = failures if failures is not None else []
    ratings = fit_season_ratings(season_files, season, config, local_failures)
    names: dict[int, str] = {}
    for prior in range(season - 3, season):
        path = season_files.get(prior)
        if path is None or not path.exists():
            continue
        for game in read_pbp_games(path, local_failures):
            names.update(game.player_names)
    by_name: dict[str, float] = {}
    for player_id, rating in ratings.ratings.items():
        name = names.get(player_id)
        if name is None:
            continue
        key = " ".join(normalize_player_name(name))
        if key not in by_name or abs(rating) > abs(by_name[key]):
            by_name[key] = rating
    return by_name


def _prepare(stints: Sequence[StintRecord]) -> list[_Observation]:
    observations: list[_Observation] = []
    for stint in stints:
        possessions = (stint.home_possessions + stint.away_possessions) / 2.0
        if possessions < _MIN_STINT_POSSESSIONS:
            continue
        margin = 100.0 * (stint.home_points - stint.away_points) / possessions
        observations.append(
            _Observation(
                home=stint.home_players,
                away=stint.away_players,
                margin=margin,
                weight=possessions,
            )
        )
    return observations


@dataclass(frozen=True, slots=True)
class _Observation:
    """One usable stint prepared for the optimizer."""

    home: tuple[int, ...]
    away: tuple[int, ...]
    margin: float
    weight: float


class _FitState:
    """Immutable fit context shared by every optimizer step."""

    def __init__(
        self,
        observations: list[_Observation],
        index: dict[int, int],
        total_weight: float,
        settings: RapmFitConfig,
    ) -> None:
        self.observations = observations
        self.index = index
        self.total_weight = total_weight
        self.settings = settings


def _batch_step(
    state: _FitState,
    batch: list[int],
    beta: list[float],
    home_advantage: float,
) -> tuple[float, list[float]]:
    grad_h = 0.0
    grad = [0.0] * len(beta)
    for position in batch:
        observation = state.observations[position]
        predicted = (
            home_advantage
            + sum(beta[state.index[p]] for p in observation.home)
            - sum(beta[state.index[p]] for p in observation.away)
        )
        residual = observation.margin - predicted
        grad_h -= 2.0 * observation.weight * residual / state.total_weight
        for player in observation.home:
            grad[state.index[player]] -= 2.0 * observation.weight * residual / state.total_weight
        for player in observation.away:
            grad[state.index[player]] += 2.0 * observation.weight * residual / state.total_weight
    player_count = len(beta)
    updated = [
        value
        - state.settings.learning_rate
        * (grad[position] + 2.0 * state.settings.ridge_lambda * value / player_count)
        for position, value in enumerate(beta)
    ]
    return home_advantage - state.settings.learning_rate * grad_h, updated
