"""Tests for the RAPM ridge fit over stints."""

from random import Random

import pytest

from forecastfm.nba_pbp import StintRecord
from forecastfm.nba_rapm import NbaRapmError, RapmFitConfig, fit_rapm_ratings

TRUE_RATINGS = {1: 8.0, 2: 4.0, 3: 0.0, 4: -4.0, 5: -8.0, 6: 6.0, 7: 2.0, 8: -2.0, 9: -6.0, 10: 1.0}
TRUE_HOME_ADVANTAGE = 3.0


def _synthetic_stints(count: int, noise_scale: float, seed: int) -> list[StintRecord]:
    random = Random(seed)
    players = sorted(TRUE_RATINGS)
    stints: list[StintRecord] = []
    for _ in range(count):
        random.shuffle(players)
        home = tuple(sorted(players[:5]))
        away = tuple(sorted(players[5:]))
        possessions = 5.0
        expected = (
            TRUE_HOME_ADVANTAGE
            + sum(TRUE_RATINGS[p] for p in home)
            - sum(TRUE_RATINGS[p] for p in away)
        )
        margin = expected + random.gauss(0.0, noise_scale)
        points = margin * possessions / 100.0
        home_points = round(max(points, 0.0))
        away_points = round(max(-points, 0.0))
        stints.append(
            StintRecord(
                home_players=home,
                away_players=away,
                seconds=120,
                home_points=home_points,
                away_points=away_points,
                home_possessions=possessions,
                away_possessions=possessions,
            )
        )
    return stints


def test_recovers_rating_order_and_home_advantage() -> None:
    stints = _synthetic_stints(4_000, noise_scale=10.0, seed=7)
    ratings = fit_rapm_ratings(stints, RapmFitConfig(learning_rate=0.5, epochs=40))
    assert ratings.home_advantage == pytest.approx(TRUE_HOME_ADVANTAGE, abs=1.0)
    ordered = sorted(TRUE_RATINGS, key=ratings.rating_for)
    assert ordered == sorted(TRUE_RATINGS, key=lambda p: TRUE_RATINGS[p])
    assert ratings.rating_for(1) > ratings.rating_for(5)
    assert ratings.rating_for(1) > 0.0 > ratings.rating_for(10 - 1)


def test_deterministic_across_runs() -> None:
    stints = _synthetic_stints(500, noise_scale=15.0, seed=11)
    first = fit_rapm_ratings(stints, RapmFitConfig(learning_rate=0.5, epochs=5))
    second = fit_rapm_ratings(stints, RapmFitConfig(learning_rate=0.5, epochs=5))
    assert first.ratings == second.ratings
    assert first.home_advantage == second.home_advantage


def test_unseen_player_rating_is_zero() -> None:
    stints = _synthetic_stints(100, noise_scale=5.0, seed=3)
    ratings = fit_rapm_ratings(stints, RapmFitConfig(learning_rate=0.5, epochs=3))
    assert ratings.rating_for(999) == 0.0


def test_rejects_empty_input_and_bad_config() -> None:
    with pytest.raises(NbaRapmError):
        fit_rapm_ratings([], RapmFitConfig())
    with pytest.raises(NbaRapmError):
        RapmFitConfig(epochs=0)
    with pytest.raises(NbaRapmError):
        RapmFitConfig(ridge_lambda=-1.0)


def test_filters_tiny_possession_stints() -> None:
    stints = [
        StintRecord(
            home_players=(1, 2, 3, 4, 5),
            away_players=(6, 7, 8, 9, 10),
            seconds=10,
            home_points=2,
            away_points=0,
            home_possessions=0.1,
            away_possessions=0.1,
        )
    ]
    with pytest.raises(NbaRapmError):
        fit_rapm_ratings(stints, RapmFitConfig())
