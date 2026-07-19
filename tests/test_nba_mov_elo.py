"""Tests for the carryover margin-of-victory Elo replay."""

import pytest

from forecastfm.nba_mov_elo import (
    EloGameResult,
    MovEloRecipe,
    NbaMovEloError,
    replay_mov_elo,
)


def _game(
    game_id: int, home_score: int = 110, away_score: int = 100, neutral: bool = False
) -> EloGameResult:
    return EloGameResult(
        game_id=game_id,
        home_abbreviation="HOM",
        away_abbreviation="AWY",
        home_score=home_score,
        away_score=away_score,
        neutral=neutral,
    )


def test_equal_ratings_home_advantage() -> None:
    replay = replay_mov_elo([[_game(1)]])
    expected = 1.0 / (1.0 + 10.0 ** (-100.0 / 400.0))
    assert replay.home_probability(1) == pytest.approx(expected)
    assert replay.ratings[(1, "HOM")] == 1500.0
    assert replay.ratings[(1, "AWY")] == 1500.0


def test_neutral_site_zeroes_advantage() -> None:
    replay = replay_mov_elo([[_game(1, neutral=True)]])
    assert replay.home_probability(1) == pytest.approx(0.5)


def test_winner_rating_rises_loser_falls() -> None:
    replay = replay_mov_elo([[_game(1)]])
    assert replay.ratings[(1, "HOM")] == 1500.0
    ratings_after = replay_mov_elo([[_game(1)], [_game(2)]])
    assert ratings_after.ratings[(2, "HOM")] != 1500.0


def test_carryover_pulls_ratings_toward_mean() -> None:
    replay = replay_mov_elo([[_game(1)], [_game(2)]])
    expected_home = 1.0 / (1.0 + 10.0 ** (-0.25))
    multiplier = 2.3978952727983707 * 2.2 / (0.001 * 100.0 + 2.2)
    shift = 20.0 * multiplier * (1.0 - expected_home)
    carried = 1500.0 + 0.75 * shift
    assert replay.ratings[(2, "HOM")] == pytest.approx(carried)


def test_upset_moves_ratings_more_than_expected_win() -> None:
    replay = replay_mov_elo([[_game(1, home_score=130, away_score=90)], [_game(2)]])
    assert replay.ratings[(2, "HOM")] > 1500.0
    assert replay.ratings[(2, "AWY")] < 1500.0
    expected = 1.0 / (1.0 + 10.0 ** (-100.0 / 400.0))
    assert replay.home_probability(1) == pytest.approx(expected)


def test_rejects_invalid_games() -> None:
    with pytest.raises(NbaMovEloError):
        replay_mov_elo([[_game(1, home_score=100, away_score=100)]])
    with pytest.raises(NbaMovEloError):
        MovEloRecipe(carryover=0.0)
    replay = replay_mov_elo([[_game(1)]])
    with pytest.raises(NbaMovEloError):
        replay.home_probability(999)
