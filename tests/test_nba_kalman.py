"""Tests for the DARKO-lite Kalman player-rating filter."""

from datetime import UTC, date, datetime, timedelta
from random import Random

from forecastfm.nba_arenas import NbaArena
from forecastfm.nba_kalman import KalmanConfig, ratings_at
from forecastfm.nba_pbp import PbpGame, PlayerGameLine, TeamGameStats, normalize_player_name
from forecastfm.nba_season_games import SeasonGame

ARENA = NbaArena("Test Arena", 40.75, -73.99, "America/New_York", "test fixture")
TRACKED_PLAYER = 1
TRACKED_SECONDS = 7200  # half the game: 50 estimated possessions per 100 team possessions


def _stats(abbreviation: str) -> TeamGameStats:
    return TeamGameStats(
        team_abbreviation=abbreviation,
        points=100,
        field_goals_attempted=90,
        free_throws_attempted=0,
        offensive_rebounds=0,
        turnovers=10,
        starters=(1, 2, 3, 4, 5),
    )


def _season_game(
    game_id: int,
    season_label: int,
    day: date,
    tracked_plus_minus: int,
    player_names: dict[int, str] | None = None,
) -> SeasonGame:
    tipoff = datetime(day.year, day.month, day.day, 19, 30, tzinfo=UTC)
    away_lines = (
        PlayerGameLine(TRACKED_PLAYER, "BOS", TRACKED_SECONDS, tracked_plus_minus),
        *(PlayerGameLine(pid, "BOS", 1440, 0) for pid in (2, 3, 4, 5)),
    )
    home_lines = tuple(PlayerGameLine(pid, "NYK", 2880, 0) for pid in range(11, 16))
    lines = away_lines + home_lines
    pbp = PbpGame(
        game_id=game_id,
        away_abbreviation="BOS",
        home_abbreviation="NYK",
        away_score=100,
        home_score=110,
        team_stats=(_stats("BOS"), _stats("NYK")),
        player_lines=lines,
        player_names=player_names or {line.player_id: f"Player {line.player_id}" for line in lines},
    )
    return SeasonGame(
        game_id=game_id,
        season_label=season_label,
        game_date=day,
        tipoff=tipoff,
        away_abbreviation="BOS",
        home_abbreviation="NYK",
        away_score=100,
        home_score=110,
        arena=ARENA,
        pbp=pbp,
    )


def _season(
    season_label: int,
    plus_minus: list[int],
    start: date = date(2021, 10, 19),
) -> list[SeasonGame]:
    base_id = (season_label - 2000 - 1) * 100000 + 20000
    return [
        _season_game(base_id + index, season_label, start + timedelta(days=2 * index), pm)
        for index, pm in enumerate(plus_minus)
    ]


def _tracked_ratings(
    games: list[SeasonGame],
    prior_rapm: dict[int, dict[int, float]] | None = None,
) -> list[float]:
    ratings = ratings_at({games[0].season_label: games}, prior_rapm=prior_rapm)
    return [ratings.by_id[(game.game_id, "BOS")][TRACKED_PLAYER] for game in games]


def test_filter_recovers_constant_rating_within_observation_noise() -> None:
    true_rating = 5.0
    played_possessions = 50.0
    noise_std = 20.0 / played_possessions**0.5  # per-game observation noise, per 100
    random = Random(20_260_720)
    plus_minus = [
        round((true_rating + random.gauss(0.0, noise_std)) * played_possessions / 100.0)
        for _ in range(80)
    ]
    games = _season(2022, plus_minus)
    estimates = _tracked_ratings(games)
    assert estimates[0] == 0.0  # no prior: pregame mean of the opener
    assert abs(estimates[-1] - true_rating) < noise_std


def test_filter_reacts_to_level_shift_faster_than_ten_game_window() -> None:
    games = _season(2022, [0] * 30 + [5] * 30)  # pm 5 over 50 poss = +10 per 100
    estimates = _tracked_ratings(games)
    shift = 30

    def kalman_games_to_cross() -> int:
        for observed in range(1, 31):
            if estimates[shift + observed] >= 5.0:
                return observed
        raise AssertionError("Kalman estimate never crossed half the shift")

    def window_games_to_cross() -> int:
        for observed in range(1, 31):
            window = [0.0] * (10 - min(observed, 10)) + [10.0] * min(observed, 10)
            if sum(window) / len(window) >= 5.0:
                return observed
        raise AssertionError("ten-game window never crossed half the shift")

    assert kalman_games_to_cross() < window_games_to_cross()


def test_season_start_resets_to_prior_rapm() -> None:
    first = _season(2022, [5] * 30)
    second = _season(2023, [0], start=date(2022, 10, 18))
    ratings = ratings_at(
        {2022: first, 2023: second},
        prior_rapm={2023: {TRACKED_PLAYER: 7.5}},
    )
    converged = ratings.by_id[(first[-1].game_id, "BOS")][TRACKED_PLAYER]
    assert converged > 5.0  # in-season filtering moved far from the prior
    assert ratings.by_id[(second[0].game_id, "BOS")][TRACKED_PLAYER] == 7.5


def test_season_start_without_prior_resets_to_zero() -> None:
    first = _season(2022, [5] * 30)
    second = _season(2023, [0], start=date(2022, 10, 18))
    ratings = ratings_at({2022: first, 2023: second})
    assert ratings.by_id[(second[0].game_id, "BOS")][TRACKED_PLAYER] == 0.0


def test_refit_is_deterministic() -> None:
    random = Random(7)
    games = {2022: _season(2022, [random.randint(-6, 6) for _ in range(40)])}
    priors = {2022: {TRACKED_PLAYER: 1.25}}
    config = KalmanConfig()
    assert ratings_at(games, priors, config) == ratings_at(games, priors, config)


def test_output_is_keyed_by_game_team_and_normalized_name() -> None:
    names = {pid: f"Player {pid}" for pid in (*range(2, 6), *range(11, 16))}
    names[TRACKED_PLAYER] = "Jayson Tatum"
    game = _season_game(22100001, 2022, date(2021, 10, 19), 4, player_names=names)
    ratings = ratings_at({2022: [game]}, prior_rapm={2022: {TRACKED_PLAYER: 3.0}})
    assert set(ratings.by_id) == {(22100001, "BOS"), (22100001, "NYK")}
    assert set(ratings.by_name) == set(ratings.by_id)
    away = ratings.by_id[(22100001, "BOS")]
    assert away[TRACKED_PLAYER] == 3.0
    away_named = ratings.by_name[(22100001, "BOS")]
    assert away_named["jayson tatum"] == 3.0
    expected = {" ".join(normalize_player_name(names[pid])) for pid in (1, 2, 3, 4, 5)}
    assert set(away_named) == expected
