"""Tests for the season-game join and the per-team rolling feature state."""

from datetime import UTC, date, datetime

import pytest

from forecastfm.nba_arenas import home_arena
from forecastfm.nba_pbp import PbpGame, PlayerGameLine, TeamGameStats
from forecastfm.nba_season_games import (
    ScheduleEntry,
    join_season_games,
)
from forecastfm.nba_team_history import GameContext, NbaTeamHistory

TIPOFF = datetime(2022, 1, 10, 0, 30, tzinfo=UTC)


def pbp_game_fixture(
    game_id: int,
    away: str = "BOS",
    home: str = "NYK",
    away_score: int = 100,
    home_score: int = 110,
) -> PbpGame:
    def stats(abbr: str, points: int, starters: tuple[int, ...]) -> TeamGameStats:
        return TeamGameStats(
            team_abbreviation=abbr,
            points=points,
            field_goals_attempted=80,
            free_throws_attempted=20,
            offensive_rebounds=10,
            turnovers=12,
            starters=starters,
        )

    lines = tuple(
        PlayerGameLine(
            player_id=player_id,
            team_abbreviation=team,
            seconds_played=2880 * 5 // 10,
            plus_minus=5 if team == home else -5,
        )
        for team, ids in (
            (away, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
            (home, (11, 12, 13, 14, 15, 16, 17, 18, 19, 20)),
        )
        for player_id in ids
    )
    return PbpGame(
        game_id=game_id,
        away_abbreviation=away,
        home_abbreviation=home,
        away_score=away_score,
        home_score=home_score,
        team_stats=(
            stats(away, away_score, (1, 2, 3, 4, 5)),
            stats(home, home_score, (11, 12, 13, 14, 15)),
        ),
        player_lines=lines,
        player_names={line.player_id: f"Player {line.player_id}" for line in lines},
    )


def schedule_entry_fixture(day: date, away: str = "BOS", home: str = "NYK") -> ScheduleEntry:
    return ScheduleEntry(
        game_date=day, away_abbreviation=away, home_abbreviation=home, tip_clock=(19, 30)
    )


def test_join_pairs_games_in_id_order() -> None:
    games = [pbp_game_fixture(22100002), pbp_game_fixture(22100001)]
    schedule = [
        schedule_entry_fixture(date(2021, 10, 20)),
        schedule_entry_fixture(date(2021, 10, 19)),
    ]
    joined, notes = join_season_games(games, schedule)
    assert notes == []
    assert [game.game_id for game in joined] == [22100001, 22100002]
    assert joined[0].game_date == date(2021, 10, 19)
    assert joined[0].home_won is True
    assert joined[0].arena.arena_name == "Madison Square Garden"


def test_join_excludes_pair_count_mismatch() -> None:
    joined, notes = join_season_games([pbp_game_fixture(22100001)], [])
    assert joined == []
    assert len(notes) == 1
    assert "excluded" in notes[0]


def _context(day: date, home: bool = True) -> GameContext:
    tipoff = datetime(day.year, day.month, day.day, 19, 30, tzinfo=UTC)
    arena = home_arena("NYK" if home else "BOS", tipoff)
    return GameContext(game_date=day, tipoff=tipoff, home=home, arena=arena)


def _record(history: NbaTeamHistory, day: date, home: bool = True, elo: float = 1500.0) -> None:
    history.record_game(pbp_game_fixture(22100001), _context(day, home), elo)


def test_opener_defaults() -> None:
    history = NbaTeamHistory("NYK")
    features = history.features_for(_context(date(2021, 10, 19)))
    assert features.rest_days == 0.0
    assert features.back_to_back == 0.0
    assert features.games_last_7 == 0.0
    assert features.travel_miles == 0.0
    assert features.roster_continuity == 1.0
    assert features.expected_lineup_continuity == 1.0
    assert features.rolling_team_net_rating == 0.0
    assert features.schedule_strength == 1500.0


def test_rest_and_back_to_back() -> None:
    history = NbaTeamHistory("NYK")
    _record(history, date(2021, 10, 19))
    assert history.features_for(_context(date(2021, 10, 20))).back_to_back == 1.0
    assert history.features_for(_context(date(2021, 10, 20))).rest_days == 0.0
    assert history.features_for(_context(date(2021, 10, 22))).rest_days == 2.0


def test_games_last_7_counts_tipoffs() -> None:
    history = NbaTeamHistory("NYK")
    for offset in (0, 1, 2, 3, 4, 8):
        _record(history, date(2021, 10, 19 + offset) if offset < 5 else date(2021, 10, 27))
    features = history.features_for(_context(date(2021, 10, 28)))
    assert features.games_last_7 == 4.0
    assert features.road_games_last_7 == 0.0


def test_rolling_net_rating_and_schedule_strength() -> None:
    history = NbaTeamHistory("NYK")
    _record(history, date(2021, 10, 19), elo=1600.0)
    features = history.features_for(_context(date(2021, 10, 20)))
    expected_rating = 10 / (80 + 0.44 * 20 - 10 + 12) * 100.0
    assert features.rolling_team_net_rating == pytest.approx(expected_rating)
    assert features.schedule_strength == 1600.0


def test_lineup_continuity_full_overlap() -> None:
    history = NbaTeamHistory("NYK")
    _record(history, date(2021, 10, 19))
    _record(history, date(2021, 10, 21))
    features = history.features_for(_context(date(2021, 10, 23)))
    assert features.expected_lineup_continuity == 1.0
    assert 0.0 < features.roster_continuity <= 1.0
