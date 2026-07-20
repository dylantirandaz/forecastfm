"""Tests for the T-15 forecast-horizon diagnostic."""

from datetime import UTC, date, datetime, timedelta

import pytest
from examples.run_t15_diagnostic import (
    HorizonInputs,
    compute_horizon_features,
    horizon_feature_vector,
    select_snapshot,
)

from forecastfm.nba_arenas import home_arena
from forecastfm.nba_feature_builder import InjurySnapshot
from forecastfm.nba_injury_report import ET_ZONE, InjuryReportRow
from forecastfm.nba_prototype_dataset import PrototypeGameRow
from forecastfm.nba_season_games import SeasonGame
from tests.test_nba_season_games import pbp_game_fixture

GAME_DATE = date(2026, 1, 10)
TIPOFF = datetime(2026, 1, 10, 19, 30, tzinfo=ET_ZONE).astimezone(UTC)


def _season_game(game_id: int, day: date) -> SeasonGame:
    tipoff = datetime(day.year, day.month, day.day, 19, 30, tzinfo=ET_ZONE).astimezone(UTC)
    return SeasonGame(
        game_id=game_id,
        season_label=2026,
        game_date=day,
        tipoff=tipoff,
        away_abbreviation="BOS",
        home_abbreviation="NYK",
        away_score=100,
        home_score=110,
        arena=home_arena("NYK", tipoff),
        pbp=pbp_game_fixture(game_id),
    )


def _snapshot(
    report_time: datetime,
    matchup: str = "BOS@NYK",
    team: str = "New York Knicks",
    player: str = "Player 11",
    status: str = "Out",
) -> InjurySnapshot:
    row = InjuryReportRow(
        report_time=report_time,
        game_date=GAME_DATE,
        game_clock_et=(19, 30),
        matchup=matchup,
        team=team,
        player_name=player,
        status=status,
    )
    return InjurySnapshot(report_time=report_time, rows=(row,))


def _et(hour: int, minute: int) -> datetime:
    return datetime(2026, 1, 10, hour, minute, tzinfo=ET_ZONE)


def test_select_snapshot_prefers_exact_cutoff_over_one_minute_earlier() -> None:
    game = _season_game(1, GAME_DATE)
    cutoff = TIPOFF - timedelta(minutes=15)
    at_cutoff = _snapshot(cutoff.astimezone(ET_ZONE))
    earlier = _snapshot((cutoff - timedelta(minutes=1)).astimezone(ET_ZONE))
    later = _snapshot((cutoff + timedelta(minutes=1)).astimezone(ET_ZONE))
    snapshots = sorted((earlier, at_cutoff, later), key=lambda snapshot: snapshot.report_time)
    assert select_snapshot(snapshots, game, 15) is at_cutoff


def test_select_snapshot_falls_back_when_later_snapshot_lacks_matchup() -> None:
    game = _season_game(1, GAME_DATE)
    containing = _snapshot(_et(18, 0))
    other_game = _snapshot(_et(19, 0), matchup="LAL@DEN", team="Los Angeles Lakers")
    assert select_snapshot([containing, other_game], game, 15) is containing


def test_select_snapshot_returns_none_when_all_snapshots_postdate_cutoff() -> None:
    game = _season_game(1, GAME_DATE)
    snapshots = [_snapshot(_et(19, 16)), _snapshot(_et(19, 20))]
    assert select_snapshot(snapshots, game, 15) is None


def test_horizon_features_end_to_end() -> None:
    days = [date(2026, 1, 5), date(2026, 1, 7), date(2026, 1, 9), GAME_DATE]
    games = [_season_game(game_id, day) for game_id, day in enumerate(days, start=1)]
    elo = {
        (game.game_id, team): rating
        for game in games
        for team, rating in (("BOS", 1500.0), ("NYK", 1520.0))
    }
    early = _snapshot(_et(18, 0), player="Player 12", status="Questionable")
    late = _snapshot(_et(19, 0), player="Player 11", status="Out")
    ratings = {"11 player": 2.0, "12 player": 2.0}
    notes: list[str] = []
    inputs = HorizonInputs(snapshots=(early, late), player_ratings=ratings, notes=notes)
    derived = compute_horizon_features(games, elo, inputs)

    report60 = derived[games[3].game_id][60]
    assert report60 is not None
    (away60, home60), projected60 = report60
    assert away60 == (0.0, 0.0)
    assert home60 == (0.0, 0.0)
    assert projected60[0] == 0.0
    assert projected60[1] == pytest.approx(0.4)

    report15 = derived[games[3].game_id][15]
    assert report15 is not None
    (away15, home15), projected15 = report15
    assert away15 == (0.0, 0.0)
    assert home15 == (24.0, 48.0)
    assert projected15[0] == 0.0
    assert projected15[1] == pytest.approx(0.2)

    # The T-30 cutoff (19:00 ET) admits the late snapshot exactly, so T-30 matches T-15.
    assert derived[games[3].game_id][30] == report15

    row = PrototypeGameRow(
        question_id="nba-4",
        game_id=games[3].game_id,
        season=2026,
        game_date=GAME_DATE,
        elo_home_probability=0.6,
        features_standard=tuple(float(index) for index in range(11)),
        features_health=None,
        home_won=True,
    )
    vector = horizon_feature_vector(row, report15)
    assert vector[:11] == row.features_standard
    assert vector[11] == 24.0
    assert vector[12] == 48.0
    assert vector[13] == pytest.approx(0.2)
    assert horizon_feature_vector(row, None) == (*row.features_standard, 0.0, 0.0, 0.0)
