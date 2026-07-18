"""Tests for the per-game feature assembly layer."""

import json
from datetime import date
from pathlib import Path

from forecastfm.nba_feature_builder import (
    TEAM_NAME_TO_ABBREVIATION,
    build_game_features,
    load_injury_index,
    normalize_player_name,
)
from forecastfm.nba_season_games import SeasonGame, join_season_games
from tests.test_nba_season_games import pbp_game_fixture, schedule_entry_fixture


def test_normalize_player_name_handles_orders_and_accents() -> None:
    assert normalize_player_name("Collins, Zach") == normalize_player_name("Zach Collins")
    assert normalize_player_name("Dončić, Luka") == normalize_player_name("Luka Doncic")
    assert normalize_player_name("McConnell, T.J.") == normalize_player_name("TJ McConnell")
    assert normalize_player_name("Trent Jr., Gary") == normalize_player_name("Gary Trent Jr.")


def test_team_name_table_covers_thirty_teams() -> None:
    assert len(set(TEAM_NAME_TO_ABBREVIATION.values())) == 30


def _write_rows(root: Path, day: date, filename: str, rows: list[dict[str, object]]) -> None:
    day_dir = root / day.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    (day_dir / filename).write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(
    player: str,
    status: str,
    team: str = "Boston Celtics",
    report_time: str = "2021-10-19T17:30:00-04:00",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "report_time": report_time,
        "game_date": "2021-10-19",
        "game_clock_et": "19:30",
        "matchup": "BOS@NYK",
        "away_team": "BOS",
        "home_team": "NYK",
        "team": team,
        "player_name": player,
        "status": status,
    }


def test_load_injury_index_groups_by_date(tmp_path: Path) -> None:
    _write_rows(tmp_path, date(2021, 10, 19), "a.rows.jsonl", [_row("Smart, Marcus", "Out")])
    _write_rows(
        tmp_path,
        date(2021, 10, 19),
        "b.rows.jsonl",
        [_row("Smart, Marcus", "Out", report_time="2021-10-19T19:30:00-04:00")],
    )
    snapshots = load_injury_index(tmp_path)
    assert len(snapshots) == 2
    assert snapshots[0].report_time < snapshots[1].report_time
    assert snapshots[0].rows[0].game_date == date(2021, 10, 19)


def _season_game(game_id: int, day: date) -> SeasonGame:
    games = [pbp_game_fixture(game_id)]
    schedule = [schedule_entry_fixture(day)]
    joined, _ = join_season_games(games, schedule)
    return joined[0]


def test_build_game_features_computes_both_sides() -> None:
    day = date(2021, 10, 19)
    game = _season_game(22100001, day)
    elo = {(game.game_id, "BOS"): 1500.0, (game.game_id, "NYK"): 1520.0}
    features, notes = build_game_features([game], elo, [])
    assert notes == [f"game {game.game_id} has no report snapshot at or before its T-60 cutoff"]
    (entry,) = features
    assert entry.game_id == game.game_id
    assert entry.away.rest_days == 0.0
    assert entry.home.rest_days == 0.0
    assert entry.health is None


def test_health_aggregates_from_selected_snapshot(tmp_path: Path) -> None:
    day = date(2021, 10, 19)
    _write_rows(
        tmp_path,
        day,
        "a.rows.jsonl",
        [_row("Player 11", "Out", team="New York Knicks"), _row("Player 1", "Out")],
    )
    snapshots = load_injury_index(tmp_path)
    game = _season_game(22100001, day)
    elo = {(game.game_id, "BOS"): 1500.0, (game.game_id, "NYK"): 1520.0}
    later = _season_game(22100002, date(2021, 10, 21))
    later_game = SeasonGame(
        game_id=later.game_id,
        season_label=later.season_label,
        game_date=day,
        tipoff=game.tipoff,
        away_abbreviation=later.away_abbreviation,
        home_abbreviation=later.home_abbreviation,
        away_score=later.away_score,
        home_score=later.home_score,
        arena=later.arena,
        pbp=later.pbp,
    )
    elo[(later_game.game_id, "BOS")] = 1500.0
    elo[(later_game.game_id, "NYK")] = 1520.0
    features, notes = build_game_features([game, later_game], elo, snapshots)
    assert len(notes) == 2
    assert all("no play-by-play history" in note for note in notes)
    first, second = features
    assert first.health is not None
    away_minutes, away_value = first.health[0]
    assert away_minutes == 0.0
    assert away_value == 0.0
    assert second.health is not None
    home_minutes, home_value = second.health[1]
    assert home_minutes == 24.0
    assert home_value != 0.0
