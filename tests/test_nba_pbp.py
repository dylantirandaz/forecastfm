"""Tests for the play-by-play derivation module using synthetic event streams."""

from dataclasses import dataclass
from pathlib import Path

from forecastfm.nba_pbp import read_pbp_games

GAME_ID = 22100001
HEADER = (
    "GAME_ID,EVENTNUM,EVENTMSGTYPE,EVENTMSGACTIONTYPE,PERIOD,WCTIMESTRING,PCTIMESTRING,"
    "HOMEDESCRIPTION,NEUTRALDESCRIPTION,VISITORDESCRIPTION,SCORE,SCOREMARGIN,PERSON1TYPE,"
    "PLAYER1_ID,PLAYER1_NAME,PLAYER1_TEAM_ID,PLAYER1_TEAM_CITY,PLAYER1_TEAM_NICKNAME,"
    "PLAYER1_TEAM_ABBREVIATION,PERSON2TYPE,PLAYER2_ID,PLAYER2_NAME,PLAYER2_TEAM_ID,"
    "PLAYER2_TEAM_CITY,PLAYER2_TEAM_NICKNAME,PLAYER2_TEAM_ABBREVIATION,PERSON3TYPE,"
    "PLAYER3_ID,PLAYER3_NAME,PLAYER3_TEAM_ID,PLAYER3_TEAM_CITY,PLAYER3_TEAM_NICKNAME,"
    "PLAYER3_TEAM_ABBREVIATION,VIDEO_AVAILABLE_FLAG"
)


@dataclass(frozen=True, slots=True)
class _Meta:
    num: int
    kind: int
    period: int
    clock: str


def _event(
    meta: _Meta,
    *,
    side: str = "",
    score: str = "",
    p1: tuple[int, str] = (0, ""),
    p2: tuple[int, str] = (0, ""),
) -> list[str]:
    row = [""] * 34
    row[0] = str(GAME_ID)
    row[1] = str(meta.num)
    row[2] = str(meta.kind)
    row[3] = "0"
    row[4] = str(meta.period)
    row[6] = meta.clock
    text = f"Player{p1[0]} Action" if p1[1] else ""
    row[7] = text if side == "home" else ""
    row[9] = text if side == "away" else ""
    row[10] = score
    row[13] = str(p1[0])
    row[15] = f"Player{p1[0]}"
    row[18] = p1[1]
    row[20] = str(p2[0])
    row[22] = f"Player{p2[0]}"
    row[25] = p2[1]
    return row


def _quiet_period(events: list[list[str]], period: int, starters: dict[str, list[int]]) -> None:
    num = len(events) * 10 + 1
    for team, players in starters.items():
        for index, player_id in enumerate(players):
            events.append(
                _event(
                    _Meta(num + index, 6 if index else 1, period, "11:00"),
                    side="home" if team == "HOM" else "away",
                    p1=(player_id, team),
                )
            )


def _write_csv(path: Path, events: list[list[str]]) -> None:
    body = "\n".join(",".join(event) for event in events)
    path.write_text(f"{HEADER}\n{body}\n", encoding="utf-8")


def _simple_game() -> list[list[str]]:
    home = [101, 102, 103, 104, 105]
    away = [201, 202, 203, 204, 205]
    events: list[list[str]] = []
    _quiet_period(events, 1, {"HOM": home, "AWY": away})
    events.append(_event(_Meta(90, 1, 1, "10:00"), side="home", score="0 - 2", p1=(101, "HOM")))
    for period in (2, 3, 4):
        _quiet_period(events, period, {"HOM": home, "AWY": away})
    events.append(_event(_Meta(91, 1, 4, "00:00"), side="away", score="2 - 2", p1=(201, "AWY")))
    events.append(_event(_Meta(92, 1, 4, "00:00"), side="home", score="2 - 5", p1=(102, "HOM")))
    return events


def test_parse_simple_game(tmp_path: Path) -> None:
    path = tmp_path / "season.csv"
    _write_csv(path, _simple_game())
    failures: list[str] = []
    (game,) = list(read_pbp_games(path, failures))
    assert failures == []
    assert game.game_id == GAME_ID
    assert game.home_abbreviation == "HOM"
    assert game.away_abbreviation == "AWY"
    assert (game.away_score, game.home_score) == (2, 5)
    assert game.season_label == 2022
    home_stats, away_stats = game.team_stats[1], game.team_stats[0]
    assert home_stats.points == 5
    assert len(home_stats.starters) == 5
    assert len(away_stats.starters) == 5
    totals: dict[str, int] = {}
    for line in game.player_lines:
        totals[line.team_abbreviation] = totals.get(line.team_abbreviation, 0) + line.seconds_played
    assert totals == {"HOM": 5 * 2880, "AWY": 5 * 2880}


def test_substitution_and_reentry_minutes(tmp_path: Path) -> None:
    events = _simple_game()
    events.insert(6, _event(_Meta(40, 8, 1, "10:00"), p1=(105, "HOM"), p2=(106, "HOM")))
    events.append(_event(_Meta(95, 8, 4, "05:00"), p1=(106, "HOM"), p2=(105, "HOM")))
    path = tmp_path / "season.csv"
    _write_csv(path, events)
    (game,) = list(read_pbp_games(path, []))
    minutes = {line.player_id: line.seconds_played for line in game.player_lines}
    assert minutes[105] == 120 + 720 + 720 + 300
    assert minutes[106] == 600 + 420


def test_plus_minus_tracks_score_changes(tmp_path: Path) -> None:
    path = tmp_path / "season.csv"
    _write_csv(path, _simple_game())
    (game,) = list(read_pbp_games(path, []))
    plus_minus = {line.player_id: line.plus_minus for line in game.player_lines}
    assert plus_minus[101] == 3
    assert plus_minus[201] == -3
    assert plus_minus[102] == 3


def test_overtime_period_uses_short_clock(tmp_path: Path) -> None:
    events = _simple_game()
    events[-2] = _event(_Meta(91, 1, 4, "00:00"), side="away", score="4 - 2", p1=(201, "AWY"))
    events[-1] = _event(_Meta(92, 1, 4, "00:00"), side="home", score="4 - 4", p1=(102, "HOM"))
    _quiet_period(events, 5, {"HOM": [101, 102, 103, 104, 105], "AWY": [201, 202, 203, 204, 205]})
    events.append(_event(_Meta(99, 1, 5, "00:00"), side="home", score="4 - 6", p1=(101, "HOM")))
    path = tmp_path / "season.csv"
    _write_csv(path, events)
    (game,) = list(read_pbp_games(path, []))
    totals: dict[str, int] = {}
    for line in game.player_lines:
        totals[line.team_abbreviation] = totals.get(line.team_abbreviation, 0) + line.seconds_played
    assert totals == {"HOM": 5 * 3180, "AWY": 5 * 3180}
    assert game.home_score == 6


def test_carryover_fills_invisible_period_starter(tmp_path: Path) -> None:
    events = _simple_game()
    _quiet_period(events, 5, {"HOM": [101, 102, 103, 104], "AWY": [201, 202, 203, 204, 205]})
    events.append(_event(_Meta(99, 1, 5, "00:00"), side="home", score="2 - 7", p1=(101, "HOM")))
    path = tmp_path / "season.csv"
    _write_csv(path, events)
    (game,) = list(read_pbp_games(path, []))
    assert game.home_score == 7
    minutes = {line.player_id: line.seconds_played for line in game.player_lines}
    assert minutes[105] == 2880 + 300


def test_non_regular_game_is_recorded(tmp_path: Path) -> None:
    events = _simple_game()
    for event in events:
        event[0] = "422100001"
    path = tmp_path / "season.csv"
    _write_csv(path, events)
    failures: list[str] = []
    games = list(read_pbp_games(path, failures))
    assert games == []
    assert len(failures) == 1
    assert "regular-season" in failures[0]
