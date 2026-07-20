"""Tests for the empirical injury-status play rates and the weighted health variant."""

from datetime import date, datetime
from pathlib import Path

import pytest

from forecastfm.nba_feature_builder import (
    InjurySnapshot,
    build_game_features,
    load_injury_index,
)
from forecastfm.nba_injury_report import ET_ZONE, InjuryReportRow
from forecastfm.nba_pbp import PbpGame, PlayerGameLine, TeamGameStats
from forecastfm.nba_season_games import SeasonGame, join_season_games
from forecastfm.nba_status_rates import (
    NbaStatusRatesError,
    StatusPlayRates,
    compute_status_play_rates,
)
from tests.test_nba_feature_builder import _row, _write_rows
from tests.test_nba_season_games import pbp_game_fixture, schedule_entry_fixture

DAY_ONE = date(2025, 11, 1)
DAY_TWO = date(2025, 11, 3)

# Listed on Boston in both games; the flags mark the games the player actually appears in
# (game one, game two). Rates: Out 0/2, Doubtful 1/4, Questionable 1/2, Probable 3/4,
# Available 4/4 (Adam on Boston plus Tony on Atlanta).
_FIXTURE_ROWS: tuple[tuple[str, str, str, tuple[bool, bool]], ...] = (
    ("Out, Oliver", "Out", "Boston Celtics", (False, False)),
    ("Doubtful, Danny", "Doubtful", "Boston Celtics", (True, False)),
    ("Doubtful, Doug", "Doubtful", "Boston Celtics", (False, False)),
    ("Questionable, Quinn", "Questionable", "Boston Celtics", (True, False)),
    ("Probable, Paul", "Probable", "Boston Celtics", (True, True)),
    ("Probable, Peter", "Probable", "Boston Celtics", (True, False)),
    ("Available, Adam", "Available", "Boston Celtics", (True, True)),
    ("Available, Tony", "Available", "Atlanta Hawks", (True, True)),
)

_EXPECTED_RATES = {
    "Out": 0.0,
    "Doubtful": 0.25,
    "Questionable": 0.5,
    "Probable": 0.75,
    "Available": 1.0,
}
_EXPECTED_COUNTS = {"Out": 2, "Doubtful": 4, "Questionable": 2, "Probable": 4, "Available": 4}


def _pbp(game_id: int, played: list[tuple[int, str, str]]) -> PbpGame:
    def stats(abbr: str, points: int) -> TeamGameStats:
        return TeamGameStats(
            team_abbreviation=abbr,
            points=points,
            field_goals_attempted=80,
            free_throws_attempted=20,
            offensive_rebounds=10,
            turnovers=12,
            starters=tuple(player_id for player_id, _, team in played if team == abbr)[:5],
        )

    lines = [
        PlayerGameLine(
            player_id=player_id,
            team_abbreviation=team,
            seconds_played=1200,
            plus_minus=0,
        )
        for player_id, _, team in played
    ]
    # A zero-second line for an unlisted player: never counts as an appearance.
    lines.append(
        PlayerGameLine(player_id=999, team_abbreviation="ATL", seconds_played=0, plus_minus=0)
    )
    names = {player_id: name for player_id, name, _ in played}
    names[999] = "Bench, Bernie"
    return PbpGame(
        game_id=game_id,
        away_abbreviation="ATL",
        home_abbreviation="BOS",
        away_score=100,
        home_score=110,
        team_stats=(stats("ATL", 100), stats("BOS", 110)),
        player_lines=tuple(lines),
        player_names=names,
    )


def _pbp_name(report_name: str) -> str:
    last, first = (part.strip() for part in report_name.split(","))
    return f"{first} {last}"


def _played_entries(
    spec: tuple[tuple[str, str, str, tuple[bool, bool]], ...],
    game_index: int,
) -> list[tuple[int, str, str]]:
    entries: list[tuple[int, str, str]] = []
    for player_id, (report_name, _, team_name, flags) in enumerate(spec, start=1):
        if flags[game_index]:
            team = "BOS" if team_name == "Boston Celtics" else "ATL"
            entries.append((player_id, _pbp_name(report_name), team))
    return entries


def _games_for(spec: tuple[tuple[str, str, str, tuple[bool, bool]], ...]) -> list[SeasonGame]:
    games = [_pbp(22600001, _played_entries(spec, 0)), _pbp(22600002, _played_entries(spec, 1))]
    schedule = [
        schedule_entry_fixture(DAY_ONE, away="ATL", home="BOS"),
        schedule_entry_fixture(DAY_TWO, away="ATL", home="BOS"),
    ]
    joined, notes = join_season_games(games, schedule)
    assert notes == []
    return joined


def _report_row(
    report_name: str,
    status: str,
    team: str,
    report_time: datetime,
    game_date: date,
) -> InjuryReportRow:
    return InjuryReportRow(
        report_time=report_time,
        game_date=game_date,
        game_clock_et=(19, 30),
        matchup="ATL@BOS",
        team=team,
        player_name=report_name,
        status=status,
    )


def _snapshots_for(
    spec: tuple[tuple[str, str, str, tuple[bool, bool]], ...],
) -> list[InjurySnapshot]:
    early = datetime(2025, 11, 1, 17, 30, tzinfo=ET_ZONE)
    stale_flipped = datetime(2025, 11, 1, 19, 0, tzinfo=ET_ZONE)
    second = datetime(2025, 11, 3, 17, 30, tzinfo=ET_ZONE)
    day_one_rows = [
        _report_row(name, status, team, early, DAY_ONE) for name, status, team, _ in spec
    ]
    # Past the T-60 cutoff for game one, so it must never be selected: it flips every
    # status to Available, and any selection bug would change the computed rates.
    flipped_rows = [
        _report_row(name, "Available", team, stale_flipped, DAY_ONE) for name, _, team, _ in spec
    ]
    day_two_rows = [
        _report_row(name, status, team, second, DAY_TWO) for name, status, team, _ in spec
    ]
    return [
        InjurySnapshot(report_time=early, rows=tuple(day_one_rows)),
        InjurySnapshot(report_time=stale_flipped, rows=tuple(flipped_rows)),
        InjurySnapshot(report_time=second, rows=tuple(day_two_rows)),
    ]


def _fixture_games() -> list[SeasonGame]:
    return _games_for(_FIXTURE_ROWS)


def _fixture_snapshots() -> list[InjurySnapshot]:
    return _snapshots_for(_FIXTURE_ROWS)


def test_compute_status_play_rates_recovers_known_fixture_rates() -> None:
    result = compute_status_play_rates(_fixture_snapshots(), _fixture_games())
    assert result.rates == _EXPECTED_RATES
    assert result.counts == _EXPECTED_COUNTS


def test_compute_status_play_rates_skips_games_without_pregame_report() -> None:
    games = _fixture_games()
    schedule = [schedule_entry_fixture(date(2025, 11, 5), away="ATL", home="BOS")]
    extra, notes = join_season_games([_pbp(22600003, _played_entries(_FIXTURE_ROWS, 0))], schedule)
    assert notes == []
    result = compute_status_play_rates(_fixture_snapshots(), [*games, *extra])
    assert result.rates == _EXPECTED_RATES
    assert result.counts == _EXPECTED_COUNTS


def test_compute_status_play_rates_accepts_available_below_probable() -> None:
    # Empirically Available sits below Probable (cleared bench players record DNP-CDs),
    # so only the injury-severity ladder plus Available > Questionable is enforced.
    spec: tuple[tuple[str, str, str, tuple[bool, bool]], ...] = (
        ("Out, Olaf", "Out", "Boston Celtics", (False, False)),
        ("Doubtful, Dave", "Doubtful", "Boston Celtics", (True, False)),
        ("Doubtful, Drew", "Doubtful", "Boston Celtics", (False, False)),
        ("Questionable, Quade", "Questionable", "Boston Celtics", (True, False)),
        ("Questionable, Quinton", "Questionable", "Boston Celtics", (True, False)),
        ("Probable, Parker", "Probable", "Boston Celtics", (True, True)),
        ("Available, Aaron", "Available", "Boston Celtics", (True, True)),
        ("Available, Ali", "Available", "Boston Celtics", (True, False)),
    )
    result = compute_status_play_rates(_snapshots_for(spec), _games_for(spec))
    assert result.rates == {
        "Out": 0.0,
        "Doubtful": 0.25,
        "Questionable": 0.5,
        "Probable": 1.0,
        "Available": 0.75,
    }


def test_compute_status_play_rates_rejects_missing_status_class() -> None:
    report_time = datetime(2025, 11, 1, 17, 30, tzinfo=ET_ZONE)
    rows = tuple(
        _report_row(name, status, team, report_time, DAY_ONE)
        for name, status, team, _ in _FIXTURE_ROWS
        if status != "Available"
    )
    snapshots = [InjurySnapshot(report_time=report_time, rows=rows)]
    with pytest.raises(NbaStatusRatesError, match="Available has no listings"):
        compute_status_play_rates(snapshots, _fixture_games())


def test_compute_status_play_rates_rejects_non_monotone_rates() -> None:
    report_time = datetime(2025, 11, 1, 17, 30, tzinfo=ET_ZONE)
    rows = (
        *(
            _report_row(name, status, team, report_time, DAY_ONE)
            for name, status, team, _ in _FIXTURE_ROWS
            if status != "Available"
        ),
        # The only Available player never appears, so Available ties Out at 0.0.
        _report_row("Absent, Abby", "Available", "Boston Celtics", report_time, DAY_ONE),
    )
    snapshots = [InjurySnapshot(report_time=report_time, rows=rows)]
    with pytest.raises(NbaStatusRatesError, match="not strictly increasing"):
        compute_status_play_rates(snapshots, _fixture_games()[:1])


def _weighted_games() -> list[SeasonGame]:
    games = [pbp_game_fixture(22100001), pbp_game_fixture(22100002)]
    schedule = [
        schedule_entry_fixture(date(2021, 10, 19)),
        schedule_entry_fixture(date(2021, 10, 20)),
    ]
    joined, notes = join_season_games(games, schedule)
    assert notes == []
    return joined


def _elo(joined: list[SeasonGame]) -> dict[tuple[int, str], float]:
    return {
        (game.game_id, team): 1500.0
        for game in joined
        for team in (game.away_abbreviation, game.home_abbreviation)
    }


def test_side_health_defaults_to_frozen_binary_rule(tmp_path: Path) -> None:
    day_two = date(2021, 10, 20)
    _write_rows(
        tmp_path,
        day_two,
        "a.rows.jsonl",
        [_row("Player 1", "Questionable", game_date=day_two.isoformat())],
    )
    joined = _weighted_games()
    features, _ = build_game_features(
        joined, _elo(joined), load_injury_index(tmp_path), {"1 player": 2.0}
    )
    second = features[1]
    assert second.health is not None
    away_minutes, away_value = second.health[0]
    # Questionable is fully available under the binary rule, so nothing is subtracted.
    assert away_minutes == 0.0
    assert away_value == 0.0


def test_side_health_status_rates_variant_weights_both_aggregates(tmp_path: Path) -> None:
    day_two = date(2021, 10, 20)
    _write_rows(
        tmp_path,
        day_two,
        "a.rows.jsonl",
        [_row("Player 1", "Questionable", game_date=day_two.isoformat())],
    )
    joined = _weighted_games()
    rates = StatusPlayRates(rates=dict(_EXPECTED_RATES), counts=dict(_EXPECTED_COUNTS))
    features, _ = build_game_features(
        joined, _elo(joined), load_injury_index(tmp_path), {"1 player": 2.0}, rates
    )
    second = features[1]
    assert second.health is not None
    away_minutes, away_value = second.health[0]
    # Player 1 has 24.0 expected minutes from game one; weight = 1 - 0.5.
    assert away_minutes == pytest.approx(12.0)
    assert away_value == pytest.approx(12.0 * 2.0)


def test_side_health_status_rates_none_matches_binary_out(tmp_path: Path) -> None:
    day_two = date(2021, 10, 20)
    _write_rows(
        tmp_path,
        day_two,
        "a.rows.jsonl",
        [
            _row(
                "Player 1",
                "Out",
                report_time="2021-10-20T17:30:00-04:00",
                game_date=day_two.isoformat(),
            )
        ],
    )
    joined = _weighted_games()
    features, _ = build_game_features(
        joined, _elo(joined), load_injury_index(tmp_path), {"1 player": 2.0}, None
    )
    second = features[1]
    assert second.health is not None
    away_minutes, away_value = second.health[0]
    assert away_minutes == pytest.approx(24.0)
    assert away_value == pytest.approx(24.0 * 2.0)
