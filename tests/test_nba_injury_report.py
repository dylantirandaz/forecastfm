"""Tests for the dependency-free injury-report availability core."""

from datetime import date, datetime

import pytest

from forecastfm.nba_evidence import NbaEvidenceError
from forecastfm.nba_injury_report import (
    ET_ZONE,
    InjuryReportRow,
    matchup_teams,
    parse_game_clock,
    parse_game_date,
    parse_report_header_time,
    rows_from_report_records,
    select_report_at_cutoff,
    unavailable_rotation_minutes,
    unavailable_rotation_value,
)

REPORT_TIME = datetime(2026, 1, 7, 17, 30, tzinfo=ET_ZONE)


def _record(
    status: object = "Out",
    player: object = "Player A",
    team: object = "BOS",
    game_time: object = "07:30 (ET)",
    game_date: object = "01/07/2026",
) -> dict[str, object]:
    return {
        "Game Date": game_date,
        "Game Time": game_time,
        "Matchup": "BOS@NYK",
        "Team": team,
        "Player Name": player,
        "Current Status": status,
        "Reason": "Injury/Illness - Test; Reason",
    }


def _row(status: str = "Out", player: str = "Player A", team: str = "BOS") -> InjuryReportRow:
    return InjuryReportRow(
        report_time=REPORT_TIME,
        game_date=date(2026, 1, 7),
        game_clock_et=(19, 30),
        matchup="BOS@NYK",
        team=team,
        player_name=player,
        status=status,
    )


def test_parse_game_clock_converts_evening_times() -> None:
    assert parse_game_clock("07:00 (ET)") == (19, 0)
    assert parse_game_clock("09:45 (ET)") == (21, 45)
    assert parse_game_clock("12:30 (ET)") == (12, 30)
    assert parse_game_clock("01:00 (ET)") == (13, 0)


def test_parse_game_clock_rejects_malformed_values() -> None:
    for raw in ["7 PM", "13:00 (ET)", "07:75 (ET)", "", "ab:cd"]:
        with pytest.raises(NbaEvidenceError):
            parse_game_clock(raw)


def test_parse_game_date() -> None:
    assert parse_game_date("03/06/2019") == date(2019, 3, 6)
    with pytest.raises(NbaEvidenceError):
        parse_game_date("2019-03-06")


def test_parse_report_header_time() -> None:
    text = "Injury Report: 03/06/19 05:30 PM\nGame Date Game Time Matchup"
    assert parse_report_header_time(text) == datetime(2019, 3, 6, 17, 30, tzinfo=ET_ZONE)
    with pytest.raises(NbaEvidenceError):
        parse_report_header_time("no header here")


def test_matchup_teams() -> None:
    assert matchup_teams("DAL@WAS") == ("DAL", "WAS")
    for raw in ["DALWAS", "A@B@C", "@WAS", "DAL@"]:
        with pytest.raises(NbaEvidenceError):
            matchup_teams(raw)


def test_rows_from_report_records_builds_validated_rows() -> None:
    parsed = rows_from_report_records([_record()], REPORT_TIME)
    assert parsed.dropped_rows == 0
    (row,) = parsed.rows
    assert row.status == "Out"
    assert row.game_tipoff() == datetime(2026, 1, 7, 19, 30, tzinfo=ET_ZONE)
    assert row.report_time == REPORT_TIME


def test_rows_from_report_records_drops_unknown_statuses() -> None:
    records = [_record(status="Out"), _record(status=None), _record(status="Suspended")]
    parsed = rows_from_report_records(records, REPORT_TIME)
    assert len(parsed.rows) == 1
    assert parsed.dropped_rows == 2


def test_rows_from_report_records_rejects_missing_structure() -> None:
    with pytest.raises(NbaEvidenceError):
        rows_from_report_records([_record(player="")], REPORT_TIME)
    with pytest.raises(NbaEvidenceError):
        rows_from_report_records([_record(game_time="noon")], REPORT_TIME)


def test_select_report_at_cutoff_picks_latest_eligible() -> None:
    tipoff = datetime(2026, 1, 7, 19, 30, tzinfo=ET_ZONE)
    times = [
        datetime(2026, 1, 7, 11, 30, tzinfo=ET_ZONE),
        datetime(2026, 1, 7, 17, 30, tzinfo=ET_ZONE),
        datetime(2026, 1, 7, 18, 30, tzinfo=ET_ZONE),
        datetime(2026, 1, 7, 19, 0, tzinfo=ET_ZONE),
    ]
    selected = select_report_at_cutoff(tipoff, times)
    assert selected == datetime(2026, 1, 7, 18, 30, tzinfo=ET_ZONE)
    early = select_report_at_cutoff(tipoff, times, cutoff_minutes=360)
    assert early == datetime(2026, 1, 7, 11, 30, tzinfo=ET_ZONE)
    assert select_report_at_cutoff(tipoff, []) is None


def test_unavailable_rotation_minutes_sums_out_and_doubtful() -> None:
    rows = [
        _row(status="Out", player="Player A"),
        _row(status="Doubtful", player="Player B"),
        _row(status="Questionable", player="Player C"),
        _row(status="Available", player="Player D"),
    ]
    minutes = {
        ("BOS", "Player A"): 31.5,
        ("BOS", "Player B"): 12.0,
        ("BOS", "Player C"): 20.0,
        ("BOS", "Player D"): 25.0,
    }
    assert unavailable_rotation_minutes(rows, minutes) == 43.5


def test_unavailable_rotation_minutes_rejects_missing_minutes() -> None:
    with pytest.raises(NbaEvidenceError):
        unavailable_rotation_minutes([_row()], {})
    with pytest.raises(NbaEvidenceError):
        unavailable_rotation_minutes([_row()], {("BOS", "Player A"): -1.0})


def test_unavailable_rotation_value_weights_minutes_by_value() -> None:
    rows = [_row(status="Out", player="Player A"), _row(status="Out", player="Player B")]
    minutes = {("BOS", "Player A"): 30.0, ("BOS", "Player B"): 10.0}
    values = {("BOS", "Player A"): 2.0, ("BOS", "Player B"): -1.5}
    assert unavailable_rotation_value(rows, minutes, values) == 45.0


def test_unavailable_rotation_value_rejects_missing_value() -> None:
    minutes = {("BOS", "Player A"): 30.0}
    with pytest.raises(NbaEvidenceError):
        unavailable_rotation_value([_row()], minutes, {})
