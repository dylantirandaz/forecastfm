"""Tests for the legacy nine-column injury-report text parser."""

from datetime import date, datetime

from forecastfm.nba_injury_report import ET_ZONE
from forecastfm.nba_injury_report_v1 import parse_legacy_text

REPORT_TIME = datetime(2019, 11, 6, 17, 30, tzinfo=ET_ZONE)

HEADER = (
    "Injury Report: 11/06/19 05:30 PM\n"
    "Game Date Game Time Matchup Team Player Name Category Reason Current Status"
    " Previous Status\n"
)
GAME_PREFIX = "11/06/2019 07:00 (ET) NYK@DET Detroit Pistons "


def test_row_with_previous_status_value() -> None:
    result = parse_legacy_text(
        HEADER
        + GAME_PREFIX
        + "Smith Jr., Dennis Injury/Illness Personal reasons Out Questionable\n",
        REPORT_TIME,
    )
    assert result.dropped_rows == 0
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.status == "Out"  # Current Status precedes Previous Status
    assert row.player_name == "Smith Jr., Dennis"
    assert row.team == "Detroit Pistons"
    assert row.matchup == "NYK@DET"
    assert row.game_date == date(2019, 11, 6)
    assert row.game_clock_et == (19, 0)
    assert row.report_time == REPORT_TIME


def test_row_with_trailing_dash_previous_status() -> None:
    result = parse_legacy_text(
        HEADER + GAME_PREFIX + "Doumbouya, Sekou G League - On Assignment - Out -\n",
        REPORT_TIME,
    )
    assert result.dropped_rows == 0
    assert [row.status for row in result.rows] == ["Out"]
    assert result.rows[0].player_name == "Doumbouya, Sekou"


def test_wrapped_multiline_reason() -> None:
    text = (
        HEADER
        + GAME_PREFIX
        + "Griffin, Blake Injury/IllnessLeft Hamstring/Posterior Knee\n"
        + "SorenessOut Out\n"
    )
    result = parse_legacy_text(text, REPORT_TIME)
    assert result.dropped_rows == 0
    assert len(result.rows) == 1
    assert result.rows[0].player_name == "Griffin, Blake"
    assert result.rows[0].status == "Out"


def test_g_league_category_row() -> None:
    result = parse_legacy_text(
        HEADER + GAME_PREFIX + "Allen, Kadeem G League - Two-Way - Out -\n",
        REPORT_TIME,
    )
    assert result.dropped_rows == 0
    assert [(row.player_name, row.status) for row in result.rows] == [("Allen, Kadeem", "Out")]


def test_row_without_status_is_dropped_and_counted() -> None:
    text = (
        HEADER
        + GAME_PREFIX
        + "Doe, John Injury/Illness Feeling ill\n"
        + "Frazier, Tim Injury/Illness Right Shoulder Strain Out -\n"
    )
    result = parse_legacy_text(text, REPORT_TIME)
    assert result.dropped_rows == 1
    assert [(row.player_name, row.status) for row in result.rows] == [("Frazier, Tim", "Out")]


def test_longest_team_name_match() -> None:
    text = (
        HEADER
        + "11/06/2019 10:30 (ET) POR@LAC LA Clippers Coffey, Amir G League - Two-Way - Out -\n"
        + "Portland Trail Blazers Bazemore, Kent Injury/Illness Left ankle sprain"
        " Questionable -\n"
    )
    result = parse_legacy_text(text, REPORT_TIME)
    assert result.dropped_rows == 0
    assert [(row.team, row.status) for row in result.rows] == [
        ("LA Clippers", "Out"),
        ("Portland Trail Blazers", "Questionable"),
    ]
    assert result.rows[0].game_clock_et == (22, 30)


def test_multi_page_report_with_repeated_headers_and_wrapped_team_name() -> None:
    page_one = (
        HEADER
        + GAME_PREFIX
        + "Doumbouya, Sekou G League - On Assignment - Out -\n"
        + "07:30 (ET) CHI@ATL Atlanta Hawks Collins, John League Suspension - Out -\n"
        + "Minnesota\n"
        + "Page 1 of \n2\n"
    )
    page_two = (
        HEADER
        + "TimberwolvesBates-Diop, Keita G League - On Assignment - Out -\n"
        + "Miami Heat NOT YET SUBMITTED\n"
        + "Page 2 of \n2\n"
    )
    result = parse_legacy_text(page_one + page_two, REPORT_TIME)
    assert result.dropped_rows == 0
    assert [(row.player_name, row.team, row.status) for row in result.rows] == [
        ("Doumbouya, Sekou", "Detroit Pistons", "Out"),
        ("Collins, John", "Atlanta Hawks", "Out"),
        ("Bates-Diop, Keita", "Minnesota Timberwolves", "Out"),
    ]
    # Clock-only game rows inherit the previous game date.
    assert {row.game_date for row in result.rows} == {date(2019, 11, 6)}
    assert result.rows[1].game_clock_et == (19, 30)
    assert result.rows[1].matchup == "CHI@ATL"
