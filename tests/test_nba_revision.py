"""Focused tests for the sequential revision-stream feature builder."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from forecastfm.nba_feature_builder import InjurySnapshot
from forecastfm.nba_injury_report import InjuryReportRow
from forecastfm.nba_revision import (
    NbaRevisionError,
    RevisionGame,
    RevisionGameContext,
    build_revision_rows,
    select_snapshot_at_horizon,
    side_revision_deltas,
    status_rank,
)

ET = ZoneInfo("America/New_York")
GAME_DATE = date(2024, 3, 6)
TIPOFF = datetime(2024, 3, 7, 0, 30, tzinfo=UTC)  # 19:30 ET
RATINGS = {"brown jaylen": 3.0, "jayson tatum": 5.0, "doncic luka": 4.0}
_STUB_TIME = datetime(2024, 3, 6, 12, 0, tzinfo=ET)


def _report_row(
    team: str,
    player: str,
    status: str,
    matchup: str = "DAL@BOS",
) -> InjuryReportRow:
    return InjuryReportRow(
        report_time=_STUB_TIME,
        game_date=GAME_DATE,
        game_clock_et=(19, 30),
        matchup=matchup,
        team=team,
        player_name=player,
        status=status,
    )


def _snapshot(hour: int, minute: int, rows: list[InjuryReportRow]) -> InjurySnapshot:
    report_time = datetime(2024, 3, 6, hour, minute, tzinfo=ET)
    return InjurySnapshot(
        report_time=report_time,
        rows=tuple(
            InjuryReportRow(
                report_time=report_time,
                game_date=row.game_date,
                game_clock_et=row.game_clock_et,
                matchup=row.matchup,
                team=row.team,
                player_name=row.player_name,
                status=row.status,
            )
            for row in rows
        ),
    )


def _game() -> RevisionGame:
    return RevisionGame(
        game_id=42,
        season=2024,
        game_date=GAME_DATE,
        tipoff=TIPOFF,
        away_abbreviation="DAL",
        home_abbreviation="BOS",
        home_won=True,
    )


def _context(horizon_minutes: int) -> RevisionGameContext:
    return RevisionGameContext(
        away_abbreviation="DAL",
        home_abbreviation="BOS",
        player_ratings=RATINGS,
        horizon_time=TIPOFF - timedelta(minutes=horizon_minutes),
        horizon_minutes=horizon_minutes,
    )


def test_status_rank_orders_availability_classes() -> None:
    ranks = [
        status_rank(status)
        for status in ("Available", "Probable", "Questionable", "Doubtful", "Out")
    ]
    assert ranks == sorted(ranks)
    assert status_rank("Available") == 0
    assert status_rank("Out") == 4


def test_status_rank_rejects_unknown_status() -> None:
    with pytest.raises(NbaRevisionError, match="unknown report status"):
        status_rank("Day-To-Day")


def test_side_revision_deltas_weight_changes_by_rapm() -> None:
    earlier = _snapshot(
        16,
        30,
        [
            _report_row("Boston Celtics", "Brown, Jaylen", "Questionable"),
            _report_row("Boston Celtics", "Tatum, Jayson", "Out"),
        ],
    )
    later = _snapshot(
        17,
        30,
        [
            _report_row("Boston Celtics", "Brown, Jaylen", "Out"),
            _report_row("Boston Celtics", "Tatum, Jayson", "Probable"),
            _report_row("Boston Celtics", "Doncic, Luka", "Doubtful"),
        ],
    )
    deltas = side_revision_deltas(later, earlier, "BOS", _context(60))
    assert deltas.downgrade_value == pytest.approx(3.0 + 4.0)  # Brown + newly listed Doncic
    assert deltas.upgrade_value == pytest.approx(5.0)  # Tatum Out -> Probable
    assert deltas.changes_count == 3
    # Change became visible at 17:30 ET, one hour before the T-60 horizon time (18:30 ET).
    assert deltas.minutes_since_last_change == pytest.approx(60.0)


def test_side_revision_deltas_ignore_other_teams_and_matchups() -> None:
    earlier = _snapshot(16, 30, [_report_row("Boston Celtics", "Brown, Jaylen", "Questionable")])
    later = _snapshot(
        17,
        30,
        [
            _report_row("Boston Celtics", "Brown, Jaylen", "Out"),
            _report_row("Boston Celtics", "Tatum, Jayson", "Out", matchup="NYK@BOS"),
            _report_row("Dallas Mavericks", "Doncic, Luka", "Out"),
        ],
    )
    deltas = side_revision_deltas(later, earlier, "BOS", _context(60))
    assert deltas.downgrade_value == pytest.approx(3.0)  # only Brown counts for BOS in DAL@BOS
    assert deltas.changes_count == 1


def test_side_revision_deltas_without_previous_snapshot_saturate() -> None:
    only = _snapshot(18, 30, [_report_row("Boston Celtics", "Brown, Jaylen", "Out")])
    deltas = side_revision_deltas(only, None, "BOS", _context(360))
    assert deltas.downgrade_value == 0.0
    assert deltas.upgrade_value == 0.0
    assert deltas.changes_count == 0
    assert deltas.minutes_since_last_change == 360.0


def test_side_revision_deltas_unchanged_reports_report_full_horizon() -> None:
    earlier = _snapshot(17, 30, [_report_row("Boston Celtics", "Brown, Jaylen", "Out")])
    later = _snapshot(18, 30, [_report_row("Boston Celtics", "Brown, Jaylen", "Out")])
    deltas = side_revision_deltas(later, earlier, "BOS", _context(120))
    assert deltas.changes_count == 0
    assert deltas.minutes_since_last_change == 120.0


def test_select_snapshot_at_horizon_falls_back_to_containing_snapshot() -> None:
    containing = _snapshot(13, 30, [_report_row("Boston Celtics", "Brown, Jaylen", "Out")])
    other_game = _snapshot(
        17, 30, [_report_row("Boston Celtics", "Brown, Jaylen", "Out", matchup="NYK@BOS")]
    )
    too_late = _snapshot(19, 0, [_report_row("Boston Celtics", "Brown, Jaylen", "Out")])
    snapshots = [containing, other_game, too_late]
    selected = select_snapshot_at_horizon(snapshots, _game(), 60)
    # The 19:00 ET snapshot is past the T-60 cutoff; the 17:30 one lacks DAL@BOS.
    assert selected is containing


def test_select_snapshot_at_horizon_returns_none_without_containment() -> None:
    other_game = _snapshot(
        17, 30, [_report_row("Boston Celtics", "Brown, Jaylen", "Out", matchup="NYK@BOS")]
    )
    assert select_snapshot_at_horizon([other_game], _game(), 60) is None


def test_build_revision_rows_skips_and_counts_uncovered_horizons() -> None:
    snapshots = [
        _snapshot(13, 31, [_report_row("Boston Celtics", "Brown, Jaylen", "Out")]),
        _snapshot(
            18,
            30,
            [
                _report_row("Boston Celtics", "Brown, Jaylen", "Questionable"),
                _report_row("Dallas Mavericks", "Doncic, Luka", "Out"),
            ],
        ),
    ]
    result = build_revision_rows(
        [_game()],
        snapshots,
        {42: 0.65},
        RATINGS,
        horizons=(360, 60),
    )
    assert result.games_total == 1
    assert result.skipped == 1  # no snapshot exists at or before tipoff minus 360 minutes
    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.horizon_minutes == 60
    assert row.question_id == "nba-42"
    assert row.elo_home_probability == 0.65
    assert row.home_won is True
    # Home side upgraded (Brown Out -> Questionable, +3.0); away side newly lists Doncic Out
    # (downgrade +4.0): downgrade diff = 0.0 - 4.0, upgrade diff = 3.0 - 0.0.
    assert len(row.features) == 4
    assert row.features[0] == pytest.approx(-4.0)
    assert row.features[1] == pytest.approx(3.0)
    assert row.features[2] == pytest.approx(0.0)  # both sides changed at the selected snapshot
    assert row.features[3] == pytest.approx(0.0)  # one change per side
