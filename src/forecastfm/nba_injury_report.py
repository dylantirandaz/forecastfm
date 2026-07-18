"""Cutoff-safe availability aggregates from official NBA injury-report snapshots.

The official report lists, for each game on a date, every player whose participation may be
affected, with a status (Out, Doubtful, Questionable, Probable, Available). Snapshots are
published at increasing frequency through the day, so a T-60 availability state is the latest
snapshot at or before tipoff minus 60 minutes.

Health-data boundary: rows carry player identifiers attached to health information and remain
local-only. Only the two aggregate floats defined in ``nba_rich.NBA_LOCAL_HEALTH_FEATURE_SPECS``
may leave this module, and they stay outside every Tinker export. Frozen policy v1 counts Out and
Doubtful as unavailable; a weighted-status ablation is deferred and must be predeclared.

All report and tip times use timezone-aware America/New_York datetimes; the source publishes
ET wall times, and the timezone is attached at parse time so DST stays correct.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from forecastfm.nba_evidence import NbaEvidenceError, require_canonical_float

ET_ZONE = ZoneInfo("America/New_York")

INJURY_REPORT_SCHEMA_VERSION = 1
REPORT_CUTOFF_MINUTES = 60

KNOWN_STATUSES = frozenset({"Available", "Out", "Doubtful", "Questionable", "Probable"})
UNAVAILABLE_STATUSES = frozenset({"Out", "Doubtful"})

PlayerKey = tuple[str, str]  # (team, player name)


@dataclass(frozen=True, slots=True)
class InjuryReportRow:
    """One player status row from one retained report snapshot. Local-only."""

    report_time: datetime
    game_date: date
    game_clock_et: tuple[int, int]
    matchup: str
    team: str
    player_name: str
    status: str

    def game_tipoff(self) -> datetime:
        """Combine the listed game date and ET clock into one aware tipoff."""
        hour, minute = self.game_clock_et
        return datetime(
            self.game_date.year,
            self.game_date.month,
            self.game_date.day,
            hour,
            minute,
            tzinfo=ET_ZONE,
        )


@dataclass(frozen=True, slots=True)
class ParsedInjuryReport:
    """Rows parsed from one snapshot plus the count of dropped malformed-status rows."""

    rows: tuple[InjuryReportRow, ...]
    dropped_rows: int


def parse_game_clock(raw: str) -> tuple[int, int]:
    """Parse a report game-time string like ``07:30 (ET)`` into 24-hour ET wall time.

    The report prints evening times in 12-hour form without a meridian marker; hours 1-11 are
    PM and 12 is noon, matching every observed report. Anything else is rejected.
    """
    clock = raw.strip().split(" ")[0]
    parts = clock.split(":")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise NbaEvidenceError(f"malformed report game time: {raw!r}")
    hour, minute = int(parts[0]), int(parts[1])
    if minute > 59:
        raise NbaEvidenceError(f"malformed report game time: {raw!r}")
    if hour == 12:
        return (12, minute)
    if 1 <= hour <= 11:
        return (hour + 12, minute)
    raise NbaEvidenceError(f"unexpected report game-time hour: {raw!r}")


def parse_game_date(raw: str) -> date:
    """Parse a report game-date string like ``03/06/2019``."""
    parts = raw.strip().split("/")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise NbaEvidenceError(f"malformed report game date: {raw!r}")
    month, day, year = (int(part) for part in parts)
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise NbaEvidenceError(f"malformed report game date: {raw!r}") from exc


def parse_report_header_time(text: str) -> datetime:
    """Parse the ``Injury Report: 03/06/19 05:30 PM`` header line into an aware ET datetime."""
    prefix = "Injury Report:"
    line = next((line for line in text.splitlines() if line.strip().startswith(prefix)), None)
    if line is None:
        raise NbaEvidenceError("report header line not found")
    stamp = line.split(prefix, 1)[1].strip()
    try:
        return datetime.strptime(stamp, "%m/%d/%y %I:%M %p").replace(tzinfo=ET_ZONE)
    except ValueError as exc:
        raise NbaEvidenceError(f"malformed report header time: {stamp!r}") from exc


def matchup_teams(matchup: str) -> tuple[str, str]:
    """Split ``DAL@WAS`` into (away, home) team abbreviations."""
    parts = matchup.strip().split("@")
    if len(parts) != 2 or not all(parts):
        raise NbaEvidenceError(f"malformed report matchup: {matchup!r}")
    return (parts[0], parts[1])


def rows_from_report_records(
    records: Iterable[Mapping[str, object]],
    report_time: datetime,
) -> ParsedInjuryReport:
    """Build validated rows from extracted report records.

    Rows with a missing or unrecognized status are parse artifacts: they are dropped and counted,
    never silently treated as available. Structurally missing date, time, matchup, team, or player
    fields fail the whole snapshot instead.
    """
    rows: list[InjuryReportRow] = []
    dropped = 0
    for record in records:
        status = record.get("Current Status")
        if not isinstance(status, str) or status not in KNOWN_STATUSES:
            dropped += 1
            continue
        rows.append(
            InjuryReportRow(
                report_time=report_time,
                game_date=parse_game_date(_require_text(record, "Game Date")),
                game_clock_et=parse_game_clock(_require_text(record, "Game Time")),
                matchup=_require_text(record, "Matchup"),
                team=_require_text(record, "Team"),
                player_name=_require_text(record, "Player Name"),
                status=status,
            )
        )
    return ParsedInjuryReport(rows=tuple(rows), dropped_rows=dropped)


def select_report_at_cutoff(
    tipoff: datetime,
    report_times: Iterable[datetime],
    cutoff_minutes: int = REPORT_CUTOFF_MINUTES,
) -> datetime | None:
    """Return the latest report published at or before ``tipoff - cutoff_minutes``.

    Returns None when no retained snapshot predates the cutoff; the caller treats that game as
    missing availability evidence rather than reading a later snapshot.
    """
    deadline = tipoff - timedelta(minutes=cutoff_minutes)
    eligible = [time for time in report_times if time <= deadline]
    return max(eligible) if eligible else None


def unavailable_rotation_minutes(
    rows: Iterable[InjuryReportRow],
    prior_minutes: Mapping[PlayerKey, float],
) -> float:
    """Sum prior-game minutes for players unavailable in the selected pre-cutoff report.

    This is the ``unavailable_rotation_minutes`` side value of the frozen rich schema. A missing
    prior-minutes entry for an unavailable player is a missing required source value and raises.
    """
    total = 0.0
    for row in rows:
        if row.status not in UNAVAILABLE_STATUSES:
            continue
        total += _require_minutes(prior_minutes, _player_key(row))
    return total


def unavailable_rotation_value(
    rows: Iterable[InjuryReportRow],
    prior_minutes: Mapping[PlayerKey, float],
    rolling_values: Mapping[PlayerKey, float],
) -> float:
    """Sum minutes times rolling value for unavailable players.

    This is the ``unavailable_rotation_value`` side value of the frozen rich schema. Both inputs
    are strictly pre-cutoff; a missing entry for an unavailable player raises.
    """
    total = 0.0
    for row in rows:
        if row.status not in UNAVAILABLE_STATUSES:
            continue
        key = _player_key(row)
        value = rolling_values.get(key)
        if value is None:
            raise NbaEvidenceError(f"missing rolling value for unavailable player {key[0]}")
        require_canonical_float(value, "rolling_player_value")
        total += _require_minutes(prior_minutes, key) * value
    return total


def _player_key(row: InjuryReportRow) -> PlayerKey:
    return (row.team, row.player_name)


def _require_minutes(prior_minutes: Mapping[PlayerKey, float], key: PlayerKey) -> float:
    minutes = prior_minutes.get(key)
    if minutes is None:
        raise NbaEvidenceError(f"missing prior-game minutes for unavailable player {key[0]}")
    require_canonical_float(minutes, "prior_game_minutes")
    if minutes < 0.0:
        raise NbaEvidenceError("prior-game minutes must be non-negative")
    return minutes


def _require_text(record: Mapping[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise NbaEvidenceError(f"missing report field: {field_name}")
    return value.strip()
