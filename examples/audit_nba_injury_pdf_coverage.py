"""Local-only coverage audit of official NBA injury-report PDFs.

Private, noncommercial feasibility test for reconstructing T-60 pregame injury state from the
NBA's published report snapshots. Downloads public PDFs to memory only, prints and stores
aggregate counts and timing margins, and never retains or prints player-level health records.
Nothing is uploaded anywhere. Request budget is bounded and spaced.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

from nbainjuries import injury  # pyright: ignore[reportMissingTypeStubs]

OUTPUT_DIR = Path("data/raw/nba_injury_pdf_audit")
REQUEST_DELAY_SECONDS = 0.3
ET_ZONE = ZoneInfo("America/New_York")

_CHECK_REPORTVALID = cast(
    "Callable[[datetime], bool]",
    injury.check_reportvalid,  # pyright: ignore[reportUnknownMemberType]
)
_GET_REPORTDATA = cast(
    "Callable[[datetime], str]",
    injury.get_reportdata,  # pyright: ignore[reportUnknownMemberType]
)

CANONICAL_SLOTS: list[tuple[int, int]] = [
    (11, 30),
    (13, 30),
    (15, 30),
    (17, 30),
    (19, 30),
    (21, 30),
]


def naive_et(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    """Build a report timestamp: ET wall time, naive, as the report API requires."""
    return datetime(year, month, day, hour, minute, tzinfo=ET_ZONE).replace(tzinfo=None)


SEASON_SAMPLE_DATES: dict[str, list[datetime]] = {
    "2021-22": [
        naive_et(2021, 11, 3),
        naive_et(2022, 1, 12),
        naive_et(2022, 3, 2),
        naive_et(2022, 4, 6),
        naive_et(2022, 5, 4),
    ],
    "2022-23": [
        naive_et(2022, 11, 2),
        naive_et(2023, 1, 11),
        naive_et(2023, 3, 1),
        naive_et(2023, 4, 5),
        naive_et(2023, 5, 3),
    ],
    "2023-24": [
        naive_et(2023, 11, 1),
        naive_et(2024, 1, 10),
        naive_et(2024, 3, 6),
        naive_et(2024, 4, 3),
        naive_et(2024, 5, 1),
    ],
    "2024-25": [
        naive_et(2024, 11, 6),
        naive_et(2025, 1, 8),
        naive_et(2025, 3, 5),
        naive_et(2025, 4, 2),
        naive_et(2025, 4, 30),
    ],
    "2025-26": [
        naive_et(2025, 11, 5),
        naive_et(2026, 1, 7),
        naive_et(2026, 2, 4),
        naive_et(2026, 3, 4),
        naive_et(2026, 4, 29),
    ],
}

OUT_OF_RANGE_DATES: list[datetime] = [naive_et(2019, 3, 6), naive_et(2021, 1, 6)]

CADENCE_DRILL_DATE = naive_et(2026, 1, 7)
CADENCE_SLOTS: list[tuple[int, int]] = [
    (hour, minute) for hour in range(12, 24) for minute in (0, 15, 30, 45)
]

PARSE_TIMESTAMPS: list[datetime] = [
    naive_et(2025, 3, 5, 17, 30),
    naive_et(2025, 4, 30, 17, 30),
    naive_et(2026, 1, 7, 17, 30),
]

T60_MINUTES = 60


class AuditState:
    """Mutable request accounting for the audit run."""

    def __init__(self) -> None:
        """Initialize empty audit state."""
        self.request_count = 0
        self.season_grid: dict[str, dict[str, dict[str, bool]]] = {}
        self.out_of_range: dict[str, bool] = {}
        self.cadence_drill: dict[str, bool] = {}
        self.parsed_reports: list[dict[str, object]] = []


def slot_key(timestamp: datetime) -> str:
    """Return the HH:MM label for one report slot."""
    return timestamp.strftime("%H:%M")


def probe_report(state: AuditState, timestamp: datetime) -> bool:
    """Check report existence at one timestamp with request spacing and accounting."""
    time.sleep(REQUEST_DELAY_SECONDS)
    state.request_count += 1
    try:
        return bool(_CHECK_REPORTVALID(timestamp))
    except Exception:
        return False


def probe_season_grid(state: AuditState) -> None:
    """Probe canonical slots across sampled in-season and out-of-range dates."""
    for season, dates in SEASON_SAMPLE_DATES.items():
        grid: dict[str, dict[str, bool]] = {}
        for day in dates:
            grid[day.date().isoformat()] = {
                slot_key(day.replace(hour=h, minute=m)): probe_report(
                    state, day.replace(hour=h, minute=m)
                )
                for h, m in CANONICAL_SLOTS
            }
        state.season_grid[season] = grid
    for day in OUT_OF_RANGE_DATES:
        state.out_of_range[day.date().isoformat()] = probe_report(
            state, day.replace(hour=17, minute=30)
        )


def probe_cadence_drill(state: AuditState) -> None:
    """Probe every 15-minute slot on one recent game day to learn snapshot cadence."""
    state.cadence_drill = {
        slot_key(CADENCE_DRILL_DATE.replace(hour=h, minute=m)): probe_report(
            state, CADENCE_DRILL_DATE.replace(hour=h, minute=m)
        )
        for h, m in CADENCE_SLOTS
    }


def parse_game_time(raw: str) -> tuple[int, int] | None:
    """Parse a report game-time string like '07:30 (ET)' into naive evening minutes."""
    text = raw.strip().split(" ")[0]
    parts = text.split(":")
    if len(parts) != 2:
        return None
    hour, minute = int(parts[0]), int(parts[1])
    if hour == 12:
        return (12, minute)
    if 1 <= hour <= 11:
        return (hour + 12, minute)
    return (hour, minute)


def margin_minutes(report_time: datetime, game_hm: tuple[int, int]) -> int:
    """Compute naive same-day minutes between the report snapshot and scheduled tip."""
    game = report_time.replace(hour=game_hm[0], minute=game_hm[1], second=0, microsecond=0)
    return int((game - report_time).total_seconds() / 60)


def parse_report_sample(state: AuditState, timestamp: datetime) -> None:
    """Parse one report and record aggregate counts and T-60 margins, never player rows."""
    time.sleep(REQUEST_DELAY_SECONDS)
    state.request_count += 1
    try:
        records = cast("list[dict[str, str]]", json.loads(_GET_REPORTDATA(timestamp)))
    except Exception as exc:
        state.parsed_reports.append(
            {"timestamp": timestamp.isoformat(), "error": type(exc).__name__}
        )
        return
    games: dict[str, tuple[int, int]] = {}
    statuses: dict[str, int] = {}
    for record in records:
        game_hm = parse_game_time(record.get("Game Time", ""))
        if game_hm is not None:
            games[record.get("Matchup", "")] = game_hm
        status = record.get("Current Status", "")
        statuses[status] = statuses.get(status, 0) + 1
    margins = [margin_minutes(timestamp, hm) for hm in games.values()]
    state.parsed_reports.append(
        {
            "timestamp": timestamp.isoformat(),
            "rows": len(records),
            "games": len(games),
            "statuses": statuses,
            "margin_minutes_min": min(margins) if margins else None,
            "margin_minutes_max": max(margins) if margins else None,
            "all_games_at_or_beyond_t60": all(m >= T60_MINUTES for m in margins)
            if margins
            else None,
        }
    )


def summarize_grid(state: AuditState) -> dict[str, object]:
    """Reduce the probe grid to per-season validity fractions."""
    summary: dict[str, object] = {}
    for season, grid in state.season_grid.items():
        flat = [valid for day in grid.values() for valid in day.values()]
        summary[season] = {"probes": len(flat), "valid": sum(flat)}
    valid_cadence = [slot for slot, valid in state.cadence_drill.items() if valid]
    summary["cadence_valid_slots"] = valid_cadence
    summary["out_of_range"] = state.out_of_range
    return summary


def main() -> None:
    """Run the bounded audit, write aggregate JSON, and print a summary."""
    state = AuditState()
    probe_season_grid(state)
    probe_cadence_drill(state)
    for timestamp in PARSE_TIMESTAMPS:
        parse_report_sample(state, timestamp)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()
    grid_summary = summarize_grid(state)
    payload = {
        "generated_at_utc": generated_at,
        "request_count": state.request_count,
        "grid_summary": grid_summary,
        "season_grid": state.season_grid,
        "cadence_drill": state.cadence_drill,
        "parsed_reports": state.parsed_reports,
    }
    output_path = OUTPUT_DIR / f"coverage_audit_{generated_at.replace(':', '-')}.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "grid_summary": grid_summary}, indent=2))
    print(json.dumps(state.parsed_reports, indent=2))


if __name__ == "__main__":
    main()
