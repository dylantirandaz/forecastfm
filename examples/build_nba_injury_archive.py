"""Build a bounded local archive of official NBA injury-report snapshots.

Private, noncommercial research use only. Downloads a fixed set of six ET snapshots per date
(11:30 through 21:30 at two-hour steps), which always yields a snapshot at or before every
evening tipoff's T-60 cutoff. Saved PDFs stay under an ignored 0700 root and are parsed locally;
only aggregate counts are printed. Named player-health rows never leave local storage and never
enter any Tinker artifact.
"""

from __future__ import annotations

import argparse
import json
import stat
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from nbainjuries import injury  # pyright: ignore[reportMissingTypeStubs]

from forecastfm.nba_injury_report import (
    ET_ZONE,
    INJURY_REPORT_SCHEMA_VERSION,
    matchup_teams,
    rows_from_report_records,
)

DEFAULT_STORAGE_ROOT = Path("data/raw/nba_injury_reports")
REQUEST_DELAY_SECONDS = 0.3
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
SNAPSHOT_SLOTS: tuple[tuple[int, int], ...] = (
    (11, 30),
    (13, 30),
    (15, 30),
    (17, 30),
    (19, 30),
    (21, 30),
)

_GET_REPORTDATA = cast(
    "Callable[..., str]",
    injury.get_reportdata,  # pyright: ignore[reportUnknownMemberType]
)


@dataclass(frozen=True, slots=True)
class _Arguments:
    start_date: str
    end_date: str
    storage_root: Path


def main(argv: Sequence[str] | None = None) -> int:
    """Download and parse the canonical snapshots for each requested date."""
    args = _parse_arguments(argv)
    root = _require_private_root(args.storage_root)
    start = _parse_day(args.start_date)
    end = _parse_day(args.end_date)
    if end < start:
        raise RuntimeError("end date must not precede start date")
    days = [start.fromordinal(day) for day in range(start.toordinal(), end.toordinal() + 1)]
    summaries = [_process_day(day, root) for day in days]
    print(json.dumps({"archive_root": str(root), "dates": summaries}, indent=2))
    return 0


def _process_day(day: datetime, root: Path) -> dict[str, object]:
    day_dir = root / day.date().isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    snapshots = [_process_snapshot(day, hour, minute, day_dir) for hour, minute in SNAPSHOT_SLOTS]
    downloaded = [snap for snap in snapshots if snap["status"] == "downloaded"]
    summary: dict[str, object] = {
        "date": day.date().isoformat(),
        "snapshots_downloaded": len(downloaded),
        "snapshots_missing": sum(1 for snap in snapshots if snap["status"] == "missing"),
        "snapshots": snapshots,
    }
    (day_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _process_snapshot(day: datetime, hour: int, minute: int, day_dir: Path) -> dict[str, object]:
    slot = day.replace(hour=hour, minute=minute, tzinfo=ET_ZONE)
    target = day_dir / _snapshot_filename(slot)
    try:
        if not target.exists() and not _download_snapshot(slot, target):
            return {"slot": slot.isoformat(), "status": "missing"}
        parsed = _parse_snapshot(slot, target, day_dir)
    except Exception as error:
        return {"slot": slot.isoformat(), "status": "error", "error": type(error).__name__}
    return {"slot": slot.isoformat(), "status": "downloaded", **parsed}


def _download_snapshot(slot: datetime, target: Path) -> bool:
    url = _snapshot_url(slot)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        time.sleep(REQUEST_DELAY_SECONDS)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in {403, 404}:
                return False
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
        else:
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(target)
            return True
    raise RuntimeError("unreachable")


def _parse_snapshot(slot: datetime, target: Path, day_dir: Path) -> dict[str, object]:
    raw_json = _GET_REPORTDATA(slot.replace(tzinfo=None), local=True, localdir=day_dir)
    records = cast("list[dict[str, object]]", json.loads(raw_json))
    parsed = rows_from_report_records(records, slot)
    games = {row.matchup for row in parsed.rows}
    rows_path = target.with_suffix(".rows.jsonl")
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in parsed.rows:
            away, home = matchup_teams(row.matchup)
            handle.write(
                json.dumps(
                    {
                        "schema_version": INJURY_REPORT_SCHEMA_VERSION,
                        "report_time": row.report_time.isoformat(),
                        "game_date": row.game_date.isoformat(),
                        "game_clock_et": f"{row.game_clock_et[0]:02d}:{row.game_clock_et[1]:02d}",
                        "matchup": row.matchup,
                        "away_team": away,
                        "home_team": home,
                        "team": row.team,
                        "player_name": row.player_name,
                        "status": row.status,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return {"rows": len(parsed.rows), "dropped_rows": parsed.dropped_rows, "games": len(games)}


def _snapshot_url(slot: datetime) -> str:
    return injury.gen_url(slot.replace(tzinfo=None))


def _snapshot_filename(slot: datetime) -> str:
    return _snapshot_url(slot).rsplit("/", 1)[-1]


def _parse_day(raw: str) -> datetime:
    try:
        return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=ET_ZONE)
    except ValueError as exc:
        raise RuntimeError(f"malformed date: {raw!r}") from exc


def _require_private_root(storage_root: Path) -> Path:
    root = storage_root.absolute()
    if root.exists() and not root.is_dir():
        raise RuntimeError("archive root is not a directory")
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(stat.S_IRWXU)
    if stat.S_IMODE(root.stat().st_mode) & (stat.S_IRWXG | stat.S_IRWXO):
        raise RuntimeError("archive root must have no group or other permissions")
    return root


def _parse_arguments(argv: Sequence[str] | None) -> _Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="first date, YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="last date, default: start date")
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    namespace = parser.parse_args(argv)
    return _Arguments(
        start_date=cast("str", namespace.start_date),
        end_date=cast("str | None", namespace.end_date) or cast("str", namespace.start_date),
        storage_root=cast("Path", namespace.storage_root),
    )


if __name__ == "__main__":
    raise SystemExit(main())
