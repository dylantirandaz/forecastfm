"""Fetch one NBA regular season from ESPN's free API into nbastats CSV form.

Private research use. Crawls one scoreboard document per date and one summary document per
game with spaced requests, retains every raw payload under an ignored root, converts each game
to the 34-column nbastats schema, and writes one season CSV plus a manifest binding synthetic
game IDs to ESPN event IDs and payload SHA-256s. Game IDs are synthetic, sequential in
chronological order, and not official NBA identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from forecastfm.nba_espn import EspnGameRef, convert_summary, parse_scoreboard, write_nbastats_csv

ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={day}"
)
ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}"
)
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
REQUEST_DELAY_SECONDS = 0.25
DEFAULT_OUTPUT_DIR = Path("data/raw/espn")
FIRST_SYNTHETIC_ID = 22500001


@dataclass(frozen=True, slots=True)
class _Arguments:
    start_date: str
    end_date: str
    output_dir: Path


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch and convert every regular-season game in the requested date range."""
    args = _parse_arguments(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    days = _date_range(args.start_date, args.end_date)
    references: list[EspnGameRef] = []
    for day in days:
        compact = day.replace("-", "")
        payload = _fetch(
            ESPN_SCOREBOARD_URL.format(day=compact),
            raw_dir / f"scoreboard-{day}.json",
        )
        references.extend(parse_scoreboard(payload))
    references.sort(key=lambda reference: reference.date_utc)
    manifest_games: list[dict[str, object]] = []
    rows_all: list[list[str]] = []
    for index, reference in enumerate(references):
        synthetic_id = FIRST_SYNTHETIC_ID + index
        payload = _fetch(
            ESPN_SUMMARY_URL.format(event_id=reference.event_id),
            raw_dir / f"summary-{reference.event_id}.json",
        )
        converted = convert_summary(payload, synthetic_id)
        rows_all.extend(converted.rows)
        manifest_games.append(
            {
                "synthetic_game_id": synthetic_id,
                "event_id": reference.event_id,
                "date_utc": reference.date_utc,
                "away": converted.away_abbreviation,
                "home": converted.home_abbreviation,
                "away_score": converted.away_score,
                "home_score": converted.home_score,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    csv_path = args.output_dir / "espn_2025.csv"
    write_nbastats_csv(csv_path, rows_all)
    manifest = {
        "schema_version": 1,
        "built_at_utc": datetime.now(UTC).isoformat(),
        "games": manifest_games,
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"games": len(manifest_games), "csv": str(csv_path)}, indent=2))
    return 0


def _fetch(url: str, target: Path) -> bytes:
    if target.exists():
        return target.read_bytes()
    time.sleep(REQUEST_DELAY_SECONDS)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
    target.write_bytes(payload)
    return payload


def _date_range(start: str, end: str) -> list[str]:
    first = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC).date()
    last = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=UTC).date()
    return [
        (first + timedelta(days=offset)).isoformat() for offset in range((last - first).days + 1)
    ]


def _parse_arguments(argv: Sequence[str] | None) -> _Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    namespace = parser.parse_args(argv)
    return _Arguments(
        start_date=str(namespace.start_date),
        end_date=str(namespace.end_date),
        output_dir=namespace.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
