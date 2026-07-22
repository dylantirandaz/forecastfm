"""Collect untouched prospective evidence for the 2026-27 NBA regular season.

Private research use. Polls the ESPN scoreboard once per day for the next three days, derives
each game's T-6h, T-60, and T-15 pre-tipoff cutoffs in America/New_York, and at each cutoff
(plus or minus two minutes) captures the current official NBA injury-report PDF snapshot and
the day's ESPN scoreboard into a create-only raw store. Every schedule poll, capture, and
tipoff amendment is appended to an append-only hash-chained ledger that reuses the
``forecastfm.ledger`` envelope. There are no daemons and no threads: ``run`` executes the work
due now and exits, so it is cron-friendly. ``plan`` is always offline. See
``prospective/COLLECTOR.md`` for the storage layout, ledger integration, cutoff rules, and
the October dry-run checklist.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import cast

from forecastfm.integrity import bytes_sha256, canonical_json, canonical_sha256
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.ledger import GENESIS_HASH
from forecastfm.nba_espn import EspnGameRef, parse_upcoming_scoreboard
from forecastfm.nba_injury_report import ET_ZONE

type JsonObject = dict[str, object]

COLLECTOR_SCHEMA_VERSION = 1
CAPTURE_LEDGER_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
DEFAULT_STORAGE_ROOT = Path("data/raw/prospective")
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={day}"
)
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
REQUEST_DELAY_SECONDS = 0.3
POLL_HORIZON_DAYS = 3
CUTOFF_WINDOW = timedelta(minutes=2)
CUTOFFS: tuple[tuple[str, timedelta], ...] = (
    ("t_6h", timedelta(hours=6)),
    ("t_60", timedelta(minutes=60)),
    ("t_15", timedelta(minutes=15)),
)
REPORT_STEP = timedelta(minutes=15)
REPORT_LOOKBACK_STEPS = 12
REPORT_FALLBACK_SLOTS: tuple[tuple[int, int], ...] = (
    (11, 30),
    (13, 30),
    (15, 30),
    (17, 30),
    (19, 30),
    (21, 30),
)
MAX_REPORT_CANDIDATES = 20
STATE_DIR = "state"
STATE_FILENAME = "schedule.json"
LEDGER_FILENAME = "capture-ledger.jsonl"
_ENVELOPE_KEYS = {
    "schema_version",
    "sequence",
    "event_type",
    "recorded_at",
    "previous_hash",
    "payload",
    "event_hash",
}
_ENVELOPE_BODY_KEYS = _ENVELOPE_KEYS - {"event_hash"}

_CUTOFF_RULES: JsonObject = {
    "timezone": "America/New_York",
    "execution_window_minutes": 2,
    "cutoffs": [
        {"name": "t_6h", "before_tipoff": "6h", "role": "optional early state"},
        {"name": "t_60", "before_tipoff": "60m", "role": "primary prospective state"},
        {"name": "t_15", "before_tipoff": "15m", "role": "optional late state"},
    ],
    "late_capture_policy": (
        "late captures are retained with their true retrieved_at; downstream eligibility "
        "uses available_at <= cutoff, so a late capture is simply ineligible for that state"
    ),
}


class ProspectiveCollectorError(RuntimeError):
    """Raised when prospective collection cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class _Arguments:
    command: str
    date: str | None
    storage_root: Path
    dry_run: bool
    now: str | None


@dataclass(frozen=True, slots=True)
class _RunContext:
    root: Path
    ledger_path: Path
    dry_run: bool
    now: datetime


@dataclass(slots=True)
class _GameState:
    event_id: str
    matchup: str
    away: str
    home: str
    game_date_et: str
    tipoff_utc: str
    original_tipoff_utc: str
    removed: bool
    captures: dict[str, str]


@dataclass(slots=True)
class _ScheduleState:
    last_poll_date: str | None
    games: list[_GameState]


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the ``plan`` and ``run`` subcommands."""
    args = _parse_arguments(argv)
    if args.command == "plan":
        if args.date is None:
            raise ProspectiveCollectorError("plan requires --date")
        return _cmd_plan(args.date, args.storage_root)
    return _cmd_run(args)


def _cmd_plan(date_text: str, storage_root: Path) -> int:
    target = _parse_iso_date(date_text, "--date")
    root = storage_root.absolute()
    state = _load_state(root / STATE_DIR / STATE_FILENAME)
    poll_dates = [(target + timedelta(days=offset)).isoformat() for offset in range(3)]
    games = [_plan_game(game) for game in state.games if game.game_date_et == target.isoformat()]
    plan: JsonObject = {
        "schema_version": COLLECTOR_SCHEMA_VERSION,
        "date": target.isoformat(),
        "network": "none: plan is fully offline and never touches the network",
        "schedule_poll": {
            "status": "completed" if state.last_poll_date == target.isoformat() else "pending",
            "last_poll_date": state.last_poll_date,
            "dates": poll_dates,
            "scoreboard_url_template": ESPN_SCOREBOARD_URL,
        },
        "games": games,
        "cutoff_rules": _CUTOFF_RULES,
    }
    if not games:
        plan["note"] = "no locally recorded games for this date; run a schedule poll first"
    print(json.dumps(plan, indent=2))
    return 0


def _cmd_run(args: _Arguments) -> int:
    now = _parse_now(args.now)
    root = (
        args.storage_root.absolute() if args.dry_run else _require_private_root(args.storage_root)
    )
    context = _RunContext(
        root=root,
        ledger_path=root / STATE_DIR / LEDGER_FILENAME,
        dry_run=args.dry_run,
        now=now,
    )
    state = _load_state(root / STATE_DIR / STATE_FILENAME)
    summary: JsonObject = {
        "schema_version": COLLECTOR_SCHEMA_VERSION,
        "dry_run": args.dry_run,
        "now": _utc_text(now),
        "schedule_poll": _maybe_poll(context, state, now.astimezone(ET_ZONE).date()),
        "captures": _run_captures(context, state),
    }
    if not args.dry_run:
        _save_state(root / STATE_DIR / STATE_FILENAME, state)
    print(json.dumps(summary, indent=2))
    return 0


def _maybe_poll(context: _RunContext, state: _ScheduleState, today_et: date) -> JsonObject:
    poll_dates = [today_et + timedelta(days=offset) for offset in range(POLL_HORIZON_DAYS)]
    if state.last_poll_date == today_et.isoformat():
        return {"action": "skipped", "reason": "already polled today", "dates": _iso(poll_dates)}
    if context.dry_run:
        return {"action": "would_poll", "dates": _iso(poll_dates)}
    return _poll_schedule(context, state, today_et, poll_dates)


def _poll_schedule(
    context: _RunContext,
    state: _ScheduleState,
    poll_date: date,
    poll_dates: list[date],
) -> JsonObject:
    payloads: list[JsonObject] = []
    references: list[EspnGameRef] = []
    for day in poll_dates:
        url = ESPN_SCOREBOARD_URL.format(day=day.strftime("%Y%m%d"))
        payload = _fetch_bytes(url)
        if payload is None:
            raise ProspectiveCollectorError(f"ESPN scoreboard returned no document for {day}")
        target = context.root / poll_date.isoformat() / f"poll-scoreboard-{day.isoformat()}.json"
        data = _ensure_create_only(target, payload)
        references.extend(parse_upcoming_scoreboard(data))
        payloads.append(
            {
                "date": day.isoformat(),
                "path": _rel(context.root, target),
                "sha256": bytes_sha256(data),
            }
        )
    amendments = _reconcile(state, references, poll_date, poll_dates[-1])
    recorded_at = datetime.now(UTC)
    for amendment in amendments:
        _append_event(context.ledger_path, "schedule_amendment", amendment, recorded_at)
    event: JsonObject = {
        "poll_date": poll_date.isoformat(),
        "dates": _iso(poll_dates),
        "payloads": payloads,
        "games": [_game_brief(reference) for reference in references],
    }
    event_hash = _append_event(context.ledger_path, "schedule_poll", event, recorded_at)
    state.last_poll_date = poll_date.isoformat()
    return {
        "action": "polled",
        "games": len(references),
        "amendments": len(amendments),
        "ledger_event_hash": event_hash,
    }


def _reconcile(
    state: _ScheduleState,
    references: list[EspnGameRef],
    first_day: date,
    last_day: date,
) -> list[JsonObject]:
    amendments: list[JsonObject] = []
    known = {game.event_id: game for game in state.games}
    seen: set[str] = set()
    for reference in references:
        seen.add(reference.event_id)
        amendment = _reconcile_game(state, known.get(reference.event_id), reference)
        if amendment is not None:
            amendments.append(amendment)
    for game in state.games:
        if not game.removed and game.event_id not in seen and _in_window(game, first_day, last_day):
            amendments.append(_amendment_payload(game, None))
            game.removed = True
    return amendments


def _reconcile_game(
    state: _ScheduleState,
    existing: _GameState | None,
    reference: EspnGameRef,
) -> JsonObject | None:
    tipoff = _utc_text(_parse_utc_text(reference.date_utc, "event date"))
    if existing is None:
        state.games.append(_new_game_state(reference, tipoff))
        return None
    if existing.tipoff_utc == tipoff:
        return None
    amendment = _amendment_payload(existing, tipoff)
    existing.tipoff_utc = tipoff
    return amendment


def _new_game_state(reference: EspnGameRef, tipoff_utc: str) -> _GameState:
    tipoff_et = _parse_utc_text(tipoff_utc, "event date").astimezone(ET_ZONE)
    return _GameState(
        event_id=reference.event_id,
        matchup=f"{reference.away_abbreviation}@{reference.home_abbreviation}",
        away=reference.away_abbreviation,
        home=reference.home_abbreviation,
        game_date_et=tipoff_et.date().isoformat(),
        tipoff_utc=tipoff_utc,
        original_tipoff_utc=tipoff_utc,
        removed=False,
        captures={},
    )


def _amendment_payload(game: _GameState, amended_tipoff: str | None) -> JsonObject:
    return {
        "event_id": game.event_id,
        "matchup": game.matchup,
        "game_date": game.game_date_et,
        "original_tipoff": game.original_tipoff_utc,
        "previous_tipoff": game.tipoff_utc,
        "amended_tipoff": amended_tipoff,
        "note": "original captures are retained; cutoffs derive from the amended tipoff",
    }


def _in_window(game: _GameState, first_day: date, last_day: date) -> bool:
    game_date = _parse_iso_date(game.game_date_et, "game_date_et")
    return first_day <= game_date <= last_day


def _game_brief(reference: EspnGameRef) -> JsonObject:
    return {
        "event_id": reference.event_id,
        "matchup": f"{reference.away_abbreviation}@{reference.home_abbreviation}",
        "tipoff_utc": _utc_text(_parse_utc_text(reference.date_utc, "event date")),
    }


def _run_captures(context: _RunContext, state: _ScheduleState) -> list[JsonObject]:
    results: list[JsonObject] = []
    for game, cutoff_name, cutoff_utc in _due_captures(state, context.now):
        if context.dry_run:
            results.append(_dry_capture(game, cutoff_name, cutoff_utc))
        else:
            results.append(_execute_capture(context, game, cutoff_name, cutoff_utc))
    return results


def _due_captures(state: _ScheduleState, now: datetime) -> list[tuple[_GameState, str, datetime]]:
    due: list[tuple[_GameState, str, datetime]] = []
    for game in state.games:
        if game.removed:
            continue
        tipoff = _parse_utc_text(game.tipoff_utc, "tipoff_utc")
        if now >= tipoff:
            continue
        for name, delta in CUTOFFS:
            cutoff = tipoff - delta
            if name not in game.captures and cutoff <= now + CUTOFF_WINDOW:
                due.append((game, name, cutoff))
    due.sort(key=lambda item: item[2])
    return due


def _dry_capture(game: _GameState, cutoff_name: str, cutoff_utc: datetime) -> JsonObject:
    return {
        "action": "would_capture",
        "event_id": game.event_id,
        "matchup": game.matchup,
        "cutoff_type": cutoff_name,
        "cutoff_utc": _utc_text(cutoff_utc),
        "cutoff_et": cutoff_utc.astimezone(ET_ZONE).isoformat(),
    }


def _execute_capture(
    context: _RunContext,
    game: _GameState,
    cutoff_name: str,
    cutoff_utc: datetime,
) -> JsonObject:
    retrieved_at = datetime.now(UTC)
    day_dir = context.root / game.game_date_et
    entries = [
        _capture_injury_report(context.root, day_dir, game, cutoff_name, cutoff_utc),
        _capture_scoreboard(context.root, day_dir, game, cutoff_name),
    ]
    drift = abs((retrieved_at - cutoff_utc).total_seconds())
    within_window = drift <= CUTOFF_WINDOW.total_seconds()
    payload: JsonObject = {
        "game_date": game.game_date_et,
        "event_id": game.event_id,
        "matchup": game.matchup,
        "cutoff_type": cutoff_name,
        "scheduled_tipoff": game.tipoff_utc,
        "cutoff_scheduled": _utc_text(cutoff_utc),
        "cutoff_scheduled_et": cutoff_utc.astimezone(ET_ZONE).isoformat(),
        "retrieved_at": _utc_text(retrieved_at),
        "within_window": within_window,
        "captures": entries,
    }
    event_hash = _append_event(context.ledger_path, "capture", payload, retrieved_at)
    sidecar = day_dir / f"capture-{game.event_id}-{cutoff_name}.json"
    record = {**payload, "ledger_event_hash": event_hash}
    _ensure_create_only(sidecar, f"{canonical_json(record)}\n".encode())
    game.captures[cutoff_name] = event_hash
    return {
        "action": "captured",
        "event_id": game.event_id,
        "cutoff_type": cutoff_name,
        "ledger_event_hash": event_hash,
        "within_window": within_window,
    }


def _capture_injury_report(
    root: Path,
    day_dir: Path,
    game: _GameState,
    cutoff_name: str,
    cutoff_utc: datetime,
) -> JsonObject:
    cutoff_et = cutoff_utc.astimezone(ET_ZONE)
    tried = 0
    for slot in _report_candidates(cutoff_et):
        tried += 1
        url = _report_url(slot)
        filename = url.rsplit("/", 1)[-1]
        target = day_dir / f"injury-report-{game.event_id}-{cutoff_name}-{filename}"
        data = _fetch_or_reuse(url, target)
        if data is None:
            continue
        return {
            "kind": "injury_report_pdf",
            "status": "captured",
            "url": url,
            "report_slot_et": slot.isoformat(),
            "path": _rel(root, target),
            "sha256": bytes_sha256(data),
        }
    return {"kind": "injury_report_pdf", "status": "missing", "report_slots_tried": tried}


def _capture_scoreboard(
    root: Path, day_dir: Path, game: _GameState, cutoff_name: str
) -> JsonObject:
    day = _parse_iso_date(game.game_date_et, "game_date_et")
    url = ESPN_SCOREBOARD_URL.format(day=day.strftime("%Y%m%d"))
    target = day_dir / f"scoreboard-{game.event_id}-{cutoff_name}.json"
    data = _fetch_or_reuse(url, target)
    if data is None:
        raise ProspectiveCollectorError("ESPN scoreboard missing for a scheduled game date")
    return {
        "kind": "espn_scoreboard",
        "status": "captured",
        "url": url,
        "path": _rel(root, target),
        "sha256": bytes_sha256(data),
    }


def _report_candidates(cutoff_et: datetime) -> list[datetime]:
    candidates = _stepped_candidates(cutoff_et) + _slot_candidates(cutoff_et)
    unique = [candidate for candidate in dict.fromkeys(candidates) if candidate <= cutoff_et]
    return unique[:MAX_REPORT_CANDIDATES]


def _stepped_candidates(cutoff_et: datetime) -> list[datetime]:
    minute = (cutoff_et.minute // 15) * 15
    slot = cutoff_et.replace(minute=minute, second=0, microsecond=0)
    return [slot - REPORT_STEP * offset for offset in range(REPORT_LOOKBACK_STEPS + 1)]


def _slot_candidates(cutoff_et: datetime) -> list[datetime]:
    candidates: list[datetime] = []
    for day_offset in (0, 1):
        day = (cutoff_et - timedelta(days=day_offset)).date()
        for hour, minute in reversed(REPORT_FALLBACK_SLOTS):
            candidates.append(datetime(day.year, day.month, day.day, hour, minute, tzinfo=ET_ZONE))
    return candidates


def _report_url(slot_et: datetime) -> str:
    from nbainjuries._util import (  # noqa: PLC0415  # pyright: ignore[reportMissingTypeStubs]
        _gen_url,  # pyright: ignore[reportPrivateUsage]
    )

    return _gen_url(slot_et.replace(tzinfo=None))


def _fetch_or_reuse(url: str, target: Path) -> bytes | None:
    if target.exists():
        return target.read_bytes()
    payload = _fetch_bytes(url)
    if payload is None:
        return None
    _write_create_only(target, payload)
    return payload


def _fetch_bytes(url: str) -> bytes | None:
    """Download with three bounded attempts; None means the object does not exist."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        time.sleep(REQUEST_DELAY_SECONDS)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in {403, 404}:
                return None
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
        except (TimeoutError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise ProspectiveCollectorError("unreachable")


def _write_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ProspectiveCollectorError(f"capture already exists: {path}") from error
    with os.fdopen(descriptor, "wb") as file:
        file.write(payload)


def _ensure_create_only(path: Path, payload: bytes) -> bytes:
    if path.exists():
        return path.read_bytes()
    _write_create_only(path, payload)
    return payload


def _append_event(path: Path, event_type: str, payload: JsonObject, recorded_at: datetime) -> str:
    count, head, last_recorded = _load_chain(path)
    if last_recorded is not None and recorded_at < last_recorded:
        raise ProspectiveCollectorError("recorded_at would regress the capture ledger")
    body: JsonObject = {
        "schema_version": CAPTURE_LEDGER_SCHEMA_VERSION,
        "sequence": count + 1,
        "event_type": event_type,
        "recorded_at": _utc_text(recorded_at),
        "previous_hash": head,
        "payload": payload,
    }
    record = {**body, "event_hash": canonical_sha256(body)}
    _append_jsonl(path, record)
    return require_string(record["event_hash"], "event_hash")


def _append_jsonl(path: Path, record: JsonObject) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    try:
        temporary.write_text(existing + canonical_json(record) + "\n", encoding="utf-8")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_chain(path: Path) -> tuple[int, str, datetime | None]:
    if not path.exists():
        return (0, GENESIS_HASH, None)
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise ProspectiveCollectorError("capture ledger must end with a newline")
    head = GENESIS_HASH
    last_recorded: datetime | None = None
    lines = text.splitlines()
    for sequence, line in enumerate(lines, start=1):
        head, recorded = _verify_line(line, sequence, head)
        if last_recorded is not None and recorded < last_recorded:
            raise ProspectiveCollectorError("capture ledger recorded_at is not monotonic")
        last_recorded = recorded
    return (len(lines), head, last_recorded)


def _verify_line(line: str, sequence: int, previous: str) -> tuple[str, datetime]:
    try:
        record = parse_json_object(line)
        return _verify_envelope(record, sequence, previous)
    except JsonFormatError as error:
        raise ProspectiveCollectorError("capture ledger record is invalid") from error


def _verify_envelope(record: JsonObject, sequence: int, previous: str) -> tuple[str, datetime]:
    require_exact_keys(record, _ENVELOPE_KEYS, "capture ledger envelope")
    if required_field(record, "schema_version") != CAPTURE_LEDGER_SCHEMA_VERSION:
        raise ProspectiveCollectorError("capture ledger schema_version is unsupported")
    if required_field(record, "sequence") != sequence:
        raise ProspectiveCollectorError("capture ledger sequence is not contiguous")
    if required_field(record, "previous_hash") != previous:
        raise ProspectiveCollectorError("capture ledger hash chain is broken")
    body = {key: record[key] for key in _ENVELOPE_BODY_KEYS}
    event_hash = require_string(required_field(record, "event_hash"), "event_hash")
    if event_hash != canonical_sha256(body):
        raise ProspectiveCollectorError("capture ledger event hash mismatch")
    recorded = require_string(required_field(record, "recorded_at"), "recorded_at")
    return event_hash, _parse_utc_text(recorded, "recorded_at")


def _load_state(path: Path) -> _ScheduleState:
    if not path.exists():
        return _ScheduleState(last_poll_date=None, games=[])
    try:
        record = parse_json_object(path.read_text(encoding="utf-8"))
        require_exact_keys(record, {"schema_version", "last_poll_date", "games"}, "schedule state")
        if required_field(record, "schema_version") != STATE_SCHEMA_VERSION:
            raise ProspectiveCollectorError("schedule state schema_version is unsupported")
        last_poll = required_field(record, "last_poll_date")
        games = require_list(required_field(record, "games"), "games")
        return _ScheduleState(
            last_poll_date=require_string(last_poll, "last_poll_date")
            if last_poll is not None
            else None,
            games=[_game_state_from_record(item) for item in games],
        )
    except JsonFormatError as error:
        raise ProspectiveCollectorError("schedule state is corrupted") from error


def _game_state_from_record(value: object) -> _GameState:
    record = require_object(value, "game state")
    keys = {
        "event_id",
        "matchup",
        "away",
        "home",
        "game_date_et",
        "tipoff_utc",
        "original_tipoff_utc",
        "removed",
        "captures",
    }
    require_exact_keys(record, keys, "game state")
    removed = required_field(record, "removed")
    if not isinstance(removed, bool):
        raise ProspectiveCollectorError("game state removed must be a boolean")
    captures = require_object(required_field(record, "captures"), "captures")
    return _GameState(
        event_id=require_string(required_field(record, "event_id"), "event_id"),
        matchup=require_string(required_field(record, "matchup"), "matchup"),
        away=require_string(required_field(record, "away"), "away"),
        home=require_string(required_field(record, "home"), "home"),
        game_date_et=require_string(required_field(record, "game_date_et"), "game_date_et"),
        tipoff_utc=require_string(required_field(record, "tipoff_utc"), "tipoff_utc"),
        original_tipoff_utc=require_string(
            required_field(record, "original_tipoff_utc"), "original_tipoff_utc"
        ),
        removed=removed,
        captures={name: require_string(item, "captures") for name, item in captures.items()},
    )


def _save_state(path: Path, state: _ScheduleState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record: JsonObject = {
        "schema_version": STATE_SCHEMA_VERSION,
        "last_poll_date": state.last_poll_date,
        "games": [_game_state_to_record(game) for game in state.games],
    }
    temporary = path.with_name(f".{path.name}.part")
    try:
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _game_state_to_record(game: _GameState) -> JsonObject:
    return {
        "event_id": game.event_id,
        "matchup": game.matchup,
        "away": game.away,
        "home": game.home,
        "game_date_et": game.game_date_et,
        "tipoff_utc": game.tipoff_utc,
        "original_tipoff_utc": game.original_tipoff_utc,
        "removed": game.removed,
        "captures": game.captures,
    }


def _plan_game(game: _GameState) -> JsonObject:
    tipoff = _parse_utc_text(game.tipoff_utc, "tipoff_utc")
    return {
        "event_id": game.event_id,
        "matchup": game.matchup,
        "tipoff_utc": game.tipoff_utc,
        "tipoff_et": tipoff.astimezone(ET_ZONE).isoformat(),
        "removed": game.removed,
        "cutoffs": [
            {
                "name": name,
                "utc": _utc_text(tipoff - delta),
                "et": (tipoff - delta).astimezone(ET_ZONE).isoformat(),
                "status": "captured" if name in game.captures else "pending",
            }
            for name, delta in CUTOFFS
        ],
    }


def _require_private_root(storage_root: Path) -> Path:
    root = storage_root.absolute()
    if root.exists() and not root.is_dir():
        raise ProspectiveCollectorError("storage root is not a directory")
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(stat.S_IRWXU)
    if stat.S_IMODE(root.stat().st_mode) & (stat.S_IRWXG | stat.S_IRWXO):
        raise ProspectiveCollectorError("storage root must have no group or other permissions")
    return root


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _iso(days: list[date]) -> list[str]:
    return [day.isoformat() for day in days]


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc_text(text: str, field_name: str) -> datetime:
    if not text.endswith("Z"):
        raise ProspectiveCollectorError(f"{field_name} must use canonical UTC")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as error:
        raise ProspectiveCollectorError(f"{field_name} is not a valid datetime") from error
    return parsed.astimezone(UTC)


def _parse_iso_date(text: str, field_name: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise ProspectiveCollectorError(f"{field_name} must be YYYY-MM-DD") from error


def _parse_now(raw: str | None) -> datetime:
    if raw is None:
        return datetime.now(UTC)
    return _parse_utc_text(raw.strip(), "--now")


def _parse_arguments(argv: Sequence[str] | None) -> _Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="print one date's schedule and cutoffs (offline)")
    plan.add_argument("--date", required=True, help="game date, YYYY-MM-DD (America/New_York)")
    plan.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    run = subparsers.add_parser("run", help="execute the captures due now, then exit")
    run.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned work without network access or writes",
    )
    run.add_argument("--now", default=None, help="override current time, ISO-8601 UTC with Z")
    namespace = parser.parse_args(argv)
    return _Arguments(
        command=cast("str", namespace.command),
        date=cast("str | None", getattr(namespace, "date", None)),
        storage_root=cast("Path", namespace.storage_root),
        dry_run=bool(cast("bool", getattr(namespace, "dry_run", False))),
        now=cast("str | None", getattr(namespace, "now", None)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
