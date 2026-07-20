"""Rolling weekly evaluation runner for the predeclared 2026-27 NBA candidate.

Operates the weekly cadence of ``prospective/PREDECLARED_2026_27_CANDIDATE.md`` (season opens
2026-10-21, evaluation label 2027):

- ``refresh`` re-fetches the ESPN season window (2026-10-21 through today) by delegating to
  ``examples.fetch_espn_season.main`` (no duplicated fetch code), then copies
  ``data/raw/espn/espn_2025.csv`` to a dated backup ``espn_2025_backup_<utcdate>.csv`` so the
  rolling 2026-27 file and the archived 2025-26 use of the same filename never alias.
- ``evaluate --as-of YYYY-MM-DD`` runs the frozen prototype pipeline (team_form excluded,
  training seasons 2022-2026, evaluation season 2027) into
  ``data/processed/prospective_2026_27``.
- ``track --as-of YYYY-MM-DD`` evaluates, then appends one canonical JSON line per date to
  ``data/processed/prospective_2026_27/tracker.jsonl``; re-tracking the same date rewrites
  that line instead of duplicating it. Tracker rows are informational weekly diagnostics
  only — the formal gate evaluates once, at season end, per the predeclared freeze.

Sanctioned seam: ``examples.run_private_prototype.SEASON_FILES`` has no 2027 entry and the
fetcher always writes ``espn_2025.csv``. This runner replaces the module attribute with a
new mapping that adds ``{2027: Path("data/raw/espn/espn_2025.csv")}`` immediately before
calling ``main()`` and restores the original attribute afterwards. The original dict is
never mutated and ``run_private_prototype.py`` is never edited; this monkeypatch is the
declared, documented seam for the prospective season.

Every command accepts ``--dry-run`` to print the plan with no network access and no writes.
``evaluate``/``track`` exit 2 when the season has not meaningfully started (fewer than
``MIN_STARTED_GAMES`` games in the season CSV).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from forecastfm.json_utils import parse_json_object, require_float, require_list, require_object

if __package__ in {None, ""}:  # direct `python examples/run_prospective_weekly.py` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples import fetch_espn_season, run_private_prototype

type JsonObject = dict[str, object]

SEASON_OPEN_DATE = date(2026, 10, 21)
EVALUATION_SEASON = 2027
TRAINING_SEASONS_ARGUMENT = "2022,2023,2024,2025,2026"
EXCLUDED_FAMILIES_ARGUMENT = "team_form"
ESPN_SEASON_CSV = Path("data/raw/espn/espn_2025.csv")
OUTPUT_DIR = Path("data/processed/prospective_2026_27")
MANIFEST_FILENAME = "manifest.json"
TRACKER_FILENAME = "tracker.jsonl"
MIN_STARTED_GAMES = 50
EXIT_NOT_STARTED = 2


@dataclass(frozen=True, slots=True)
class _Arguments:
    command: str
    as_of: str | None
    dry_run: bool


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the ``refresh``, ``evaluate``, and ``track`` subcommands."""
    args = _parse_arguments(argv)
    if args.command == "refresh":
        return _cmd_refresh(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    return _cmd_track(args)


def _cmd_refresh(args: _Arguments) -> int:
    end = _parse_as_of(args.as_of) if args.as_of is not None else datetime.now(UTC).date()
    backup = _backup_path(datetime.now(UTC).date())
    plan: JsonObject = {
        "command": "refresh",
        "start_date": SEASON_OPEN_DATE.isoformat(),
        "end_date": end.isoformat(),
        "season_csv": str(ESPN_SEASON_CSV),
        "backup_csv": str(backup),
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    code = fetch_espn_season.main(
        ["--start-date", SEASON_OPEN_DATE.isoformat(), "--end-date", end.isoformat()]
    )
    if code != 0:
        return code
    if not ESPN_SEASON_CSV.exists():
        print(f"refresh failed: {ESPN_SEASON_CSV} was not written")
        return 1
    shutil.copyfile(ESPN_SEASON_CSV, backup)
    print(json.dumps({**plan, "games": _count_csv_games(ESPN_SEASON_CSV)}, indent=2))
    return 0


def _cmd_evaluate(args: _Arguments) -> int:
    code, _report = _run_evaluation("evaluate", args.as_of, dry_run=args.dry_run)
    return code


def _cmd_track(args: _Arguments) -> int:
    code, report = _run_evaluation("track", args.as_of, dry_run=args.dry_run)
    if code != 0 or report is None:
        return code
    if args.as_of is None:
        raise RuntimeError("unreachable: --as-of is required for track")
    row = tracker_row(args.as_of, report)
    path = OUTPUT_DIR / TRACKER_FILENAME
    append_tracker_row(path, row)
    print(json.dumps({"tracker": str(path), "row": row}, indent=2))
    return 0


def _run_evaluation(
    command: str, as_of: str | None, *, dry_run: bool
) -> tuple[int, JsonObject | None]:
    """Run the patched prototype pipeline, or print the plan under ``--dry-run``."""
    games = _count_csv_games(ESPN_SEASON_CSV)
    if dry_run:
        print(json.dumps(_evaluation_plan(command, as_of, games), indent=2))
        return 0, None
    if games < MIN_STARTED_GAMES:
        print(
            f"2026-27 season has not meaningfully started: {games} games in "
            f"{ESPN_SEASON_CSV} (< {MIN_STARTED_GAMES}); run refresh and retry after "
            "more games have been played"
        )
        return EXIT_NOT_STARTED, None
    original = run_private_prototype.SEASON_FILES
    run_private_prototype.SEASON_FILES = patched_season_files(original)
    try:
        code = run_private_prototype.main(_driver_argv())
    finally:
        run_private_prototype.SEASON_FILES = original
    if code != 0:
        return code, None
    report = parse_json_object((OUTPUT_DIR / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    return 0, report


def _evaluation_plan(command: str, as_of: str | None, games: int) -> JsonObject:
    plan: JsonObject = {
        "command": command,
        "as_of": as_of,
        "dry_run": True,
        "games_in_season_csv": games,
        "season_csv": str(ESPN_SEASON_CSV),
        "driver": "examples.run_private_prototype",
        "driver_argv": _driver_argv(),
        "season_files_patch": {str(EVALUATION_SEASON): str(ESPN_SEASON_CSV)},
        "output_dir": str(OUTPUT_DIR),
    }
    if command == "track":
        plan["tracker"] = str(OUTPUT_DIR / TRACKER_FILENAME)
    return plan


def _driver_argv() -> list[str]:
    return [
        "--exclude-families",
        EXCLUDED_FAMILIES_ARGUMENT,
        "--training-seasons",
        TRAINING_SEASONS_ARGUMENT,
        "--evaluation-seasons",
        str(EVALUATION_SEASON),
        "--output-dir",
        str(OUTPUT_DIR),
    ]


def patched_season_files(original: Mapping[int, Path]) -> dict[int, Path]:
    """Return a new SEASON_FILES mapping adding the 2027 label; never mutate the input."""
    patched = dict(original)
    patched[EVALUATION_SEASON] = ESPN_SEASON_CSV
    return patched


def tracker_row(as_of: str, report: JsonObject) -> JsonObject:
    """Distil one pipeline report into the canonical weekly tracker line."""
    variants = require_object(report.get("variants"), "variants")
    by_variant: dict[str, dict[str, JsonObject]] = {}
    for key, payload in variants.items():
        variant, baseline = key.split("_vs_", maxsplit=1)
        arm = require_object(payload, key)
        for season_entry in require_list(arm.get("seasons"), f"{key}.seasons"):
            season = require_object(season_entry, f"{key}.seasons[]")
            label = str(int(require_float(season.get("season"), "season")))
            seasons = by_variant.setdefault(variant, {})
            record = seasons.get(label)
            if record is None:
                record = _season_record(season)
                seasons[label] = record
            record[f"vs_{baseline}"] = {
                "mean_baseline_relative_log_score": require_float(
                    season.get("mean_baseline_relative_log_score"),
                    "mean_baseline_relative_log_score",
                ),
                "lower_one_sided_95": require_float(
                    season.get("lower_one_sided_95"), "lower_one_sided_95"
                ),
                "passes": season.get("passes"),
            }
    return {
        "as_of": as_of,
        "games": int(require_float(report.get("evaluation_games"), "evaluation_games")),
        "variants": by_variant,
    }


def _season_record(season: JsonObject) -> JsonObject:
    model = require_object(season.get("model"), "model")
    return {
        "games": int(require_float(season.get("game_count"), "game_count")),
        "mean_log_loss": require_float(model.get("mean_log_loss"), "mean_log_loss"),
    }


def append_tracker_row(path: Path, row: JsonObject) -> None:
    """Append ``row`` canonically, rewriting any existing line for the same ``as_of`` date."""
    kept: list[JsonObject] = []
    if path.exists():
        kept.extend(
            parse_json_object(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    kept = [existing for existing in kept if existing.get("as_of") != row["as_of"]]
    kept.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(existing, sort_keys=True) + "\n" for existing in kept),
        encoding="utf-8",
    )


def _count_csv_games(path: Path) -> int:
    """Count games in a nbastats CSV (one data row per game, minus the header)."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as file:
        return max(sum(1 for _ in file) - 1, 0)


def _backup_path(day: date) -> Path:
    return ESPN_SEASON_CSV.with_name(f"espn_2025_backup_{day:%Y%m%d}.csv")


def _parse_as_of(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise SystemExit(f"--as-of must be YYYY-MM-DD, got {text!r}") from error


def _parse_arguments(argv: Sequence[str] | None) -> _Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    refresh = subparsers.add_parser(
        "refresh", help="refetch the 2026-27 season window and back up the season CSV"
    )
    refresh.add_argument("--as-of", default=None, help="end date YYYY-MM-DD (default: today UTC)")
    refresh.add_argument("--dry-run", action="store_true")
    for command in ("evaluate", "track"):
        subparser = subparsers.add_parser(
            command,
            help=(
                "run the frozen pipeline on the 2026-27 games played so far"
                if command == "evaluate"
                else "evaluate, then append the canonical weekly tracker line"
            ),
        )
        subparser.add_argument("--as-of", required=True, help="label date YYYY-MM-DD")
        subparser.add_argument("--dry-run", action="store_true")
    namespace = parser.parse_args(argv)
    as_of = None if namespace.as_of is None else str(namespace.as_of)
    if as_of is not None:
        _parse_as_of(as_of)
    return _Arguments(
        command=str(namespace.command),
        as_of=as_of,
        dry_run=bool(namespace.dry_run),
    )


if __name__ == "__main__":
    raise SystemExit(main())
