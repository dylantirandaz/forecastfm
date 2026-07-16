"""One-way preparation of the protocol-frozen open-modern NBA source."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from math import isclose, isfinite
from pathlib import Path

from forecastfm.integrity import canonical_sha256, file_sha256

OPEN_MODERN_SOURCE_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/checking-our-work-data/"
    "f6d5b2e1d6da2889345d381c41431f9a4ee208dd/nba_games.csv"
)
OPEN_MODERN_SOURCE_COMMIT = "f6d5b2e1d6da2889345d381c41431f9a4ee208dd"
OPEN_MODERN_SOURCE_GIT_BLOB = "cd4308f1753377b4b1e197640539233a328e6b33"
OPEN_MODERN_SOURCE_SHA256 = "fb99d5a4870ae761684313b36433926f9f53b0e0d272b922c73aa287da20ab20"
OPEN_MODERN_SOURCE_BYTES = 655_771
OPEN_MODERN_SOURCE_ROWS = 8_886
OPEN_MODERN_PROTOCOL_SHA256 = "36bed36f546e1a7f9fe6dbc6f9c3582f1a41746cbee738ed89f92918cacf6aae"

TRAIN_SEASONS = (2016, 2017, 2018, 2019)
VALIDATION_SEASONS = (2020,)
TEST_SEASONS = (2021, 2022)
DEVELOPMENT_SEASONS = TRAIN_SEASONS + VALIDATION_SEASONS
ALL_SEASONS = DEVELOPMENT_SEASONS + TEST_SEASONS

SOURCE_COLUMNS = (
    "season",
    "date",
    "team1",
    "team2",
    "prob1",
    "prob1_outcome",
    "prob2",
    "prob2_outcome",
)
DEVELOPMENT_COLUMNS = ("game_id", *SOURCE_COLUMNS)
TEST_INPUT_COLUMNS = (
    "game_id",
    "season",
    "date",
    "team1",
    "team2",
    "prob1",
    "prob2",
)

_FIRST_GAME_DATE = date(2015, 10, 27)
_LAST_GAME_DATE = date(2022, 6, 16)


class OpenModernError(ValueError):
    """Raised when the open-modern source or seal violates its contract."""


@dataclass(frozen=True, slots=True)
class OpenModernGame:
    """One validated source game, including its development-only answer."""

    game_id: str
    season: int
    game_date: date
    team1: str
    team2: str
    prob1: float
    prob2: float
    team1_won: bool


@dataclass(frozen=True, slots=True)
class OpenModernSealResult:
    """Safe structural summary returned by the one-way sealer."""

    development_count: int
    test_input_count: int
    development_sha256: str
    test_inputs_sha256: str
    seal_sha256: str


@dataclass(frozen=True, slots=True)
class _WrittenArtifact:
    filename: str
    sha256: str


def load_open_modern_games(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[OpenModernGame, ...]:
    """Load and validate an exact source file, then sort it chronologically."""
    if file_sha256(path) != expected_sha256:
        raise OpenModernError("open-modern source SHA-256 does not match")

    games: list[OpenModernGame] = []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != SOURCE_COLUMNS:
            raise OpenModernError("open-modern source columns do not match")
        for line_number, row in enumerate(reader, start=2):
            games.append(_parse_game(row, line_number))

    ordered = tuple(sorted(games, key=_game_sort_key))
    _require_unique_games(ordered)
    return ordered


def seal_open_modern_source(
    source_path: Path,
    development_path: Path,
    test_inputs_path: Path,
    protocol_path: Path,
    seal_path: Path,
) -> OpenModernSealResult:
    """Write labeled development rows and target-free test inputs from pinned bytes."""
    if source_path.stat().st_size != OPEN_MODERN_SOURCE_BYTES:
        raise OpenModernError("open-modern source byte size does not match")
    if file_sha256(protocol_path) != OPEN_MODERN_PROTOCOL_SHA256:
        raise OpenModernError("open-modern protocol SHA-256 does not match")

    games = load_open_modern_games(
        source_path,
        expected_sha256=OPEN_MODERN_SOURCE_SHA256,
    )
    _require_pinned_cohort(games)
    return write_open_modern_artifacts(
        games,
        development_path,
        test_inputs_path,
        protocol_path,
        seal_path,
    )


def write_open_modern_artifacts(
    games: Sequence[OpenModernGame],
    development_path: Path,
    test_inputs_path: Path,
    protocol_path: Path,
    seal_path: Path,
) -> OpenModernSealResult:
    """Write deterministic artifacts without materializing test answers."""
    ordered = tuple(sorted(games, key=_game_sort_key))
    _require_unique_games(ordered)
    development = tuple(game for game in ordered if game.season in DEVELOPMENT_SEASONS)
    test = tuple(game for game in ordered if game.season in TEST_SEASONS)
    if len(development) + len(test) != len(ordered):
        raise OpenModernError("open-modern source contains an undeclared season")
    if {game.season for game in ordered} != set(ALL_SEASONS):
        raise OpenModernError("open-modern source must cover every declared season")

    development_path.parent.mkdir(parents=True, exist_ok=True)
    test_inputs_path.parent.mkdir(parents=True, exist_ok=True)
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    _write_development(development_path, development)
    _write_test_inputs(test_inputs_path, test)

    development_hash = file_sha256(development_path)
    test_inputs_hash = file_sha256(test_inputs_path)
    seal = _seal_dict(
        development,
        test,
        _WrittenArtifact(development_path.name, development_hash),
        _WrittenArtifact(test_inputs_path.name, test_inputs_hash),
        file_sha256(protocol_path),
    )
    seal_path.write_text(
        json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return OpenModernSealResult(
        development_count=len(development),
        test_input_count=len(test),
        development_sha256=development_hash,
        test_inputs_sha256=test_inputs_hash,
        seal_sha256=file_sha256(seal_path),
    )


def _parse_game(row: Mapping[str, str | None], line_number: int) -> OpenModernGame:
    season = _integer(_required(row, "season", line_number), "season", line_number)
    try:
        game_date = date.fromisoformat(_required(row, "date", line_number))
    except ValueError as error:
        raise OpenModernError(f"line {line_number}: date is invalid") from error
    if game_date.year not in {season - 1, season}:
        raise OpenModernError(f"line {line_number}: date and season disagree")

    team1 = _required(row, "team1", line_number)
    team2 = _required(row, "team2", line_number)
    if team1 == team2:
        raise OpenModernError(f"line {line_number}: teams must differ")
    prob1 = _probability(_required(row, "prob1", line_number), "prob1", line_number)
    prob2 = _probability(_required(row, "prob2", line_number), "prob2", line_number)
    if not isclose(prob1 + prob2, 1.0, abs_tol=1e-9):
        raise OpenModernError(f"line {line_number}: probabilities must sum to one")

    outcome1 = _binary(_required(row, "prob1_outcome", line_number), line_number)
    outcome2 = _binary(_required(row, "prob2_outcome", line_number), line_number)
    if outcome1 + outcome2 != 1:
        raise OpenModernError(f"line {line_number}: outcomes must be complementary")

    return OpenModernGame(
        game_id=_game_id(season, game_date, team1, team2),
        season=season,
        game_date=game_date,
        team1=team1,
        team2=team2,
        prob1=prob1,
        prob2=prob2,
        team1_won=outcome1 == 1,
    )


def _required(row: Mapping[str, str | None], field: str, line_number: int) -> str:
    value = row.get(field)
    if value is None or not value.strip() or value != value.strip():
        raise OpenModernError(f"line {line_number}: {field} is missing or untrimmed")
    return value


def _integer(value: str, field: str, line_number: int) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise OpenModernError(f"line {line_number}: {field} is not an integer") from error
    if str(result) != value:
        raise OpenModernError(f"line {line_number}: {field} is not canonical")
    return result


def _probability(value: str, field: str, line_number: int) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise OpenModernError(f"line {line_number}: {field} is not numeric") from error
    if not isfinite(result) or not 0.0 < result < 1.0:
        raise OpenModernError(f"line {line_number}: {field} must be an interior probability")
    return result


def _binary(value: str, line_number: int) -> int:
    if value == "0":
        return 0
    if value == "1":
        return 1
    raise OpenModernError(f"line {line_number}: outcome must be zero or one")


def _game_id(season: int, game_date: date, team1: str, team2: str) -> str:
    first, second = sorted((team1, team2))
    identity = f"{season}|{game_date.isoformat()}|{first}|{second}"
    return f"nba-538-{sha256(identity.encode('utf-8')).hexdigest()[:24]}"


def _game_sort_key(game: OpenModernGame) -> tuple[date, str, str, str]:
    return (game.game_date, game.team1, game.team2, game.game_id)


def _require_unique_games(games: Sequence[OpenModernGame]) -> None:
    ids = {game.game_id for game in games}
    if len(ids) != len(games):
        raise OpenModernError("open-modern game identities must be unique")


def _require_pinned_cohort(games: Sequence[OpenModernGame]) -> None:
    if len(games) != OPEN_MODERN_SOURCE_ROWS:
        raise OpenModernError("open-modern source row count does not match")
    dates = tuple(game.game_date for game in games)
    if not dates or min(dates) != _FIRST_GAME_DATE or max(dates) != _LAST_GAME_DATE:
        raise OpenModernError("open-modern source date range does not match")


def _write_development(path: Path, games: Sequence[OpenModernGame]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(DEVELOPMENT_COLUMNS)
        for game in games:
            writer.writerow(
                (
                    game.game_id,
                    game.season,
                    game.game_date.isoformat(),
                    game.team1,
                    game.team2,
                    str(game.prob1),
                    int(game.team1_won),
                    str(game.prob2),
                    int(not game.team1_won),
                )
            )


def _write_test_inputs(path: Path, games: Sequence[OpenModernGame]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(TEST_INPUT_COLUMNS)
        for game in games:
            writer.writerow(
                (
                    game.game_id,
                    game.season,
                    game.game_date.isoformat(),
                    game.team1,
                    game.team2,
                    str(game.prob1),
                    str(game.prob2),
                )
            )


def _seal_dict(
    development: Sequence[OpenModernGame],
    test: Sequence[OpenModernGame],
    development_artifact: _WrittenArtifact,
    test_inputs_artifact: _WrittenArtifact,
    protocol_hash: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "sealed_for_development",
        "protocol_sha256": protocol_hash,
        "source": {
            "url": OPEN_MODERN_SOURCE_URL,
            "commit": OPEN_MODERN_SOURCE_COMMIT,
            "git_blob_sha": OPEN_MODERN_SOURCE_GIT_BLOB,
            "sha256": OPEN_MODERN_SOURCE_SHA256,
            "byte_size": OPEN_MODERN_SOURCE_BYTES,
            "row_count": len(development) + len(test),
        },
        "development": _split_dict(
            development,
            DEVELOPMENT_SEASONS,
            development_artifact.filename,
            development_artifact.sha256,
        ),
        "test_inputs": _split_dict(
            test,
            TEST_SEASONS,
            test_inputs_artifact.filename,
            test_inputs_artifact.sha256,
        ),
        "test_answers": {
            "materialized": False,
            "row_count": len(test),
            "ordered_ids_sha256": _ordered_ids_sha256(test),
            "sha256": _test_answers_sha256(test),
        },
        "safety": {
            "test_input_columns": list(TEST_INPUT_COLUMNS),
            "test_labels_written": False,
            "test_metrics_computed": False,
        },
    }


def _split_dict(
    games: Sequence[OpenModernGame],
    seasons: Sequence[int],
    filename: str,
    file_hash: str,
) -> dict[str, object]:
    return {
        "seasons": list(seasons),
        "row_count": len(games),
        "season_counts": {
            str(season): sum(game.season == season for game in games) for season in seasons
        },
        "ordered_ids_sha256": _ordered_ids_sha256(games),
        "filename": filename,
        "sha256": file_hash,
    }


def _ordered_ids_sha256(games: Sequence[OpenModernGame]) -> str:
    return canonical_sha256([game.game_id for game in games])


def _test_answers_sha256(games: Sequence[OpenModernGame]) -> str:
    answers = [
        {
            "game_id": game.game_id,
            "prob1_outcome": int(game.team1_won),
            "prob2_outcome": int(not game.team1_won),
        }
        for game in games
    ]
    return canonical_sha256(answers)
