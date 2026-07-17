"""One-way preparation of the protocol-frozen open-modern NBA source."""

from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from math import isclose, isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile

from forecastfm.integrity import canonical_sha256, file_sha256, text_sha256

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
OPEN_MODERN_EXPOSURE_SHA256 = "ca4e036989f700fc94a0e367f1754cdd807d4df232450adcd73ddaef9e8c5c5e"
OPEN_MODERN_SOURCE_SEAL_SHA256 = "b5871f4139882b7333b07b843d7d511b41d47bb476cd689ee0fd9b73deaab265"
OPEN_MODERN_DEVELOPMENT_SHA256 = "02baf591b285f0537088b2fa05dfc0b98020b2a04a7441ee0eb7061a75881690"
OPEN_MODERN_DEVELOPMENT_IDS_SHA256 = (
    "5befb0fe1dc497826384ac1c05852f3c5ce13d00b5cc0f4a879564456baf1a1a"
)
OPEN_MODERN_TRAIN_COUNT = 5_249
OPEN_MODERN_TRAIN_IDS_SHA256 = "f52d201bfe90cf88c1046275584ad197a7d7b880f54b5bd5b2f5c3007163649c"
OPEN_MODERN_VALIDATION_COUNT = 1_143
OPEN_MODERN_VALIDATION_IDS_SHA256 = (
    "63216edec7012648bf886471dca23304b213ae35624a7da1601e9eac73558a98"
)
OPEN_MODERN_DEVELOPMENT_COUNT = OPEN_MODERN_TRAIN_COUNT + OPEN_MODERN_VALIDATION_COUNT
OPEN_MODERN_TEST_COUNT = 2_494
OPEN_MODERN_TEST_INPUTS_SHA256 = "e92bab54ce46d2cb38de2969e529092bad607394528a68faf0a58231b089b0ed"
OPEN_MODERN_TEST_IDS_SHA256 = "74fffad360eaa74d80f2a8f7f838173cee21658c8967b0adb2312ef3f3b76afb"

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
class _SourceGame:
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


@dataclass(frozen=True, slots=True)
class _ArtifactContract:
    columns: tuple[str, ...]
    sha256: str
    row_count: int
    ordered_ids_sha256: str


_DEVELOPMENT_CONTRACT = _ArtifactContract(
    DEVELOPMENT_COLUMNS,
    OPEN_MODERN_DEVELOPMENT_SHA256,
    OPEN_MODERN_DEVELOPMENT_COUNT,
    OPEN_MODERN_DEVELOPMENT_IDS_SHA256,
)
_TEST_INPUTS_CONTRACT = _ArtifactContract(
    TEST_INPUT_COLUMNS,
    OPEN_MODERN_TEST_INPUTS_SHA256,
    OPEN_MODERN_TEST_COUNT,
    OPEN_MODERN_TEST_IDS_SHA256,
)


def _load_source_games(
    source_bytes: bytes,
    *,
    expected_sha256: str,
) -> tuple[_SourceGame, ...]:
    """Validate and parse one immutable source snapshot chronologically."""
    if sha256(source_bytes).hexdigest() != expected_sha256:
        raise OpenModernError("open-modern source SHA-256 does not match")
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OpenModernError("open-modern source is not valid UTF-8") from error

    games: list[_SourceGame] = []
    with io.StringIO(source_text, newline="") as file:
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
    source_bytes = source_path.read_bytes()
    if len(source_bytes) != OPEN_MODERN_SOURCE_BYTES:
        raise OpenModernError("open-modern source byte size does not match")
    if file_sha256(protocol_path) != OPEN_MODERN_PROTOCOL_SHA256:
        raise OpenModernError("open-modern protocol SHA-256 does not match")

    games = _load_source_games(
        source_bytes,
        expected_sha256=OPEN_MODERN_SOURCE_SHA256,
    )
    _require_pinned_cohort(games)
    return _write_artifacts(
        games,
        development_path,
        test_inputs_path,
        protocol_path,
        seal_path,
    )


def require_open_modern_development(
    development_path: Path,
    seal_path: Path,
    protocol_path: Path,
    exposure_path: Path,
) -> None:
    """Require the exact committed controls and labeled development artifact."""
    _require_control_files(seal_path, protocol_path, exposure_path)
    _require_csv_artifact(development_path, _DEVELOPMENT_CONTRACT)


def require_open_modern_test_inputs(
    test_inputs_path: Path,
    seal_path: Path,
    protocol_path: Path,
    exposure_path: Path,
) -> None:
    """Require the exact committed controls and target-free holdout inputs."""
    _require_control_files(seal_path, protocol_path, exposure_path)
    _require_csv_artifact(test_inputs_path, _TEST_INPUTS_CONTRACT)


def _require_control_files(seal_path: Path, protocol_path: Path, exposure_path: Path) -> None:
    expected = (
        (seal_path, OPEN_MODERN_SOURCE_SEAL_SHA256, "source seal"),
        (protocol_path, OPEN_MODERN_PROTOCOL_SHA256, "protocol"),
        (exposure_path, OPEN_MODERN_EXPOSURE_SHA256, "exposure record"),
    )
    for path, expected_hash, name in expected:
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise OpenModernError(f"open-modern {name} does not match")


def _require_csv_artifact(path: Path, contract: _ArtifactContract) -> None:
    if not path.is_file():
        raise OpenModernError(f"open-modern artifact does not match: {path.name}")
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != contract.sha256:
        raise OpenModernError(f"open-modern artifact does not match: {path.name}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OpenModernError(f"open-modern artifact is not UTF-8: {path.name}") from error
    with io.StringIO(text, newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != contract.columns:
            raise OpenModernError(f"open-modern artifact columns differ: {path.name}")
        ids = tuple(_required(row, "game_id", line) for line, row in enumerate(reader, start=2))
    if len(ids) != contract.row_count or len(set(ids)) != len(ids):
        raise OpenModernError(f"open-modern artifact cohort differs: {path.name}")
    if canonical_sha256(ids) != contract.ordered_ids_sha256:
        raise OpenModernError(f"open-modern artifact ID order differs: {path.name}")


def _write_artifacts(
    games: Sequence[_SourceGame],
    development_path: Path,
    test_inputs_path: Path,
    protocol_path: Path,
    seal_path: Path,
) -> OpenModernSealResult:
    """Write deterministic artifacts without persisting test answers."""
    ordered = tuple(sorted(games, key=_game_sort_key))
    _require_unique_games(ordered)
    development = tuple(game for game in ordered if game.season in DEVELOPMENT_SEASONS)
    test = tuple(game for game in ordered if game.season in TEST_SEASONS)
    if len(development) + len(test) != len(ordered):
        raise OpenModernError("open-modern source contains an undeclared season")
    if {game.season for game in ordered} != set(ALL_SEASONS):
        raise OpenModernError("open-modern source must cover every declared season")

    development_text = _development_text(development)
    test_inputs_text = _test_inputs_text(test)
    development_hash = text_sha256(development_text)
    test_inputs_hash = text_sha256(test_inputs_text)
    seal = _seal_dict(
        development,
        test,
        _WrittenArtifact(development_path.name, development_hash),
        _WrittenArtifact(test_inputs_path.name, test_inputs_hash),
        protocol_path,
    )
    seal_text = json.dumps(seal, indent=2, sort_keys=True, allow_nan=False) + "\n"
    artifacts = (
        (development_path, development_text),
        (test_inputs_path, test_inputs_text),
        (seal_path, seal_text),
    )
    _write_exact_artifacts(artifacts)

    return OpenModernSealResult(
        development_count=len(development),
        test_input_count=len(test),
        development_sha256=development_hash,
        test_inputs_sha256=test_inputs_hash,
        seal_sha256=text_sha256(seal_text),
    )


def _parse_game(row: Mapping[str, str | None], line_number: int) -> _SourceGame:
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

    return _SourceGame(
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


def _game_sort_key(game: _SourceGame) -> tuple[date, str, str, str]:
    return (game.game_date, game.team1, game.team2, game.game_id)


def _require_unique_games(games: Sequence[_SourceGame]) -> None:
    ids = {game.game_id for game in games}
    if len(ids) != len(games):
        raise OpenModernError("open-modern game identities must be unique")


def _require_pinned_cohort(games: Sequence[_SourceGame]) -> None:
    if len(games) != OPEN_MODERN_SOURCE_ROWS:
        raise OpenModernError("open-modern source row count does not match")
    dates = tuple(game.game_date for game in games)
    if not dates or min(dates) != _FIRST_GAME_DATE or max(dates) != _LAST_GAME_DATE:
        raise OpenModernError("open-modern source date range does not match")


def _development_text(games: Sequence[_SourceGame]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
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
    return output.getvalue()


def _test_inputs_text(games: Sequence[_SourceGame]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
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
    return output.getvalue()


def _preflight_artifacts(artifacts: Sequence[tuple[Path, str]]) -> None:
    """Reject every conflicting destination before creating any artifact."""
    for path, text in artifacts:
        if path.exists() and (not path.is_file() or path.read_bytes() != text.encode("utf-8")):
            raise OpenModernError(f"refusing to replace different artifact: {path.name}")


def _write_exact_artifacts(artifacts: Sequence[tuple[Path, str]]) -> None:
    """Atomically publish a preflighted set, rolling back incomplete sets."""
    _preflight_artifacts(artifacts)
    for path, _ in artifacts:
        path.parent.mkdir(parents=True, exist_ok=True)

    staged = _stage_artifacts(artifacts)
    try:
        _preflight_artifacts(artifacts)
        _publish_staged_artifacts(staged, artifacts)
    finally:
        _remove_paths(partial_path for _, partial_path in staged)


def _stage_artifacts(
    artifacts: Sequence[tuple[Path, str]],
) -> tuple[tuple[Path, Path], ...]:
    """Flush every payload to a temporary file before publishing any target."""
    staged: list[tuple[Path, Path]] = []
    complete = False
    try:
        for path, text in artifacts:
            with NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".part",
                delete=False,
            ) as file:
                partial_path = Path(file.name)
                staged.append((path, partial_path))
                file.write(text.encode("utf-8"))
                file.flush()
                os.fsync(file.fileno())
        complete = True
        return tuple(staged)
    finally:
        if not complete:
            _remove_paths(partial_path for _, partial_path in staged)


def _publish_staged_artifacts(
    staged: Sequence[tuple[Path, Path]],
    artifacts: Sequence[tuple[Path, str]],
) -> None:
    """Link staged files exclusively and remove every new target on failure."""
    published: list[Path] = []
    complete = False
    try:
        for path, partial_path in staged:
            if path.exists():
                continue
            _link_exclusively(partial_path, path)
            published.append(path)

        _preflight_artifacts(artifacts)
        if any(not path.is_file() for path, _ in artifacts):
            raise OpenModernError("an artifact disappeared during immutable write")
        complete = True
    finally:
        if not complete:
            _remove_paths(reversed(published))


def _link_exclusively(partial_path: Path, path: Path) -> None:
    try:
        os.link(partial_path, path)
    except FileExistsError as error:
        raise OpenModernError(f"artifact appeared during exclusive write: {path.name}") from error


def _remove_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _seal_dict(
    development: Sequence[_SourceGame],
    test: Sequence[_SourceGame],
    development_artifact: _WrittenArtifact,
    test_inputs_artifact: _WrittenArtifact,
    protocol_path: Path,
) -> dict[str, object]:
    exposure_path = protocol_path.with_name("EXPOSURE.md")
    return {
        "schema_version": 2,
        "status": "sealed_for_development",
        "protocol_sha256": file_sha256(protocol_path),
        "claim": {
            "classification": "protocol_frozen_historical_holdout",
            "literally_unopened": False,
            "known_test_label_exposures": 1,
            "exposure_record": exposure_path.name,
            "exposure_record_sha256": file_sha256(exposure_path),
        },
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
        "selection_splits": {
            "train": _cohort_dict(
                tuple(game for game in development if game.season in TRAIN_SEASONS),
                TRAIN_SEASONS,
            ),
            "validation": _cohort_dict(
                tuple(game for game in development if game.season in VALIDATION_SEASONS),
                VALIDATION_SEASONS,
            ),
        },
        "test_inputs": _split_dict(
            test,
            TEST_SEASONS,
            test_inputs_artifact.filename,
            test_inputs_artifact.sha256,
        ),
        "test_answers": {
            "answer_artifact_written": False,
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
    games: Sequence[_SourceGame],
    seasons: Sequence[int],
    filename: str,
    file_hash: str,
) -> dict[str, object]:
    return {
        **_cohort_dict(games, seasons),
        "filename": filename,
        "sha256": file_hash,
    }


def _cohort_dict(
    games: Sequence[_SourceGame],
    seasons: Sequence[int],
) -> dict[str, object]:
    return {
        "seasons": list(seasons),
        "row_count": len(games),
        "season_counts": {
            str(season): sum(game.season == season for game in games) for season in seasons
        },
        "ordered_ids_sha256": _ordered_ids_sha256(games),
    }


def _ordered_ids_sha256(games: Sequence[_SourceGame]) -> str:
    return canonical_sha256([game.game_id for game in games])


def _test_answers_sha256(games: Sequence[_SourceGame]) -> str:
    answers = [
        {
            "game_id": game.game_id,
            "prob1_outcome": int(game.team1_won),
            "prob2_outcome": int(not game.team1_won),
        }
        for game in games
    ]
    return canonical_sha256(answers)
