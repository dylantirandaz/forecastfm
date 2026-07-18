"""Independent sealed evaluation gate for untouched NBA seasons.

This module proves local artifact integrity, chronological calibration, exact cohort
coverage, and deterministic scoring. It does not prove that artifacts were externally
precommitted before outcomes became known, provide a trusted timestamp, or derive fields
from raw provider bytes. Those remain separate external and connector proofs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from math import copysign, exp, fsum, isfinite, log
from pathlib import Path

from forecastfm.integrity import canonical_json, canonical_sha256, text_sha256
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_string,
    required_field,
)
from forecastfm.outcome_v2_metrics import (
    BinaryForecast,
    DatedBinaryCohortMember,
    MultiSeasonEvaluation,
    OutcomeV2MetricsError,
    evaluate_multi_season,
)

NBA_EVALUATION_GATE_SCHEMA_VERSION = 1
NBA_EVALUATION_COHORT_SCHEMA_VERSION = 1
NBA_EVALUATION_ANSWER_SCHEMA_VERSION = 1
NBA_EVALUATION_FORECAST_SCHEMA_VERSION = 1
NBA_RECALIBRATION_ROW_SCHEMA_VERSION = 1

_HASH_CHARACTERS = frozenset("0123456789abcdef")
_COHORT_KEYS = {
    "schema_version",
    "question_id",
    "season",
    "game_date",
    "raw_elo_team_probability",
}
_ANSWER_KEYS = {"schema_version", "question_id", "realized_team_win"}
_FORECAST_KEYS = {
    "schema_version",
    "question_id",
    "team_probability",
    "failure_reason",
}
_CALIBRATION_KEYS = {
    "schema_version",
    "question_id",
    "season",
    "game_date",
    "raw_elo_team_probability",
    "realized_team_win",
}

type JsonObject = dict[str, object]


class NbaEvaluationGateError(ValueError):
    """Raised when a sealed NBA evaluation cannot authorize progression."""


@dataclass(frozen=True, slots=True)
class NbaEvaluationGatePolicy:
    """Frozen sample-size requirements and recalibration optimizer recipe."""

    minimum_games_per_season: int
    minimum_calendar_blocks_per_season: int
    recalibration_gradient_steps: int
    recalibration_learning_rate: float
    recalibration_initial_intercept: float
    recalibration_initial_slope: float

    def __post_init__(self) -> None:
        _require_positive_integer(
            self.minimum_games_per_season,
            "minimum_games_per_season",
        )
        _require_positive_integer(
            self.minimum_calendar_blocks_per_season,
            "minimum_calendar_blocks_per_season",
        )
        _require_positive_integer(
            self.recalibration_gradient_steps,
            "recalibration_gradient_steps",
        )
        _require_positive_float(
            self.recalibration_learning_rate,
            "recalibration_learning_rate",
        )
        _require_finite_float(
            self.recalibration_initial_intercept,
            "recalibration_initial_intercept",
        )
        _require_finite_float(
            self.recalibration_initial_slope,
            "recalibration_initial_slope",
        )

    def canonical_payload(self) -> JsonObject:
        """Return the complete canonical policy configuration."""
        return {
            "minimum_games_per_season": self.minimum_games_per_season,
            "minimum_calendar_blocks_per_season": self.minimum_calendar_blocks_per_season,
            "recalibration_gradient_steps": self.recalibration_gradient_steps,
            "recalibration_learning_rate": self.recalibration_learning_rate,
            "recalibration_initial_intercept": self.recalibration_initial_intercept,
            "recalibration_initial_slope": self.recalibration_initial_slope,
        }

    @property
    def policy_sha256(self) -> str:
        """Hash the full sample-size and optimizer policy."""
        return canonical_sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class NbaEvaluationCohortInput:
    """One outcome-free member of the frozen untouched evaluation cohort."""

    question_id: str
    season: int
    game_date: date
    raw_elo_team_probability: float

    def __post_init__(self) -> None:
        _require_question_id(self.question_id)
        _require_season_date(self.season, self.game_date)
        _require_probability(self.raw_elo_team_probability, "raw_elo_team_probability")


@dataclass(frozen=True, slots=True)
class NbaEvaluationAnswer:
    """One separately sealed realized outcome keyed to the frozen cohort."""

    question_id: str
    realized_team_win: bool

    def __post_init__(self) -> None:
        _require_question_id(self.question_id)
        _require_boolean(self.realized_team_win, "realized_team_win")


@dataclass(frozen=True, slots=True)
class NbaRecalibrationRow:
    """One resolved training-only row for fitting the frozen Elo recalibrator."""

    question_id: str
    season: int
    game_date: date
    raw_elo_team_probability: float
    realized_team_win: bool

    def __post_init__(self) -> None:
        _require_question_id(self.question_id)
        _require_season_date(self.season, self.game_date)
        _require_probability(self.raw_elo_team_probability, "raw_elo_team_probability")
        _require_boolean(self.realized_team_win, "realized_team_win")


@dataclass(frozen=True, slots=True)
class LogitRecalibrationModel:
    """Frozen deterministic parameters and training identity for Elo recalibration."""

    intercept: float
    slope: float
    training_ids_sha256: str
    policy_sha256: str

    def __post_init__(self) -> None:
        _require_finite_float(self.intercept, "recalibration intercept")
        _require_finite_float(self.slope, "recalibration slope")
        _require_sha256(self.training_ids_sha256, "training_ids_sha256")
        _require_sha256(self.policy_sha256, "policy_sha256")

    def team_probability(self, raw_elo_team_probability: float) -> float:
        """Apply the frozen logit correction to one strict interior Elo probability."""
        _require_probability(raw_elo_team_probability, "raw_elo_team_probability")
        value = _sigmoid(self.intercept + self.slope * _logit(raw_elo_team_probability))
        _require_probability(value, "recalibrated_team_probability")
        return value

    def canonical_payload(self) -> JsonObject:
        """Return the complete hashable model recipe and fitted state."""
        return {
            "algorithm": "full_batch_cross_entropy_logit_recalibration",
            "intercept": self.intercept,
            "policy_sha256": self.policy_sha256,
            "slope": self.slope,
            "training_ids_sha256": self.training_ids_sha256,
        }

    @property
    def model_sha256(self) -> str:
        """Hash the frozen algorithm, config, parameters, and training identity."""
        return canonical_sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class NbaEvaluationGateArtifacts:
    """Paths to all independently sealed inputs and an optional claimed report."""

    cohort_path: Path
    answers_path: Path
    forecasts_path: Path
    calibration_path: Path
    supplied_report_path: Path | None = None


@dataclass(frozen=True, slots=True)
class NbaEvaluationGateReport:
    """Canonical passing report and its deterministic SHA-256 digest."""

    canonical_text: str
    sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.sha256, "report sha256")
        try:
            payload = parse_json_object(self.canonical_text)
        except JsonFormatError as error:
            raise NbaEvaluationGateError("report must contain one JSON object") from error
        if self.canonical_text != canonical_json(payload):
            raise NbaEvaluationGateError("report must use canonical JSON encoding")
        if text_sha256(self.canonical_text) != self.sha256:
            raise NbaEvaluationGateError("report sha256 does not match its canonical text")

    @property
    def payload(self) -> JsonObject:
        """Return a fresh decoded copy of the canonical report payload."""
        return parse_json_object(self.canonical_text)


@dataclass(frozen=True, slots=True)
class _LoadedRows[T]:
    rows: tuple[T, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class _LoadedGateInputs:
    cohort: _LoadedRows[NbaEvaluationCohortInput]
    answers: _LoadedRows[NbaEvaluationAnswer]
    forecasts: _LoadedRows[BinaryForecast]
    calibration: _LoadedRows[NbaRecalibrationRow]


def write_nba_evaluation_cohort_jsonl(
    path: Path,
    rows: Iterable[NbaEvaluationCohortInput],
) -> None:
    """Create a canonical outcome-free untouched-cohort JSONL file."""
    checked = _require_evaluation_cohort(tuple(rows), minimum_seasons=1)
    _write_jsonl(path, checked, _cohort_payload, "NBA evaluation cohort")


def nba_evaluation_cohort_jsonl_bytes(
    rows: Iterable[NbaEvaluationCohortInput],
) -> bytes:
    """Return the exact canonical bytes used for an evaluation cohort."""
    checked = _require_evaluation_cohort(tuple(rows), minimum_seasons=1)
    return _jsonl_bytes(checked, _cohort_payload)


def read_nba_evaluation_cohort_jsonl(path: Path) -> tuple[NbaEvaluationCohortInput, ...]:
    """Load a strict canonical outcome-free untouched-cohort JSONL file."""
    return _load_cohort(path).rows


def write_nba_evaluation_answers_jsonl(
    path: Path,
    rows: Iterable[NbaEvaluationAnswer],
) -> None:
    """Create a canonical answer JSONL file separate from cohort inputs."""
    checked = _require_unique_answers(tuple(rows))
    _write_jsonl(path, checked, _answer_payload, "NBA evaluation answers")


def nba_evaluation_answers_jsonl_bytes(
    rows: Iterable[NbaEvaluationAnswer],
) -> bytes:
    """Return the exact canonical bytes used for evaluation answers."""
    checked = _require_unique_answers(tuple(rows))
    return _jsonl_bytes(checked, _answer_payload)


def read_nba_evaluation_answers_jsonl(path: Path) -> tuple[NbaEvaluationAnswer, ...]:
    """Load strict canonical answer rows without joining them to inputs."""
    return _load_answers(path).rows


def write_nba_evaluation_forecasts_jsonl(
    path: Path,
    rows: Iterable[BinaryForecast],
) -> None:
    """Create canonical model forecasts, retaining every explicit failure."""
    checked = _require_unique_forecasts(tuple(rows))
    _write_jsonl(path, checked, _forecast_payload, "NBA evaluation forecasts")


def nba_evaluation_forecasts_jsonl_bytes(
    rows: Iterable[BinaryForecast],
) -> bytes:
    """Return the exact canonical bytes used for evaluation forecasts."""
    checked = _require_unique_forecasts(tuple(rows))
    return _jsonl_bytes(checked, _forecast_payload)


def read_nba_evaluation_forecasts_jsonl(path: Path) -> tuple[BinaryForecast, ...]:
    """Load strict canonical model forecasts and explicit failures."""
    return _load_forecasts(path).rows


def write_nba_recalibration_rows_jsonl(
    path: Path,
    rows: Iterable[NbaRecalibrationRow],
) -> None:
    """Create canonical resolved rows used only to fit the Elo recalibrator."""
    checked = _require_calibration_rows(tuple(rows))
    _write_jsonl(path, checked, _calibration_payload, "NBA recalibration rows")


def nba_recalibration_rows_jsonl_bytes(
    rows: Iterable[NbaRecalibrationRow],
) -> bytes:
    """Return the exact canonical bytes used for Elo recalibration rows."""
    checked = _require_calibration_rows(tuple(rows))
    return _jsonl_bytes(checked, _calibration_payload)


def read_nba_recalibration_rows_jsonl(path: Path) -> tuple[NbaRecalibrationRow, ...]:
    """Load strict canonical training-only Elo recalibration rows."""
    return _load_calibration(path).rows


def fit_training_only_logit_recalibrator(
    rows: Sequence[NbaRecalibrationRow],
    *,
    policy: NbaEvaluationGatePolicy,
) -> LogitRecalibrationModel:
    """Fit the frozen full-batch cross-entropy logit correction deterministically."""
    checked = _require_calibration_rows(tuple(rows))
    inputs = tuple(_logit(row.raw_elo_team_probability) for row in checked)
    targets = tuple(1.0 if row.realized_team_win else 0.0 for row in checked)
    intercept = policy.recalibration_initial_intercept
    slope = policy.recalibration_initial_slope
    row_count = float(len(checked))

    for _ in range(policy.recalibration_gradient_steps):
        errors = tuple(
            _sigmoid(intercept + slope * value) - target
            for value, target in zip(inputs, targets, strict=True)
        )
        intercept -= policy.recalibration_learning_rate * fsum(errors) / row_count
        slope -= (
            policy.recalibration_learning_rate
            * fsum(error * value for error, value in zip(errors, inputs, strict=True))
            / row_count
        )

    training_ids = tuple(row.question_id for row in checked)
    model = LogitRecalibrationModel(
        intercept=intercept,
        slope=slope,
        training_ids_sha256=canonical_sha256(list(training_ids)),
        policy_sha256=policy.policy_sha256,
    )
    _require_sha256(model.model_sha256, "recalibration model sha256")
    return model


def verify_untouched_nba_evaluation_gate(
    artifacts: NbaEvaluationGateArtifacts,
    *,
    policy: NbaEvaluationGatePolicy,
) -> NbaEvaluationGateReport:
    """Recompute both untouched-season gates and require every season to pass."""
    cohort = _load_cohort(artifacts.cohort_path)
    _require_evaluation_cohort(cohort.rows, minimum_seasons=2)
    answers = _load_answers(artifacts.answers_path)
    forecasts = _load_forecasts(artifacts.forecasts_path)
    calibration = _load_calibration(artifacts.calibration_path)

    evaluation_ids = tuple(row.question_id for row in cohort.rows)
    _require_exact_ids(evaluation_ids, answers.rows, "answer")
    _require_exact_ids(evaluation_ids, forecasts.rows, "forecast")
    evaluation_seasons = tuple(sorted({row.season for row in cohort.rows}))
    _require_calibration_precedes_evaluation(
        calibration.rows,
        evaluation_ids,
        evaluation_seasons,
    )

    model = fit_training_only_logit_recalibrator(calibration.rows, policy=policy)
    raw_cohort = _resolved_cohort(cohort.rows, answers.rows)
    recalibrated_cohort = tuple(
        replace(
            member,
            baseline_team_probability=model.team_probability(member.baseline_team_probability),
        )
        for member in raw_cohort
    )
    try:
        candidate_forecasts_vs_raw = evaluate_multi_season(
            forecasts.rows,
            raw_cohort,
            evaluation_seasons,
        )
        candidate_forecasts_vs_recalibrated = evaluate_multi_season(
            forecasts.rows,
            recalibrated_cohort,
            evaluation_seasons,
        )
    except OutcomeV2MetricsError as error:
        raise NbaEvaluationGateError("cannot score the sealed evaluation cohort") from error
    _require_policy_sample_minimums(candidate_forecasts_vs_raw, policy)
    if not candidate_forecasts_vs_raw.passes or not candidate_forecasts_vs_recalibrated.passes:
        raise NbaEvaluationGateError(
            "candidate forecasts must beat raw and training-only-recalibrated Elo in every season"
        )

    inputs = _LoadedGateInputs(cohort, answers, forecasts, calibration)
    payload = _report_payload(
        inputs,
        policy,
        model,
        candidate_forecasts_vs_raw,
        candidate_forecasts_vs_recalibrated,
    )
    report_text = canonical_json(payload)
    report = NbaEvaluationGateReport(
        canonical_text=report_text,
        sha256=text_sha256(report_text),
    )
    if artifacts.supplied_report_path is not None:
        _require_matching_report(artifacts.supplied_report_path, report.canonical_text)
    return report


def write_nba_evaluation_gate_report(path: Path, report: NbaEvaluationGateReport) -> None:
    """Create the exact canonical passing report without overwriting an existing claim."""
    try:
        with path.open("x", encoding="utf-8", newline="") as file:
            file.write(report.canonical_text)
    except FileExistsError as error:
        raise NbaEvaluationGateError("NBA evaluation gate report already exists") from error
    except OSError as error:
        raise NbaEvaluationGateError("cannot write NBA evaluation gate report") from error


def _load_cohort(path: Path) -> _LoadedRows[NbaEvaluationCohortInput]:
    loaded = _read_jsonl(path, _cohort_from_payload, _cohort_payload, "NBA evaluation cohort")
    return replace(
        loaded,
        rows=_require_evaluation_cohort(loaded.rows, minimum_seasons=1),
    )


def _load_answers(path: Path) -> _LoadedRows[NbaEvaluationAnswer]:
    loaded = _read_jsonl(path, _answer_from_payload, _answer_payload, "NBA evaluation answers")
    return replace(loaded, rows=_require_unique_answers(loaded.rows))


def _load_forecasts(path: Path) -> _LoadedRows[BinaryForecast]:
    loaded = _read_jsonl(
        path,
        _forecast_from_payload,
        _forecast_payload,
        "NBA evaluation forecasts",
    )
    return replace(loaded, rows=_require_unique_forecasts(loaded.rows))


def _load_calibration(path: Path) -> _LoadedRows[NbaRecalibrationRow]:
    loaded = _read_jsonl(
        path,
        _calibration_from_payload,
        _calibration_payload,
        "NBA recalibration rows",
    )
    return replace(loaded, rows=_require_calibration_rows(loaded.rows))


def _write_jsonl[T](
    path: Path,
    rows: Sequence[T],
    to_payload: Callable[[T], JsonObject],
    description: str,
) -> None:
    text = _jsonl_bytes(rows, to_payload).decode("utf-8")
    try:
        with path.open("x", encoding="utf-8", newline="") as file:
            file.write(text)
    except FileExistsError as error:
        raise NbaEvaluationGateError(f"{description} already exists") from error
    except OSError as error:
        raise NbaEvaluationGateError(f"cannot write {description}") from error


def _jsonl_bytes[T](
    rows: Sequence[T],
    to_payload: Callable[[T], JsonObject],
) -> bytes:
    return "".join(f"{canonical_json(to_payload(row))}\n" for row in rows).encode("utf-8")


def _read_jsonl[T](
    path: Path,
    from_payload: Callable[[Mapping[str, object]], T],
    to_payload: Callable[[T], JsonObject],
    description: str,
) -> _LoadedRows[T]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise NbaEvaluationGateError(f"cannot read {description}") from error
    if not text or not text.endswith("\n"):
        raise NbaEvaluationGateError(f"{description} must be nonempty and end with a newline")

    rows: list[T] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise NbaEvaluationGateError(f"blank {description} row on line {line_number}")
        try:
            rows.append(from_payload(parse_json_object(line)))
        except ValueError as error:
            raise NbaEvaluationGateError(
                f"invalid {description} row on line {line_number}"
            ) from error
    checked = tuple(rows)
    expected = "".join(f"{canonical_json(to_payload(row))}\n" for row in checked)
    if text != expected:
        raise NbaEvaluationGateError(f"{description} must use canonical JSONL encoding")
    return _LoadedRows(rows=checked, sha256=text_sha256(text))


def _require_evaluation_cohort(
    rows: tuple[NbaEvaluationCohortInput, ...],
    *,
    minimum_seasons: int,
) -> tuple[NbaEvaluationCohortInput, ...]:
    _require_unique_ids(tuple(row.question_id for row in rows), "evaluation cohort")
    row_seasons = tuple(row.season for row in rows)
    if row_seasons != tuple(sorted(row_seasons)):
        raise NbaEvaluationGateError("evaluation cohort seasons must be in increasing order")
    seasons = tuple(sorted({row.season for row in rows}))
    _require_increasing_seasons(seasons, minimum_count=minimum_seasons)
    return rows


def _require_unique_answers(
    rows: tuple[NbaEvaluationAnswer, ...],
) -> tuple[NbaEvaluationAnswer, ...]:
    _require_unique_ids(tuple(row.question_id for row in rows), "evaluation answers")
    return rows


def _require_unique_forecasts(
    rows: tuple[BinaryForecast, ...],
) -> tuple[BinaryForecast, ...]:
    if not rows:
        raise NbaEvaluationGateError("evaluation forecasts must not be empty")
    for row in rows:
        _require_question_id(row.question_id)
        if row.team_probability is None:
            if not isinstance(row.failure_reason, str) or not row.failure_reason.strip():
                raise NbaEvaluationGateError("failed forecast requires a string reason")
        else:
            _require_probability(row.team_probability, "team_probability")
            if row.failure_reason is not None:
                raise NbaEvaluationGateError("valid forecast cannot include a failure reason")
    _require_unique_ids(tuple(row.question_id for row in rows), "evaluation forecasts")
    return rows


def _require_calibration_rows(
    rows: tuple[NbaRecalibrationRow, ...],
) -> tuple[NbaRecalibrationRow, ...]:
    _require_unique_ids(tuple(row.question_id for row in rows), "recalibration rows")
    return rows


def _require_unique_ids(question_ids: tuple[str, ...], description: str) -> None:
    if not question_ids:
        raise NbaEvaluationGateError(f"{description} must not be empty")
    if len(set(question_ids)) != len(question_ids):
        raise NbaEvaluationGateError(f"{description} contains a duplicate question ID")


def _require_exact_ids(
    expected_ids: tuple[str, ...],
    rows: Sequence[NbaEvaluationAnswer] | Sequence[BinaryForecast],
    description: str,
) -> None:
    actual_ids = tuple(row.question_id for row in rows)
    if actual_ids != expected_ids:
        raise NbaEvaluationGateError(
            f"{description} IDs, order, or coverage differ from the frozen cohort"
        )


def _require_calibration_precedes_evaluation(
    calibration: tuple[NbaRecalibrationRow, ...],
    evaluation_ids: tuple[str, ...],
    evaluation_seasons: tuple[int, ...],
) -> None:
    calibration_ids = {row.question_id for row in calibration}
    if calibration_ids & set(evaluation_ids):
        raise NbaEvaluationGateError("recalibration and evaluation question IDs must be disjoint")
    calibration_seasons = tuple(sorted({row.season for row in calibration}))
    if max(calibration_seasons) >= min(evaluation_seasons):
        raise NbaEvaluationGateError(
            "every recalibration season must be strictly before every evaluation season"
        )


def _require_policy_sample_minimums(
    evaluation: MultiSeasonEvaluation,
    policy: NbaEvaluationGatePolicy,
) -> None:
    for season in evaluation.seasons:
        if season.game_count < policy.minimum_games_per_season:
            raise NbaEvaluationGateError(
                f"season {season.season} has fewer than minimum_games_per_season"
            )
    for season in evaluation.seasons:
        if season.calendar_block_count < policy.minimum_calendar_blocks_per_season:
            raise NbaEvaluationGateError(
                f"season {season.season} has fewer than minimum_calendar_blocks_per_season"
            )


def _resolved_cohort(
    inputs: tuple[NbaEvaluationCohortInput, ...],
    answers: tuple[NbaEvaluationAnswer, ...],
) -> tuple[DatedBinaryCohortMember, ...]:
    return tuple(
        DatedBinaryCohortMember(
            question_id=input_row.question_id,
            season=input_row.season,
            game_date=input_row.game_date,
            realized_team_win=answer.realized_team_win,
            baseline_team_probability=input_row.raw_elo_team_probability,
        )
        for input_row, answer in zip(inputs, answers, strict=True)
    )


def _report_payload(
    inputs: _LoadedGateInputs,
    policy: NbaEvaluationGatePolicy,
    model: LogitRecalibrationModel,
    candidate_forecasts_vs_raw: MultiSeasonEvaluation,
    candidate_forecasts_vs_recalibrated: MultiSeasonEvaluation,
) -> JsonObject:
    evaluation_ids = tuple(row.question_id for row in inputs.cohort.rows)
    evaluation_seasons = tuple(sorted({row.season for row in inputs.cohort.rows}))
    return {
        "schema_version": NBA_EVALUATION_GATE_SCHEMA_VERSION,
        "kind": "forecastfm_nba_untouched_evaluation_gate",
        "status": "passed",
        "proof_scope": {
            "candidate_model_and_run_provenance": "required_separately",
            "external_precommit_and_timestamp": "required_separately",
            "raw_provider_derivation": "required_separately",
        },
        "artifacts": {
            "cohort_sha256": inputs.cohort.sha256,
            "answers_sha256": inputs.answers.sha256,
            "forecasts_sha256": inputs.forecasts.sha256,
            "calibration_sha256": inputs.calibration.sha256,
        },
        "evaluation": {
            "question_ids_sha256": canonical_sha256(list(evaluation_ids)),
            "seasons": list(evaluation_seasons),
            "candidate_forecasts_vs_raw_elo": asdict(candidate_forecasts_vs_raw),
            "candidate_forecasts_vs_recalibrated_elo": asdict(candidate_forecasts_vs_recalibrated),
        },
        "policy": {
            "config": policy.canonical_payload(),
            "policy_sha256": policy.policy_sha256,
        },
        "recalibration": {
            **model.canonical_payload(),
            "model_sha256": model.model_sha256,
            "training_seasons": sorted({row.season for row in inputs.calibration.rows}),
        },
    }


def _require_matching_report(path: Path, expected_text: str) -> None:
    try:
        text = path.read_text(encoding="utf-8")
        payload = parse_json_object(text)
    except (JsonFormatError, OSError, UnicodeError) as error:
        raise NbaEvaluationGateError("cannot read a valid supplied evaluation report") from error
    if text != canonical_json(payload):
        raise NbaEvaluationGateError("supplied evaluation report must use canonical JSON encoding")
    if text != expected_text:
        raise NbaEvaluationGateError("supplied evaluation report differs from recomputed results")


def _cohort_payload(row: NbaEvaluationCohortInput) -> JsonObject:
    return {
        "schema_version": NBA_EVALUATION_COHORT_SCHEMA_VERSION,
        "question_id": row.question_id,
        "season": row.season,
        "game_date": row.game_date.isoformat(),
        "raw_elo_team_probability": row.raw_elo_team_probability,
    }


def _answer_payload(row: NbaEvaluationAnswer) -> JsonObject:
    return {
        "schema_version": NBA_EVALUATION_ANSWER_SCHEMA_VERSION,
        "question_id": row.question_id,
        "realized_team_win": row.realized_team_win,
    }


def _forecast_payload(row: BinaryForecast) -> JsonObject:
    return {
        "schema_version": NBA_EVALUATION_FORECAST_SCHEMA_VERSION,
        "question_id": row.question_id,
        "team_probability": row.team_probability,
        "failure_reason": row.failure_reason,
    }


def _calibration_payload(row: NbaRecalibrationRow) -> JsonObject:
    return {
        "schema_version": NBA_RECALIBRATION_ROW_SCHEMA_VERSION,
        "question_id": row.question_id,
        "season": row.season,
        "game_date": row.game_date.isoformat(),
        "raw_elo_team_probability": row.raw_elo_team_probability,
        "realized_team_win": row.realized_team_win,
    }


def _cohort_from_payload(payload: Mapping[str, object]) -> NbaEvaluationCohortInput:
    require_exact_keys(payload, _COHORT_KEYS, "evaluation cohort row")
    _require_schema_version(payload, NBA_EVALUATION_COHORT_SCHEMA_VERSION)
    return NbaEvaluationCohortInput(
        question_id=_string_field(payload, "question_id"),
        season=_integer_field(payload, "season"),
        game_date=_date_field(payload, "game_date"),
        raw_elo_team_probability=_float_field(payload, "raw_elo_team_probability"),
    )


def _answer_from_payload(payload: Mapping[str, object]) -> NbaEvaluationAnswer:
    require_exact_keys(payload, _ANSWER_KEYS, "evaluation answer row")
    _require_schema_version(payload, NBA_EVALUATION_ANSWER_SCHEMA_VERSION)
    return NbaEvaluationAnswer(
        question_id=_string_field(payload, "question_id"),
        realized_team_win=_boolean_field(payload, "realized_team_win"),
    )


def _forecast_from_payload(payload: Mapping[str, object]) -> BinaryForecast:
    require_exact_keys(payload, _FORECAST_KEYS, "evaluation forecast row")
    _require_schema_version(payload, NBA_EVALUATION_FORECAST_SCHEMA_VERSION)
    probability_value = required_field(payload, "team_probability")
    reason_value = required_field(payload, "failure_reason")
    probability = (
        None if probability_value is None else _strict_float(probability_value, "team_probability")
    )
    reason = None if reason_value is None else require_string(reason_value, "failure_reason")
    return BinaryForecast(
        question_id=_string_field(payload, "question_id"),
        team_probability=probability,
        failure_reason=reason,
    )


def _calibration_from_payload(payload: Mapping[str, object]) -> NbaRecalibrationRow:
    require_exact_keys(payload, _CALIBRATION_KEYS, "recalibration row")
    _require_schema_version(payload, NBA_RECALIBRATION_ROW_SCHEMA_VERSION)
    return NbaRecalibrationRow(
        question_id=_string_field(payload, "question_id"),
        season=_integer_field(payload, "season"),
        game_date=_date_field(payload, "game_date"),
        raw_elo_team_probability=_float_field(payload, "raw_elo_team_probability"),
        realized_team_win=_boolean_field(payload, "realized_team_win"),
    )


def _require_schema_version(payload: Mapping[str, object], expected: int) -> None:
    if _integer_field(payload, "schema_version") != expected:
        raise JsonFormatError("unsupported schema_version")


def _string_field(payload: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(payload, field_name), field_name)


def _integer_field(payload: Mapping[str, object], field_name: str) -> int:
    value = required_field(payload, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise JsonFormatError(f"{field_name} must be an integer")
    return value


def _boolean_field(payload: Mapping[str, object], field_name: str) -> bool:
    value = required_field(payload, field_name)
    if not isinstance(value, bool):
        raise JsonFormatError(f"{field_name} must be a boolean")
    return value


def _float_field(payload: Mapping[str, object], field_name: str) -> float:
    return _strict_float(required_field(payload, field_name), field_name)


def _strict_float(value: object, field_name: str) -> float:
    if not isinstance(value, float) or not isfinite(value):
        raise JsonFormatError(f"{field_name} must be a finite JSON float")
    if value == 0.0 and copysign(1.0, value) < 0.0:
        raise JsonFormatError(f"{field_name} cannot use negative zero")
    return value


def _date_field(payload: Mapping[str, object], field_name: str) -> date:
    text = _string_field(payload, field_name)
    try:
        value = date.fromisoformat(text)
    except ValueError as error:
        raise JsonFormatError(f"{field_name} must be an ISO 8601 date") from error
    if text != value.isoformat():
        raise JsonFormatError(f"{field_name} must use canonical ISO 8601 date notation")
    return value


def _require_question_id(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise NbaEvaluationGateError("question_id must be a nonempty string")
    if value != value.strip():
        raise NbaEvaluationGateError("question_id cannot have surrounding whitespace")


def _require_season_date(season: object, game_date: object) -> None:
    if isinstance(season, bool) or not isinstance(season, int) or season <= 0:
        raise NbaEvaluationGateError("season must be a positive integer")
    if not isinstance(game_date, date) or isinstance(game_date, datetime):
        raise NbaEvaluationGateError("game_date must be a date")
    expected_season = game_date.year + 1 if game_date.month >= 7 else game_date.year
    if season != expected_season:
        raise NbaEvaluationGateError("season does not match game_date")


def _require_probability(value: object, field_name: str) -> None:
    _require_finite_float(value, field_name)
    assert isinstance(value, float)
    if not 0.0 < value < 1.0:
        raise NbaEvaluationGateError(f"{field_name} must be strictly between zero and one")


def _require_finite_float(value: object, field_name: str) -> None:
    if not isinstance(value, float) or not isfinite(value):
        raise NbaEvaluationGateError(f"{field_name} must be a finite float")
    if value == 0.0 and copysign(1.0, value) < 0.0:
        raise NbaEvaluationGateError(f"{field_name} cannot use negative zero")


def _require_sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str):
        raise NbaEvaluationGateError(f"{field_name} must be a string")
    if len(value) != 64 or any(character not in _HASH_CHARACTERS for character in value):
        raise NbaEvaluationGateError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_increasing_seasons(seasons: tuple[int, ...], *, minimum_count: int) -> None:
    if len(seasons) < minimum_count:
        raise NbaEvaluationGateError(f"at least {minimum_count} increasing seasons are required")
    for season in seasons:
        _require_positive_integer(season, "season")
    if seasons != tuple(sorted(set(seasons))):
        raise NbaEvaluationGateError("seasons must be unique and increasing")


def _require_boolean(value: object, field_name: str) -> None:
    if not isinstance(value, bool):
        raise NbaEvaluationGateError(f"{field_name} must be a boolean")


def _require_positive_integer(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NbaEvaluationGateError(f"{field_name} must be a positive integer")


def _require_positive_float(value: object, field_name: str) -> None:
    _require_finite_float(value, field_name)
    assert isinstance(value, float)
    if value <= 0.0:
        raise NbaEvaluationGateError(f"{field_name} must be positive")


def _logit(probability: float) -> float:
    return log(probability) - log(1.0 - probability)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    ratio = exp(value)
    return ratio / (1.0 + ratio)
