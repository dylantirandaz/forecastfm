"""Append-only prospective forecast ledger with a verifiable hash chain."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from forecastfm.integrity import canonical_json, canonical_sha256, text_sha256
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.models import ForecastPrediction
from forecastfm.prompting import parse_prediction

LEDGER_SCHEMA_VERSION = 1
GENESIS_HASH = "0" * 64
_HASH_CHARS = frozenset("0123456789abcdef")
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

type EventType = Literal["forecast_batch", "resolution_batch"]
type JsonObject = dict[str, object]


class LedgerValidationError(ValueError):
    """Raised when a prospective ledger or submission violates its schema."""


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise LedgerValidationError(f"{field_name} must not be empty")


def _require_hash(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in _HASH_CHARS for character in value):
        raise LedgerValidationError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise LedgerValidationError(f"{field_name} must be in UTC")


def _utc_text(value: datetime) -> str:
    _require_utc(value, "datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: object, field_name: str) -> datetime:
    text = require_string(value, field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise LedgerValidationError(f"{field_name} must be an ISO 8601 datetime") from error
    _require_utc(parsed, field_name)
    return parsed.astimezone(UTC)


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    values = require_list(value, field_name)
    result = tuple(
        require_string(item, f"{field_name}[{index}]") for index, item in enumerate(values)
    )
    for item in result:
        _require_text(item, field_name)
    return result


def _field(record: Mapping[str, object], name: str) -> object:
    return required_field(record, name)


def _string(record: Mapping[str, object], name: str) -> str:
    return require_string(_field(record, name), name)


def _object(record: Mapping[str, object], name: str) -> JsonObject:
    return require_object(_field(record, name), name)


def _list(record: Mapping[str, object], name: str) -> list[object]:
    return require_list(_field(record, name), name)


def _time(record: Mapping[str, object], name: str) -> datetime:
    return _parse_utc(_field(record, name), name)


@dataclass(frozen=True, slots=True)
class CohortGame:
    """One frozen game in a prospective evaluation cohort."""

    question_id: str
    source_game_id: str
    matchup: str
    outcomes: tuple[str, ...]
    forecast_deadline: datetime
    scheduled_tipoff: datetime

    def __post_init__(self) -> None:
        _require_text(self.question_id, "question_id")
        _require_text(self.source_game_id, "source_game_id")
        _require_text(self.matchup, "matchup")
        if len(self.outcomes) < 2 or len(set(self.outcomes)) != len(self.outcomes):
            raise LedgerValidationError("outcomes must contain at least two unique values")
        for outcome in self.outcomes:
            _require_text(outcome, "outcome")
        _require_utc(self.forecast_deadline, "forecast_deadline")
        _require_utc(self.scheduled_tipoff, "scheduled_tipoff")
        if self.forecast_deadline >= self.scheduled_tipoff:
            raise LedgerValidationError("forecast_deadline must precede scheduled_tipoff")


@dataclass(frozen=True, slots=True)
class Cohort:
    """A schedule snapshot and its complete, frozen set of forecast questions."""

    cohort_id: str
    experiment_sha256: str
    schedule_source: str
    schedule_snapshot_sha256: str
    schedule_retrieved: datetime
    inclusion_rule: str
    games: tuple[CohortGame, ...]

    def __post_init__(self) -> None:
        _require_text(self.cohort_id, "cohort_id")
        _require_hash(self.experiment_sha256, "experiment_sha256")
        _require_text(self.schedule_source, "schedule_source")
        _require_hash(self.schedule_snapshot_sha256, "schedule_snapshot_sha256")
        _require_utc(self.schedule_retrieved, "schedule_retrieved")
        _require_text(self.inclusion_rule, "inclusion_rule")
        if not self.games:
            raise LedgerValidationError("a cohort must contain at least one game")
        question_ids = [game.question_id for game in self.games]
        source_ids = [game.source_game_id for game in self.games]
        if len(set(question_ids)) != len(question_ids):
            raise LedgerValidationError("cohort question_id values must be unique")
        if len(set(source_ids)) != len(source_ids):
            raise LedgerValidationError("cohort source_game_id values must be unique")
        if any(self.schedule_retrieved >= game.forecast_deadline for game in self.games):
            raise LedgerValidationError("schedule_retrieved must precede every forecast_deadline")


@dataclass(frozen=True, slots=True)
class ForecastSubmission:
    """One provider response and the exact prompt from which it was generated."""

    question_id: str
    input_as_of: datetime
    generated_at: datetime
    prompt: str
    prompt_sha256: str
    raw_response: str
    prediction: ForecastPrediction
    provider_request_id: str

    def __post_init__(self) -> None:
        _require_text(self.question_id, "question_id")
        _require_utc(self.input_as_of, "input_as_of")
        _require_utc(self.generated_at, "generated_at")
        _require_text(self.prompt, "prompt")
        _require_hash(self.prompt_sha256, "prompt_sha256")
        if text_sha256(self.prompt) != self.prompt_sha256:
            raise LedgerValidationError("prompt_sha256 does not match the exact prompt")
        _require_text(self.raw_response, "raw_response")
        _require_text(self.provider_request_id, "provider_request_id")
        try:
            parsed = parse_prediction(self.raw_response, self.prediction.distribution.outcomes)
        except ValueError as error:
            raise LedgerValidationError("raw_response is not a valid stored prediction") from error
        if parsed != self.prediction:
            raise LedgerValidationError("raw_response does not match the stored prediction")


@dataclass(frozen=True, slots=True)
class ResolutionSubmission:
    """One independently sourced resolution for a frozen cohort question."""

    question_id: str
    realized_outcome: str
    resolved_at: datetime
    resolution_source: str
    resolution_source_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.question_id, "question_id")
        _require_text(self.realized_outcome, "realized_outcome")
        _require_utc(self.resolved_at, "resolved_at")
        _require_text(self.resolution_source, "resolution_source")
        _require_hash(self.resolution_source_sha256, "resolution_source_sha256")


@dataclass(frozen=True, slots=True)
class LedgerAudit:
    """Summary returned after every ledger record has been verified."""

    event_count: int
    cohort_count: int
    resolution_count: int
    unresolved_cohort_ids: tuple[str, ...]
    head_sha256: str


def _empty_cohorts() -> dict[str, Cohort]:
    return {}


def _empty_cohort_ids() -> set[str]:
    return set()


@dataclass(slots=True)
class _LedgerState:
    head_sha256: str = GENESIS_HASH
    event_count: int = 0
    last_recorded_at: datetime | None = None
    cohorts: dict[str, Cohort] = field(default_factory=_empty_cohorts)
    resolved: set[str] = field(default_factory=_empty_cohort_ids)


def _cohort_to_dict(cohort: Cohort) -> JsonObject:
    return {
        "cohort_id": cohort.cohort_id,
        "experiment_sha256": cohort.experiment_sha256,
        "schedule_source": cohort.schedule_source,
        "schedule_snapshot_sha256": cohort.schedule_snapshot_sha256,
        "schedule_retrieved": _utc_text(cohort.schedule_retrieved),
        "inclusion_rule": cohort.inclusion_rule,
        "games": [
            {
                "question_id": game.question_id,
                "source_game_id": game.source_game_id,
                "matchup": game.matchup,
                "outcomes": list(game.outcomes),
                "forecast_deadline": _utc_text(game.forecast_deadline),
                "scheduled_tipoff": _utc_text(game.scheduled_tipoff),
            }
            for game in cohort.games
        ],
    }


def cohort_sha256(cohort: Cohort) -> str:
    """Hash a cohort's canonical JSON representation."""
    return canonical_sha256(_cohort_to_dict(cohort))


def _cohort_from_dict(record: Mapping[str, object]) -> Cohort:
    keys = {
        "cohort_id",
        "experiment_sha256",
        "schedule_source",
        "schedule_snapshot_sha256",
        "schedule_retrieved",
        "inclusion_rule",
        "games",
    }
    require_exact_keys(record, keys, "cohort")
    games = _list(record, "games")
    return Cohort(
        cohort_id=_string(record, "cohort_id"),
        experiment_sha256=_string(record, "experiment_sha256"),
        schedule_source=_string(record, "schedule_source"),
        schedule_snapshot_sha256=_string(record, "schedule_snapshot_sha256"),
        schedule_retrieved=_time(record, "schedule_retrieved"),
        inclusion_rule=_string(record, "inclusion_rule"),
        games=tuple(_cohort_game_from_object(item, index) for index, item in enumerate(games)),
    )


def _cohort_game_from_object(value: object, index: int) -> CohortGame:
    field_name = f"games[{index}]"
    record = require_object(value, field_name)
    keys = {
        "question_id",
        "source_game_id",
        "matchup",
        "outcomes",
        "forecast_deadline",
        "scheduled_tipoff",
    }
    require_exact_keys(record, keys, field_name)
    return CohortGame(
        question_id=_string(record, "question_id"),
        source_game_id=_string(record, "source_game_id"),
        matchup=_string(record, "matchup"),
        outcomes=_string_tuple(_field(record, "outcomes"), "outcomes"),
        forecast_deadline=_time(record, "forecast_deadline"),
        scheduled_tipoff=_time(record, "scheduled_tipoff"),
    )


def load_cohort(path: Path) -> Cohort:
    """Load and validate a cohort JSON file."""
    return _cohort_from_dict(parse_json_object(path.read_text(encoding="utf-8")))


def _forecast_submission_to_dict(submission: ForecastSubmission) -> JsonObject:
    return {
        "question_id": submission.question_id,
        "input_as_of": _utc_text(submission.input_as_of),
        "generated_at": _utc_text(submission.generated_at),
        "prompt": submission.prompt,
        "prompt_sha256": submission.prompt_sha256,
        "raw_response": submission.raw_response,
        "prediction": {
            "probabilities": submission.prediction.distribution.as_dict(),
        },
        "provider_request_id": submission.provider_request_id,
    }


def _forecast_submission_from_record(
    record: Mapping[str, object],
    game: CohortGame,
) -> ForecastSubmission:
    keys = {
        "question_id",
        "input_as_of",
        "generated_at",
        "prompt",
        "prompt_sha256",
        "raw_response",
        "prediction",
        "provider_request_id",
    }
    require_exact_keys(record, keys, "forecast submission")
    prediction_record = _object(record, "prediction")
    try:
        prediction = parse_prediction(canonical_json(prediction_record), game.outcomes)
    except ValueError as error:
        raise LedgerValidationError("stored prediction is invalid") from error
    return ForecastSubmission(
        question_id=_string(record, "question_id"),
        input_as_of=_time(record, "input_as_of"),
        generated_at=_time(record, "generated_at"),
        prompt=_string(record, "prompt"),
        prompt_sha256=_string(record, "prompt_sha256"),
        raw_response=_string(record, "raw_response"),
        prediction=prediction,
        provider_request_id=_string(record, "provider_request_id"),
    )


def _resolution_submission_to_dict(submission: ResolutionSubmission) -> JsonObject:
    return {
        "question_id": submission.question_id,
        "realized_outcome": submission.realized_outcome,
        "resolved_at": _utc_text(submission.resolved_at),
        "resolution_source": submission.resolution_source,
        "resolution_source_sha256": submission.resolution_source_sha256,
    }


def _resolution_submission_from_record(record: Mapping[str, object]) -> ResolutionSubmission:
    keys = {
        "question_id",
        "realized_outcome",
        "resolved_at",
        "resolution_source",
        "resolution_source_sha256",
    }
    require_exact_keys(record, keys, "resolution submission")
    return ResolutionSubmission(
        question_id=_string(record, "question_id"),
        realized_outcome=_string(record, "realized_outcome"),
        resolved_at=_time(record, "resolved_at"),
        resolution_source=_string(record, "resolution_source"),
        resolution_source_sha256=_string(record, "resolution_source_sha256"),
    )


def _load_jsonl_objects(path: Path) -> tuple[JsonObject, ...]:
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise LedgerValidationError("JSONL file must end with a newline")
    records: list[JsonObject] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise LedgerValidationError(f"blank JSONL record on line {line_number}")
        try:
            records.append(parse_json_object(line))
        except JsonFormatError as error:
            raise LedgerValidationError(f"invalid JSONL record on line {line_number}") from error
    return tuple(records)


def append_forecast_batch(
    path: Path,
    cohort: Cohort,
    submissions: Sequence[ForecastSubmission],
    recorded_at: datetime,
    expected_head: str | None = None,
) -> str:
    """Atomically append one complete forecast batch and return its event hash."""
    state = _audit_state(path, expected_head)
    if cohort.cohort_id in state.cohorts:
        raise LedgerValidationError(f"duplicate forecast batch for cohort: {cohort.cohort_id}")
    _require_event_time(state, recorded_at)
    _validate_forecast_batch(cohort, submissions, recorded_at)
    cohort_hash = cohort_sha256(cohort)
    payload: JsonObject = {
        "experiment_sha256": cohort.experiment_sha256,
        "cohort_sha256": cohort_hash,
        "cohort": _cohort_to_dict(cohort),
        "submissions": [_forecast_submission_to_dict(item) for item in submissions],
    }
    record = _make_event(state, "forecast_batch", recorded_at, payload)
    _atomic_append(path, record)
    return _string(record, "event_hash")


def append_resolution_batch(
    path: Path,
    cohort: Cohort,
    submissions: Sequence[ResolutionSubmission],
    recorded_at: datetime,
    expected_head: str | None = None,
) -> str:
    """Atomically append one complete resolution batch and return its event hash."""
    state = _audit_state(path, expected_head)
    forecast_cohort = state.cohorts.get(cohort.cohort_id)
    if forecast_cohort is None:
        raise LedgerValidationError("a resolution batch must follow its forecast batch")
    if cohort.cohort_id in state.resolved:
        raise LedgerValidationError(f"duplicate resolution batch for cohort: {cohort.cohort_id}")
    if cohort_sha256(cohort) != cohort_sha256(forecast_cohort):
        raise LedgerValidationError("resolution cohort differs from the forecasted cohort")
    _require_event_time(state, recorded_at)
    _validate_resolution_batch(cohort, submissions, recorded_at)
    payload: JsonObject = {
        "experiment_sha256": cohort.experiment_sha256,
        "cohort_sha256": cohort_sha256(cohort),
        "cohort_id": cohort.cohort_id,
        "submissions": [_resolution_submission_to_dict(item) for item in submissions],
    }
    record = _make_event(state, "resolution_batch", recorded_at, payload)
    _atomic_append(path, record)
    return _string(record, "event_hash")


def audit_ledger(
    path: Path,
    expected_head: str | None = None,
    expected_experiment_sha256: str | None = None,
) -> LedgerAudit:
    """Verify the chain and optional head and experiment commitments."""
    state = _audit_state(path, expected_head)
    if expected_experiment_sha256 is not None:
        _require_hash(expected_experiment_sha256, "expected_experiment_sha256")
        if any(
            cohort.experiment_sha256 != expected_experiment_sha256
            for cohort in state.cohorts.values()
        ):
            raise LedgerValidationError("ledger is bound to a different experiment")
    unresolved = tuple(sorted(set(state.cohorts) - state.resolved))
    return LedgerAudit(
        event_count=state.event_count,
        cohort_count=len(state.cohorts),
        resolution_count=len(state.resolved),
        unresolved_cohort_ids=unresolved,
        head_sha256=state.head_sha256,
    )


def _require_exact_order(cohort: Cohort, actual: tuple[str, ...], kind: str) -> None:
    expected = tuple(game.question_id for game in cohort.games)
    if actual != expected:
        raise LedgerValidationError(f"{kind} submissions must exactly match frozen cohort order")


def _validate_forecast_batch(
    cohort: Cohort,
    submissions: Sequence[ForecastSubmission],
    recorded_at: datetime,
) -> None:
    _require_utc(recorded_at, "recorded_at")
    _require_exact_order(cohort, tuple(item.question_id for item in submissions), "forecast")
    for game, submission in zip(cohort.games, submissions, strict=True):
        if submission.prediction.distribution.outcomes != game.outcomes:
            raise LedgerValidationError("prediction outcomes differ from the frozen game outcomes")
        if not (
            cohort.schedule_retrieved
            <= submission.input_as_of
            <= submission.generated_at
            <= recorded_at
            < game.forecast_deadline
            < game.scheduled_tipoff
        ):
            raise LedgerValidationError("forecast timestamps violate prospective ordering")


def _validate_resolution_batch(
    cohort: Cohort,
    submissions: Sequence[ResolutionSubmission],
    recorded_at: datetime,
) -> None:
    _require_utc(recorded_at, "recorded_at")
    _require_exact_order(cohort, tuple(item.question_id for item in submissions), "resolution")
    for game, submission in zip(cohort.games, submissions, strict=True):
        if submission.realized_outcome not in game.outcomes:
            raise LedgerValidationError("realized_outcome is not a frozen game outcome")
        if not game.scheduled_tipoff <= submission.resolved_at <= recorded_at:
            raise LedgerValidationError("resolution timestamps violate prospective ordering")


def _require_event_time(state: _LedgerState, recorded_at: datetime) -> None:
    _require_utc(recorded_at, "recorded_at")
    if state.last_recorded_at is not None and recorded_at < state.last_recorded_at:
        raise LedgerValidationError("ledger recorded_at values must be nondecreasing")


def _make_event(
    state: _LedgerState,
    event_type: EventType,
    recorded_at: datetime,
    payload: JsonObject,
) -> JsonObject:
    body: JsonObject = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "sequence": state.event_count + 1,
        "event_type": event_type,
        "recorded_at": _utc_text(recorded_at),
        "previous_hash": state.head_sha256,
        "payload": payload,
    }
    return {**body, "event_hash": canonical_sha256(body)}


def _atomic_append(path: Path, record: JsonObject) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".part",
            delete=False,
        ) as file:
            partial_path = Path(file.name)
            file.write(existing)
            file.write(canonical_json(record))
            file.write("\n")
        partial_path.replace(path)
    except Exception:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)
        raise


def _audit_state(path: Path, expected_head: str | None) -> _LedgerState:
    if expected_head is not None:
        _require_hash(expected_head, "expected_head")
    state = _LedgerState()
    records = _load_ledger_records(path)
    for record in records:
        _audit_envelope(record, state)
    if expected_head is not None and state.head_sha256 != expected_head:
        raise LedgerValidationError(
            "ledger head differs from expected_head; ledger may be truncated"
        )
    return state


def _load_ledger_records(path: Path) -> tuple[JsonObject, ...]:
    if not path.exists():
        return ()
    text = path.read_text(encoding="utf-8")
    records = _load_jsonl_objects(path)
    pairs = zip(text.splitlines(), records, strict=True)
    for line_number, (line, record) in enumerate(pairs, start=1):
        if line != canonical_json(record):
            raise LedgerValidationError(f"non-canonical ledger record on line {line_number}")
    return records


def _audit_envelope(record: JsonObject, state: _LedgerState) -> None:
    require_exact_keys(record, _ENVELOPE_KEYS, "ledger envelope")
    schema_version = _field(record, "schema_version")
    sequence = _field(record, "sequence")
    if isinstance(schema_version, bool) or schema_version != LEDGER_SCHEMA_VERSION:
        raise LedgerValidationError(f"unsupported ledger schema version: {schema_version}")
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise LedgerValidationError("ledger sequence must be an integer")
    if sequence != state.event_count + 1:
        raise LedgerValidationError("ledger sequence is missing, duplicated, or out of order")
    previous_hash = _string(record, "previous_hash")
    event_hash = _string(record, "event_hash")
    _require_hash(previous_hash, "previous_hash")
    _require_hash(event_hash, "event_hash")
    if previous_hash != state.head_sha256:
        raise LedgerValidationError("ledger previous_hash chain is broken")
    body = {key: _field(record, key) for key in _ENVELOPE_BODY_KEYS}
    if canonical_sha256(body) != event_hash:
        raise LedgerValidationError("ledger event_hash does not match its canonical event")
    _audit_event(record, state)
    state.event_count += 1
    state.head_sha256 = event_hash


def _audit_event(record: JsonObject, state: _LedgerState) -> None:
    recorded_at = _time(record, "recorded_at")
    _require_event_time(state, recorded_at)
    event_type = _string(record, "event_type")
    payload = _object(record, "payload")
    if event_type == "forecast_batch":
        _audit_forecast_payload(payload, recorded_at, state)
    elif event_type == "resolution_batch":
        _audit_resolution_payload(payload, recorded_at, state)
    else:
        raise LedgerValidationError(f"unsupported ledger event_type: {event_type}")
    state.last_recorded_at = recorded_at


def _audit_forecast_payload(
    payload: JsonObject,
    recorded_at: datetime,
    state: _LedgerState,
) -> None:
    keys = {"experiment_sha256", "cohort_sha256", "cohort", "submissions"}
    require_exact_keys(payload, keys, "forecast batch payload")
    cohort = _cohort_from_dict(_object(payload, "cohort"))
    if cohort.cohort_id in state.cohorts:
        raise LedgerValidationError(f"duplicate forecast batch for cohort: {cohort.cohort_id}")
    experiment_hash = _string(payload, "experiment_sha256")
    cohort_hash = _string(payload, "cohort_sha256")
    if experiment_hash != cohort.experiment_sha256 or cohort_hash != cohort_sha256(cohort):
        raise LedgerValidationError("forecast payload hashes do not match its embedded cohort")
    values = _list(payload, "submissions")
    games = {game.question_id: game for game in cohort.games}
    submissions = tuple(
        _forecast_submission_from_record(
            require_object(value, f"submissions[{index}]"),
            _game_for_submission(value, games, index),
        )
        for index, value in enumerate(values)
    )
    _validate_forecast_batch(cohort, submissions, recorded_at)
    state.cohorts[cohort.cohort_id] = cohort


def _game_for_submission(
    value: object,
    games: Mapping[str, CohortGame],
    index: int,
) -> CohortGame:
    record = require_object(value, f"submissions[{index}]")
    question_id = _string(record, "question_id")
    try:
        return games[question_id]
    except KeyError as error:
        raise LedgerValidationError(f"unknown cohort question_id: {question_id}") from error


def _audit_resolution_payload(
    payload: JsonObject,
    recorded_at: datetime,
    state: _LedgerState,
) -> None:
    keys = {"experiment_sha256", "cohort_sha256", "cohort_id", "submissions"}
    require_exact_keys(payload, keys, "resolution batch payload")
    cohort_id = _string(payload, "cohort_id")
    try:
        cohort = state.cohorts[cohort_id]
    except KeyError as error:
        raise LedgerValidationError("a resolution batch must follow its forecast batch") from error
    if cohort_id in state.resolved:
        raise LedgerValidationError(f"duplicate resolution batch for cohort: {cohort_id}")
    experiment_hash = _string(payload, "experiment_sha256")
    cohort_hash = _string(payload, "cohort_sha256")
    if experiment_hash != cohort.experiment_sha256 or cohort_hash != cohort_sha256(cohort):
        raise LedgerValidationError("resolution payload hashes differ from its forecast batch")
    values = _list(payload, "submissions")
    submissions = tuple(
        _resolution_submission_from_record(require_object(value, f"submissions[{index}]"))
        for index, value in enumerate(values)
    )
    _validate_resolution_batch(cohort, submissions, recorded_at)
    state.resolved.add(cohort_id)
