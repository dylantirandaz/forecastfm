"""Tinker-free commitments and terminal records for outcome-v2 inference."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import exp, isclose, isfinite
from pathlib import Path
from typing import Literal, cast

from forecastfm.integrity import bytes_sha256, canonical_json, canonical_sha256
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_float,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_feature_rows import (
    NbaFeatureRowError,
    NbaRichFeatureRow,
    read_nba_feature_rows_jsonl_bytes,
)
from forecastfm.outcome import (
    OPPONENT_LABEL,
    TEAM_LABEL,
    OutcomeForecastError,
    elo_offset_team_probability_from_logprobs,
    symmetric_team_probability,
)
from forecastfm.outcome_v2_config import outcome_v2_inference_settings
from forecastfm.outcome_v2_experiment import (
    OutcomeV2ExperimentError,
    verify_outcome_v2_experiment_lock,
)
from forecastfm.outcome_v2_metrics import BinaryForecast
from forecastfm.outcome_v2_prompt import build_outcome_v2_messages
from forecastfm.outcome_v2_run import (
    OutcomeV2RunError,
    OutcomeV2RunLock,
    require_outcome_v2_run_static_contract,
)
from forecastfm.tinker_data import ForecastRecord

OUTCOME_V2_GENERATION_LOCK_SCHEMA_VERSION = 1
OUTCOME_V2_INFERENCE_RECORD_SCHEMA_VERSION = 1

_LOCK_KIND = "forecastfm_outcome_v2_generation_lock"
_LOCK_STATUS = "committed_before_remote_calls"
_CANDIDATE_ROLE = "forecastfm_outcome_v2_sft_adapter"
_RECORD_KIND = "forecastfm_outcome_v2_inference_record"
_SEASON_RELATION = "strictly_later_than_run_lock_tabular_seasons"
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_FAILURE_PATTERN = re.compile(r"candidate_call_exception:[A-Za-z][A-Za-z0-9_]*")
_FIXED_FAILURE_REASONS = frozenset(
    {
        "candidate_output_invalid",
        "interrupted_after_start",
    }
)
_LOCK_KEYS = {
    "schema_version",
    "kind",
    "status",
    "candidate_role",
    "created_at",
    "outcome_v2_run_lock_sha256",
    "outcome_v2_experiment_lock_sha256",
    "sampler_path",
    "feature_rows_sha256",
    "feature_row_sha256s",
    "prompt_pairs_sha256",
    "question_ids",
    "question_ids_sha256",
    "tabular_seasons",
    "evaluation_seasons",
    "season_relation",
    "game_count",
    "orientation_count",
    "renderer_name",
    "label_token_ids",
    "inference_settings",
    "inference_settings_sha256",
    "call_policy",
}
_LABEL_TOKEN_KEYS = {TEAM_LABEL, OPPONENT_LABEL}
_CALL_POLICY_KEYS = {
    "logical_calls_per_game",
    "expected_logical_calls",
    "application_attempts_per_game",
    "application_retries",
    "sdk_retry_logic_enabled",
    "sdk_internal_retransmission_window_seconds",
    "generated_text_used",
    "sdk_internal_unused_generated_tokens_per_call",
    "transport_retry_note",
}
_ORIENTATION_KEYS = {
    "elo_team_probability",
    "team_logprob",
    "opponent_logprob",
    "valid_label_mass",
    "team_probability",
}
_RECORD_KEYS = {
    "schema_version",
    "kind",
    "sequence",
    "generation_lock_sha256",
    "question_id",
    "swapped_question_id",
    "feature_row_sha256",
    "original_prompt_token_ids_sha256",
    "swapped_prompt_token_ids_sha256",
    "status",
    "failure_reason",
    "original",
    "swapped",
    "team_probability",
    "pre_average_side_swap_gap",
}

type JsonObject = dict[str, object]
type InferenceStatus = Literal["completed", "failed"]


class OutcomeV2InferenceError(ValueError):
    """Raised when answer-free inference provenance or output is invalid."""


@dataclass(frozen=True, slots=True)
class OutcomeV2GenerationArtifacts:
    """Only model locks and new original-only feature rows used before calls."""

    project_root: Path
    run_lock_path: Path
    experiment_lock_path: Path
    feature_rows_path: Path


@dataclass(frozen=True, slots=True)
class OutcomeV2GenerationLock:
    """Canonical pre-call binding for one fixed post-SFT generation attempt."""

    canonical_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_immutable_bytes(self.canonical_bytes)
        _generation_lock_record(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        """Return the digest of the exact canonical lock bytes."""
        return bytes_sha256(self.canonical_bytes)

    def to_record(self) -> JsonObject:
        """Return a newly decoded strict lock record."""
        return _generation_lock_record(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class OrientationScore:
    """Raw two-label scores and their deterministic Elo-offset probability."""

    elo_team_probability: float
    team_logprob: float
    opponent_logprob: float
    valid_label_mass: float
    team_probability: float

    def __post_init__(self) -> None:
        _validate_orientation_score(self)


@dataclass(frozen=True, slots=True)
class InferenceRecord:
    """One completed or explicitly failed terminal result for a frozen game."""

    sequence: int
    generation_lock_sha256: str
    question_id: str
    swapped_question_id: str
    feature_row_sha256: str
    original_prompt_token_ids_sha256: str
    swapped_prompt_token_ids_sha256: str
    status: InferenceStatus
    failure_reason: str | None
    original: OrientationScore | None
    swapped: OrientationScore | None
    team_probability: float | None
    pre_average_side_swap_gap: float | None

    def __post_init__(self) -> None:
        _validate_inference_record(self)


@dataclass(frozen=True, slots=True)
class _ModelLocks:
    run_bytes: bytes = field(repr=False)
    experiment_bytes: bytes = field(repr=False)
    run_record: JsonObject
    experiment_record: JsonObject


def build_outcome_v2_prompt_records(
    rows: Sequence[NbaRichFeatureRow],
) -> tuple[ForecastRecord, ...]:
    """Build adjacent original/swap records from target-free feature rows only."""
    checked = _require_original_rows(tuple(rows))
    records: list[ForecastRecord] = []
    for row in checked:
        records.extend(
            ForecastRecord(
                question_id=orientation.question_id,
                messages=list(build_outcome_v2_messages(orientation)),
            )
            for orientation in (row, row.side_swap())
        )
    return tuple(records)


def outcome_v2_prompt_pairs_jsonl_bytes(rows: Sequence[NbaRichFeatureRow]) -> bytes:
    """Return the exact canonical prompt-pair JSONL bytes bound before calls."""
    records = build_outcome_v2_prompt_records(rows)
    return "".join(f"{canonical_json(record)}\n" for record in records).encode("utf-8")


def build_outcome_v2_generation_lock(
    artifacts: OutcomeV2GenerationArtifacts,
    label_token_ids: tuple[int, int],
    created_at: datetime,
) -> OutcomeV2GenerationLock:
    """Bind verified model locks and new target-free prompts before remote calls."""
    created_at_text = _utc_text(created_at, "created_at")
    model_locks = _load_model_locks(artifacts)
    feature_bytes, feature_rows = _load_feature_rows(artifacts.feature_rows_path)
    run_record = model_locks.run_record
    experiment_record = model_locks.experiment_record
    experiment_at = _parse_utc(
        _string(experiment_record, "created_at"),
        "experiment.created_at",
    )
    if created_at < experiment_at:
        raise OutcomeV2InferenceError("generation lock cannot predate the trained experiment")

    settings = _locked_inference_settings(run_record)
    renderer_name = _locked_renderer(run_record)
    tabular_seasons = _tabular_seasons(run_record)
    evaluation_seasons = tuple(sorted({row.season for row in feature_rows}))
    _require_new_later_seasons(tabular_seasons, evaluation_seasons)
    _require_new_feature_rows(run_record, bytes_sha256(feature_bytes))
    team_token, opponent_token = _require_label_token_ids(label_token_ids)
    question_ids = tuple(row.question_id for row in feature_rows)
    feature_row_sha256s = tuple(row.row_sha256 for row in feature_rows)
    prompt_bytes = outcome_v2_prompt_pairs_jsonl_bytes(feature_rows)
    call_policy = _call_policy(settings, len(feature_rows))

    record: JsonObject = {
        "schema_version": OUTCOME_V2_GENERATION_LOCK_SCHEMA_VERSION,
        "kind": _LOCK_KIND,
        "status": _LOCK_STATUS,
        "candidate_role": _CANDIDATE_ROLE,
        "created_at": created_at_text,
        "outcome_v2_run_lock_sha256": bytes_sha256(model_locks.run_bytes),
        "outcome_v2_experiment_lock_sha256": bytes_sha256(model_locks.experiment_bytes),
        "sampler_path": _string(experiment_record, "sampler_path"),
        "feature_rows_sha256": bytes_sha256(feature_bytes),
        "feature_row_sha256s": list(feature_row_sha256s),
        "prompt_pairs_sha256": bytes_sha256(prompt_bytes),
        "question_ids": list(question_ids),
        "question_ids_sha256": canonical_sha256(list(question_ids)),
        "tabular_seasons": list(tabular_seasons),
        "evaluation_seasons": list(evaluation_seasons),
        "season_relation": _SEASON_RELATION,
        "game_count": len(feature_rows),
        "orientation_count": len(feature_rows) * 2,
        "renderer_name": renderer_name,
        "label_token_ids": {
            TEAM_LABEL: team_token,
            OPPONENT_LABEL: opponent_token,
        },
        "inference_settings": settings,
        "inference_settings_sha256": canonical_sha256(settings),
        "call_policy": call_policy,
    }
    lock = OutcomeV2GenerationLock(canonical_json(record).encode("utf-8"))
    _require_unchanged_inputs(artifacts, model_locks, feature_bytes)
    return lock


def write_outcome_v2_generation_lock(
    path: Path,
    lock: OutcomeV2GenerationLock,
) -> str:
    """Create and durably flush one canonical generation lock without replacement."""
    _write_once(path, lock.canonical_bytes, "generation lock")
    return lock.sha256


def read_outcome_v2_generation_lock(path: Path) -> OutcomeV2GenerationLock:
    """Read one strict canonical generation lock."""
    return OutcomeV2GenerationLock(_read_bytes(path, "generation lock"))


def verify_outcome_v2_generation_lock(
    artifacts: OutcomeV2GenerationArtifacts,
    path: Path,
) -> OutcomeV2GenerationLock:
    """Rebuild a generation lock from current answer-free inputs and compare bytes."""
    actual = read_outcome_v2_generation_lock(path)
    record = actual.to_record()
    labels = _object(record, "label_token_ids")
    label_token_ids = (
        _nonnegative_integer(labels, TEAM_LABEL),
        _nonnegative_integer(labels, OPPONENT_LABEL),
    )
    created_at = _parse_utc(_string(record, "created_at"), "created_at")
    expected = build_outcome_v2_generation_lock(
        artifacts,
        label_token_ids,
        created_at,
    )
    if actual.canonical_bytes != expected.canonical_bytes:
        raise OutcomeV2InferenceError(
            "generation lock differs from current model locks or target-free inputs"
        )
    return actual


def build_orientation_score(
    elo_team_probability: float,
    team_logprob: float,
    opponent_logprob: float,
) -> OrientationScore:
    """Build one validated Elo-offset score from the two fixed label log-probabilities."""
    try:
        valid_label_mass = exp(team_logprob) + exp(opponent_logprob)
        team_probability = elo_offset_team_probability_from_logprobs(
            elo_team_probability,
            team_logprob,
            opponent_logprob,
        )
    except (OverflowError, OutcomeForecastError) as error:
        raise OutcomeV2InferenceError("cannot derive an Elo-offset orientation score") from error
    return OrientationScore(
        elo_team_probability=elo_team_probability,
        team_logprob=team_logprob,
        opponent_logprob=opponent_logprob,
        valid_label_mass=valid_label_mass,
        team_probability=team_probability,
    )


def completed_inference_record(  # noqa: PLR0913
    generation_lock: OutcomeV2GenerationLock,
    sequence: int,
    row: NbaRichFeatureRow,
    *,
    original_prompt_token_ids_sha256: str,
    swapped_prompt_token_ids_sha256: str,
    original: OrientationScore,
    swapped: OrientationScore,
) -> InferenceRecord:
    """Build a terminal success using exact probability-space side-swap averaging."""
    _require_lock_row(generation_lock, sequence, row)
    if original.elo_team_probability != row.elo_team_win_probability:
        raise OutcomeV2InferenceError("original score uses the wrong sealed Elo prior")
    if swapped.elo_team_probability != row.elo_opponent_win_probability:
        raise OutcomeV2InferenceError("swapped score uses the wrong sealed Elo prior")
    team_probability = symmetric_team_probability(
        original.team_probability,
        swapped.team_probability,
    )
    gap = abs(original.team_probability - (1.0 - swapped.team_probability))
    return InferenceRecord(
        sequence=sequence,
        generation_lock_sha256=generation_lock.sha256,
        question_id=row.question_id,
        swapped_question_id=row.side_swap().question_id,
        feature_row_sha256=row.row_sha256,
        original_prompt_token_ids_sha256=original_prompt_token_ids_sha256,
        swapped_prompt_token_ids_sha256=swapped_prompt_token_ids_sha256,
        status="completed",
        failure_reason=None,
        original=original,
        swapped=swapped,
        team_probability=team_probability,
        pre_average_side_swap_gap=gap,
    )


def failed_inference_record(  # noqa: PLR0913
    generation_lock: OutcomeV2GenerationLock,
    sequence: int,
    row: NbaRichFeatureRow,
    *,
    original_prompt_token_ids_sha256: str,
    swapped_prompt_token_ids_sha256: str,
    failure_reason: str,
) -> InferenceRecord:
    """Build one terminal failure without retaining ambiguous partial scores."""
    _require_lock_row(generation_lock, sequence, row)
    return InferenceRecord(
        sequence=sequence,
        generation_lock_sha256=generation_lock.sha256,
        question_id=row.question_id,
        swapped_question_id=row.side_swap().question_id,
        feature_row_sha256=row.row_sha256,
        original_prompt_token_ids_sha256=original_prompt_token_ids_sha256,
        swapped_prompt_token_ids_sha256=swapped_prompt_token_ids_sha256,
        status="failed",
        failure_reason=failure_reason,
        original=None,
        swapped=None,
        team_probability=None,
        pre_average_side_swap_gap=None,
    )


def sanitize_inference_failure(error: BaseException) -> str:
    """Return only a stable exception type, never a provider message or secret."""
    type_name = "".join(
        character if character.isascii() and (character.isalnum() or character == "_") else "_"
        for character in type(error).__name__
    )
    if not type_name or not type_name[0].isascii() or not type_name[0].isalpha():
        type_name = "Exception"
    return f"candidate_call_exception:{type_name}"


def rendered_prompt_token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash one nonempty rendered prompt-token sequence with canonical JSON encoding."""
    values = tuple(token_ids)
    if not values:
        raise OutcomeV2InferenceError("rendered prompt token IDs must not be empty")
    if any(_invalid_token_id(token) for token in values):
        raise OutcomeV2InferenceError("rendered prompt token IDs must be non-negative integers")
    return canonical_sha256(list(values))


def binary_forecasts_from_inference_records(
    records: Sequence[InferenceRecord],
    generation_lock: OutcomeV2GenerationLock,
) -> tuple[BinaryForecast, ...]:
    """Map exact ordered terminal coverage to the one frozen scoring type."""
    checked = _require_record_coverage(tuple(records), generation_lock)
    return tuple(
        BinaryForecast(
            question_id=record.question_id,
            team_probability=record.team_probability,
            failure_reason=record.failure_reason,
        )
        for record in checked
    )


def outcome_v2_inference_record_payload(record: InferenceRecord) -> JsonObject:
    """Return the strict canonical JSON payload for one terminal record."""
    return _record_payload(record)


def outcome_v2_inference_record_from_payload(
    payload: Mapping[str, object],
) -> InferenceRecord:
    """Parse and validate one strict terminal-record JSON payload."""
    return _record_from_payload(payload)


def write_outcome_v2_inference_records(
    path: Path,
    records: Sequence[InferenceRecord],
    generation_lock: OutcomeV2GenerationLock,
) -> str:
    """Create canonical terminal JSONL with exact ordered generation coverage."""
    checked = _require_record_coverage(tuple(records), generation_lock)
    value = "".join(f"{canonical_json(_record_payload(record))}\n" for record in checked).encode(
        "utf-8"
    )
    _write_once(path, value, "inference records")
    return bytes_sha256(value)


def read_outcome_v2_inference_records(
    path: Path,
    generation_lock: OutcomeV2GenerationLock,
) -> tuple[InferenceRecord, ...]:
    """Read strict canonical terminal JSONL with exact ordered generation coverage."""
    value = _read_bytes(path, "inference records")
    return read_outcome_v2_inference_records_jsonl_bytes(value, generation_lock)


def read_outcome_v2_inference_records_jsonl_bytes(
    value: bytes,
    generation_lock: OutcomeV2GenerationLock,
) -> tuple[InferenceRecord, ...]:
    """Parse terminal records from one captured canonical JSONL byte buffer."""
    try:
        text = value.decode("utf-8")
    except UnicodeError as error:
        raise OutcomeV2InferenceError("inference records must be valid UTF-8") from error
    if not text or not text.endswith("\n"):
        raise OutcomeV2InferenceError("inference records must be nonempty canonical JSONL")
    records: list[InferenceRecord] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise OutcomeV2InferenceError(f"blank inference record at line {line_number}")
        try:
            payload = parse_json_object(line)
            if line != canonical_json(payload):
                raise OutcomeV2InferenceError("inference record must use canonical JSON")
            records.append(_record_from_payload(payload))
        except (JsonFormatError, OutcomeV2InferenceError) as error:
            raise OutcomeV2InferenceError(
                f"invalid inference record at line {line_number}"
            ) from error
    checked = _require_record_coverage(tuple(records), generation_lock)
    canonical_bytes = "".join(
        f"{canonical_json(_record_payload(record))}\n" for record in checked
    ).encode("utf-8")
    if value != canonical_bytes:
        raise OutcomeV2InferenceError("inference records must use canonical JSONL bytes")
    return checked


def _load_model_locks(artifacts: OutcomeV2GenerationArtifacts) -> _ModelLocks:
    try:
        verified_experiment = verify_outcome_v2_experiment_lock(
            artifacts.run_lock_path,
            artifacts.experiment_lock_path,
        )
        run_bytes = artifacts.run_lock_path.read_bytes()
        experiment_bytes = artifacts.experiment_lock_path.read_bytes()
        run_lock = OutcomeV2RunLock(run_bytes)
    except (OSError, OutcomeV2ExperimentError, OutcomeV2RunError) as error:
        raise OutcomeV2InferenceError("cannot verify the outcome-v2 model locks") from error
    if verified_experiment.canonical_bytes != experiment_bytes:
        raise OutcomeV2InferenceError("experiment lock changed during verification")
    experiment_record = verified_experiment.to_record()
    if _string(experiment_record, "outcome_v2_run_lock_sha256") != bytes_sha256(run_bytes):
        raise OutcomeV2InferenceError("experiment lock does not bind the exact run lock")
    run_record = run_lock.to_record()
    try:
        require_outcome_v2_run_static_contract(artifacts.project_root, run_lock)
    except OutcomeV2RunError as error:
        raise OutcomeV2InferenceError(
            "run lock differs from the current static contract"
        ) from error
    return _ModelLocks(
        run_bytes=run_bytes,
        experiment_bytes=experiment_bytes,
        run_record=run_record,
        experiment_record=experiment_record,
    )


def _load_feature_rows(path: Path) -> tuple[bytes, tuple[NbaRichFeatureRow, ...]]:
    try:
        value = path.read_bytes()
        rows = read_nba_feature_rows_jsonl_bytes(value)
    except (OSError, NbaFeatureRowError) as error:
        raise OutcomeV2InferenceError("cannot load canonical original-only feature rows") from error
    return value, rows


def _require_unchanged_inputs(
    artifacts: OutcomeV2GenerationArtifacts,
    model_locks: _ModelLocks,
    feature_bytes: bytes,
) -> None:
    expected = (
        model_locks.run_bytes,
        model_locks.experiment_bytes,
        feature_bytes,
    )
    try:
        actual = (
            artifacts.run_lock_path.read_bytes(),
            artifacts.experiment_lock_path.read_bytes(),
            artifacts.feature_rows_path.read_bytes(),
        )
    except OSError as error:
        raise OutcomeV2InferenceError("cannot recheck generation inputs") from error
    if actual != expected:
        raise OutcomeV2InferenceError("generation inputs changed while building the lock")


def _locked_inference_settings(run_record: Mapping[str, object]) -> JsonObject:
    settings = _object(run_record, "inference_settings")
    if settings != outcome_v2_inference_settings():
        raise OutcomeV2InferenceError("run lock uses different outcome-v2 inference settings")
    return settings


def _locked_renderer(run_record: Mapping[str, object]) -> str:
    renderer_name = _string(_object(run_record, "model"), "renderer")
    if not renderer_name.strip() or renderer_name != renderer_name.strip():
        raise OutcomeV2InferenceError("run-lock renderer name is invalid")
    return renderer_name


def _tabular_seasons(run_record: Mapping[str, object]) -> tuple[int, ...]:
    preflight = _object(run_record, "preflight")
    seasons = _integer_tuple(preflight, "untouched_evaluation_seasons")
    _require_seasons(seasons, "tabular")
    return seasons


def _require_new_later_seasons(
    tabular_seasons: tuple[int, ...],
    evaluation_seasons: tuple[int, ...],
) -> None:
    _require_nonempty_seasons(evaluation_seasons, "post-SFT evaluation")
    if min(evaluation_seasons) <= max(tabular_seasons):
        raise OutcomeV2InferenceError(
            "every post-SFT evaluation season must be later than every tabular season"
        )


def _require_new_feature_rows(run_record: Mapping[str, object], feature_sha256: str) -> None:
    preflight = _object(run_record, "preflight")
    tabular_feature_sha256 = _string(preflight, "evaluation_feature_rows_sha256")
    if feature_sha256 == tabular_feature_sha256:
        raise OutcomeV2InferenceError("post-SFT generation requires new feature rows")


def _require_original_rows(
    rows: tuple[NbaRichFeatureRow, ...],
) -> tuple[NbaRichFeatureRow, ...]:
    if not rows:
        raise OutcomeV2InferenceError("post-SFT feature rows must not be empty")
    question_ids = tuple(row.question_id for row in rows)
    if len(set(question_ids)) != len(question_ids):
        raise OutcomeV2InferenceError("post-SFT feature-row IDs must be unique")
    if any(question_id.endswith(SIDE_SWAP_SUFFIX) for question_id in question_ids):
        raise OutcomeV2InferenceError("post-SFT feature rows must contain originals only")
    return rows


def _call_policy(settings: Mapping[str, object], game_count: int) -> JsonObject:
    logical_calls = _settings_integer(settings, "logical_calls_per_game")
    application_attempts = _settings_integer(settings, "application_attempts_per_game")
    application_retries = _settings_integer(settings, "application_retries")
    unused_tokens = _settings_integer(
        settings,
        "sdk_internal_unused_generated_tokens_per_call",
    )
    retry_logic = _settings_boolean(settings, "sdk_retry_logic_enabled")
    retransmission_window = _settings_integer(
        settings,
        "sdk_internal_retransmission_window_seconds",
    )
    generated_text_used = _settings_boolean(settings, "generated_text_used")
    transport_note = _settings_string(settings, "transport_retry_note")
    if logical_calls != 4:
        raise OutcomeV2InferenceError("outcome-v2 inference must use four logical calls per game")
    if application_attempts != 1 or application_retries != 0 or retry_logic:
        raise OutcomeV2InferenceError("outcome-v2 inference must disable application retries")
    if retransmission_window != 300:
        raise OutcomeV2InferenceError("outcome-v2 SDK retransmission disclosure is inconsistent")
    if generated_text_used or unused_tokens != 1:
        raise OutcomeV2InferenceError("outcome-v2 SDK generation disclosure is inconsistent")
    if not transport_note.strip():
        raise OutcomeV2InferenceError("transport retransmission disclosure must not be empty")
    return {
        "logical_calls_per_game": logical_calls,
        "expected_logical_calls": game_count * logical_calls,
        "application_attempts_per_game": application_attempts,
        "application_retries": application_retries,
        "sdk_retry_logic_enabled": retry_logic,
        "sdk_internal_retransmission_window_seconds": retransmission_window,
        "generated_text_used": generated_text_used,
        "sdk_internal_unused_generated_tokens_per_call": unused_tokens,
        "transport_retry_note": transport_note,
    }


def _require_label_token_ids(value: tuple[int, int]) -> tuple[int, int]:
    team_token, opponent_token = value
    for token in value:
        if _invalid_token_id(token):
            raise OutcomeV2InferenceError("label token IDs must be non-negative integers")
    if team_token == opponent_token:
        raise OutcomeV2InferenceError("label token IDs must be distinct")
    return team_token, opponent_token


def _require_lock_row(
    generation_lock: OutcomeV2GenerationLock,
    sequence: int,
    row: NbaRichFeatureRow,
) -> None:
    lock_record = generation_lock.to_record()
    question_ids = _string_tuple(lock_record, "question_ids")
    feature_row_sha256s = _string_tuple(lock_record, "feature_row_sha256s")
    if _invalid_sequence(sequence) or not 0 <= sequence < len(question_ids):
        raise OutcomeV2InferenceError("inference sequence is outside the generation lock")
    if question_ids[sequence] != row.question_id:
        raise OutcomeV2InferenceError("inference row differs from the locked ID order")
    if feature_row_sha256s[sequence] != row.row_sha256:
        raise OutcomeV2InferenceError("inference row differs from the locked feature row")


def _validate_orientation_score(score: OrientationScore) -> None:
    values = (
        score.elo_team_probability,
        score.team_logprob,
        score.opponent_logprob,
        score.valid_label_mass,
        score.team_probability,
    )
    if any(isinstance(value, bool) or not isfinite(value) for value in values):
        raise OutcomeV2InferenceError("orientation scores must be finite numbers")
    if not 0.0 < score.elo_team_probability < 1.0:
        raise OutcomeV2InferenceError("orientation Elo probability must be interior")
    if score.team_logprob > 0.0 or score.opponent_logprob > 0.0:
        raise OutcomeV2InferenceError("orientation label log-probabilities cannot be positive")
    expected_mass = exp(score.team_logprob) + exp(score.opponent_logprob)
    if not 0.0 <= score.valid_label_mass <= 1.000001:
        raise OutcomeV2InferenceError("orientation valid-label mass exceeds one")
    if not isclose(score.valid_label_mass, expected_mass, rel_tol=1e-12, abs_tol=1e-12):
        raise OutcomeV2InferenceError("orientation valid-label mass differs from raw scores")
    try:
        expected_probability = elo_offset_team_probability_from_logprobs(
            score.elo_team_probability,
            score.team_logprob,
            score.opponent_logprob,
        )
    except OutcomeForecastError as error:
        raise OutcomeV2InferenceError("orientation probability cannot be reconstructed") from error
    if not 0.0 < score.team_probability < 1.0 or not isclose(
        score.team_probability,
        expected_probability,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise OutcomeV2InferenceError("orientation probability differs from Elo and raw scores")


def _validate_inference_record(record: InferenceRecord) -> None:
    if _invalid_sequence(record.sequence):
        raise OutcomeV2InferenceError("inference sequence must be an integer")
    if record.sequence < 0:
        raise OutcomeV2InferenceError("inference sequence must be non-negative")
    _require_hash(record.generation_lock_sha256, "generation_lock_sha256")
    _require_hash(record.feature_row_sha256, "feature_row_sha256")
    _require_hash(
        record.original_prompt_token_ids_sha256,
        "original_prompt_token_ids_sha256",
    )
    _require_hash(
        record.swapped_prompt_token_ids_sha256,
        "swapped_prompt_token_ids_sha256",
    )
    if not record.question_id.strip() or (
        record.swapped_question_id != f"{record.question_id}{SIDE_SWAP_SUFFIX}"
    ):
        raise OutcomeV2InferenceError("inference record IDs must form one side-swap pair")
    if record.status == "failed":
        _validate_failed_record(record)
        return
    if record.status != "completed":
        raise OutcomeV2InferenceError("unsupported inference record status")
    _validate_completed_record(record)


def _validate_failed_record(record: InferenceRecord) -> None:
    reason = record.failure_reason
    if reason is None or (
        reason not in _FIXED_FAILURE_REASONS and _FAILURE_PATTERN.fullmatch(reason) is None
    ):
        raise OutcomeV2InferenceError("failed inference record requires a sanitized reason")
    if any(
        value is not None
        for value in (
            record.original,
            record.swapped,
            record.team_probability,
            record.pre_average_side_swap_gap,
        )
    ):
        raise OutcomeV2InferenceError("failed inference record cannot retain partial scores")


def _validate_completed_record(record: InferenceRecord) -> None:
    original = record.original
    swapped = record.swapped
    probability = record.team_probability
    gap = record.pre_average_side_swap_gap
    if record.failure_reason is not None or original is None or swapped is None:
        raise OutcomeV2InferenceError("completed inference record is missing raw scores")
    if probability is None or gap is None or not isfinite(probability) or not isfinite(gap):
        raise OutcomeV2InferenceError("completed inference record has invalid diagnostics")
    if not 0.0 < probability < 1.0:
        raise OutcomeV2InferenceError("completed inference probability must be interior")
    if not isclose(
        original.elo_team_probability + swapped.elo_team_probability,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise OutcomeV2InferenceError("completed orientations do not use complementary Elo priors")
    expected_probability = symmetric_team_probability(
        original.team_probability,
        swapped.team_probability,
    )
    expected_gap = abs(original.team_probability - (1.0 - swapped.team_probability))
    if not isclose(probability, expected_probability, rel_tol=0.0, abs_tol=1e-12):
        raise OutcomeV2InferenceError("inference probability was not side-swap averaged")
    if not isclose(gap, expected_gap, rel_tol=0.0, abs_tol=1e-12):
        raise OutcomeV2InferenceError("inference side-swap gap differs from raw orientations")


def _require_record_coverage(
    records: tuple[InferenceRecord, ...],
    generation_lock: OutcomeV2GenerationLock,
) -> tuple[InferenceRecord, ...]:
    lock_record = generation_lock.to_record()
    question_ids = _string_tuple(lock_record, "question_ids")
    feature_row_sha256s = _string_tuple(lock_record, "feature_row_sha256s")
    if len(records) != len(question_ids):
        raise OutcomeV2InferenceError("terminal inference records have incomplete coverage")
    for sequence, (record, question_id) in enumerate(zip(records, question_ids, strict=True)):
        if record.sequence != sequence or record.question_id != question_id:
            raise OutcomeV2InferenceError("terminal inference record order differs from the lock")
        if record.generation_lock_sha256 != generation_lock.sha256:
            raise OutcomeV2InferenceError("terminal inference record binds a different lock")
        if record.feature_row_sha256 != feature_row_sha256s[sequence]:
            raise OutcomeV2InferenceError("terminal inference record binds a different feature row")
    return records


def _record_payload(record: InferenceRecord) -> JsonObject:
    return {
        "schema_version": OUTCOME_V2_INFERENCE_RECORD_SCHEMA_VERSION,
        "kind": _RECORD_KIND,
        "sequence": record.sequence,
        "generation_lock_sha256": record.generation_lock_sha256,
        "question_id": record.question_id,
        "swapped_question_id": record.swapped_question_id,
        "feature_row_sha256": record.feature_row_sha256,
        "original_prompt_token_ids_sha256": record.original_prompt_token_ids_sha256,
        "swapped_prompt_token_ids_sha256": record.swapped_prompt_token_ids_sha256,
        "status": record.status,
        "failure_reason": record.failure_reason,
        "original": None if record.original is None else _orientation_payload(record.original),
        "swapped": None if record.swapped is None else _orientation_payload(record.swapped),
        "team_probability": record.team_probability,
        "pre_average_side_swap_gap": record.pre_average_side_swap_gap,
    }


def _record_from_payload(payload: Mapping[str, object]) -> InferenceRecord:
    require_exact_keys(payload, _RECORD_KEYS, "inference record")
    if _integer(payload, "schema_version") != OUTCOME_V2_INFERENCE_RECORD_SCHEMA_VERSION:
        raise OutcomeV2InferenceError("unsupported inference-record schema")
    if _string(payload, "kind") != _RECORD_KIND:
        raise OutcomeV2InferenceError("unexpected inference-record kind")
    status = _inference_status(_string(payload, "status"))
    reason_value = required_field(payload, "failure_reason")
    probability_value = required_field(payload, "team_probability")
    gap_value = required_field(payload, "pre_average_side_swap_gap")
    return InferenceRecord(
        sequence=_integer(payload, "sequence"),
        generation_lock_sha256=_string(payload, "generation_lock_sha256"),
        question_id=_string(payload, "question_id"),
        swapped_question_id=_string(payload, "swapped_question_id"),
        feature_row_sha256=_string(payload, "feature_row_sha256"),
        original_prompt_token_ids_sha256=_string(
            payload,
            "original_prompt_token_ids_sha256",
        ),
        swapped_prompt_token_ids_sha256=_string(
            payload,
            "swapped_prompt_token_ids_sha256",
        ),
        status=status,
        failure_reason=(
            None if reason_value is None else require_string(reason_value, "failure_reason")
        ),
        original=_optional_orientation(required_field(payload, "original"), "original"),
        swapped=_optional_orientation(required_field(payload, "swapped"), "swapped"),
        team_probability=(
            None
            if probability_value is None
            else require_float(probability_value, "team_probability")
        ),
        pre_average_side_swap_gap=(
            None if gap_value is None else require_float(gap_value, "pre_average_side_swap_gap")
        ),
    )


def _orientation_payload(score: OrientationScore) -> JsonObject:
    return {
        "elo_team_probability": score.elo_team_probability,
        "team_logprob": score.team_logprob,
        "opponent_logprob": score.opponent_logprob,
        "valid_label_mass": score.valid_label_mass,
        "team_probability": score.team_probability,
    }


def _optional_orientation(value: object, field_name: str) -> OrientationScore | None:
    if value is None:
        return None
    payload = require_object(value, field_name)
    require_exact_keys(payload, _ORIENTATION_KEYS, field_name)
    return OrientationScore(
        elo_team_probability=require_float(
            required_field(payload, "elo_team_probability"),
            f"{field_name}.elo_team_probability",
        ),
        team_logprob=require_float(
            required_field(payload, "team_logprob"),
            f"{field_name}.team_logprob",
        ),
        opponent_logprob=require_float(
            required_field(payload, "opponent_logprob"),
            f"{field_name}.opponent_logprob",
        ),
        valid_label_mass=require_float(
            required_field(payload, "valid_label_mass"),
            f"{field_name}.valid_label_mass",
        ),
        team_probability=require_float(
            required_field(payload, "team_probability"),
            f"{field_name}.team_probability",
        ),
    )


def _generation_lock_record(value: bytes) -> JsonObject:
    try:
        text = value.decode("utf-8")
        record = parse_json_object(text)
    except (JsonFormatError, UnicodeError) as error:
        raise OutcomeV2InferenceError("generation lock must be one UTF-8 JSON object") from error
    if text != canonical_json(record):
        raise OutcomeV2InferenceError("generation lock must use canonical JSON bytes")
    _validate_generation_lock_record(record)
    return record


def _validate_generation_lock_record(record: Mapping[str, object]) -> None:
    try:
        require_exact_keys(record, _LOCK_KEYS, "generation lock")
        if _integer(record, "schema_version") != OUTCOME_V2_GENERATION_LOCK_SCHEMA_VERSION:
            raise OutcomeV2InferenceError("unsupported generation-lock schema")
        if _string(record, "kind") != _LOCK_KIND:
            raise OutcomeV2InferenceError("unexpected generation-lock kind")
        if _string(record, "status") != _LOCK_STATUS:
            raise OutcomeV2InferenceError("generation lock was not committed before calls")
        if _string(record, "candidate_role") != _CANDIDATE_ROLE:
            raise OutcomeV2InferenceError("generation lock uses the wrong candidate role")
        _parse_utc(_string(record, "created_at"), "created_at")
        for field_name in (
            "outcome_v2_run_lock_sha256",
            "outcome_v2_experiment_lock_sha256",
            "feature_rows_sha256",
            "prompt_pairs_sha256",
            "question_ids_sha256",
            "inference_settings_sha256",
        ):
            _require_hash(_string(record, field_name), field_name)
        _require_tinker_path(_string(record, "sampler_path"))
        _validate_generation_identity(record)
        _validate_generation_settings(record)
    except JsonFormatError as error:
        raise OutcomeV2InferenceError("invalid generation-lock structure") from error


def _validate_generation_identity(record: Mapping[str, object]) -> None:
    question_ids = _string_tuple(record, "question_ids")
    if not question_ids or len(question_ids) != len(set(question_ids)):
        raise OutcomeV2InferenceError("generation-lock question IDs must be nonempty and unique")
    if any(not value.strip() or value.endswith(SIDE_SWAP_SUFFIX) for value in question_ids):
        raise OutcomeV2InferenceError("generation-lock question IDs must contain originals only")
    if canonical_sha256(list(question_ids)) != _string(record, "question_ids_sha256"):
        raise OutcomeV2InferenceError("generation-lock question IDs differ from their hash")
    game_count = _positive_integer(record, "game_count")
    orientation_count = _positive_integer(record, "orientation_count")
    if game_count != len(question_ids) or orientation_count != game_count * 2:
        raise OutcomeV2InferenceError("generation-lock row counts are inconsistent")
    _validate_feature_row_hashes(record, game_count)
    tabular_seasons = _integer_tuple(record, "tabular_seasons")
    evaluation_seasons = _integer_tuple(record, "evaluation_seasons")
    _require_seasons(tabular_seasons, "tabular")
    _require_new_later_seasons(tabular_seasons, evaluation_seasons)
    if _string(record, "season_relation") != _SEASON_RELATION:
        raise OutcomeV2InferenceError("generation-lock season relation is invalid")
    renderer_name = _string(record, "renderer_name")
    if not renderer_name.strip() or renderer_name != renderer_name.strip():
        raise OutcomeV2InferenceError("generation-lock renderer name is invalid")
    labels = _object(record, "label_token_ids")
    require_exact_keys(labels, _LABEL_TOKEN_KEYS, "label_token_ids")
    _require_label_token_ids(
        (
            _nonnegative_integer(labels, TEAM_LABEL),
            _nonnegative_integer(labels, OPPONENT_LABEL),
        )
    )


def _validate_feature_row_hashes(
    record: Mapping[str, object],
    game_count: int,
) -> None:
    values = _string_tuple(record, "feature_row_sha256s")
    if len(values) != game_count:
        raise OutcomeV2InferenceError("generation-lock feature-row hashes are incomplete")
    for index, value in enumerate(values):
        _require_hash(value, f"feature_row_sha256s[{index}]")


def _validate_generation_settings(record: Mapping[str, object]) -> None:
    settings = _object(record, "inference_settings")
    if settings != outcome_v2_inference_settings():
        raise OutcomeV2InferenceError("generation-lock inference settings are not frozen")
    if canonical_sha256(settings) != _string(record, "inference_settings_sha256"):
        raise OutcomeV2InferenceError("generation-lock inference settings hash is invalid")
    call_policy = _object(record, "call_policy")
    require_exact_keys(call_policy, _CALL_POLICY_KEYS, "call_policy")
    expected = _call_policy(settings, _positive_integer(record, "game_count"))
    if call_policy != expected:
        raise OutcomeV2InferenceError("generation-lock call policy differs from its settings")


def _write_once(path: Path, value: bytes, description: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError:
        raise
    except OSError as error:
        raise OutcomeV2InferenceError(f"cannot write {description}") from error


def _require_immutable_bytes(value: object) -> None:
    if not isinstance(value, bytes):
        raise OutcomeV2InferenceError("generation lock requires immutable bytes")


def _invalid_token_id(value: object) -> bool:
    return isinstance(value, bool) or not isinstance(value, int) or value < 0


def _invalid_sequence(value: object) -> bool:
    return isinstance(value, bool) or not isinstance(value, int)


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise OutcomeV2InferenceError(f"cannot read {description}") from error


def _object(record: Mapping[str, object], field_name: str) -> JsonObject:
    return require_object(required_field(record, field_name), field_name)


def _string(record: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = required_field(record, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutcomeV2InferenceError(f"{field_name} must be an integer")
    return value


def _positive_integer(record: Mapping[str, object], field_name: str) -> int:
    value = _integer(record, field_name)
    if value <= 0:
        raise OutcomeV2InferenceError(f"{field_name} must be positive")
    return value


def _nonnegative_integer(record: Mapping[str, object], field_name: str) -> int:
    value = _integer(record, field_name)
    if value < 0:
        raise OutcomeV2InferenceError(f"{field_name} must be non-negative")
    return value


def _integer_tuple(record: Mapping[str, object], field_name: str) -> tuple[int, ...]:
    values = require_list(required_field(record, field_name), field_name)
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise OutcomeV2InferenceError(f"{field_name}[{index}] must be an integer")
        result.append(value)
    return tuple(result)


def _string_tuple(record: Mapping[str, object], field_name: str) -> tuple[str, ...]:
    values = require_list(required_field(record, field_name), field_name)
    return tuple(
        require_string(value, f"{field_name}[{index}]") for index, value in enumerate(values)
    )


def _require_seasons(seasons: tuple[int, ...], description: str) -> None:
    if (
        len(seasons) < 2
        or seasons != tuple(sorted(set(seasons)))
        or any(season <= 0 for season in seasons)
    ):
        raise OutcomeV2InferenceError(
            f"{description} seasons must contain at least two increasing positive values"
        )


def _require_nonempty_seasons(seasons: tuple[int, ...], description: str) -> None:
    if (
        not seasons
        or seasons != tuple(sorted(set(seasons)))
        or any(season <= 0 for season in seasons)
    ):
        raise OutcomeV2InferenceError(
            f"{description} seasons must contain increasing positive values"
        )


def _require_hash(value: str, field_name: str) -> None:
    if _HASH_PATTERN.fullmatch(value) is None:
        raise OutcomeV2InferenceError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_tinker_path(value: str) -> None:
    suffix = value.removeprefix("tinker://")
    has_control = any(character.isspace() or ord(character) < 32 for character in value)
    if not value.startswith("tinker://") or not suffix or has_control:
        raise OutcomeV2InferenceError("sampler_path must be a permanent tinker:// path")
    if "?" in suffix or "#" in suffix:
        raise OutcomeV2InferenceError("sampler_path must not contain query or fragment data")


def _settings_integer(settings: Mapping[str, object], field_name: str) -> int:
    value = required_field(settings, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutcomeV2InferenceError(f"inference setting {field_name} must be an integer")
    return value


def _settings_boolean(settings: Mapping[str, object], field_name: str) -> bool:
    value = required_field(settings, field_name)
    if not isinstance(value, bool):
        raise OutcomeV2InferenceError(f"inference setting {field_name} must be a boolean")
    return value


def _settings_string(settings: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(settings, field_name), field_name)


def _inference_status(value: str) -> InferenceStatus:
    if value not in {"completed", "failed"}:
        raise OutcomeV2InferenceError("unsupported inference record status")
    return cast(InferenceStatus, value)


def _parse_utc(value: str, field_name: str) -> datetime:
    if not value.endswith("Z"):
        raise OutcomeV2InferenceError(f"{field_name} must use canonical UTC Z notation")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise OutcomeV2InferenceError(f"{field_name} must be an ISO 8601 datetime") from error
    if _utc_text(parsed, field_name) != value:
        raise OutcomeV2InferenceError(f"{field_name} must use canonical UTC notation")
    return parsed.astimezone(UTC)


def _utc_text(value: object, field_name: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise OutcomeV2InferenceError(f"{field_name} must be a UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise OutcomeV2InferenceError(f"{field_name} must be a UTC datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
