"""Immutable answer-blind artifacts for outcome-model evaluation."""

import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from math import exp, isclose, isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, cast

from forecastfm.integrity import canonical_sha256, file_sha256
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
from forecastfm.tinker_data import (
    ForecastRecord,
    pair_outcome_forecast_records,
    read_outcome_forecast_jsonl,
)

EVALUATION_SCHEMA_VERSION = 1
type ModelRole = Literal["base", "adapter"]
type RecordStatus = Literal["completed", "failed"]

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40,64}")
_MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "status",
    "created_at",
    "protocol_revision",
    "source_manifest_sha256",
    "source_prompts_sha256",
    "source_answers_sha256",
    "frozen_prompts_sha256",
    "training_lock_sha256",
    "experiment_sha256",
    "base_model",
    "adapter_sampler_path",
    "renderer_name",
    "team_token_id",
    "opponent_token_id",
    "game_count",
    "orientation_count",
    "logical_calls_per_game_per_arm",
    "expected_total_logical_calls",
    "max_active_arms",
    "application_retries",
    "transport_retry_note",
    "question_ids",
    "question_ids_sha256",
    "scoring_policy",
}
_RECORD_KEYS = {
    "schema_version",
    "sequence",
    "model_role",
    "question_id",
    "swapped_question_id",
    "status",
    "error",
    "original_prompt_tokens",
    "swapped_prompt_tokens",
    "original",
    "swapped",
    "symmetric_team_probability",
    "pre_average_side_swap_gap",
}
_ORIENTATION_KEYS = {
    "team_logprob",
    "opponent_logprob",
    "team_probability",
    "valid_label_mass",
}
_ATTEMPT_KEYS = {
    "schema_version",
    "kind",
    "status",
    "created_at",
    "manifest_sha256",
    "prompts_sha256",
}
_SEAL_KEYS = {
    "schema_version",
    "kind",
    "status",
    "created_at",
    "manifest_sha256",
    "prompts_sha256",
    "attempt_sha256",
    "journal_sha256",
    "base_sha256",
    "adapter_sha256",
    "game_count",
    "base_completed",
    "adapter_completed",
    "question_ids_sha256",
}


class OutcomeEvaluationError(ValueError):
    """Raised when an outcome evaluation artifact is incomplete or changed."""


@dataclass(frozen=True, slots=True)
class OutcomeEvaluationManifest:
    """Every answer-blind commitment needed before remote inference."""

    created_at: str
    protocol_revision: str
    source_manifest_sha256: str
    source_prompts_sha256: str
    source_answers_sha256: str
    frozen_prompts_sha256: str
    training_lock_sha256: str
    experiment_sha256: str
    base_model: str
    adapter_sampler_path: str
    renderer_name: str
    team_token_id: int
    opponent_token_id: int
    game_count: int
    orientation_count: int
    logical_calls_per_game_per_arm: int
    expected_total_logical_calls: int
    max_active_arms: int
    application_retries: int
    transport_retry_note: str
    question_ids: tuple[str, ...]
    question_ids_sha256: str
    scoring_policy: dict[str, object]

    def __post_init__(self) -> None:
        _validate_manifest_identity(self)
        _validate_manifest_counts(self)


@dataclass(frozen=True, slots=True)
class EvaluationPaths:
    """All published and raw paths for one outcome evaluation run."""

    manifest: Path
    prompts: Path
    attempt: Path
    journal: Path
    base: Path
    adapter: Path
    seal: Path


def _validate_manifest_identity(manifest: OutcomeEvaluationManifest) -> None:
    _require_utc(manifest.created_at)
    _require_revision(manifest.protocol_revision)
    for value in (
        manifest.source_manifest_sha256,
        manifest.source_prompts_sha256,
        manifest.source_answers_sha256,
        manifest.frozen_prompts_sha256,
        manifest.training_lock_sha256,
        manifest.experiment_sha256,
        manifest.question_ids_sha256,
    ):
        _require_hash(value)
    if not manifest.base_model.strip() or not manifest.renderer_name.strip():
        raise OutcomeEvaluationError("evaluation model settings must not be empty")
    if not manifest.adapter_sampler_path.startswith("tinker://") or not (
        manifest.adapter_sampler_path.removeprefix("tinker://").strip()
    ):
        raise OutcomeEvaluationError("adapter sampler path must be immutable")
    if manifest.team_token_id < 0 or manifest.opponent_token_id < 0:
        raise OutcomeEvaluationError("label token IDs must be non-negative")
    if manifest.team_token_id == manifest.opponent_token_id:
        raise OutcomeEvaluationError("label token IDs must differ")


def _validate_manifest_counts(manifest: OutcomeEvaluationManifest) -> None:
    _validate_manifest_cohort(manifest)
    _validate_manifest_call_policy(manifest)


def _validate_manifest_cohort(manifest: OutcomeEvaluationManifest) -> None:
    if manifest.game_count <= 0 or manifest.orientation_count != manifest.game_count * 2:
        raise OutcomeEvaluationError("evaluation cohort counts are inconsistent")
    if len(manifest.question_ids) != manifest.game_count:
        raise OutcomeEvaluationError("evaluation question IDs are incomplete")
    if len(set(manifest.question_ids)) != manifest.game_count:
        raise OutcomeEvaluationError("evaluation question IDs must be unique")
    if canonical_sha256(list(manifest.question_ids)) != manifest.question_ids_sha256:
        raise OutcomeEvaluationError("evaluation question IDs differ from their commitment")
    if any(not question_id.strip() for question_id in manifest.question_ids):
        raise OutcomeEvaluationError("evaluation question IDs must not be blank")
    if not manifest.transport_retry_note.strip():
        raise OutcomeEvaluationError("transport retry disclosure must not be blank")


def _validate_manifest_call_policy(manifest: OutcomeEvaluationManifest) -> None:
    if manifest.logical_calls_per_game_per_arm != 4:
        raise OutcomeEvaluationError("outcome inference requires four logical calls per arm")
    expected_calls = manifest.game_count * manifest.logical_calls_per_game_per_arm * 2
    if manifest.expected_total_logical_calls != expected_calls:
        raise OutcomeEvaluationError("expected logical-call count is inconsistent")
    if manifest.max_active_arms != 1:
        raise OutcomeEvaluationError("outcome evaluation requires one active arm")
    if manifest.application_retries != 0:
        raise OutcomeEvaluationError("outcome evaluation forbids application retries")


@dataclass(frozen=True, slots=True)
class OrientationResult:
    """Raw two-label diagnostics for one frozen prompt orientation."""

    team_logprob: float
    opponent_logprob: float
    team_probability: float
    valid_label_mass: float

    def __post_init__(self) -> None:
        values = (
            self.team_logprob,
            self.opponent_logprob,
            self.team_probability,
            self.valid_label_mass,
        )
        if not all(isfinite(value) for value in values):
            raise OutcomeEvaluationError("orientation diagnostics must be finite")
        if self.team_logprob > 0.000001 or self.opponent_logprob > 0.000001:
            raise OutcomeEvaluationError("label log-probabilities cannot be positive")
        if not 0.0 <= self.team_probability <= 1.0:
            raise OutcomeEvaluationError("orientation probability is outside [0, 1]")
        if not 0.0 <= self.valid_label_mass <= 1.000001:
            raise OutcomeEvaluationError("valid-label mass is outside [0, 1]")
        expected_mass = exp(self.team_logprob) + exp(self.opponent_logprob)
        maximum = max(self.team_logprob, self.opponent_logprob)
        team_weight = exp(self.team_logprob - maximum)
        expected_probability = team_weight / (team_weight + exp(self.opponent_logprob - maximum))
        if not isclose(self.valid_label_mass, expected_mass, rel_tol=1e-12, abs_tol=1e-12):
            raise OutcomeEvaluationError("valid-label mass differs from the raw log-probabilities")
        if not isclose(self.team_probability, expected_probability, rel_tol=1e-12, abs_tol=1e-12):
            raise OutcomeEvaluationError("team probability differs from the raw log-probabilities")


@dataclass(frozen=True, slots=True)
class OutcomeEvaluationRecord:
    """One terminal base or adapter result for one original/swap game pair."""

    sequence: int
    model_role: ModelRole
    question_id: str
    swapped_question_id: str
    status: RecordStatus
    error: str | None
    original_prompt_tokens: tuple[int, ...]
    swapped_prompt_tokens: tuple[int, ...]
    original: OrientationResult | None
    swapped: OrientationResult | None
    symmetric_team_probability: float | None
    pre_average_side_swap_gap: float | None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise OutcomeEvaluationError("record sequence must be non-negative")
        if self.model_role not in {"base", "adapter"}:
            raise OutcomeEvaluationError("record model role is invalid")
        if self.status not in {"completed", "failed"}:
            raise OutcomeEvaluationError("record status is invalid")
        if (
            not self.question_id.strip()
            or self.swapped_question_id != f"{self.question_id}-side-swap"
        ):
            raise OutcomeEvaluationError("record question IDs do not form a side-swap pair")
        _validate_prompt_tokens(self.original_prompt_tokens)
        _validate_prompt_tokens(self.swapped_prompt_tokens)
        if self.status == "failed":
            _validate_failed_record(self)
            return
        _validate_completed_record(self)


def _validate_failed_record(record: OutcomeEvaluationRecord) -> None:
    if record.error is None or not record.error.strip():
        raise OutcomeEvaluationError("failed outcome record must retain an error")
    if any(
        value is not None
        for value in (
            record.original,
            record.swapped,
            record.symmetric_team_probability,
            record.pre_average_side_swap_gap,
        )
    ):
        raise OutcomeEvaluationError("failed outcome record cannot retain partial scores")


def _validate_completed_record(record: OutcomeEvaluationRecord) -> None:
    if record.error is not None or record.original is None or record.swapped is None:
        raise OutcomeEvaluationError("completed outcome record is missing raw scores")
    probability = record.symmetric_team_probability
    gap = record.pre_average_side_swap_gap
    if probability is None or gap is None or not isfinite(probability) or not isfinite(gap):
        raise OutcomeEvaluationError("completed outcome record has invalid diagnostics")
    expected_probability = (
        record.original.team_probability + 1.0 - record.swapped.team_probability
    ) / 2.0
    expected_gap = abs(record.original.team_probability - (1.0 - record.swapped.team_probability))
    if not isclose(probability, expected_probability, abs_tol=1e-12):
        raise OutcomeEvaluationError("symmetric probability was not derived from both sides")
    if not isclose(gap, expected_gap, abs_tol=1e-12):
        raise OutcomeEvaluationError("side-swap gap was not derived from both sides")


@dataclass(frozen=True, slots=True)
class SealedOutcomeEvaluation:
    """Verified manifest, prompts, and paired base/adapter raw records."""

    manifest: OutcomeEvaluationManifest
    prompt_pairs: tuple[tuple[ForecastRecord, ForecastRecord], ...]
    base: tuple[OutcomeEvaluationRecord, ...]
    adapter: tuple[OutcomeEvaluationRecord, ...]
    seal: dict[str, object]


def write_manifest(path: Path, manifest: OutcomeEvaluationManifest) -> None:
    """Exclusively write one frozen answer-blind evaluation manifest."""
    write_json_exclusively(path, _manifest_to_dict(manifest))


def read_manifest(path: Path) -> OutcomeEvaluationManifest:
    """Read and strictly validate a frozen evaluation manifest."""
    try:
        record = parse_json_object(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise OutcomeEvaluationError(f"evaluation manifest is missing: {path}") from error
    require_exact_keys(record, _MANIFEST_KEYS, "outcome evaluation manifest")
    if required_field(record, "schema_version") != EVALUATION_SCHEMA_VERSION:
        raise OutcomeEvaluationError("unsupported outcome evaluation schema")
    if required_field(record, "kind") != "forecastfm_outcome_evaluation":
        raise OutcomeEvaluationError("unexpected outcome evaluation kind")
    if required_field(record, "status") != "frozen_before_answer_access":
        raise OutcomeEvaluationError("outcome evaluation is not frozen")
    question_ids = _string_tuple(required_field(record, "question_ids"), "question_ids")
    scoring_policy = require_object(required_field(record, "scoring_policy"), "scoring_policy")
    return OutcomeEvaluationManifest(
        created_at=require_string(required_field(record, "created_at"), "created_at"),
        protocol_revision=require_string(
            required_field(record, "protocol_revision"), "protocol_revision"
        ),
        source_manifest_sha256=_string_field(record, "source_manifest_sha256"),
        source_prompts_sha256=_string_field(record, "source_prompts_sha256"),
        source_answers_sha256=_string_field(record, "source_answers_sha256"),
        frozen_prompts_sha256=_string_field(record, "frozen_prompts_sha256"),
        training_lock_sha256=_string_field(record, "training_lock_sha256"),
        experiment_sha256=_string_field(record, "experiment_sha256"),
        base_model=_string_field(record, "base_model"),
        adapter_sampler_path=_string_field(record, "adapter_sampler_path"),
        renderer_name=_string_field(record, "renderer_name"),
        team_token_id=_integer_field(record, "team_token_id"),
        opponent_token_id=_integer_field(record, "opponent_token_id"),
        game_count=_integer_field(record, "game_count"),
        orientation_count=_integer_field(record, "orientation_count"),
        logical_calls_per_game_per_arm=_integer_field(record, "logical_calls_per_game_per_arm"),
        expected_total_logical_calls=_integer_field(record, "expected_total_logical_calls"),
        max_active_arms=_integer_field(record, "max_active_arms"),
        application_retries=_integer_field(record, "application_retries"),
        transport_retry_note=_string_field(record, "transport_retry_note"),
        question_ids=question_ids,
        question_ids_sha256=_string_field(record, "question_ids_sha256"),
        scoring_policy=scoring_policy,
    )


def load_prompt_pairs(
    manifest: OutcomeEvaluationManifest,
    prompts_path: Path,
) -> tuple[tuple[ForecastRecord, ForecastRecord], ...]:
    """Load the frozen prompt bytes and verify exact ordered pair coverage."""
    if file_sha256(prompts_path) != manifest.frozen_prompts_sha256:
        raise OutcomeEvaluationError("frozen prompts differ from the manifest")
    pairs = pair_outcome_forecast_records(read_outcome_forecast_jsonl(prompts_path))
    question_ids = tuple(original["question_id"] for original, _swapped in pairs)
    if question_ids != manifest.question_ids:
        raise OutcomeEvaluationError("frozen prompt IDs differ from the manifest")
    return pairs


def completed_record(
    sequence: int,
    model_role: ModelRole,
    question_ids: tuple[str, str],
    prompt_tokens: tuple[tuple[int, ...], tuple[int, ...]],
    orientations: tuple[OrientationResult, OrientationResult],
) -> OutcomeEvaluationRecord:
    """Build a completed record from both raw orientation results."""
    original, swapped = orientations
    probability = (original.team_probability + 1.0 - swapped.team_probability) / 2.0
    gap = abs(original.team_probability - (1.0 - swapped.team_probability))
    return OutcomeEvaluationRecord(
        sequence=sequence,
        model_role=model_role,
        question_id=question_ids[0],
        swapped_question_id=question_ids[1],
        status="completed",
        error=None,
        original_prompt_tokens=prompt_tokens[0],
        swapped_prompt_tokens=prompt_tokens[1],
        original=original,
        swapped=swapped,
        symmetric_team_probability=probability,
        pre_average_side_swap_gap=gap,
    )


def failed_record(
    sequence: int,
    model_role: ModelRole,
    question_ids: tuple[str, str],
    prompt_tokens: tuple[tuple[int, ...], tuple[int, ...]],
    error: str,
) -> OutcomeEvaluationRecord:
    """Build one terminal failure with prompts but no ambiguous partial scores."""
    return OutcomeEvaluationRecord(
        sequence=sequence,
        model_role=model_role,
        question_id=question_ids[0],
        swapped_question_id=question_ids[1],
        status="failed",
        error=error,
        original_prompt_tokens=prompt_tokens[0],
        swapped_prompt_tokens=prompt_tokens[1],
        original=None,
        swapped=None,
        symmetric_team_probability=None,
        pre_average_side_swap_gap=None,
    )


def record_to_dict(record: OutcomeEvaluationRecord) -> dict[str, object]:
    """Convert one validated raw record to JSON-compatible values."""
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "sequence": record.sequence,
        "model_role": record.model_role,
        "question_id": record.question_id,
        "swapped_question_id": record.swapped_question_id,
        "status": record.status,
        "error": record.error,
        "original_prompt_tokens": list(record.original_prompt_tokens),
        "swapped_prompt_tokens": list(record.swapped_prompt_tokens),
        "original": None if record.original is None else _orientation_to_dict(record.original),
        "swapped": None if record.swapped is None else _orientation_to_dict(record.swapped),
        "symmetric_team_probability": record.symmetric_team_probability,
        "pre_average_side_swap_gap": record.pre_average_side_swap_gap,
    }


def record_from_dict(value: object) -> OutcomeEvaluationRecord:
    """Strictly parse one raw outcome evaluation record."""
    record = require_object(value, "outcome evaluation record")
    require_exact_keys(record, _RECORD_KEYS, "outcome evaluation record")
    if required_field(record, "schema_version") != EVALUATION_SCHEMA_VERSION:
        raise OutcomeEvaluationError("unsupported outcome record schema")
    model_role = _model_role(required_field(record, "model_role"))
    status = _record_status(required_field(record, "status"))
    error_value = required_field(record, "error")
    error = None if error_value is None else require_string(error_value, "error")
    original = _optional_orientation(required_field(record, "original"), "original")
    swapped = _optional_orientation(required_field(record, "swapped"), "swapped")
    probability = _optional_float(
        required_field(record, "symmetric_team_probability"),
        "symmetric_team_probability",
    )
    gap = _optional_float(
        required_field(record, "pre_average_side_swap_gap"),
        "pre_average_side_swap_gap",
    )
    return OutcomeEvaluationRecord(
        sequence=_integer_field(record, "sequence"),
        model_role=model_role,
        question_id=_string_field(record, "question_id"),
        swapped_question_id=_string_field(record, "swapped_question_id"),
        status=status,
        error=error,
        original_prompt_tokens=_integer_tuple(
            required_field(record, "original_prompt_tokens"),
            "original_prompt_tokens",
        ),
        swapped_prompt_tokens=_integer_tuple(
            required_field(record, "swapped_prompt_tokens"),
            "swapped_prompt_tokens",
        ),
        original=original,
        swapped=swapped,
        symmetric_team_probability=probability,
        pre_average_side_swap_gap=gap,
    )


def write_records(
    path: Path,
    records: tuple[OutcomeEvaluationRecord, ...],
    manifest: OutcomeEvaluationManifest,
    model_role: ModelRole,
) -> str:
    """Exclusively write exact ordered coverage for one model arm."""
    _validate_record_coverage(records, manifest, model_role)
    text = "".join(
        f"{json.dumps(record_to_dict(record), sort_keys=True, allow_nan=False)}\n"
        for record in records
    )
    write_text_exclusively(path, text)
    return file_sha256(path)


def read_records(
    path: Path,
    manifest: OutcomeEvaluationManifest,
    model_role: ModelRole,
) -> tuple[OutcomeEvaluationRecord, ...]:
    """Read and validate one complete ordered model arm."""
    records: list[OutcomeEvaluationRecord] = []
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise OutcomeEvaluationError(f"evaluation output is missing: {path}") from error
    if not text.endswith("\n"):
        raise OutcomeEvaluationError("evaluation output must end with a newline")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise OutcomeEvaluationError(f"blank evaluation output line: {line_number}")
        try:
            records.append(record_from_dict(parse_json_object(line)))
        except (JsonFormatError, OutcomeEvaluationError) as error:
            raise OutcomeEvaluationError(
                f"invalid evaluation output on line {line_number}"
            ) from error
    result = tuple(records)
    _validate_record_coverage(result, manifest, model_role)
    return result


def write_attempt_marker(
    path: Path,
    manifest_path: Path,
    prompts_path: Path,
    created_at: datetime,
) -> str:
    """Exclusively commit one attempt before any remote client exists."""
    manifest = read_manifest(manifest_path)
    load_prompt_pairs(manifest, prompts_path)
    record = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": "forecastfm_outcome_evaluation_attempt",
        "status": "committed_before_remote_clients",
        "created_at": _utc_text(created_at),
        "manifest_sha256": file_sha256(manifest_path),
        "prompts_sha256": file_sha256(prompts_path),
    }
    write_json_exclusively(path, record)
    return file_sha256(path)


def verify_attempt_marker(path: Path, manifest_path: Path, prompts_path: Path) -> None:
    """Verify that an existing attempt binds the current frozen inputs."""
    record = parse_json_object(path.read_text(encoding="utf-8"))
    require_exact_keys(record, _ATTEMPT_KEYS, "outcome evaluation attempt")
    expected = (
        EVALUATION_SCHEMA_VERSION,
        "forecastfm_outcome_evaluation_attempt",
        "committed_before_remote_clients",
        file_sha256(manifest_path),
        file_sha256(prompts_path),
    )
    actual = (
        required_field(record, "schema_version"),
        required_field(record, "kind"),
        required_field(record, "status"),
        _string_field(record, "manifest_sha256"),
        _string_field(record, "prompts_sha256"),
    )
    if actual != expected:
        raise OutcomeEvaluationError("attempt marker differs from frozen inputs")
    _require_utc(_string_field(record, "created_at"))


def seal_outputs(
    paths: EvaluationPaths,
    created_at: datetime,
) -> dict[str, object]:
    """Validate complete paired outputs and exclusively create their seal."""
    manifest = read_manifest(paths.manifest)
    load_prompt_pairs(manifest, paths.prompts)
    verify_attempt_marker(paths.attempt, paths.manifest, paths.prompts)
    base = read_records(paths.base, manifest, "base")
    adapter = read_records(paths.adapter, manifest, "adapter")
    _validate_cross_arm_tokens(base, adapter)
    seal: dict[str, object] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": "forecastfm_outcome_evaluation_seal",
        "status": "sealed_before_answer_access",
        "created_at": _utc_text(created_at),
        "manifest_sha256": file_sha256(paths.manifest),
        "prompts_sha256": file_sha256(paths.prompts),
        "attempt_sha256": file_sha256(paths.attempt),
        "journal_sha256": file_sha256(paths.journal),
        "base_sha256": file_sha256(paths.base),
        "adapter_sha256": file_sha256(paths.adapter),
        "game_count": manifest.game_count,
        "base_completed": sum(record.status == "completed" for record in base),
        "adapter_completed": sum(record.status == "completed" for record in adapter),
        "question_ids_sha256": manifest.question_ids_sha256,
    }
    write_json_exclusively(paths.seal, seal)
    return seal


def load_sealed_evaluation(
    paths: EvaluationPaths,
) -> SealedOutcomeEvaluation:
    """Load raw outputs only when every frozen hash and row still matches."""
    manifest = read_manifest(paths.manifest)
    prompt_pairs = load_prompt_pairs(manifest, paths.prompts)
    verify_attempt_marker(paths.attempt, paths.manifest, paths.prompts)
    seal = parse_json_object(paths.seal.read_text(encoding="utf-8"))
    require_exact_keys(seal, _SEAL_KEYS, "outcome evaluation seal")
    expected = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": "forecastfm_outcome_evaluation_seal",
        "status": "sealed_before_answer_access",
        "manifest_sha256": file_sha256(paths.manifest),
        "prompts_sha256": file_sha256(paths.prompts),
        "attempt_sha256": file_sha256(paths.attempt),
        "journal_sha256": file_sha256(paths.journal),
        "base_sha256": file_sha256(paths.base),
        "adapter_sha256": file_sha256(paths.adapter),
        "game_count": manifest.game_count,
        "question_ids_sha256": manifest.question_ids_sha256,
    }
    for key, value in expected.items():
        if required_field(seal, key) != value:
            raise OutcomeEvaluationError(f"evaluation seal field differs: {key}")
    _require_utc(_string_field(seal, "created_at"))
    base = read_records(paths.base, manifest, "base")
    adapter = read_records(paths.adapter, manifest, "adapter")
    completed = (
        sum(record.status == "completed" for record in base),
        sum(record.status == "completed" for record in adapter),
    )
    if completed != (
        _integer_field(seal, "base_completed"),
        _integer_field(seal, "adapter_completed"),
    ):
        raise OutcomeEvaluationError("evaluation completion counts differ from the seal")
    _validate_cross_arm_tokens(base, adapter)
    return SealedOutcomeEvaluation(manifest, prompt_pairs, base, adapter, seal)


def _manifest_to_dict(manifest: OutcomeEvaluationManifest) -> dict[str, object]:
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": "forecastfm_outcome_evaluation",
        "status": "frozen_before_answer_access",
        "created_at": manifest.created_at,
        "protocol_revision": manifest.protocol_revision,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "source_prompts_sha256": manifest.source_prompts_sha256,
        "source_answers_sha256": manifest.source_answers_sha256,
        "frozen_prompts_sha256": manifest.frozen_prompts_sha256,
        "training_lock_sha256": manifest.training_lock_sha256,
        "experiment_sha256": manifest.experiment_sha256,
        "base_model": manifest.base_model,
        "adapter_sampler_path": manifest.adapter_sampler_path,
        "renderer_name": manifest.renderer_name,
        "team_token_id": manifest.team_token_id,
        "opponent_token_id": manifest.opponent_token_id,
        "game_count": manifest.game_count,
        "orientation_count": manifest.orientation_count,
        "logical_calls_per_game_per_arm": manifest.logical_calls_per_game_per_arm,
        "expected_total_logical_calls": manifest.expected_total_logical_calls,
        "max_active_arms": manifest.max_active_arms,
        "application_retries": manifest.application_retries,
        "transport_retry_note": manifest.transport_retry_note,
        "question_ids": list(manifest.question_ids),
        "question_ids_sha256": manifest.question_ids_sha256,
        "scoring_policy": manifest.scoring_policy,
    }


def _orientation_to_dict(value: OrientationResult) -> dict[str, object]:
    return {
        "team_logprob": value.team_logprob,
        "opponent_logprob": value.opponent_logprob,
        "team_probability": value.team_probability,
        "valid_label_mass": value.valid_label_mass,
    }


def _optional_orientation(value: object, field_name: str) -> OrientationResult | None:
    if value is None:
        return None
    record = require_object(value, field_name)
    require_exact_keys(record, _ORIENTATION_KEYS, field_name)
    return OrientationResult(
        team_logprob=require_float(required_field(record, "team_logprob"), "team_logprob"),
        opponent_logprob=require_float(
            required_field(record, "opponent_logprob"), "opponent_logprob"
        ),
        team_probability=require_float(
            required_field(record, "team_probability"), "team_probability"
        ),
        valid_label_mass=require_float(
            required_field(record, "valid_label_mass"), "valid_label_mass"
        ),
    )


def _validate_record_coverage(
    records: tuple[OutcomeEvaluationRecord, ...],
    manifest: OutcomeEvaluationManifest,
    model_role: ModelRole,
) -> None:
    if len(records) != manifest.game_count:
        raise OutcomeEvaluationError(f"{model_role} output has incomplete game coverage")
    for sequence, (record, question_id) in enumerate(
        zip(records, manifest.question_ids, strict=True)
    ):
        if (
            record.sequence != sequence
            or record.model_role != model_role
            or record.question_id != question_id
        ):
            raise OutcomeEvaluationError(f"{model_role} output order differs from the manifest")


def _validate_cross_arm_tokens(
    base: tuple[OutcomeEvaluationRecord, ...],
    adapter: tuple[OutcomeEvaluationRecord, ...],
) -> None:
    for base_record, adapter_record in zip(base, adapter, strict=True):
        if base_record.question_id != adapter_record.question_id:
            raise OutcomeEvaluationError("base and adapter game coverage differs")
        if (
            base_record.original_prompt_tokens != adapter_record.original_prompt_tokens
            or base_record.swapped_prompt_tokens != adapter_record.swapped_prompt_tokens
        ):
            raise OutcomeEvaluationError("base and adapter rendered different prompt tokens")


def write_json_exclusively(path: Path, value: object) -> None:
    """Atomically create one canonical indented JSON artifact."""
    text = f"{json.dumps(value, indent=2, sort_keys=True, allow_nan=False)}\n"
    write_text_exclusively(path, text)


def write_text_exclusively(path: Path, text: str) -> None:
    """Atomically create one UTF-8 text artifact without replacement."""
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
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.link(partial_path, path)
    except FileExistsError as error:
        raise OutcomeEvaluationError(f"refusing to replace evaluation artifact: {path}") from error
    finally:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    values = require_list(value, field_name)
    return tuple(
        require_string(item, f"{field_name}[{index}]") for index, item in enumerate(values)
    )


def _integer_tuple(value: object, field_name: str) -> tuple[int, ...]:
    values = require_list(value, field_name)
    return tuple(
        _require_integer(item, f"{field_name}[{index}]") for index, item in enumerate(values)
    )


def _string_field(record: dict[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _integer_field(record: dict[str, object], field_name: str) -> int:
    return _require_integer(required_field(record, field_name), field_name)


def _require_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise JsonFormatError(f"{field_name} must be an integer")
    return value


def _validate_prompt_tokens(tokens: tuple[object, ...]) -> None:
    if not tokens or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens
    ):
        raise OutcomeEvaluationError("prompt tokens must be non-empty and non-negative")


def _optional_float(value: object, field_name: str) -> float | None:
    return None if value is None else require_float(value, field_name)


def _model_role(value: object) -> ModelRole:
    role = require_string(value, "model_role")
    if role not in {"base", "adapter"}:
        raise OutcomeEvaluationError(f"unsupported model role: {role}")
    return cast(ModelRole, role)


def _record_status(value: object) -> RecordStatus:
    status = require_string(value, "status")
    if status not in {"completed", "failed"}:
        raise OutcomeEvaluationError(f"unsupported outcome record status: {status}")
    return cast(RecordStatus, status)


def _require_hash(value: str) -> None:
    if _HASH_PATTERN.fullmatch(value) is None:
        raise OutcomeEvaluationError("evaluation digest must be a lowercase SHA-256")


def _require_revision(value: str) -> None:
    if _REVISION_PATTERN.fullmatch(value) is None:
        raise OutcomeEvaluationError("evaluation revision must be a Git object ID")


def _require_utc(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise OutcomeEvaluationError("evaluation datetime must be ISO 8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise OutcomeEvaluationError("evaluation datetime must use UTC")


def _utc_text(value: datetime) -> str:
    text = value.isoformat()
    _require_utc(text)
    return text
