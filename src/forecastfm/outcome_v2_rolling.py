"""Externally timed rolling batches for genuinely prospective outcome-v2 forecasts."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from forecastfm.github_actions_receipt import (
    GitHubActionsReceipt,
    GitHubActionsReceiptPolicy,
    GitHubActionsReceiptRequest,
    read_github_actions_receipt,
    verify_github_actions_receipt,
)
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
from forecastfm.nba_feature_rows import (
    NBA_PRIMARY_STATE_ID,
    NbaFeatureRowError,
    NbaRichFeatureRow,
    read_nba_feature_rows_jsonl_bytes,
)
from forecastfm.outcome_v2_config import outcome_v2_coverage_policy
from forecastfm.outcome_v2_experiment import OutcomeV2ExperimentError, OutcomeV2ExperimentLock
from forecastfm.outcome_v2_inference import OutcomeV2GenerationLock
from forecastfm.outcome_v2_run import (
    OutcomeV2RunError,
    OutcomeV2RunLock,
    require_outcome_v2_run_static_contract,
)
from forecastfm.outcome_v2_sft_gate import (
    OutcomeV2SftForecastArtifacts,
    OutcomeV2SftForecastSeal,
    OutcomeV2SftGateError,
    verify_outcome_v2_sft_forecast_seal,
)

OUTCOME_V2_PROSPECTIVE_PLAN_SCHEMA_VERSION = 2
OUTCOME_V2_PROSPECTIVE_BATCH_SCHEMA_VERSION = 1
OUTCOME_V2_GITHUB_REPOSITORY = "dylantirandaz/forecastfm"
OUTCOME_V2_GITHUB_BRANCH = "main"
OUTCOME_V2_GITHUB_WORKFLOW_PATH = ".github/workflows/outcome-v2-publication-timestamp.yml"

_PLAN_KIND = "forecastfm_outcome_v2_prospective_plan"
_PLAN_STATUS = "committed_before_first_prospective_batch"
_BATCH_KIND = "forecastfm_outcome_v2_prospective_batch_seal"
_BATCH_STATUS = "terminal_records_sealed_before_external_receipt"
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40,64}")
_PLAN_KEYS = {
    "calibration_sha256",
    "coverage_policy",
    "coverage_policy_sha256",
    "created_at",
    "evaluation_policy_sha256",
    "experiment_lock_sha256",
    "inclusion_rule",
    "kind",
    "outcome_v2_run_lock_sha256",
    "protocol_revision",
    "receipt_policy",
    "receipt_policy_sha256",
    "sampler_path",
    "schema_version",
    "seasons",
    "state_id",
    "status",
}
_RECEIPT_POLICY_KEYS = {
    "branch",
    "event",
    "repository",
    "workflow_path",
    "workflow_sha256",
    "workflow_id",
}
_BATCH_KEYS = {
    "batch_id",
    "coverage_policy_sha256",
    "created_at",
    "earliest_input_available_at",
    "earliest_forecast_cutoff",
    "evaluation_feature_rows_sha256",
    "evaluation_generation_lock_sha256",
    "evaluation_inference_journal_sha256",
    "evaluation_inference_records_sha256",
    "evaluation_prompts_sha256",
    "experiment_lock_sha256",
    "failed_record_count",
    "game_count",
    "kind",
    "latest_input_available_at",
    "outcome_v2_run_lock_sha256",
    "plan_sha256",
    "question_ids_sha256",
    "sampler_path",
    "schema_version",
    "seasons",
    "sft_forecast_seal_sha256",
    "state_id",
    "status",
}

type JsonObject = dict[str, object]


class OutcomeV2RollingError(ValueError):
    """Raised when a rolling plan, batch, or external receipt is invalid."""


@dataclass(frozen=True, slots=True)
class OutcomeV2ProspectivePlanArtifacts:
    """Frozen model files used to construct one multi-season plan."""

    project_root: Path
    run_lock_path: Path
    experiment_lock_path: Path


@dataclass(frozen=True, slots=True)
class OutcomeV2ProspectivePlanConfig:
    """Human-chosen scope frozen before the first prospective batch."""

    seasons: tuple[int, ...]
    inclusion_rule: str
    created_at: datetime
    receipt_workflow_id: int

    def __post_init__(self) -> None:
        _require_multi_seasons(self.seasons)
        _require_text(self.inclusion_rule, "inclusion_rule")
        _require_utc(self.created_at, "created_at")
        _require_workflow_id(self.receipt_workflow_id)


@dataclass(frozen=True, slots=True)
class OutcomeV2ProspectivePlan:
    """Canonical multi-season model, evaluation, and receipt commitment."""

    canonical_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_immutable_bytes(self.canonical_bytes, "prospective plan")
        _plan_record(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        """Return the digest of the exact canonical plan bytes."""
        return bytes_sha256(self.canonical_bytes)

    def to_record(self) -> JsonObject:
        """Return a newly decoded strict plan record."""
        return _plan_record(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class OutcomeV2ProspectiveBatchArtifacts:
    """One plan, answer-free forecast chain, and terminal SFT seal."""

    plan_path: Path
    forecast: OutcomeV2SftForecastArtifacts
    forecast_seal_path: Path


@dataclass(frozen=True, slots=True)
class OutcomeV2ProspectiveBatchSeal:
    """Canonical terminal-record commitment for one causal forecast window."""

    canonical_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_immutable_bytes(self.canonical_bytes, "prospective batch seal")
        _batch_record(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        """Return the digest of the exact canonical batch-seal bytes."""
        return bytes_sha256(self.canonical_bytes)

    def to_record(self) -> JsonObject:
        """Return a newly decoded strict batch-seal record."""
        return _batch_record(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class OutcomeV2ProspectiveReceiptArtifacts:
    """One externally frozen plan and one externally timed terminal batch seal."""

    batch: OutcomeV2ProspectiveBatchArtifacts
    batch_seal_path: Path
    batch_seal_repository_path: str
    plan_receipt_path: Path
    plan_repository_path: str
    terminal_receipt_path: Path

    def __post_init__(self) -> None:
        _require_rolling_repository_path(self.batch_seal_repository_path)
        _require_rolling_repository_path(self.plan_repository_path)


@dataclass(frozen=True, slots=True)
class VerifiedOutcomeV2ProspectiveBatch:
    """One locally complete batch with a live externally verified deadline proof."""

    seal: OutcomeV2ProspectiveBatchSeal
    plan_receipt: GitHubActionsReceipt
    receipt: GitHubActionsReceipt
    externally_committed_at: datetime


def build_outcome_v2_prospective_plan(
    artifacts: OutcomeV2ProspectivePlanArtifacts,
    config: OutcomeV2ProspectivePlanConfig,
) -> OutcomeV2ProspectivePlan:
    """Freeze model identity, seasons, inclusion, evaluation, and receipt policy."""
    run_bytes = _read_bytes(artifacts.run_lock_path, "outcome-v2 run lock")
    experiment_bytes = _read_bytes(
        artifacts.experiment_lock_path,
        "outcome-v2 experiment lock",
    )
    try:
        run_lock = OutcomeV2RunLock(run_bytes)
        experiment_lock = OutcomeV2ExperimentLock(experiment_bytes)
        require_outcome_v2_run_static_contract(artifacts.project_root, run_lock)
    except (OutcomeV2ExperimentError, OutcomeV2RunError) as error:
        raise OutcomeV2RollingError("cannot verify prospective-plan model locks") from error
    run = run_lock.to_record()
    experiment = experiment_lock.to_record()
    if _string(experiment, "outcome_v2_run_lock_sha256") != run_lock.sha256:
        raise OutcomeV2RollingError("experiment does not bind the prospective-plan run lock")
    experiment_at = _parse_utc(_string(experiment, "created_at"), "experiment.created_at")
    if config.created_at < experiment_at:
        raise OutcomeV2RollingError("prospective plan cannot predate the trained experiment")

    tabular_seasons = _integer_tuple(_object(run, "preflight"), "untouched_evaluation_seasons")
    _require_multi_seasons(tabular_seasons)
    if min(config.seasons) <= max(tabular_seasons):
        raise OutcomeV2RollingError("prospective seasons must be later than tabular seasons")
    receipt_policy = _receipt_policy_from_run(run, config.receipt_workflow_id)
    evaluation_policy = _object(run, "evaluation_policy")
    coverage_policy = outcome_v2_coverage_policy()
    record: JsonObject = {
        "schema_version": OUTCOME_V2_PROSPECTIVE_PLAN_SCHEMA_VERSION,
        "kind": _PLAN_KIND,
        "status": _PLAN_STATUS,
        "created_at": _utc_text(config.created_at, "created_at"),
        "outcome_v2_run_lock_sha256": run_lock.sha256,
        "experiment_lock_sha256": experiment_lock.sha256,
        "sampler_path": _string(experiment, "sampler_path"),
        "protocol_revision": _string(run, "code_revision"),
        "state_id": NBA_PRIMARY_STATE_ID,
        "seasons": list(config.seasons),
        "inclusion_rule": config.inclusion_rule,
        "evaluation_policy_sha256": _string(evaluation_policy, "sha256"),
        "calibration_sha256": _string(_object(run, "preflight"), "calibration_sha256"),
        "coverage_policy": coverage_policy,
        "coverage_policy_sha256": canonical_sha256(coverage_policy),
        "receipt_policy": receipt_policy.canonical_payload(),
        "receipt_policy_sha256": receipt_policy.policy_sha256,
    }
    plan = OutcomeV2ProspectivePlan(canonical_json(record).encode("utf-8"))
    _require_unchanged_model_files(artifacts, run_bytes, experiment_bytes)
    return plan


def write_outcome_v2_prospective_plan(
    path: Path,
    plan: OutcomeV2ProspectivePlan,
) -> str:
    """Create and durably flush one prospective plan without replacement."""
    _write_once(path, plan.canonical_bytes, "prospective plan")
    return plan.sha256


def read_outcome_v2_prospective_plan(path: Path) -> OutcomeV2ProspectivePlan:
    """Read one strict canonical prospective plan."""
    return OutcomeV2ProspectivePlan(_read_bytes(path, "prospective plan"))


def verify_outcome_v2_prospective_plan(
    artifacts: OutcomeV2ProspectivePlanArtifacts,
    path: Path,
) -> OutcomeV2ProspectivePlan:
    """Rebuild a prospective plan from current locks and compare exact bytes."""
    actual = read_outcome_v2_prospective_plan(path)
    record = actual.to_record()
    expected = build_outcome_v2_prospective_plan(
        artifacts,
        OutcomeV2ProspectivePlanConfig(
            seasons=_integer_tuple(record, "seasons"),
            inclusion_rule=_string(record, "inclusion_rule"),
            created_at=_parse_utc(_string(record, "created_at"), "created_at"),
            receipt_workflow_id=_positive_integer(
                _object(record, "receipt_policy"),
                "workflow_id",
            ),
        ),
    )
    if actual.canonical_bytes != expected.canonical_bytes:
        raise OutcomeV2RollingError("prospective plan differs from current model locks")
    return actual


def build_outcome_v2_prospective_batch_seal(
    artifacts: OutcomeV2ProspectiveBatchArtifacts,
    batch_id: str,
    created_at: datetime,
) -> OutcomeV2ProspectiveBatchSeal:
    """Bind terminal records for one batch whose causal windows overlap."""
    _require_text(batch_id, "batch_id")
    _require_utc(created_at, "created_at")
    plan = verify_outcome_v2_prospective_plan(
        OutcomeV2ProspectivePlanArtifacts(
            project_root=artifacts.forecast.project_root,
            run_lock_path=artifacts.forecast.run_lock_path,
            experiment_lock_path=artifacts.forecast.experiment_lock_path,
        ),
        artifacts.plan_path,
    )
    try:
        forecast_seal = verify_outcome_v2_sft_forecast_seal(
            artifacts.forecast,
            artifacts.forecast_seal_path,
        )
    except OutcomeV2SftGateError as error:
        raise OutcomeV2RollingError("cannot verify terminal answer-free forecast files") from error
    rows = _load_batch_rows(artifacts.forecast.feature_rows_path)
    generation_lock = OutcomeV2GenerationLock(
        _read_bytes(artifacts.forecast.generation_lock_path, "generation lock")
    )
    _require_batch_bindings(plan, forecast_seal, generation_lock, rows, artifacts.forecast)
    latest_input = max(row.input_available_at for row in rows)
    earliest_input = min(row.input_available_at for row in rows)
    earliest_cutoff = min(row.forecast_cutoff for row in rows)
    generation_at = _parse_utc(
        _string(generation_lock.to_record(), "created_at"),
        "generation.created_at",
    )
    forecast_at = _parse_utc(
        _string(forecast_seal.to_record(), "created_at"),
        "forecast_seal.created_at",
    )
    if not latest_input <= generation_at <= forecast_at <= created_at < earliest_cutoff:
        raise OutcomeV2RollingError("batch timestamps do not fit one prospective causal window")

    forecast = forecast_seal.to_record()
    plan_record = plan.to_record()
    question_ids = tuple(row.question_id for row in rows)
    record: JsonObject = {
        "schema_version": OUTCOME_V2_PROSPECTIVE_BATCH_SCHEMA_VERSION,
        "kind": _BATCH_KIND,
        "status": _BATCH_STATUS,
        "created_at": _utc_text(created_at, "created_at"),
        "batch_id": batch_id,
        "plan_sha256": plan.sha256,
        "outcome_v2_run_lock_sha256": _string(
            forecast,
            "outcome_v2_run_lock_sha256",
        ),
        "experiment_lock_sha256": _string(
            forecast,
            "outcome_v2_experiment_lock_sha256",
        ),
        "sampler_path": _string(forecast, "sampler_path"),
        "state_id": _string(plan_record, "state_id"),
        "coverage_policy_sha256": _string(plan_record, "coverage_policy_sha256"),
        "seasons": sorted({row.season for row in rows}),
        "question_ids_sha256": canonical_sha256(list(question_ids)),
        "game_count": len(rows),
        "failed_record_count": _nonnegative_integer(forecast, "failed_record_count"),
        "earliest_input_available_at": _utc_text(
            earliest_input,
            "earliest_input_available_at",
        ),
        "latest_input_available_at": _utc_text(latest_input, "latest_input_available_at"),
        "earliest_forecast_cutoff": _utc_text(earliest_cutoff, "earliest_forecast_cutoff"),
        "sft_forecast_seal_sha256": forecast_seal.sha256,
        "evaluation_feature_rows_sha256": _string(
            forecast,
            "evaluation_feature_rows_sha256",
        ),
        "evaluation_prompts_sha256": _string(forecast, "evaluation_prompts_sha256"),
        "evaluation_generation_lock_sha256": _string(
            forecast,
            "evaluation_generation_lock_sha256",
        ),
        "evaluation_inference_journal_sha256": _string(
            forecast,
            "evaluation_inference_journal_sha256",
        ),
        "evaluation_inference_records_sha256": _string(
            forecast,
            "evaluation_inference_records_sha256",
        ),
    }
    return OutcomeV2ProspectiveBatchSeal(canonical_json(record).encode("utf-8"))


def write_outcome_v2_prospective_batch_seal(
    path: Path,
    seal: OutcomeV2ProspectiveBatchSeal,
) -> str:
    """Create and durably flush one terminal batch seal without replacement."""
    _write_once(path, seal.canonical_bytes, "prospective batch seal")
    return seal.sha256


def read_outcome_v2_prospective_batch_seal(path: Path) -> OutcomeV2ProspectiveBatchSeal:
    """Read one strict canonical terminal batch seal."""
    return OutcomeV2ProspectiveBatchSeal(_read_bytes(path, "prospective batch seal"))


def verify_outcome_v2_prospective_batch_seal(
    artifacts: OutcomeV2ProspectiveBatchArtifacts,
    path: Path,
) -> OutcomeV2ProspectiveBatchSeal:
    """Rebuild one terminal batch seal and compare exact bytes."""
    actual = read_outcome_v2_prospective_batch_seal(path)
    record = actual.to_record()
    expected = build_outcome_v2_prospective_batch_seal(
        artifacts,
        _string(record, "batch_id"),
        _parse_utc(_string(record, "created_at"), "created_at"),
    )
    if actual.canonical_bytes != expected.canonical_bytes:
        raise OutcomeV2RollingError("prospective batch seal differs from terminal forecast files")
    return actual


def verify_outcome_v2_prospective_batch_receipt(
    artifacts: OutcomeV2ProspectiveReceiptArtifacts,
    token: str | None = None,
) -> VerifiedOutcomeV2ProspectiveBatch:
    """Re-fetch GitHub and require a terminal batch seal before its T-60 cutoff."""
    seal = verify_outcome_v2_prospective_batch_seal(
        artifacts.batch,
        artifacts.batch_seal_path,
    )
    plan = verify_outcome_v2_prospective_plan(
        OutcomeV2ProspectivePlanArtifacts(
            project_root=artifacts.batch.forecast.project_root,
            run_lock_path=artifacts.batch.forecast.run_lock_path,
            experiment_lock_path=artifacts.batch.forecast.experiment_lock_path,
        ),
        artifacts.batch.plan_path,
    )
    if _string(seal.to_record(), "plan_sha256") != plan.sha256:
        raise OutcomeV2RollingError("terminal batch seal differs from the verified plan")
    policy = _receipt_policy_from_plan(plan.to_record())
    seal_record = seal.to_record()
    earliest_input = _parse_utc(
        _string(seal_record, "earliest_input_available_at"),
        "batch.earliest_input_available_at",
    )
    plan_receipt = read_github_actions_receipt(artifacts.plan_receipt_path)
    plan_receipt_record = plan_receipt.to_record()
    plan_deadline = _parse_utc(
        _string(plan_receipt_record, "deadline"),
        "plan_receipt.deadline",
    )
    if plan_deadline > earliest_input:
        raise OutcomeV2RollingError("prospective plan receipt deadline is after batch inputs")
    plan_request = GitHubActionsReceiptRequest(
        run_id=_receipt_run_id(plan_receipt),
        artifact_path=artifacts.plan_repository_path,
        artifact_bytes=plan.canonical_bytes,
        not_before=_parse_utc(_string(plan.to_record(), "created_at"), "plan.created_at"),
        deadline=plan_deadline,
    )
    verify_github_actions_receipt(policy, plan_request, plan_receipt, token)

    receipt = read_github_actions_receipt(artifacts.terminal_receipt_path)
    request = GitHubActionsReceiptRequest(
        run_id=_receipt_run_id(receipt),
        artifact_path=artifacts.batch_seal_repository_path,
        artifact_bytes=seal.canonical_bytes,
        not_before=_parse_utc(_string(seal_record, "created_at"), "batch.created_at"),
        deadline=_parse_utc(
            _string(seal_record, "earliest_forecast_cutoff"),
            "batch.earliest_forecast_cutoff",
        ),
    )
    verify_github_actions_receipt(policy, request, receipt, token)
    committed_at = _parse_utc(
        _string(_object(receipt.to_record(), "run"), "created_at"),
        "receipt.run.created_at",
    )
    return VerifiedOutcomeV2ProspectiveBatch(seal, plan_receipt, receipt, committed_at)


def _receipt_run_id(receipt: GitHubActionsReceipt) -> int:
    return _positive_integer(_object(receipt.to_record(), "run"), "id")


def _require_batch_bindings(
    plan: OutcomeV2ProspectivePlan,
    forecast_seal: OutcomeV2SftForecastSeal,
    generation_lock: OutcomeV2GenerationLock,
    rows: tuple[NbaRichFeatureRow, ...],
    artifacts: OutcomeV2SftForecastArtifacts,
) -> None:
    plan_record = plan.to_record()
    forecast = forecast_seal.to_record()
    generation = generation_lock.to_record()
    expected_model = (
        _string(plan_record, "outcome_v2_run_lock_sha256"),
        _string(plan_record, "experiment_lock_sha256"),
        _string(plan_record, "sampler_path"),
    )
    actual_model = (
        _string(forecast, "outcome_v2_run_lock_sha256"),
        _string(forecast, "outcome_v2_experiment_lock_sha256"),
        _string(forecast, "sampler_path"),
    )
    if actual_model != expected_model:
        raise OutcomeV2RollingError("batch model identity differs from the prospective plan")
    plan_at = _parse_utc(_string(plan_record, "created_at"), "plan.created_at")
    generation_at = _parse_utc(_string(generation, "created_at"), "generation.created_at")
    if plan_at > generation_at:
        raise OutcomeV2RollingError("prospective plan was created after batch generation")
    question_ids = tuple(row.question_id for row in rows)
    if tuple(_string_tuple(generation, "question_ids")) != question_ids:
        raise OutcomeV2RollingError("batch feature rows differ from the generation lock")
    allowed_seasons = set(_integer_tuple(plan_record, "seasons"))
    if any(row.season not in allowed_seasons for row in rows):
        raise OutcomeV2RollingError("batch contains a season outside the prospective plan")
    if any(row.state_id != _string(plan_record, "state_id") for row in rows):
        raise OutcomeV2RollingError("batch state differs from the prospective plan")
    _require_file_hash(
        artifacts.feature_rows_path,
        _string(forecast, "evaluation_feature_rows_sha256"),
        "feature rows",
    )
    _require_file_hash(
        artifacts.prompts_path,
        _string(forecast, "evaluation_prompts_sha256"),
        "prompts",
    )
    _require_file_hash(
        artifacts.generation_lock_path,
        _string(forecast, "evaluation_generation_lock_sha256"),
        "generation lock",
    )
    _require_file_hash(
        artifacts.inference_journal_path,
        _string(forecast, "evaluation_inference_journal_sha256"),
        "inference journal",
    )
    _require_file_hash(
        artifacts.inference_records_path,
        _string(forecast, "evaluation_inference_records_sha256"),
        "inference records",
    )


def _receipt_policy_from_run(
    run: Mapping[str, object],
    workflow_id: int,
) -> GitHubActionsReceiptPolicy:
    code_hashes = _object(run, "code_sha256")
    workflow_sha256 = _string(code_hashes, "publication_workflow")
    return GitHubActionsReceiptPolicy(
        repository=OUTCOME_V2_GITHUB_REPOSITORY,
        branch=OUTCOME_V2_GITHUB_BRANCH,
        workflow_path=OUTCOME_V2_GITHUB_WORKFLOW_PATH,
        workflow_sha256=workflow_sha256,
        workflow_id=workflow_id,
    )


def outcome_v2_receipt_policy(
    plan: OutcomeV2ProspectivePlan,
) -> GitHubActionsReceiptPolicy:
    """Return the exact trusted GitHub policy frozen in a verified plan."""
    return _receipt_policy_from_plan(plan.to_record())


def _receipt_policy_from_plan(plan: Mapping[str, object]) -> GitHubActionsReceiptPolicy:
    payload = _object(plan, "receipt_policy")
    policy = GitHubActionsReceiptPolicy(
        repository=_string(payload, "repository"),
        branch=_string(payload, "branch"),
        workflow_path=_string(payload, "workflow_path"),
        workflow_sha256=_string(payload, "workflow_sha256"),
        workflow_id=_positive_integer(payload, "workflow_id"),
        event=_string(payload, "event"),
    )
    if policy.policy_sha256 != _string(plan, "receipt_policy_sha256"):
        raise OutcomeV2RollingError("prospective-plan receipt policy hash is invalid")
    return policy


def _plan_record(value: bytes) -> JsonObject:
    record = _canonical_record(value, "prospective plan")
    try:
        require_exact_keys(record, _PLAN_KEYS, "prospective plan")
        if _integer(record, "schema_version") != OUTCOME_V2_PROSPECTIVE_PLAN_SCHEMA_VERSION:
            raise OutcomeV2RollingError("unsupported prospective-plan schema")
        _require_value(record, "kind", _PLAN_KIND)
        _require_value(record, "status", _PLAN_STATUS)
        _parse_utc(_string(record, "created_at"), "created_at")
        for field_name in (
            "outcome_v2_run_lock_sha256",
            "experiment_lock_sha256",
            "evaluation_policy_sha256",
            "calibration_sha256",
            "receipt_policy_sha256",
            "coverage_policy_sha256",
        ):
            _require_hash(_string(record, field_name), field_name)
        _require_tinker_path(_string(record, "sampler_path"))
        if _REVISION_PATTERN.fullmatch(_string(record, "protocol_revision")) is None:
            raise OutcomeV2RollingError("protocol_revision is invalid")
        _require_value(record, "state_id", NBA_PRIMARY_STATE_ID)
        _require_multi_seasons(_integer_tuple(record, "seasons"))
        _require_text(_string(record, "inclusion_rule"), "inclusion_rule")
        coverage_policy = _object(record, "coverage_policy")
        if coverage_policy != outcome_v2_coverage_policy():
            raise OutcomeV2RollingError("prospective-plan coverage policy is stale")
        if canonical_sha256(coverage_policy) != _string(
            record,
            "coverage_policy_sha256",
        ):
            raise OutcomeV2RollingError("prospective-plan coverage policy hash is invalid")
        payload = _object(record, "receipt_policy")
        require_exact_keys(payload, _RECEIPT_POLICY_KEYS, "receipt_policy")
        _receipt_policy_from_plan(record)
    except JsonFormatError as error:
        raise OutcomeV2RollingError("invalid prospective-plan structure") from error
    return record


def _batch_record(value: bytes) -> JsonObject:
    record = _canonical_record(value, "prospective batch seal")
    try:
        require_exact_keys(record, _BATCH_KEYS, "prospective batch seal")
        if _integer(record, "schema_version") != OUTCOME_V2_PROSPECTIVE_BATCH_SCHEMA_VERSION:
            raise OutcomeV2RollingError("unsupported prospective-batch schema")
        _require_value(record, "kind", _BATCH_KIND)
        _require_value(record, "status", _BATCH_STATUS)
        _require_text(_string(record, "batch_id"), "batch_id")
        _require_value(record, "state_id", NBA_PRIMARY_STATE_ID)
        for field_name in _BATCH_KEYS & {
            "plan_sha256",
            "outcome_v2_run_lock_sha256",
            "experiment_lock_sha256",
            "question_ids_sha256",
            "sft_forecast_seal_sha256",
            "evaluation_feature_rows_sha256",
            "evaluation_prompts_sha256",
            "evaluation_generation_lock_sha256",
            "evaluation_inference_journal_sha256",
            "evaluation_inference_records_sha256",
            "coverage_policy_sha256",
        }:
            _require_hash(_string(record, field_name), field_name)
        _require_tinker_path(_string(record, "sampler_path"))
        _require_nonempty_seasons(_integer_tuple(record, "seasons"))
        game_count = _positive_integer(record, "game_count")
        if _nonnegative_integer(record, "failed_record_count") > game_count:
            raise OutcomeV2RollingError("failed_record_count exceeds game_count")
        latest_input = _parse_utc(
            _string(record, "latest_input_available_at"),
            "latest_input_available_at",
        )
        earliest_input = _parse_utc(
            _string(record, "earliest_input_available_at"),
            "earliest_input_available_at",
        )
        created_at = _parse_utc(_string(record, "created_at"), "created_at")
        cutoff = _parse_utc(
            _string(record, "earliest_forecast_cutoff"),
            "earliest_forecast_cutoff",
        )
        if not earliest_input <= latest_input <= created_at < cutoff:
            raise OutcomeV2RollingError("batch-seal timestamps violate the causal window")
    except JsonFormatError as error:
        raise OutcomeV2RollingError("invalid prospective-batch structure") from error
    return record


def _canonical_record(value: bytes, description: str) -> JsonObject:
    try:
        text = value.decode("utf-8")
        record = parse_json_object(text)
    except (JsonFormatError, UnicodeError) as error:
        raise OutcomeV2RollingError(f"{description} must be one UTF-8 JSON object") from error
    if text != canonical_json(record):
        raise OutcomeV2RollingError(f"{description} must use canonical JSON bytes")
    return record


def _load_batch_rows(path: Path) -> tuple[NbaRichFeatureRow, ...]:
    try:
        return read_nba_feature_rows_jsonl_bytes(_read_bytes(path, "feature rows"))
    except NbaFeatureRowError as error:
        raise OutcomeV2RollingError("cannot load prospective batch feature rows") from error


def _require_file_hash(path: Path, expected: str, description: str) -> None:
    if bytes_sha256(_read_bytes(path, description)) != expected:
        raise OutcomeV2RollingError(f"{description} differ from the terminal forecast seal")


def _require_unchanged_model_files(
    artifacts: OutcomeV2ProspectivePlanArtifacts,
    run_bytes: bytes,
    experiment_bytes: bytes,
) -> None:
    if (
        _read_bytes(artifacts.run_lock_path, "outcome-v2 run lock") != run_bytes
        or _read_bytes(artifacts.experiment_lock_path, "outcome-v2 experiment lock")
        != experiment_bytes
    ):
        raise OutcomeV2RollingError("model locks changed while building the prospective plan")


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
        raise OutcomeV2RollingError(f"cannot write {description}") from error


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise OutcomeV2RollingError(f"cannot read {description}") from error


def _require_immutable_bytes(value: object, description: str) -> None:
    if not isinstance(value, bytes):
        raise OutcomeV2RollingError(f"{description} requires immutable bytes")


def _require_text(value: str, field_name: str) -> None:
    if not value.strip() or value != value.strip():
        raise OutcomeV2RollingError(f"{field_name} must be a nonempty trimmed string")


def _require_workflow_id(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OutcomeV2RollingError("receipt_workflow_id must be a positive integer")


def _require_rolling_repository_path(value: str) -> None:
    prefix = "prospective/outcome_v2/rolling/"
    if not value.startswith(prefix) or value.endswith("/") or ".." in value.split("/"):
        raise OutcomeV2RollingError(
            "receipt artifact paths must be files under prospective/outcome_v2/rolling"
        )


def _require_hash(value: str, field_name: str) -> None:
    if _HASH_PATTERN.fullmatch(value) is None:
        raise OutcomeV2RollingError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_tinker_path(value: str) -> None:
    suffix = value.removeprefix("tinker://")
    if (
        not value.startswith("tinker://")
        or not suffix
        or any(character.isspace() or ord(character) < 32 for character in value)
        or "?" in suffix
        or "#" in suffix
    ):
        raise OutcomeV2RollingError("sampler_path must be a permanent tinker:// path")


def _require_multi_seasons(seasons: tuple[int, ...]) -> None:
    if len(seasons) < 2:
        raise OutcomeV2RollingError("prospective plan requires at least two seasons")
    _require_nonempty_seasons(seasons)


def _require_nonempty_seasons(seasons: tuple[int, ...]) -> None:
    if (
        not seasons
        or seasons != tuple(sorted(set(seasons)))
        or any(value <= 0 for value in seasons)
    ):
        raise OutcomeV2RollingError("seasons must be increasing unique positive values")


def _require_utc(value: object, field_name: str) -> None:
    _utc_text(value, field_name)


def _parse_utc(value: str, field_name: str) -> datetime:
    if not value.endswith("Z"):
        raise OutcomeV2RollingError(f"{field_name} must use canonical UTC Z notation")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise OutcomeV2RollingError(f"{field_name} must be an ISO 8601 datetime") from error
    if _utc_text(parsed, field_name) != value:
        raise OutcomeV2RollingError(f"{field_name} must use canonical UTC notation")
    return parsed.astimezone(UTC)


def _utc_text(value: object, field_name: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise OutcomeV2RollingError(f"{field_name} must be a UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise OutcomeV2RollingError(f"{field_name} must be a UTC datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _require_value(record: Mapping[str, object], field_name: str, expected: str) -> None:
    if _string(record, field_name) != expected:
        raise OutcomeV2RollingError(f"{field_name} has an unexpected value")


def _object(record: Mapping[str, object], field_name: str) -> JsonObject:
    return require_object(required_field(record, field_name), field_name)


def _string(record: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = required_field(record, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutcomeV2RollingError(f"{field_name} must be an integer")
    return value


def _positive_integer(record: Mapping[str, object], field_name: str) -> int:
    value = _integer(record, field_name)
    if value <= 0:
        raise OutcomeV2RollingError(f"{field_name} must be positive")
    return value


def _nonnegative_integer(record: Mapping[str, object], field_name: str) -> int:
    value = _integer(record, field_name)
    if value < 0:
        raise OutcomeV2RollingError(f"{field_name} must be non-negative")
    return value


def _integer_tuple(record: Mapping[str, object], field_name: str) -> tuple[int, ...]:
    values = require_list(required_field(record, field_name), field_name)
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise OutcomeV2RollingError(f"{field_name}[{index}] must be an integer")
        result.append(value)
    return tuple(result)


def _string_tuple(record: Mapping[str, object], field_name: str) -> tuple[str, ...]:
    values = require_list(required_field(record, field_name), field_name)
    return tuple(
        require_string(value, f"{field_name}[{index}]") for index, value in enumerate(values)
    )
