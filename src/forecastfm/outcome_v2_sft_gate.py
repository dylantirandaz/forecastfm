"""Answer-free SFT sealing and a retrospective post-SFT NBA holdout gate.

The local chain binds a durable journal as tamper evidence. It does not prove remote
execution, single-application-attempt history, absence of pretraining contamination, or rolling
prospective performance.
"""

from __future__ import annotations

import os
import re
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

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
from forecastfm.nba_evaluation_gate import (
    NbaEvaluationCohortInput,
    NbaEvaluationGateArtifacts,
    NbaEvaluationGateError,
    NbaEvaluationGatePolicy,
    NbaEvaluationGateReport,
    read_nba_evaluation_cohort_jsonl,
    read_nba_evaluation_forecasts_jsonl,
    verify_untouched_nba_evaluation_gate,
)
from forecastfm.nba_feature_rows import (
    NbaFeatureRowError,
    NbaRichFeatureRow,
    read_nba_feature_rows_jsonl_bytes,
)
from forecastfm.outcome_v2_config import outcome_v2_evaluation_policy
from forecastfm.outcome_v2_experiment import (
    OutcomeV2ExperimentError,
    OutcomeV2ExperimentLock,
)
from forecastfm.outcome_v2_inference import (
    InferenceRecord,
    OutcomeV2GenerationArtifacts,
    OutcomeV2GenerationLock,
    OutcomeV2InferenceError,
    binary_forecasts_from_inference_records,
    outcome_v2_prompt_pairs_jsonl_bytes,
    read_outcome_v2_inference_records,
    verify_outcome_v2_generation_lock,
)
from forecastfm.outcome_v2_metrics import BinaryForecast
from forecastfm.outcome_v2_run import OutcomeV2RunError, OutcomeV2RunLock

OUTCOME_V2_SFT_FORECAST_SEAL_SCHEMA_VERSION = 1
OUTCOME_V2_POST_SFT_GATE_SCHEMA_VERSION = 1
OUTCOME_V2_SFT_CANDIDATE_ROLE = "forecastfm_outcome_v2_sft_adapter"

_SEAL_KIND = "forecastfm_outcome_v2_sft_forecast_seal"
_SEAL_STATUS = "answer_free_forecasts_sealed"
_REPORT_KIND = "forecastfm_nba_outcome_v2_post_sft_gate"
_REPORT_STATUS = "passed"
_RELATION = "disjoint_and_strictly_later"
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_SEAL_KEYS = {
    "schema_version",
    "kind",
    "status",
    "created_at",
    "candidate_role",
    "outcome_v2_run_lock_sha256",
    "outcome_v2_experiment_lock_sha256",
    "sampler_path",
    "evaluation_cohort_sha256",
    "evaluation_feature_rows_sha256",
    "evaluation_prompts_sha256",
    "evaluation_forecasts_sha256",
    "evaluation_generation_lock_sha256",
    "evaluation_inference_journal_sha256",
    "evaluation_inference_records_sha256",
    "evaluation_question_ids_sha256",
    "evaluation_seasons",
    "forecast_count",
    "orientation_count",
    "failed_record_count",
}
_REPORT_KEYS = {
    "schema_version",
    "kind",
    "status",
    "candidate_role",
    "proof_scope",
    "candidate",
    "cohorts",
    "artifacts",
    "evaluation",
}
_PROOF_SCOPE_KEYS = {
    "candidate_model_and_run_provenance",
    "external_precommit_and_timestamp",
    "remote_inference_attestation",
    "raw_provider_derivation",
    "pretraining_contamination",
    "rolling_prospective_proof",
    "durable_inference_journal",
}
_CANDIDATE_KEYS = {
    "outcome_v2_run_lock_sha256",
    "outcome_v2_experiment_lock_sha256",
    "sft_forecast_seal_sha256",
    "sampler_path",
}
_COHORT_KEYS = {
    "relation",
    "tabular_question_ids_sha256",
    "tabular_seasons",
    "sft_question_ids_sha256",
    "sft_seasons",
}
_ARTIFACT_KEYS = {
    "tabular_cohort_sha256",
    "tabular_evaluation_report_sha256",
    "sft_cohort_sha256",
    "sft_feature_rows_sha256",
    "sft_prompts_sha256",
    "sft_forecasts_sha256",
    "sft_generation_lock_sha256",
    "sft_inference_journal_sha256",
    "sft_inference_records_sha256",
    "sft_failed_record_count",
    "sft_answers_sha256",
    "sft_calibration_sha256",
    "sft_generic_gate_report_sha256",
}
_EVALUATION_KEYS = {
    "policy_sha256",
    "mode",
    "tabular_gate_role",
    "post_sft_gate_role",
}
_ARTIFACT_HASH_KEYS = _ARTIFACT_KEYS - {"sft_failed_record_count"}

type JsonObject = dict[str, object]


class OutcomeV2SftGateError(ValueError):
    """Raised when post-SFT provenance or evaluation is incomplete."""


@dataclass(frozen=True, slots=True)
class OutcomeV2SftForecastArtifacts:
    """Only the answer-free files and model locks used during generation."""

    project_root: Path
    cohort_path: Path
    feature_rows_path: Path
    prompts_path: Path
    forecasts_path: Path
    run_lock_path: Path
    experiment_lock_path: Path
    generation_lock_path: Path
    inference_journal_path: Path
    inference_records_path: Path


@dataclass(frozen=True, slots=True)
class OutcomeV2SftForecastSeal:
    """Canonical binding from one trained SFT adapter to answer-free forecasts."""

    canonical_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_bytes(self.canonical_bytes, "SFT forecast seal")
        _seal_record_from_bytes(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        """Return the digest of the exact canonical seal bytes."""
        return bytes_sha256(self.canonical_bytes)

    def to_record(self) -> JsonObject:
        """Return a newly decoded strict seal record."""
        return _seal_record_from_bytes(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class OutcomeV2PostSftGateArtifacts:
    """Scorer-side files for a gate that is separate from the tabular gate."""

    tabular_cohort_path: Path
    tabular_evaluation_report_path: Path
    sft_forecast: OutcomeV2SftForecastArtifacts
    sft_forecast_seal_path: Path
    sft_evaluation: NbaEvaluationGateArtifacts
    supplied_report_path: Path | None = None


@dataclass(frozen=True, slots=True)
class OutcomeV2PostSftGateReport:
    """Canonical passing report with an explicitly post-SFT identity."""

    canonical_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_bytes(self.canonical_bytes, "post-SFT report")
        _report_record_from_bytes(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        """Return the digest of the exact canonical report bytes."""
        return bytes_sha256(self.canonical_bytes)

    @property
    def payload(self) -> JsonObject:
        """Return a newly decoded strict report record."""
        return _report_record_from_bytes(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class _ModelProvenance:
    run_lock_sha256: str
    experiment_lock_sha256: str
    sampler_path: str
    run_record: JsonObject


@dataclass(frozen=True, slots=True)
class _ForecastInputs:
    cohort_sha256: str
    feature_rows_sha256: str
    prompts_sha256: str
    forecasts_sha256: str
    generation_lock_sha256: str
    inference_journal_sha256: str
    inference_records_sha256: str
    question_ids: tuple[str, ...]
    seasons: tuple[int, ...]
    forecast_count: int
    orientation_count: int
    failed_record_count: int
    generation_created_at: datetime


@dataclass(frozen=True, slots=True)
class _AnswerFreeBytes:
    cohort: bytes
    feature_rows: bytes
    prompts: bytes
    forecasts: bytes
    generation_lock: bytes
    inference_journal: bytes
    inference_records: bytes
    run_lock: bytes
    experiment_lock: bytes


@dataclass(frozen=True, slots=True)
class _LoadedAnswerFreeFiles:
    cohort: tuple[NbaEvaluationCohortInput, ...]
    feature_rows: tuple[NbaRichFeatureRow, ...]
    forecasts: tuple[BinaryForecast, ...]
    generation_lock: OutcomeV2GenerationLock
    inference_records: tuple[InferenceRecord, ...]


@dataclass(frozen=True, slots=True)
class _TabularGate:
    cohort_sha256: str
    report_sha256: str
    calibration_sha256: str
    question_ids: tuple[str, ...]
    seasons: tuple[int, ...]


@dataclass(slots=True)
class _ArtifactSnapshots:
    """Read every source path once and retain its immutable bytes."""

    values: dict[Path, bytes] = field(default_factory=dict[Path, bytes])

    def capture(self, path: Path, description: str) -> bytes:
        """Return the one captured byte snapshot for ``path``."""
        if path not in self.values:
            self.values[path] = _read_bytes(path, description)
        return self.values[path]


@dataclass(frozen=True, slots=True)
class _VerifiedForecastBundle:
    """One model-and-input snapshot verified against one forecast seal."""

    seal: OutcomeV2SftForecastSeal
    provenance: _ModelProvenance
    inputs: _ForecastInputs


def build_outcome_v2_sft_forecast_seal(
    artifacts: OutcomeV2SftForecastArtifacts,
    created_at: datetime,
) -> OutcomeV2SftForecastSeal:
    """Build one seal without accepting an answer or resolution artifact."""
    snapshots = _ArtifactSnapshots()
    provenance = _verify_model_provenance(artifacts, snapshots)
    inputs = _load_forecast_inputs(artifacts, snapshots)
    _require_utc(created_at, "created_at")
    if created_at < inputs.generation_created_at:
        raise OutcomeV2SftGateError("forecast seal cannot predate its generation lock")
    record = _forecast_seal_payload(provenance, inputs, created_at)
    return OutcomeV2SftForecastSeal(canonical_json(record).encode("utf-8"))


def write_outcome_v2_sft_forecast_seal(
    path: Path,
    seal: OutcomeV2SftForecastSeal,
) -> str:
    """Create and durably flush one SFT forecast seal without replacement."""
    _write_once(path, seal.canonical_bytes, "SFT forecast seal")
    return seal.sha256


def read_outcome_v2_sft_forecast_seal(path: Path) -> OutcomeV2SftForecastSeal:
    """Read one strict canonical SFT forecast seal."""
    return OutcomeV2SftForecastSeal(_read_bytes(path, "SFT forecast seal"))


def verify_outcome_v2_sft_forecast_seal(
    artifacts: OutcomeV2SftForecastArtifacts,
    seal_path: Path,
) -> OutcomeV2SftForecastSeal:
    """Verify exact model, cohort, prompt, and forecast bindings."""
    snapshots = _ArtifactSnapshots()
    return _verify_forecast_bundle(artifacts, seal_path, snapshots).seal


def _verify_forecast_bundle(
    artifacts: OutcomeV2SftForecastArtifacts,
    seal_path: Path,
    snapshots: _ArtifactSnapshots,
) -> _VerifiedForecastBundle:
    seal = OutcomeV2SftForecastSeal(snapshots.capture(seal_path, "SFT forecast seal"))
    record = seal.to_record()
    provenance = _verify_model_provenance(artifacts, snapshots)
    inputs = _load_forecast_inputs(artifacts, snapshots)
    created_at = _parse_utc(_string(record, "created_at"), "created_at")
    if created_at < inputs.generation_created_at:
        raise OutcomeV2SftGateError("forecast seal predates its generation lock")
    expected = _forecast_seal_payload(provenance, inputs, created_at)
    if record != expected:
        raise OutcomeV2SftGateError(
            "SFT forecast seal differs from the verified model or answer-free files"
        )
    return _VerifiedForecastBundle(seal, provenance, inputs)


def verify_outcome_v2_post_sft_gate(
    artifacts: OutcomeV2PostSftGateArtifacts,
    *,
    policy: NbaEvaluationGatePolicy,
) -> OutcomeV2PostSftGateReport:
    """Score new later answer-held seasons without claiming prospective proof."""
    snapshots = _ArtifactSnapshots()
    forecast = _verify_forecast_bundle(
        artifacts.sft_forecast,
        artifacts.sft_forecast_seal_path,
        snapshots,
    )
    seal_record = forecast.seal.to_record()
    _require_policy_matches_run_lock(policy, forecast.provenance.run_record)
    tabular = _verify_tabular_gate(
        artifacts.tabular_cohort_path,
        artifacts.tabular_evaluation_report_path,
        forecast.provenance.run_record,
        snapshots,
    )
    _require_new_later_cohort(tabular, forecast.inputs)

    generic_report = _verify_generic_gate(
        artifacts.sft_evaluation,
        forecast.inputs,
        tabular.calibration_sha256,
        policy,
        snapshots,
    )
    generic_artifacts = _verify_generic_report_bindings(generic_report, forecast.inputs)
    if generic_report.sha256 == tabular.report_sha256:
        raise OutcomeV2SftGateError("tabular and post-SFT reports must be distinct")

    report_record = {
        "schema_version": OUTCOME_V2_POST_SFT_GATE_SCHEMA_VERSION,
        "kind": _REPORT_KIND,
        "status": _REPORT_STATUS,
        "candidate_role": OUTCOME_V2_SFT_CANDIDATE_ROLE,
        "proof_scope": {
            "candidate_model_and_run_provenance": "verified_local_lock_chain",
            "external_precommit_and_timestamp": "required_separately",
            "remote_inference_attestation": "required_separately",
            "raw_provider_derivation": "required_separately",
            "pretraining_contamination": "possible_not_ruled_out",
            "rolling_prospective_proof": "not_satisfied",
            "durable_inference_journal": (
                "locally_sha256_bound_not_remote_execution_or_one_attempt_proof"
            ),
        },
        "candidate": {
            "outcome_v2_run_lock_sha256": forecast.provenance.run_lock_sha256,
            "outcome_v2_experiment_lock_sha256": forecast.provenance.experiment_lock_sha256,
            "sft_forecast_seal_sha256": forecast.seal.sha256,
            "sampler_path": _string(seal_record, "sampler_path"),
        },
        "cohorts": {
            "relation": _RELATION,
            "tabular_question_ids_sha256": canonical_sha256(list(tabular.question_ids)),
            "tabular_seasons": list(tabular.seasons),
            "sft_question_ids_sha256": canonical_sha256(list(forecast.inputs.question_ids)),
            "sft_seasons": list(forecast.inputs.seasons),
        },
        "artifacts": {
            "tabular_cohort_sha256": tabular.cohort_sha256,
            "tabular_evaluation_report_sha256": tabular.report_sha256,
            "sft_cohort_sha256": forecast.inputs.cohort_sha256,
            "sft_feature_rows_sha256": forecast.inputs.feature_rows_sha256,
            "sft_prompts_sha256": forecast.inputs.prompts_sha256,
            "sft_forecasts_sha256": forecast.inputs.forecasts_sha256,
            "sft_generation_lock_sha256": forecast.inputs.generation_lock_sha256,
            "sft_inference_journal_sha256": forecast.inputs.inference_journal_sha256,
            "sft_inference_records_sha256": forecast.inputs.inference_records_sha256,
            "sft_failed_record_count": forecast.inputs.failed_record_count,
            "sft_answers_sha256": generic_artifacts["answers_sha256"],
            "sft_calibration_sha256": generic_artifacts["calibration_sha256"],
            "sft_generic_gate_report_sha256": generic_report.sha256,
        },
        "evaluation": {
            "policy_sha256": policy.policy_sha256,
            "mode": "retrospective_answer_held_holdout",
            "tabular_gate_role": "sft_training_prerequisite_only",
            "post_sft_gate_role": "sft_candidate_advancement",
        },
    }
    report = OutcomeV2PostSftGateReport(canonical_json(report_record).encode("utf-8"))
    _require_matching_supplied_report(artifacts.supplied_report_path, report, snapshots)
    return report


def write_outcome_v2_post_sft_gate_report(
    path: Path,
    report: OutcomeV2PostSftGateReport,
) -> str:
    """Create one canonical post-SFT report without replacement."""
    _write_once(path, report.canonical_bytes, "post-SFT gate report")
    return report.sha256


def read_outcome_v2_post_sft_gate_report(path: Path) -> OutcomeV2PostSftGateReport:
    """Read one strict canonical post-SFT report."""
    return OutcomeV2PostSftGateReport(_read_bytes(path, "post-SFT gate report"))


def _verify_model_provenance(
    artifacts: OutcomeV2SftForecastArtifacts,
    snapshots: _ArtifactSnapshots,
) -> _ModelProvenance:
    try:
        run_bytes = snapshots.capture(
            artifacts.run_lock_path,
            "outcome-v2 run lock",
        )
        experiment_bytes = snapshots.capture(
            artifacts.experiment_lock_path,
            "outcome-v2 experiment lock",
        )
        run_lock = OutcomeV2RunLock(run_bytes)
        experiment = OutcomeV2ExperimentLock(experiment_bytes)
    except (OSError, OutcomeV2ExperimentError, OutcomeV2RunError) as error:
        raise OutcomeV2SftGateError("cannot verify the outcome-v2 model locks") from error
    experiment_record = experiment.to_record()
    run_sha256 = bytes_sha256(run_bytes)
    if _string(experiment_record, "outcome_v2_run_lock_sha256") != run_sha256:
        raise OutcomeV2SftGateError("experiment does not bind the exact run lock")
    return _ModelProvenance(
        run_lock_sha256=run_sha256,
        experiment_lock_sha256=experiment.sha256,
        sampler_path=_string(experiment_record, "sampler_path"),
        run_record=run_lock.to_record(),
    )


def _load_forecast_inputs(
    artifacts: OutcomeV2SftForecastArtifacts,
    snapshots: _ArtifactSnapshots,
) -> _ForecastInputs:
    values = _capture_answer_free_bytes(artifacts, snapshots)
    loaded = _load_answer_free_files(artifacts, values)
    question_ids, seasons = _require_answer_free_alignment(values, loaded)
    generation_record = loaded.generation_lock.to_record()
    return _ForecastInputs(
        cohort_sha256=bytes_sha256(values.cohort),
        feature_rows_sha256=bytes_sha256(values.feature_rows),
        prompts_sha256=bytes_sha256(values.prompts),
        forecasts_sha256=bytes_sha256(values.forecasts),
        generation_lock_sha256=bytes_sha256(values.generation_lock),
        inference_journal_sha256=bytes_sha256(values.inference_journal),
        inference_records_sha256=bytes_sha256(values.inference_records),
        question_ids=question_ids,
        seasons=seasons,
        forecast_count=len(loaded.forecasts),
        orientation_count=_positive_integer(generation_record, "orientation_count"),
        failed_record_count=sum(record.status == "failed" for record in loaded.inference_records),
        generation_created_at=_parse_utc(
            _string(generation_record, "created_at"),
            "generation.created_at",
        ),
    )


def _capture_answer_free_bytes(
    artifacts: OutcomeV2SftForecastArtifacts,
    snapshots: _ArtifactSnapshots,
) -> _AnswerFreeBytes:
    return _AnswerFreeBytes(
        cohort=snapshots.capture(artifacts.cohort_path, "SFT evaluation cohort"),
        feature_rows=snapshots.capture(
            artifacts.feature_rows_path,
            "SFT evaluation feature rows",
        ),
        prompts=snapshots.capture(artifacts.prompts_path, "SFT evaluation prompts"),
        forecasts=snapshots.capture(
            artifacts.forecasts_path,
            "SFT evaluation forecasts",
        ),
        generation_lock=snapshots.capture(
            artifacts.generation_lock_path,
            "SFT generation lock",
        ),
        inference_journal=_capture_nonempty(
            snapshots,
            artifacts.inference_journal_path,
            "SFT inference journal",
        ),
        inference_records=snapshots.capture(
            artifacts.inference_records_path,
            "SFT inference records",
        ),
        run_lock=snapshots.capture(artifacts.run_lock_path, "outcome-v2 run lock"),
        experiment_lock=snapshots.capture(
            artifacts.experiment_lock_path,
            "outcome-v2 experiment lock",
        ),
    )


def _capture_nonempty(
    snapshots: _ArtifactSnapshots,
    path: Path,
    description: str,
) -> bytes:
    value = snapshots.capture(path, description)
    if not value:
        raise OutcomeV2SftGateError(f"{description} must not be empty")
    return value


def _load_answer_free_files(
    artifacts: OutcomeV2SftForecastArtifacts,
    values: _AnswerFreeBytes,
) -> _LoadedAnswerFreeFiles:
    try:
        with _stage_files(
            {
                "cohort.jsonl": values.cohort,
                "forecasts.jsonl": values.forecasts,
                "feature-rows.jsonl": values.feature_rows,
                "run-lock.json": values.run_lock,
                "experiment-lock.json": values.experiment_lock,
                "generation-lock.json": values.generation_lock,
                "inference-records.jsonl": values.inference_records,
            }
        ) as staged:
            cohort = read_nba_evaluation_cohort_jsonl(staged["cohort.jsonl"])
            feature_rows = read_nba_feature_rows_jsonl_bytes(values.feature_rows)
            forecasts = read_nba_evaluation_forecasts_jsonl(staged["forecasts.jsonl"])
            generation_lock = verify_outcome_v2_generation_lock(
                OutcomeV2GenerationArtifacts(
                    project_root=artifacts.project_root,
                    run_lock_path=staged["run-lock.json"],
                    experiment_lock_path=staged["experiment-lock.json"],
                    feature_rows_path=staged["feature-rows.jsonl"],
                ),
                staged["generation-lock.json"],
            )
            inference_records = read_outcome_v2_inference_records(
                staged["inference-records.jsonl"],
                generation_lock,
            )
            derived_forecasts = binary_forecasts_from_inference_records(
                inference_records,
                generation_lock,
            )
    except (
        JsonFormatError,
        NbaEvaluationGateError,
        NbaFeatureRowError,
        OutcomeV2InferenceError,
        OSError,
    ) as error:
        raise OutcomeV2SftGateError("cannot load answer-free SFT forecast files") from error
    if forecasts != derived_forecasts:
        raise OutcomeV2SftGateError(
            "SFT forecasts differ from the sealed terminal inference records"
        )
    return _LoadedAnswerFreeFiles(
        cohort=cohort,
        feature_rows=feature_rows,
        forecasts=forecasts,
        generation_lock=generation_lock,
        inference_records=inference_records,
    )


def _require_answer_free_alignment(
    values: _AnswerFreeBytes,
    loaded: _LoadedAnswerFreeFiles,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    generation_record = loaded.generation_lock.to_record()
    if _string(generation_record, "prompt_pairs_sha256") != bytes_sha256(values.prompts):
        raise OutcomeV2SftGateError("SFT prompts differ from the generation lock")
    expected_prompts = outcome_v2_prompt_pairs_jsonl_bytes(loaded.feature_rows)
    if values.prompts != expected_prompts:
        raise OutcomeV2SftGateError(
            "SFT prompts differ from the canonical target-free feature-row rendering"
        )
    question_ids = tuple(row.question_id for row in loaded.cohort)
    feature_ids = tuple(row.question_id for row in loaded.feature_rows)
    forecast_ids = tuple(row.question_id for row in loaded.forecasts)
    if feature_ids != question_ids:
        raise OutcomeV2SftGateError("SFT feature-row IDs differ from the frozen cohort")
    if forecast_ids != question_ids:
        raise OutcomeV2SftGateError("SFT forecast IDs differ from the frozen cohort")
    for member, row in zip(loaded.cohort, loaded.feature_rows, strict=True):
        _require_feature_cohort_alignment(member, row)
    seasons = tuple(sorted({row.season for row in loaded.cohort}))
    if _integer_tuple(generation_record, "evaluation_seasons") != seasons:
        raise OutcomeV2SftGateError("SFT cohort seasons differ from the generation lock")
    return question_ids, seasons


def _require_feature_cohort_alignment(
    member: NbaEvaluationCohortInput,
    row: NbaRichFeatureRow,
) -> None:
    if row.season != member.season:
        raise OutcomeV2SftGateError("SFT feature-row season differs from the cohort")
    if row.scheduled_tipoff.date() != member.game_date:
        raise OutcomeV2SftGateError("SFT feature-row date differs from the cohort")
    if row.elo_team_win_probability != member.raw_elo_team_probability:
        raise OutcomeV2SftGateError("SFT feature-row Elo prior differs from the cohort")


def _forecast_seal_payload(
    provenance: _ModelProvenance,
    inputs: _ForecastInputs,
    created_at: datetime,
) -> JsonObject:
    return {
        "schema_version": OUTCOME_V2_SFT_FORECAST_SEAL_SCHEMA_VERSION,
        "kind": _SEAL_KIND,
        "status": _SEAL_STATUS,
        "created_at": _utc_text(created_at, "created_at"),
        "candidate_role": OUTCOME_V2_SFT_CANDIDATE_ROLE,
        "outcome_v2_run_lock_sha256": provenance.run_lock_sha256,
        "outcome_v2_experiment_lock_sha256": provenance.experiment_lock_sha256,
        "sampler_path": provenance.sampler_path,
        "evaluation_cohort_sha256": inputs.cohort_sha256,
        "evaluation_feature_rows_sha256": inputs.feature_rows_sha256,
        "evaluation_prompts_sha256": inputs.prompts_sha256,
        "evaluation_forecasts_sha256": inputs.forecasts_sha256,
        "evaluation_generation_lock_sha256": inputs.generation_lock_sha256,
        "evaluation_inference_journal_sha256": inputs.inference_journal_sha256,
        "evaluation_inference_records_sha256": inputs.inference_records_sha256,
        "evaluation_question_ids_sha256": canonical_sha256(list(inputs.question_ids)),
        "evaluation_seasons": list(inputs.seasons),
        "forecast_count": inputs.forecast_count,
        "orientation_count": inputs.orientation_count,
        "failed_record_count": inputs.failed_record_count,
    }


def _verify_tabular_gate(
    cohort_path: Path,
    report_path: Path,
    run_record: Mapping[str, object],
    snapshots: _ArtifactSnapshots,
) -> _TabularGate:
    cohort_bytes = snapshots.capture(cohort_path, "tabular evaluation cohort")
    report_bytes = snapshots.capture(report_path, "tabular evaluation report")
    try:
        with _stage_files({"cohort.jsonl": cohort_bytes}) as staged:
            cohort = read_nba_evaluation_cohort_jsonl(staged["cohort.jsonl"])
        report_text = report_bytes.decode("utf-8")
        report = NbaEvaluationGateReport(report_text, bytes_sha256(report_bytes))
        preflight = require_object(required_field(run_record, "preflight"), "preflight")
        expected_report_sha256 = require_string(
            required_field(preflight, "evaluation_report_sha256"),
            "preflight.evaluation_report_sha256",
        )
        run_lock_seasons = _integer_tuple(preflight, "untouched_evaluation_seasons")
        payload = report.payload
        reported_artifacts = require_object(
            required_field(payload, "artifacts"),
            "tabular report artifacts",
        )
        reported_evaluation = require_object(
            required_field(payload, "evaluation"),
            "tabular report evaluation",
        )
    except (JsonFormatError, NbaEvaluationGateError, OSError, UnicodeError) as error:
        raise OutcomeV2SftGateError("cannot verify the tabular prerequisite gate") from error
    if report.sha256 != expected_report_sha256:
        raise OutcomeV2SftGateError("tabular report differs from the SFT run-lock prerequisite")
    if payload.get("kind") != "forecastfm_nba_untouched_evaluation_gate" or (
        payload.get("status") != "passed"
    ):
        raise OutcomeV2SftGateError("run-lock prerequisite is not a passing NBA gate")
    cohort_sha256 = bytes_sha256(cohort_bytes)
    if _string(reported_artifacts, "cohort_sha256") != cohort_sha256:
        raise OutcomeV2SftGateError("tabular cohort differs from its run-lock-bound report")
    question_ids = tuple(row.question_id for row in cohort)
    seasons = tuple(sorted({row.season for row in cohort}))
    if _string(reported_evaluation, "question_ids_sha256") != canonical_sha256(list(question_ids)):
        raise OutcomeV2SftGateError("tabular cohort IDs differ from its report")
    if _integer_tuple(reported_evaluation, "seasons") != seasons:
        raise OutcomeV2SftGateError("tabular cohort seasons differ from its report")
    if seasons != run_lock_seasons:
        raise OutcomeV2SftGateError("tabular cohort seasons differ from the SFT run lock")
    calibration_sha256 = _string(reported_artifacts, "calibration_sha256")
    _require_hash(calibration_sha256, "tabular calibration_sha256")
    return _TabularGate(
        cohort_sha256=cohort_sha256,
        report_sha256=report.sha256,
        calibration_sha256=calibration_sha256,
        question_ids=question_ids,
        seasons=seasons,
    )


def _require_policy_matches_run_lock(
    policy: NbaEvaluationGatePolicy,
    run_record: Mapping[str, object],
) -> None:
    if policy != outcome_v2_evaluation_policy():
        raise OutcomeV2SftGateError(
            "post-SFT evaluation policy differs from the frozen production policy"
        )
    try:
        committed = require_object(
            required_field(run_record, "evaluation_policy"),
            "evaluation_policy",
        )
        config = require_object(
            required_field(committed, "config"),
            "evaluation_policy.config",
        )
        digest = _string(committed, "sha256")
    except JsonFormatError as error:
        raise OutcomeV2SftGateError("run lock has an invalid evaluation policy") from error
    if config != policy.canonical_payload() or digest != policy.policy_sha256:
        raise OutcomeV2SftGateError(
            "post-SFT evaluation policy differs from the policy committed in the run lock"
        )


def _verify_generic_gate(
    artifacts: NbaEvaluationGateArtifacts,
    inputs: _ForecastInputs,
    tabular_calibration_sha256: str,
    policy: NbaEvaluationGatePolicy,
    snapshots: _ArtifactSnapshots,
) -> NbaEvaluationGateReport:
    cohort_bytes = snapshots.capture(artifacts.cohort_path, "SFT scorer cohort")
    forecasts_bytes = snapshots.capture(artifacts.forecasts_path, "SFT scorer forecasts")
    if bytes_sha256(cohort_bytes) != inputs.cohort_sha256:
        raise OutcomeV2SftGateError("SFT scorer cohort differs from the forecast seal")
    if bytes_sha256(forecasts_bytes) != inputs.forecasts_sha256:
        raise OutcomeV2SftGateError("SFT scorer forecasts differ from the forecast seal")

    calibration_bytes = snapshots.capture(
        artifacts.calibration_path,
        "SFT scorer calibration rows",
    )
    if bytes_sha256(calibration_bytes) != tabular_calibration_sha256:
        raise OutcomeV2SftGateError(
            "post-SFT recalibrated-Elo comparator differs from the tabular prerequisite"
        )
    files = {
        "cohort.jsonl": cohort_bytes,
        "answers.jsonl": snapshots.capture(artifacts.answers_path, "SFT scorer answers"),
        "forecasts.jsonl": forecasts_bytes,
        "calibration.jsonl": calibration_bytes,
    }
    if artifacts.supplied_report_path is not None:
        files["supplied-report.json"] = snapshots.capture(
            artifacts.supplied_report_path,
            "supplied generic SFT gate report",
        )
    try:
        with _stage_files(files) as staged:
            staged_artifacts = NbaEvaluationGateArtifacts(
                cohort_path=staged["cohort.jsonl"],
                answers_path=staged["answers.jsonl"],
                forecasts_path=staged["forecasts.jsonl"],
                calibration_path=staged["calibration.jsonl"],
                supplied_report_path=(
                    staged["supplied-report.json"]
                    if artifacts.supplied_report_path is not None
                    else None
                ),
            )
            return verify_untouched_nba_evaluation_gate(
                staged_artifacts,
                policy=policy,
            )
    except NbaEvaluationGateError as error:
        raise OutcomeV2SftGateError("post-SFT generic evaluation gate failed") from error


def _require_new_later_cohort(tabular: _TabularGate, sft: _ForecastInputs) -> None:
    if set(tabular.question_ids) & set(sft.question_ids):
        raise OutcomeV2SftGateError("tabular and post-SFT cohort IDs must be disjoint")
    if min(sft.seasons) <= max(tabular.seasons):
        raise OutcomeV2SftGateError(
            "every post-SFT evaluation season must be later than every tabular season"
        )


def _verify_generic_report_bindings(
    report: NbaEvaluationGateReport,
    inputs: _ForecastInputs,
) -> JsonObject:
    try:
        artifacts = require_object(required_field(report.payload, "artifacts"), "artifacts")
        require_exact_keys(
            artifacts,
            {"cohort_sha256", "answers_sha256", "forecasts_sha256", "calibration_sha256"},
            "generic gate artifacts",
        )
    except JsonFormatError as error:
        raise OutcomeV2SftGateError("generic SFT gate has invalid artifact bindings") from error
    if _string(artifacts, "cohort_sha256") != inputs.cohort_sha256:
        raise OutcomeV2SftGateError("generic SFT gate binds a different cohort")
    if _string(artifacts, "forecasts_sha256") != inputs.forecasts_sha256:
        raise OutcomeV2SftGateError("generic SFT gate binds different forecasts")
    _require_hash(_string(artifacts, "answers_sha256"), "answers_sha256")
    _require_hash(_string(artifacts, "calibration_sha256"), "calibration_sha256")
    return artifacts


def _require_matching_supplied_report(
    path: Path | None,
    expected: OutcomeV2PostSftGateReport,
    snapshots: _ArtifactSnapshots,
) -> None:
    if path is None:
        return
    supplied = OutcomeV2PostSftGateReport(snapshots.capture(path, "supplied post-SFT report"))
    if supplied.canonical_bytes != expected.canonical_bytes:
        raise OutcomeV2SftGateError("supplied post-SFT report differs from recomputed results")


@contextmanager
def _stage_files(values: Mapping[str, bytes]) -> Generator[dict[str, Path]]:
    """Expose captured bytes to existing strict path-based parsers."""
    try:
        with TemporaryDirectory(prefix="forecastfm-sft-gate-") as directory:
            root = Path(directory)
            paths = {name: root / name for name in values}
            for name, value in values.items():
                paths[name].write_bytes(value)
            yield paths
    except OSError as error:
        raise OutcomeV2SftGateError("cannot stage captured gate artifacts") from error


def _seal_record_from_bytes(value: bytes) -> JsonObject:
    record = _canonical_record(value, "SFT forecast seal")
    try:
        require_exact_keys(record, _SEAL_KEYS, "SFT forecast seal")
        if _integer(record, "schema_version") != OUTCOME_V2_SFT_FORECAST_SEAL_SCHEMA_VERSION:
            raise OutcomeV2SftGateError("unsupported SFT forecast-seal schema")
        _require_value(record, "kind", _SEAL_KIND)
        _require_value(record, "status", _SEAL_STATUS)
        _require_value(record, "candidate_role", OUTCOME_V2_SFT_CANDIDATE_ROLE)
        _parse_utc(_string(record, "created_at"), "created_at")
        for field_name in (
            "outcome_v2_run_lock_sha256",
            "outcome_v2_experiment_lock_sha256",
            "evaluation_cohort_sha256",
            "evaluation_feature_rows_sha256",
            "evaluation_prompts_sha256",
            "evaluation_forecasts_sha256",
            "evaluation_generation_lock_sha256",
            "evaluation_inference_journal_sha256",
            "evaluation_inference_records_sha256",
            "evaluation_question_ids_sha256",
        ):
            _require_hash(_string(record, field_name), field_name)
        _require_tinker_path(_string(record, "sampler_path"))
        _require_nonempty_seasons(_integer_tuple(record, "evaluation_seasons"))
        forecast_count = _positive_integer(record, "forecast_count")
        if _positive_integer(record, "orientation_count") != forecast_count * 2:
            raise OutcomeV2SftGateError("orientation_count must equal two per forecast")
        if _nonnegative_integer(record, "failed_record_count") > forecast_count:
            raise OutcomeV2SftGateError("failed_record_count exceeds forecast_count")
    except JsonFormatError as error:
        raise OutcomeV2SftGateError("invalid SFT forecast seal structure") from error
    return record


def _report_record_from_bytes(value: bytes) -> JsonObject:
    record = _canonical_record(value, "post-SFT gate report")
    try:
        require_exact_keys(record, _REPORT_KEYS, "post-SFT gate report")
        if _integer(record, "schema_version") != OUTCOME_V2_POST_SFT_GATE_SCHEMA_VERSION:
            raise OutcomeV2SftGateError("unsupported post-SFT gate schema")
        _require_value(record, "kind", _REPORT_KIND)
        _require_value(record, "status", _REPORT_STATUS)
        _require_value(record, "candidate_role", OUTCOME_V2_SFT_CANDIDATE_ROLE)
        proof_scope = require_object(required_field(record, "proof_scope"), "proof_scope")
        candidate = require_object(required_field(record, "candidate"), "candidate")
        cohorts = require_object(required_field(record, "cohorts"), "cohorts")
        artifacts = require_object(required_field(record, "artifacts"), "artifacts")
        evaluation = require_object(required_field(record, "evaluation"), "evaluation")
        require_exact_keys(proof_scope, _PROOF_SCOPE_KEYS, "proof_scope")
        require_exact_keys(candidate, _CANDIDATE_KEYS, "candidate")
        require_exact_keys(cohorts, _COHORT_KEYS, "cohorts")
        require_exact_keys(artifacts, _ARTIFACT_KEYS, "artifacts")
        require_exact_keys(evaluation, _EVALUATION_KEYS, "evaluation")
        _require_post_sft_proof_scope(proof_scope)
        for field_name in _CANDIDATE_KEYS - {"sampler_path"}:
            _require_hash(_string(candidate, field_name), f"candidate.{field_name}")
        _require_tinker_path(_string(candidate, "sampler_path"))
        _require_value(cohorts, "relation", _RELATION)
        for field_name in ("tabular_question_ids_sha256", "sft_question_ids_sha256"):
            _require_hash(_string(cohorts, field_name), f"cohorts.{field_name}")
        tabular_seasons = _integer_tuple(cohorts, "tabular_seasons")
        sft_seasons = _integer_tuple(cohorts, "sft_seasons")
        _require_seasons(tabular_seasons)
        _require_seasons(sft_seasons)
        if min(sft_seasons) <= max(tabular_seasons):
            raise OutcomeV2SftGateError("post-SFT report seasons are not strictly later")
        for field_name in _ARTIFACT_HASH_KEYS:
            _require_hash(_string(artifacts, field_name), f"artifacts.{field_name}")
        _nonnegative_integer(artifacts, "sft_failed_record_count")
        _require_hash(_string(evaluation, "policy_sha256"), "evaluation.policy_sha256")
        _require_value(
            evaluation,
            "mode",
            "retrospective_answer_held_holdout",
        )
        _require_value(
            evaluation,
            "tabular_gate_role",
            "sft_training_prerequisite_only",
        )
        _require_value(
            evaluation,
            "post_sft_gate_role",
            "sft_candidate_advancement",
        )
    except JsonFormatError as error:
        raise OutcomeV2SftGateError("invalid post-SFT report structure") from error
    return record


def _require_post_sft_proof_scope(proof_scope: Mapping[str, object]) -> None:
    expected = {
        "candidate_model_and_run_provenance": "verified_local_lock_chain",
        "external_precommit_and_timestamp": "required_separately",
        "remote_inference_attestation": "required_separately",
        "raw_provider_derivation": "required_separately",
        "pretraining_contamination": "possible_not_ruled_out",
        "rolling_prospective_proof": "not_satisfied",
        "durable_inference_journal": (
            "locally_sha256_bound_not_remote_execution_or_one_attempt_proof"
        ),
    }
    if proof_scope != expected:
        raise OutcomeV2SftGateError("post-SFT report proof scope is invalid")


def _canonical_record(value: bytes, description: str) -> JsonObject:
    try:
        text = value.decode("utf-8")
        record = parse_json_object(text)
    except (JsonFormatError, UnicodeError) as error:
        raise OutcomeV2SftGateError(f"{description} must be one UTF-8 JSON object") from error
    if text != canonical_json(record):
        raise OutcomeV2SftGateError(f"{description} must use canonical JSON bytes")
    return record


def _require_bytes(value: object, description: str) -> None:
    if not isinstance(value, bytes):
        raise OutcomeV2SftGateError(f"{description} requires immutable bytes")


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
        raise OutcomeV2SftGateError(f"cannot write {description}") from error


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise OutcomeV2SftGateError(f"cannot read {description}") from error


def _require_value(record: Mapping[str, object], field_name: str, expected: str) -> None:
    if _string(record, field_name) != expected:
        raise OutcomeV2SftGateError(f"unexpected {field_name}")


def _string(record: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = required_field(record, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutcomeV2SftGateError(f"{field_name} must be an integer")
    return value


def _positive_integer(record: Mapping[str, object], field_name: str) -> int:
    value = _integer(record, field_name)
    if value <= 0:
        raise OutcomeV2SftGateError(f"{field_name} must be positive")
    return value


def _nonnegative_integer(record: Mapping[str, object], field_name: str) -> int:
    value = _integer(record, field_name)
    if value < 0:
        raise OutcomeV2SftGateError(f"{field_name} must be non-negative")
    return value


def _integer_tuple(record: Mapping[str, object], field_name: str) -> tuple[int, ...]:
    values = require_list(required_field(record, field_name), field_name)
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise OutcomeV2SftGateError(f"{field_name} must contain integers")
        result.append(value)
    return tuple(result)


def _require_seasons(seasons: tuple[int, ...]) -> None:
    if len(seasons) < 2 or seasons != tuple(sorted(set(seasons))):
        raise OutcomeV2SftGateError("evaluation seasons must be unique and increasing")
    if any(season <= 0 for season in seasons):
        raise OutcomeV2SftGateError("evaluation seasons must be positive")


def _require_nonempty_seasons(seasons: tuple[int, ...]) -> None:
    if not seasons or seasons != tuple(sorted(set(seasons))):
        raise OutcomeV2SftGateError("evaluation seasons must be nonempty, unique, and increasing")
    if any(season <= 0 for season in seasons):
        raise OutcomeV2SftGateError("evaluation seasons must be positive")


def _require_hash(value: str, field_name: str) -> None:
    if _HASH_PATTERN.fullmatch(value) is None:
        raise OutcomeV2SftGateError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_tinker_path(value: str) -> None:
    suffix = value.removeprefix("tinker://")
    if (
        not value.startswith("tinker://")
        or not suffix
        or any(character.isspace() or ord(character) < 32 for character in value)
        or "?" in suffix
        or "#" in suffix
    ):
        raise OutcomeV2SftGateError("sampler_path must be a permanent tinker:// path")


def _parse_utc(value: str, field_name: str) -> datetime:
    if not value.endswith("Z"):
        raise OutcomeV2SftGateError(f"{field_name} must use canonical UTC Z notation")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise OutcomeV2SftGateError(f"{field_name} must be an ISO 8601 datetime") from error
    if _utc_text(parsed, field_name) != value:
        raise OutcomeV2SftGateError(f"{field_name} must use canonical UTC notation")
    return parsed.astimezone(UTC)


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise OutcomeV2SftGateError(f"{field_name} must be in UTC")


def _utc_text(value: datetime, field_name: str) -> str:
    _require_utc(value, field_name)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
