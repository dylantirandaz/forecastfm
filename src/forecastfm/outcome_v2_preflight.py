"""Offline readiness checks for an eventual outcome-v2 SFT run."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

from forecastfm.integrity import (
    bytes_sha256,
    canonical_json,
    canonical_sha256,
    file_sha256,
    text_sha256,
)
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_elo_replay import (
    NbaEloReplayError,
    NbaEloReplayRow,
    read_nba_elo_replay_rows_jsonl,
    validate_nba_elo_replay_states,
)
from forecastfm.nba_elo_state import (
    NbaEloState,
    NbaEloStateError,
    read_nba_elo_states_jsonl,
    validate_elo_states_against_feature_rows,
)
from forecastfm.nba_evaluation_gate import (
    NbaEvaluationAnswer,
    NbaEvaluationCohortInput,
    NbaEvaluationGateArtifacts,
    NbaEvaluationGateError,
    NbaEvaluationGateReport,
    NbaRecalibrationRow,
    read_nba_evaluation_answers_jsonl,
    read_nba_evaluation_cohort_jsonl,
    read_nba_evaluation_forecasts_jsonl,
    read_nba_recalibration_rows_jsonl,
    verify_untouched_nba_evaluation_gate,
)
from forecastfm.nba_evidence import NbaEvidenceBundle, NbaEvidenceError
from forecastfm.nba_evidence_io import (
    NbaEvidenceIoError,
    read_nba_evidence_bundles_jsonl,
    validate_tinker_feature_rows_from_bundles,
)
from forecastfm.nba_feature_rows import (
    NbaFeatureRowError,
    NbaRichFeatureRow,
    read_nba_feature_rows_jsonl,
)
from forecastfm.nba_resolutions import (
    NbaResolution,
    NbaResolutionError,
    read_nba_resolutions_jsonl,
    validate_outcome_training_labels,
)
from forecastfm.nba_rich import (
    NBA_RICH_FEATURE_NAMES,
    NBA_RICH_SCHEMA_SHA256,
    NBA_RICH_SCHEMA_VERSION,
)
from forecastfm.nba_rich_baseline import (
    NbaRichBaselineError,
    NbaRichBaselineModel,
    build_nba_rich_baseline_forecast_lock,
    fit_nba_rich_baseline,
    predict_nba_rich_baseline,
    read_nba_rich_baseline_forecast_lock,
    read_nba_rich_baseline_model,
)
from forecastfm.nba_rights_lock import (
    NbaRightsApprovalError,
    NbaRightsApprovalLock,
    load_nba_rights_approval_lock,
    require_approved_action,
    require_snapshot_index_rights,
)
from forecastfm.nba_snapshot_pack import (
    NbaSnapshotIndex,
    SnapshotPackError,
    load_snapshot_pack,
)
from forecastfm.outcome import OUTCOME_INPUT_SCHEMA_VERSION
from forecastfm.outcome_v2_config import (
    BATCH_SIZE,
    ELO_REPLAY_FILENAME,
    ELO_STATES_FILENAME,
    EVALUATION_ANSWERS_FILENAME,
    EVALUATION_COHORT_FILENAME,
    EVALUATION_ELO_REPLAY_FILENAME,
    EVALUATION_ELO_STATES_FILENAME,
    EVALUATION_FEATURE_ROWS_FILENAME,
    EVALUATION_FORECASTS_FILENAME,
    EVALUATION_REPORT_FILENAME,
    EVALUATION_RESOLUTIONS_FILENAME,
    EVIDENCE_BUNDLES_FILENAME,
    FEATURE_ROWS_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    RECALIBRATION_FILENAME,
    RESOLUTIONS_FILENAME,
    RICH_BASELINE_FORECAST_LOCK_FILENAME,
    RICH_BASELINE_MODEL_FILENAME,
    RIGHTS_LOCK_FILENAME,
    SEASONS_FILENAME,
    SNAPSHOT_PACK_FILENAME,
    TRAINING_FILENAME,
    outcome_v2_elo_recipe,
    outcome_v2_evaluation_policy,
    outcome_v2_rich_baseline_fit_config,
)
from forecastfm.outcome_v2_prompt import OUTCOME_V2_SYSTEM_PROMPT, build_outcome_v2_messages
from forecastfm.tinker_data import (
    OutcomeTrainingRecord,
    read_outcome_training_jsonl_bytes,
)
from forecastfm.tinker_screening import (
    TinkerScreeningError,
    require_text_health_screen_passes,
)

_SHA256_LENGTH = 64
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_SEASONS_SCHEMA_VERSION = 1
_SEASONS_KEYS = {"schema_version", "seasons"}
_SEASON_KEYS = {"question_id", "season"}

type _EvaluationAlignment = tuple[
    NbaEvaluationCohortInput,
    NbaEloReplayRow,
    NbaEloState,
    NbaRichFeatureRow,
    NbaResolution,
    NbaEvaluationAnswer,
]
type OutcomeV2ActionTimeSource = Literal[
    "caller_supplied_offline_check", "internal_paid_preparation"
]


class OutcomeV2PreflightError(ValueError):
    """Raised when an outcome-v2 artifact is not safe to hand to a trainer."""


@dataclass(frozen=True, slots=True)
class OutcomeV2Preflight:
    """The exact local artifact accepted by the offline gate."""

    manifest_sha256: str
    action_at: datetime
    action_time_source: OutcomeV2ActionTimeSource
    untouched_evaluation_seasons: tuple[int, ...]
    training_sha256: str
    feature_rows_sha256: str
    snapshot_pack_sha256: str
    evidence_bundles_sha256: str
    elo_states_sha256: str
    elo_replay_sha256: str
    seasons_sha256: str
    resolutions_sha256: str
    rights_lock_sha256: str
    evaluation_feature_rows_sha256: str
    evaluation_elo_replay_sha256: str
    evaluation_elo_states_sha256: str
    evaluation_resolutions_sha256: str
    calibration_sha256: str
    rich_baseline_model_sha256: str
    rich_baseline_forecast_lock_sha256: str
    evaluation_report_sha256: str
    row_count: int
    pair_count: int
    batch_size: int

    def canonical_payload(self) -> dict[str, object]:
        """Return the JSON-ready proof representation owned by this type."""
        _require_utc(self.action_at, "action_at")
        payload = cast(dict[str, object], asdict(self))
        payload["action_at"] = self.action_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
        payload["untouched_evaluation_seasons"] = list(self.untouched_evaluation_seasons)
        return payload


@dataclass(frozen=True, slots=True)
class OutcomeV2Artifacts:
    """All sealed local files required by a readiness-true SFT run."""

    feature_rows_path: Path
    snapshot_pack_path: Path
    evidence_bundles_path: Path
    elo_states_path: Path
    elo_replay_path: Path
    seasons_path: Path
    resolutions_path: Path
    rights_lock_path: Path
    agreement_path: Path
    evaluation: NbaEvaluationGateArtifacts
    evaluation_feature_rows_path: Path
    evaluation_elo_replay_path: Path
    evaluation_elo_states_path: Path
    evaluation_resolutions_path: Path
    rich_baseline_model_path: Path
    rich_baseline_forecast_lock_path: Path

    def sealed_paths(self) -> dict[str, Path]:
        """Map every manifest-bound filename to its exact local path."""
        report_path = self.evaluation.supplied_report_path
        if report_path is None:
            raise OutcomeV2PreflightError("the sealed NBA evaluation report is required")
        return {
            FEATURE_ROWS_FILENAME: self.feature_rows_path,
            SNAPSHOT_PACK_FILENAME: self.snapshot_pack_path,
            EVIDENCE_BUNDLES_FILENAME: self.evidence_bundles_path,
            ELO_STATES_FILENAME: self.elo_states_path,
            ELO_REPLAY_FILENAME: self.elo_replay_path,
            SEASONS_FILENAME: self.seasons_path,
            RESOLUTIONS_FILENAME: self.resolutions_path,
            RIGHTS_LOCK_FILENAME: self.rights_lock_path,
            EVALUATION_FEATURE_ROWS_FILENAME: self.evaluation_feature_rows_path,
            EVALUATION_ELO_REPLAY_FILENAME: self.evaluation_elo_replay_path,
            EVALUATION_ELO_STATES_FILENAME: self.evaluation_elo_states_path,
            EVALUATION_RESOLUTIONS_FILENAME: self.evaluation_resolutions_path,
            EVALUATION_COHORT_FILENAME: self.evaluation.cohort_path,
            EVALUATION_ANSWERS_FILENAME: self.evaluation.answers_path,
            EVALUATION_FORECASTS_FILENAME: self.evaluation.forecasts_path,
            RECALIBRATION_FILENAME: self.evaluation.calibration_path,
            RICH_BASELINE_MODEL_FILENAME: self.rich_baseline_model_path,
            RICH_BASELINE_FORECAST_LOCK_FILENAME: self.rich_baseline_forecast_lock_path,
            EVALUATION_REPORT_FILENAME: report_path,
        }


@dataclass(frozen=True, slots=True)
class PreparedOutcomeV2Run:
    """One passing proof paired with the exact immutable paid-training bytes."""

    proof: OutcomeV2Preflight
    training_jsonl: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if self.proof.action_time_source != "internal_paid_preparation":
            raise OutcomeV2PreflightError(
                "a prepared paid run requires an internally derived action time"
            )
        if bytes_sha256(self.training_jsonl) != self.proof.training_sha256:
            raise OutcomeV2PreflightError("prepared training bytes differ from the preflight proof")
        if self.proof.batch_size != BATCH_SIZE:
            raise OutcomeV2PreflightError(
                "prepared batch size differs from frozen outcome-v2 config"
            )


@dataclass(frozen=True, slots=True)
class _ProvenanceInputs:
    records: tuple[OutcomeTrainingRecord, ...]
    feature_rows: tuple[NbaRichFeatureRow, ...]
    frozen_seasons: dict[str, int]


def require_outcome_v2_sft_ready(
    manifest_path: Path,
    training_path: Path,
    artifacts: OutcomeV2Artifacts | None = None,
    *,
    action_at: datetime | None = None,
) -> OutcomeV2Preflight:
    """Validate the complete local SFT boundary without importing Tinker."""
    proof, _ = _validate_outcome_v2_sft_run(
        manifest_path,
        training_path,
        artifacts,
        action_at=action_at,
        derive_action_at=False,
    )
    return proof


def prepare_outcome_v2_sft_run(
    manifest_path: Path,
    training_path: Path,
    artifacts: OutcomeV2Artifacts | None = None,
) -> PreparedOutcomeV2Run:
    """Derive the action time and retain the exact bytes accepted for a paid run."""
    proof, training_jsonl = _validate_outcome_v2_sft_run(
        manifest_path,
        training_path,
        artifacts,
        action_at=None,
        derive_action_at=True,
    )
    return PreparedOutcomeV2Run(proof=proof, training_jsonl=training_jsonl)


def _validate_outcome_v2_sft_run(
    manifest_path: Path,
    training_path: Path,
    artifacts: OutcomeV2Artifacts | None,
    *,
    action_at: datetime | None,
    derive_action_at: bool,
) -> tuple[OutcomeV2Preflight, bytes]:
    manifest, manifest_sha256 = _read_manifest(manifest_path)
    _require_schema_versions(manifest)
    untouched_seasons = _require_full_readiness(manifest)
    checked_artifacts = _require_artifact_set(artifacts)
    _require_requested_action_time(action_at, derive_action_at=derive_action_at)
    _require_feature_schema(manifest)

    training_jsonl = _read_training_bytes(training_path)
    training_sha256 = bytes_sha256(training_jsonl)
    artifact_hashes = _verify_artifact_hashes(
        manifest,
        training_path,
        checked_artifacts,
        training_sha256=training_sha256,
    )
    protected_action_at = _protected_action_time(action_at, derive_action_at=derive_action_at)
    approval = _load_reviewed_rights(checked_artifacts)
    _require_upload_rights(manifest, approval)

    feature_rows = _read_feature_rows(checked_artifacts.feature_rows_path)
    records = _read_training_records(training_jsonl)
    _require_health_screen(records)
    original_ids = _require_exact_pairs(records)
    _require_row_contract(manifest, len(records), len(original_ids))
    _require_exact_original_id_order(manifest, original_ids)
    _require_feature_row_binding(records, feature_rows, original_ids)
    frozen_seasons = _read_frozen_seasons(checked_artifacts.seasons_path, original_ids)
    _require_disjoint_training_seasons(frozen_seasons, untouched_seasons)
    evaluation_report = _require_complete_provenance(
        checked_artifacts,
        approval,
        _ProvenanceInputs(records, feature_rows, frozen_seasons),
        untouched_seasons=untouched_seasons,
        action_at=protected_action_at,
    )
    _require_evaluation_report_hashes(evaluation_report, artifact_hashes)
    _require_reviewed_external_proofs()
    final_hashes = _verify_artifact_hashes(
        manifest,
        training_path,
        checked_artifacts,
        training_sha256=training_sha256,
    )
    _require_same_hashes(artifact_hashes, final_hashes)
    _require_unchanged_file(
        manifest_path,
        manifest_sha256,
        "outcome-v2 manifest",
        "outcome-v2 manifest changed during preflight",
    )
    _require_unchanged_file(
        training_path,
        training_sha256,
        "outcome-v2 training data",
        "outcome-v2 training path changed during preflight",
    )
    _require_unchanged_file(
        checked_artifacts.agreement_path,
        approval.agreement_sha256,
        "reviewed agreement",
        "reviewed agreement changed during preflight",
    )

    proof = OutcomeV2Preflight(
        manifest_sha256=manifest_sha256,
        action_at=protected_action_at,
        action_time_source=(
            "internal_paid_preparation" if derive_action_at else "caller_supplied_offline_check"
        ),
        untouched_evaluation_seasons=untouched_seasons,
        training_sha256=artifact_hashes[TRAINING_FILENAME],
        feature_rows_sha256=artifact_hashes[FEATURE_ROWS_FILENAME],
        snapshot_pack_sha256=artifact_hashes[SNAPSHOT_PACK_FILENAME],
        evidence_bundles_sha256=artifact_hashes[EVIDENCE_BUNDLES_FILENAME],
        elo_states_sha256=artifact_hashes[ELO_STATES_FILENAME],
        elo_replay_sha256=artifact_hashes[ELO_REPLAY_FILENAME],
        seasons_sha256=artifact_hashes[SEASONS_FILENAME],
        resolutions_sha256=artifact_hashes[RESOLUTIONS_FILENAME],
        rights_lock_sha256=artifact_hashes[RIGHTS_LOCK_FILENAME],
        evaluation_feature_rows_sha256=artifact_hashes[EVALUATION_FEATURE_ROWS_FILENAME],
        evaluation_elo_replay_sha256=artifact_hashes[EVALUATION_ELO_REPLAY_FILENAME],
        evaluation_elo_states_sha256=artifact_hashes[EVALUATION_ELO_STATES_FILENAME],
        evaluation_resolutions_sha256=artifact_hashes[EVALUATION_RESOLUTIONS_FILENAME],
        calibration_sha256=artifact_hashes[RECALIBRATION_FILENAME],
        rich_baseline_model_sha256=artifact_hashes[RICH_BASELINE_MODEL_FILENAME],
        rich_baseline_forecast_lock_sha256=artifact_hashes[RICH_BASELINE_FORECAST_LOCK_FILENAME],
        evaluation_report_sha256=evaluation_report.sha256,
        row_count=len(records),
        pair_count=len(original_ids),
        batch_size=BATCH_SIZE,
    )
    return proof, training_jsonl


def _require_artifact_set(
    artifacts: OutcomeV2Artifacts | None,
) -> OutcomeV2Artifacts:
    if artifacts is None:
        raise OutcomeV2PreflightError("the complete sealed NBA artifact set is required")
    return artifacts


def _require_requested_action_time(
    action_at: datetime | None,
    *,
    derive_action_at: bool,
) -> None:
    if action_at is None:
        if not derive_action_at:
            raise OutcomeV2PreflightError("the protected Tinker action time is required")
        return
    _require_utc(action_at, "action_at")
    if action_at > _utc_now():
        raise OutcomeV2PreflightError("the protected Tinker action time cannot be in the future")


def _protected_action_time(
    requested: datetime | None,
    *,
    derive_action_at: bool,
) -> datetime:
    protected = _utc_now() if derive_action_at else requested
    if protected is None:
        raise OutcomeV2PreflightError("the protected Tinker action time is required")
    _require_utc(protected, "action_at")
    return protected


def _require_disjoint_training_seasons(
    frozen_seasons: dict[str, int],
    untouched_seasons: tuple[int, ...],
) -> None:
    if set(frozen_seasons.values()) & set(untouched_seasons):
        raise OutcomeV2PreflightError(
            "training seasons overlap the declared untouched evaluation seasons"
        )


def _require_reviewed_external_proofs() -> None:
    """Fail closed until production connector and pre-event receipt verifiers exist."""
    raise OutcomeV2PreflightError(
        "outcome-v2 has no reviewed production connector and pre-event commitment verifier"
    )


def _require_evaluation_report_hashes(
    report: NbaEvaluationGateReport,
    artifact_hashes: dict[str, str],
) -> None:
    try:
        payload = parse_json_object(report.canonical_text)
        reported = require_object(required_field(payload, "artifacts"), "artifacts")
        require_exact_keys(
            reported,
            {
                "cohort_sha256",
                "answers_sha256",
                "forecasts_sha256",
                "calibration_sha256",
            },
            "evaluation report artifacts",
        )
        expected = {
            "cohort_sha256": artifact_hashes[EVALUATION_COHORT_FILENAME],
            "answers_sha256": artifact_hashes[EVALUATION_ANSWERS_FILENAME],
            "forecasts_sha256": artifact_hashes[EVALUATION_FORECASTS_FILENAME],
            "calibration_sha256": artifact_hashes[RECALIBRATION_FILENAME],
        }
    except JsonFormatError as error:
        raise OutcomeV2PreflightError("evaluation report artifact hashes are invalid") from error
    if reported != expected:
        raise OutcomeV2PreflightError(
            "evaluation report hashes differ from the initially sealed artifacts"
        )


def _require_same_hashes(
    initial_hashes: dict[str, str],
    final_hashes: dict[str, str],
) -> None:
    if final_hashes != initial_hashes:
        raise OutcomeV2PreflightError("sealed NBA artifacts changed during preflight")


def _require_unchanged_file(
    path: Path,
    expected_sha256: str,
    description: str,
    error_message: str,
) -> None:
    if _file_hash(path, description) != expected_sha256:
        raise OutcomeV2PreflightError(error_message)


def _read_manifest(path: Path) -> tuple[dict[str, object], str]:
    try:
        text = path.read_bytes().decode("utf-8")
        return parse_json_object(text), text_sha256(text)
    except (JsonFormatError, OSError, UnicodeError) as error:
        raise OutcomeV2PreflightError("cannot read a valid outcome-v2 manifest") from error


def _read_training_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise OutcomeV2PreflightError("cannot read outcome-v2 training bytes") from error


def _read_training_records(data: bytes) -> tuple[OutcomeTrainingRecord, ...]:
    try:
        return read_outcome_training_jsonl_bytes(
            data,
            expected_system_prompt=OUTCOME_V2_SYSTEM_PROMPT,
        )
    except (JsonFormatError, UnicodeError) as error:
        raise OutcomeV2PreflightError("cannot read valid outcome-v2 training rows") from error


def _read_feature_rows(path: Path) -> tuple[NbaRichFeatureRow, ...]:
    try:
        return read_nba_feature_rows_jsonl(path)
    except NbaFeatureRowError as error:
        raise OutcomeV2PreflightError("cannot read valid sealed NBA feature rows") from error


def _require_schema_versions(manifest: dict[str, object]) -> None:
    manifest_version = _integer_field(manifest, "schema_version")
    if manifest_version != MANIFEST_SCHEMA_VERSION:
        raise OutcomeV2PreflightError("unsupported outcome-v2 manifest schema_version")

    input_version = _integer_field(manifest, "outcome_input_schema_version")
    if input_version != OUTCOME_INPUT_SCHEMA_VERSION:
        raise OutcomeV2PreflightError("unsupported outcome-v2 input schema version")


def _require_full_readiness(manifest: dict[str, object]) -> tuple[int, ...]:
    evaluation = _object_field(manifest, "evaluation")
    ready = _boolean_field(evaluation, "full_outcome_v2_ready")
    if not ready:
        raise OutcomeV2PreflightError("full_outcome_v2_ready is false")
    missing = _string_tuple_field(evaluation, "full_outcome_v2_missing")
    if missing:
        raise OutcomeV2PreflightError("full_outcome_v2_missing must be empty")
    untouched_seasons = _integer_tuple_field(evaluation, "untouched_evaluation_seasons")
    if len(untouched_seasons) < 2 or len(set(untouched_seasons)) != len(untouched_seasons):
        raise OutcomeV2PreflightError(
            "at least two unique untouched evaluation seasons are required"
        )
    return untouched_seasons


def _require_upload_rights(
    manifest: dict[str, object],
    approval: NbaRightsApprovalLock,
) -> None:
    rights = _object_field(manifest, "upload_rights")
    if _boolean_field(rights, "player_health_included"):
        raise OutcomeV2PreflightError("standard Tinker training cannot include player health")
    try:
        require_approved_action(approval, "tinker_processing")
    except NbaRightsApprovalError as error:
        message = f"reviewed rights do not allow Tinker processing: {error}"
        raise OutcomeV2PreflightError(message) from error
    reviewed_permissions = {
        "third_party_processing": approval.third_party_processing,
        "tinker_processing": approval.tinker_processing,
    }
    for field_name, reviewed_permission in reviewed_permissions.items():
        manifest_permission = _string_field(rights, field_name)
        if manifest_permission != reviewed_permission:
            message = f"upload_rights.{field_name} differs from the reviewed rights lock"
            raise OutcomeV2PreflightError(message)


def _load_reviewed_rights(artifacts: OutcomeV2Artifacts) -> NbaRightsApprovalLock:
    try:
        approval = load_nba_rights_approval_lock(
            artifacts.rights_lock_path,
            artifacts.agreement_path,
        )
    except NbaRightsApprovalError as error:
        message = f"cannot verify reviewed NBA rights artifacts: {error}"
        raise OutcomeV2PreflightError(message) from error
    return approval


def _verify_artifact_hashes(
    manifest: dict[str, object],
    training_path: Path,
    artifacts: OutcomeV2Artifacts,
    *,
    training_sha256: str,
) -> dict[str, str]:
    hashes = {
        filename: _verify_output_hash(manifest, path, filename)
        for filename, path in artifacts.sealed_paths().items()
    }
    hashes[TRAINING_FILENAME] = _verify_captured_training_hash(
        manifest,
        training_path,
        training_sha256,
    )
    return hashes


def _verify_captured_training_hash(
    manifest: dict[str, object],
    path: Path,
    actual_sha256: str,
) -> str:
    if path.name != TRAINING_FILENAME:
        raise OutcomeV2PreflightError(f"artifact file must be named {TRAINING_FILENAME}")
    outputs = _object_field(manifest, "outputs")
    expected_sha256 = _string_field(outputs, TRAINING_FILENAME)
    _require_sha256(expected_sha256, f"outputs.{TRAINING_FILENAME}")
    if actual_sha256 != expected_sha256:
        raise OutcomeV2PreflightError(f"{TRAINING_FILENAME} SHA-256 does not match the manifest")
    return actual_sha256


def _verify_output_hash(
    manifest: dict[str, object],
    path: Path,
    expected_filename: str,
) -> str:
    if path.name != expected_filename:
        raise OutcomeV2PreflightError(f"artifact file must be named {expected_filename}")
    outputs = _object_field(manifest, "outputs")
    expected_sha256 = _string_field(outputs, expected_filename)
    _require_sha256(expected_sha256, f"outputs.{expected_filename}")
    actual_sha256 = _file_hash(path, f"sealed artifact {expected_filename}")
    if actual_sha256 != expected_sha256:
        raise OutcomeV2PreflightError(f"{expected_filename} SHA-256 does not match the manifest")
    return actual_sha256


def _file_hash(path: Path, description: str) -> str:
    try:
        return file_sha256(path)
    except OSError as error:
        raise OutcomeV2PreflightError(f"cannot read {description}") from error


def _require_complete_provenance(
    artifacts: OutcomeV2Artifacts,
    approval: NbaRightsApprovalLock,
    inputs: _ProvenanceInputs,
    *,
    untouched_seasons: tuple[int, ...],
    action_at: datetime,
) -> NbaEvaluationGateReport:
    try:
        snapshot_index = load_snapshot_pack(artifacts.snapshot_pack_path)
        require_snapshot_index_rights(
            snapshot_index,
            approval,
            action="tinker_processing",
            action_at=action_at,
        )
        bundles = read_nba_evidence_bundles_jsonl(
            artifacts.evidence_bundles_path,
            snapshot_index=snapshot_index,
        )
        validate_tinker_feature_rows_from_bundles(
            bundles,
            inputs.feature_rows,
            inputs.frozen_seasons,
            action_at=action_at,
        )
        replay_rows = read_nba_elo_replay_rows_jsonl(artifacts.elo_replay_path)
        _validate_replay_rows_against_training(
            replay_rows,
            bundles,
            inputs.frozen_seasons,
        )
        elo_states = read_nba_elo_states_jsonl(artifacts.elo_states_path)
        validate_elo_states_against_feature_rows(
            elo_states,
            inputs.feature_rows,
            action_at=action_at,
        )
        resolutions = read_nba_resolutions_jsonl(
            artifacts.resolutions_path,
            snapshot_index=snapshot_index,
        )
        validate_nba_elo_replay_states(
            replay_rows,
            resolutions,
            outcome_v2_elo_recipe(),
            elo_states,
        )
        validate_outcome_training_labels(
            bundles,
            resolutions,
            inputs.records,
            snapshot_index=snapshot_index,
            action_at=action_at,
        )
        rich_baseline_model = _require_exact_rich_baseline_model(
            artifacts,
            inputs.feature_rows,
            resolutions,
        )
        calibration_rows = read_nba_recalibration_rows_jsonl(artifacts.evaluation.calibration_path)
        _validate_recalibration_against_training(
            calibration_rows,
            replay_rows,
            elo_states,
            resolutions,
            inputs.frozen_seasons,
        )
        evaluation_cohort = read_nba_evaluation_cohort_jsonl(artifacts.evaluation.cohort_path)
        _require_declared_evaluation_seasons(evaluation_cohort, untouched_seasons)
        _require_evaluation_lineage(
            artifacts,
            snapshot_index,
            evaluation_cohort,
            rich_baseline_model,
            action_at=action_at,
        )
        return verify_untouched_nba_evaluation_gate(
            artifacts.evaluation,
            policy=outcome_v2_evaluation_policy(),
        )
    except (
        NbaEloReplayError,
        NbaEloStateError,
        NbaEvaluationGateError,
        NbaEvidenceError,
        NbaEvidenceIoError,
        NbaFeatureRowError,
        NbaResolutionError,
        NbaRichBaselineError,
        NbaRightsApprovalError,
        SnapshotPackError,
    ) as error:
        raise OutcomeV2PreflightError(f"NBA provenance validation failed: {error}") from error


def _validate_replay_rows_against_training(
    rows: tuple[NbaEloReplayRow, ...],
    bundles: tuple[NbaEvidenceBundle, ...],
    seasons: dict[str, int],
) -> None:
    row_ids = tuple(row.question_id for row in rows)
    bundle_ids = tuple(bundle.game.question_id for bundle in bundles)
    if row_ids != bundle_ids:
        raise NbaEloReplayError(
            "Elo replay rows and training bundles must have identical IDs and order"
        )
    for row, bundle in zip(rows, bundles, strict=True):
        if row.source_game_id != bundle.game.source_game_id:
            raise NbaEloReplayError("Elo replay source_game_id differs from its training bundle")
        if row.forecast_cutoff != bundle.game.forecast_deadline:
            raise NbaEloReplayError("Elo replay cutoff differs from its training bundle")
        if row.scheduled_tipoff != bundle.game.scheduled_tipoff:
            raise NbaEloReplayError("Elo replay tipoff differs from its training bundle")
        if row.season != seasons[row.question_id]:
            raise NbaEloReplayError("Elo replay season differs from the frozen training season")


def _validate_recalibration_against_training(
    rows: tuple[NbaRecalibrationRow, ...],
    replay_rows: tuple[NbaEloReplayRow, ...],
    elo_states: tuple[NbaEloState, ...],
    resolutions: tuple[NbaResolution, ...],
    seasons: dict[str, int],
) -> None:
    expected_ids = tuple(row.question_id for row in replay_rows)
    if tuple(row.question_id for row in rows) != expected_ids:
        raise NbaEvaluationGateError(
            "recalibration rows must exactly match the training IDs and order"
        )
    for row, replay, state, resolution in zip(
        rows,
        replay_rows,
        elo_states,
        resolutions,
        strict=True,
    ):
        if row.season != seasons[row.question_id]:
            raise NbaEvaluationGateError(
                "recalibration season differs from the frozen training season"
            )
        if row.game_date != replay.scheduled_tipoff.date():
            raise NbaEvaluationGateError("recalibration date differs from the Elo replay")
        if row.raw_elo_team_probability != state.team_win_probability:
            raise NbaEvaluationGateError("recalibration Elo probability differs from training")
        if row.realized_team_win != resolution.team_won:
            raise NbaEvaluationGateError("recalibration answer differs from the sealed resolution")


def _require_declared_evaluation_seasons(
    cohort: tuple[NbaEvaluationCohortInput, ...],
    declared_seasons: tuple[int, ...],
) -> None:
    actual_seasons = tuple(sorted({row.season for row in cohort}))
    if actual_seasons != declared_seasons:
        raise NbaEvaluationGateError(
            "sealed evaluation seasons differ from the manifest declaration"
        )


def _require_exact_rich_baseline_model(
    artifacts: OutcomeV2Artifacts,
    feature_rows: tuple[NbaRichFeatureRow, ...],
    resolutions: tuple[NbaResolution, ...],
) -> NbaRichBaselineModel:
    expected = fit_nba_rich_baseline(
        feature_rows,
        resolutions,
        outcome_v2_rich_baseline_fit_config(),
    )
    supplied = read_nba_rich_baseline_model(artifacts.rich_baseline_model_path)
    if supplied.canonical_bytes != expected.canonical_bytes:
        raise NbaRichBaselineError(
            "sealed rich baseline model differs from deterministic training replay"
        )
    return supplied


def _require_evaluation_lineage(
    artifacts: OutcomeV2Artifacts,
    snapshot_index: NbaSnapshotIndex,
    cohort: tuple[NbaEvaluationCohortInput, ...],
    model: NbaRichBaselineModel,
    *,
    action_at: datetime,
) -> None:
    """Verify sealed relationships; raw provider-byte derivation remains external."""
    replay_rows = read_nba_elo_replay_rows_jsonl(artifacts.evaluation_elo_replay_path)
    states = read_nba_elo_states_jsonl(artifacts.evaluation_elo_states_path)
    feature_rows = read_nba_feature_rows_jsonl(artifacts.evaluation_feature_rows_path)
    resolutions = read_nba_resolutions_jsonl(
        artifacts.evaluation_resolutions_path,
        snapshot_index=snapshot_index,
    )
    answers = read_nba_evaluation_answers_jsonl(artifacts.evaluation.answers_path)
    forecasts = read_nba_evaluation_forecasts_jsonl(artifacts.evaluation.forecasts_path)
    validate_nba_elo_replay_states(
        replay_rows,
        resolutions,
        outcome_v2_elo_recipe(),
        states,
    )
    validate_elo_states_against_feature_rows(
        states,
        feature_rows,
        action_at=action_at,
    )
    cohort_ids = tuple(row.question_id for row in cohort)
    _require_evaluation_lineage_ids(
        cohort_ids,
        replay_rows,
        feature_rows,
        resolutions,
        answers,
    )
    for aligned in zip(
        cohort,
        replay_rows,
        states,
        feature_rows,
        resolutions,
        answers,
        strict=True,
    ):
        _require_evaluation_alignment(aligned, action_at=action_at)
    expected_forecasts = predict_nba_rich_baseline(model, feature_rows)
    if forecasts != expected_forecasts:
        raise NbaRichBaselineError(
            "sealed evaluation forecasts differ from deterministic rich baseline inference"
        )
    expected_lock = build_nba_rich_baseline_forecast_lock(
        model,
        feature_rows,
        expected_forecasts,
    )
    supplied_lock = read_nba_rich_baseline_forecast_lock(artifacts.rich_baseline_forecast_lock_path)
    if supplied_lock.canonical_bytes != expected_lock.canonical_bytes:
        raise NbaRichBaselineError(
            "sealed rich baseline forecast lock differs from deterministic inference"
        )


def _require_evaluation_lineage_ids(
    cohort_ids: tuple[str, ...],
    replay_rows: tuple[NbaEloReplayRow, ...],
    feature_rows: tuple[NbaRichFeatureRow, ...],
    resolutions: tuple[NbaResolution, ...],
    answers: tuple[NbaEvaluationAnswer, ...],
) -> None:
    if tuple(row.question_id for row in replay_rows) != cohort_ids:
        raise NbaEvaluationGateError("evaluation replay IDs or order differ from the cohort")
    if tuple(row.question_id for row in feature_rows) != cohort_ids:
        raise NbaEvaluationGateError("evaluation feature IDs or order differ from the cohort")
    if tuple(row.question_id for row in resolutions) != cohort_ids:
        raise NbaEvaluationGateError("evaluation resolution IDs or order differ from the cohort")
    if tuple(row.question_id for row in answers) != cohort_ids:
        raise NbaEvaluationGateError("evaluation answer IDs or order differ from the cohort")


def _require_evaluation_alignment(
    aligned: _EvaluationAlignment,
    *,
    action_at: datetime,
) -> None:
    member, replay, state, feature, resolution, answer = aligned
    if member.season != replay.season or member.season != feature.season:
        raise NbaEvaluationGateError("evaluation season differs from the Elo replay")
    if member.game_date != replay.scheduled_tipoff.date() or (
        member.game_date != feature.scheduled_tipoff.date()
    ):
        raise NbaEvaluationGateError("evaluation date differs from the Elo replay")
    if feature.forecast_cutoff != replay.forecast_cutoff:
        raise NbaEvaluationGateError("evaluation feature cutoff differs from the Elo replay")
    if feature.scheduled_tipoff != replay.scheduled_tipoff:
        raise NbaEvaluationGateError("evaluation feature tipoff differs from the Elo replay")
    if member.raw_elo_team_probability != state.team_win_probability:
        raise NbaEvaluationGateError("evaluation raw Elo differs from the replayed state")
    if answer.realized_team_win != resolution.team_won:
        raise NbaEvaluationGateError("evaluation answer differs from the sealed resolution")
    if resolution.resolved_at > action_at:
        raise NbaEvaluationGateError("evaluation resolution postdates the protected action")


def _read_frozen_seasons(
    path: Path,
    original_ids: tuple[str, ...],
) -> dict[str, int]:
    try:
        text = path.read_text(encoding="utf-8")
        payload = parse_json_object(text)
        require_exact_keys(payload, _SEASONS_KEYS, "frozen seasons")
        version = required_field(payload, "schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise JsonFormatError("season schema_version must be an integer")
        if version != _SEASONS_SCHEMA_VERSION:
            raise JsonFormatError(f"unsupported season schema version: {version}")
        raw_seasons = require_list(required_field(payload, "seasons"), "seasons")
        seasons = tuple(_season_from_payload(item) for item in raw_seasons)
    except (JsonFormatError, OSError, UnicodeError) as error:
        raise OutcomeV2PreflightError("cannot read the frozen NBA season mapping") from error
    if text != canonical_json(payload):
        raise OutcomeV2PreflightError("frozen NBA seasons must use canonical JSON encoding")
    season_ids = tuple(question_id for question_id, _ in seasons)
    if season_ids != original_ids:
        raise OutcomeV2PreflightError("frozen season IDs or order differ from the training rows")
    return dict(seasons)


def _season_from_payload(value: object) -> tuple[str, int]:
    record = require_object(value, "season")
    require_exact_keys(record, _SEASON_KEYS, "season")
    question_id = require_string(required_field(record, "question_id"), "question_id")
    season = required_field(record, "season")
    if isinstance(season, bool) or not isinstance(season, int) or season <= 0:
        raise JsonFormatError("season must be a positive integer")
    return question_id, season


def _require_feature_schema(manifest: dict[str, object]) -> None:
    features = _object_field(manifest, "features")
    full_schema = _object_field(features, "full_schema")
    if _integer_field(full_schema, "version") != NBA_RICH_SCHEMA_VERSION:
        raise OutcomeV2PreflightError("outcome-v2 richer feature schema version differs")
    schema_hash = _string_field(full_schema, "sha256")
    _require_sha256(schema_hash, "features.full_schema.sha256")
    if schema_hash != NBA_RICH_SCHEMA_SHA256:
        raise OutcomeV2PreflightError("outcome-v2 richer feature schema hash differs")
    standard_names = _string_tuple_field(full_schema, "standard_names")
    if standard_names != NBA_RICH_FEATURE_NAMES:
        raise OutcomeV2PreflightError("outcome-v2 richer feature names or order differ")
    if not _boolean_field(full_schema, "current_artifact_contains_full_schema"):
        raise OutcomeV2PreflightError("current artifact does not contain the richer feature schema")


def _require_exact_pairs(records: tuple[OutcomeTrainingRecord, ...]) -> tuple[str, ...]:
    if not records:
        raise OutcomeV2PreflightError("training file must contain at least one side-swap pair")
    if len(records) % 2 != 0:
        raise OutcomeV2PreflightError("training rows contain an incomplete side-swap pair")

    original_ids: list[str] = []
    seen_ids: set[str] = set()
    for index in range(0, len(records), 2):
        original_id, swapped_id = _require_pair(records[index], records[index + 1], seen_ids)
        original_ids.append(original_id)
        seen_ids.update((original_id, swapped_id))
    return tuple(original_ids)


def _require_health_screen(records: tuple[OutcomeTrainingRecord, ...]) -> None:
    texts = (
        text
        for record in records
        for text in (
            record["question_id"],
            *(message["content"] for message in record["messages"]),
        )
    )
    try:
        require_text_health_screen_passes(texts)
    except TinkerScreeningError as error:
        raise OutcomeV2PreflightError("training rows fail the health-language screen") from error


def _require_pair(
    original: OutcomeTrainingRecord,
    swapped: OutcomeTrainingRecord,
    seen_ids: set[str],
) -> tuple[str, str]:
    original_id = original["question_id"]
    swapped_id = swapped["question_id"]
    if original_id.endswith(SIDE_SWAP_SUFFIX):
        raise OutcomeV2PreflightError("each pair must put the original row first")
    if swapped_id != f"{original_id}{SIDE_SWAP_SUFFIX}":
        raise OutcomeV2PreflightError("training rows are not exact adjacent side-swap pairs")
    if original_id in seen_ids or swapped_id in seen_ids:
        raise OutcomeV2PreflightError("training rows contain a duplicate question ID")
    if original["label"] == swapped["label"]:
        raise OutcomeV2PreflightError("side-swap pair labels must be complementary")
    return original_id, swapped_id


def _require_row_contract(
    manifest: dict[str, object],
    row_count: int,
    pair_count: int,
) -> None:
    train = _training_split(manifest)
    expected_rows = _integer_field(train, "side_swapped_training_rows")
    expected_pairs = _integer_field(train, "original_games")
    if row_count != expected_rows or pair_count != expected_pairs:
        raise OutcomeV2PreflightError("training row counts do not match the manifest")


def _require_exact_original_id_order(
    manifest: dict[str, object],
    original_ids: tuple[str, ...],
) -> None:
    train = _training_split(manifest)
    expected_hash = _string_field(train, "question_ids_sha256")
    _require_sha256(expected_hash, "splits.train.question_ids_sha256")
    if canonical_sha256(list(original_ids)) != expected_hash:
        raise OutcomeV2PreflightError(
            "training question IDs or their order differ from the manifest"
        )


def _require_feature_row_binding(
    records: tuple[OutcomeTrainingRecord, ...],
    feature_rows: tuple[NbaRichFeatureRow, ...],
    original_ids: tuple[str, ...],
) -> None:
    feature_ids = tuple(row.question_id for row in feature_rows)
    if feature_ids != original_ids:
        raise OutcomeV2PreflightError(
            "sealed feature-row IDs, order, or count differ from the training rows"
        )
    for pair_index, row in enumerate(feature_rows):
        record_index = pair_index * 2
        _require_prompt_matches_feature_row(records[record_index], row)
        _require_prompt_matches_feature_row(records[record_index + 1], row.side_swap())


def _require_prompt_matches_feature_row(
    record: OutcomeTrainingRecord,
    row: NbaRichFeatureRow,
) -> None:
    if record["question_id"] != row.question_id:
        raise OutcomeV2PreflightError("training prompt ID differs from its sealed feature row")
    if tuple(record["messages"]) != build_outcome_v2_messages(row):
        raise OutcomeV2PreflightError("training prompt content differs from its sealed feature row")


def _training_split(manifest: dict[str, object]) -> dict[str, object]:
    splits = _object_field(manifest, "splits")
    return _object_field(splits, "train")


def _object_field(mapping: dict[str, object], field_name: str) -> dict[str, object]:
    try:
        return require_object(required_field(mapping, field_name), field_name)
    except JsonFormatError as error:
        raise OutcomeV2PreflightError(f"manifest field {field_name} must be an object") from error


def _string_field(mapping: dict[str, object], field_name: str) -> str:
    try:
        return require_string(required_field(mapping, field_name), field_name)
    except JsonFormatError as error:
        raise OutcomeV2PreflightError(f"manifest field {field_name} must be a string") from error


def _string_tuple_field(mapping: dict[str, object], field_name: str) -> tuple[str, ...]:
    try:
        values = require_list(required_field(mapping, field_name), field_name)
        return tuple(require_string(value, field_name) for value in values)
    except JsonFormatError as error:
        raise OutcomeV2PreflightError(
            f"manifest field {field_name} must be a list of strings"
        ) from error


def _integer_tuple_field(mapping: dict[str, object], field_name: str) -> tuple[int, ...]:
    try:
        values = require_list(required_field(mapping, field_name), field_name)
    except JsonFormatError as error:
        raise OutcomeV2PreflightError(
            f"manifest field {field_name} must be a list of integers"
        ) from error
    integers: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise OutcomeV2PreflightError(
                f"manifest field {field_name} must contain positive integers"
            )
        integers.append(value)
    return tuple(integers)


def _boolean_field(mapping: dict[str, object], field_name: str) -> bool:
    try:
        value = required_field(mapping, field_name)
    except JsonFormatError as error:
        raise OutcomeV2PreflightError(f"manifest field {field_name} must be a boolean") from error
    if not isinstance(value, bool):
        raise OutcomeV2PreflightError(f"manifest field {field_name} must be a boolean")
    return value


def _integer_field(mapping: dict[str, object], field_name: str) -> int:
    try:
        value = required_field(mapping, field_name)
    except JsonFormatError as error:
        raise OutcomeV2PreflightError(f"manifest field {field_name} must be an integer") from error
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutcomeV2PreflightError(f"manifest field {field_name} must be an integer")
    return value


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        character not in _SHA256_CHARACTERS for character in value
    ):
        raise OutcomeV2PreflightError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise OutcomeV2PreflightError(f"{field_name} must be in UTC")


def _utc_now() -> datetime:
    return datetime.now(UTC)
