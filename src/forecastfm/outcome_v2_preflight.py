"""Offline readiness checks for an eventual outcome-v2 SFT run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from forecastfm.integrity import canonical_json, canonical_sha256, file_sha256
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
from forecastfm.nba_elo_state import (
    NbaEloStateError,
    read_nba_elo_states_jsonl,
    validate_elo_states_against_feature_rows,
)
from forecastfm.nba_evidence import NbaEvidenceError
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
    NbaResolutionError,
    read_nba_resolutions_jsonl,
    validate_outcome_training_labels,
)
from forecastfm.nba_rich import (
    NBA_RICH_FEATURE_NAMES,
    NBA_RICH_SCHEMA_SHA256,
    NBA_RICH_SCHEMA_VERSION,
)
from forecastfm.nba_rights_lock import (
    NbaRightsApprovalError,
    NbaRightsApprovalLock,
    load_nba_rights_approval_lock,
    require_approved_action,
    require_snapshot_index_rights,
)
from forecastfm.nba_snapshot_pack import SnapshotPackError, load_snapshot_pack
from forecastfm.outcome import OPPONENT_OUTCOME, OUTCOME_INPUT_SCHEMA_VERSION, TEAM_OUTCOME
from forecastfm.outcome_v2_config import (
    BATCH_SIZE,
    ELO_STATES_FILENAME,
    EVIDENCE_BUNDLES_FILENAME,
    FEATURE_ROWS_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    QUESTION_TEXT,
    RESOLUTION_RULE,
    RESOLUTIONS_FILENAME,
    RIGHTS_LOCK_FILENAME,
    SEASONS_FILENAME,
    SNAPSHOT_PACK_FILENAME,
    TRAINING_FILENAME,
)
from forecastfm.tinker_data import OutcomeTrainingRecord, read_outcome_training_jsonl
from forecastfm.tinker_screening import (
    TinkerScreeningError,
    require_text_health_screen_passes,
)

_SHA256_LENGTH = 64
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_FEATURE_CARD_PREFIX = "Pregame numeric feature: "
_SEASONS_SCHEMA_VERSION = 1
_SEASONS_KEYS = {"schema_version", "seasons"}
_SEASON_KEYS = {"question_id", "season"}


class OutcomeV2PreflightError(ValueError):
    """Raised when an outcome-v2 artifact is not safe to hand to a trainer."""


@dataclass(frozen=True, slots=True)
class OutcomeV2Preflight:
    """The exact local artifact accepted by the offline gate."""

    manifest_sha256: str
    action_at: datetime
    training_sha256: str
    feature_rows_sha256: str
    snapshot_pack_sha256: str
    evidence_bundles_sha256: str
    elo_states_sha256: str
    seasons_sha256: str
    resolutions_sha256: str
    rights_lock_sha256: str
    row_count: int
    pair_count: int
    batch_size: int


@dataclass(frozen=True, slots=True)
class OutcomeV2Artifacts:
    """All sealed local files required by a readiness-true SFT run."""

    feature_rows_path: Path
    snapshot_pack_path: Path
    evidence_bundles_path: Path
    elo_states_path: Path
    seasons_path: Path
    resolutions_path: Path
    rights_lock_path: Path
    agreement_path: Path


def require_outcome_v2_sft_ready(
    manifest_path: Path,
    training_path: Path,
    artifacts: OutcomeV2Artifacts | None = None,
    *,
    action_at: datetime | None = None,
) -> OutcomeV2Preflight:
    """Validate the complete local SFT boundary without importing Tinker."""
    manifest = _read_manifest(manifest_path)
    _require_schema_versions(manifest)
    _require_full_readiness(manifest)
    if artifacts is None:
        raise OutcomeV2PreflightError("the complete sealed NBA artifact set is required")
    if action_at is None:
        raise OutcomeV2PreflightError("the protected Tinker action time is required")
    _require_utc(action_at, "action_at")
    if action_at > datetime.now(UTC):
        raise OutcomeV2PreflightError("the protected Tinker action time cannot be in the future")
    _require_feature_schema(manifest)

    artifact_hashes = _verify_artifact_hashes(manifest, training_path, artifacts)
    approval = _load_reviewed_rights(artifacts)
    _require_upload_rights(manifest, approval)

    feature_rows = _read_feature_rows(artifacts.feature_rows_path)
    records = _read_training_records(training_path)
    _require_health_screen(records)
    original_ids = _require_exact_pairs(records)
    _require_row_contract(manifest, len(records), len(original_ids))
    _require_exact_original_id_order(manifest, original_ids)
    _require_feature_row_binding(records, feature_rows, original_ids)
    _require_complete_provenance(
        artifacts,
        approval,
        records,
        feature_rows,
        action_at=action_at,
    )

    return OutcomeV2Preflight(
        manifest_sha256=_file_hash(manifest_path, "outcome-v2 manifest"),
        action_at=action_at,
        training_sha256=artifact_hashes[TRAINING_FILENAME],
        feature_rows_sha256=artifact_hashes[FEATURE_ROWS_FILENAME],
        snapshot_pack_sha256=artifact_hashes[SNAPSHOT_PACK_FILENAME],
        evidence_bundles_sha256=artifact_hashes[EVIDENCE_BUNDLES_FILENAME],
        elo_states_sha256=artifact_hashes[ELO_STATES_FILENAME],
        seasons_sha256=artifact_hashes[SEASONS_FILENAME],
        resolutions_sha256=artifact_hashes[RESOLUTIONS_FILENAME],
        rights_lock_sha256=artifact_hashes[RIGHTS_LOCK_FILENAME],
        row_count=len(records),
        pair_count=len(original_ids),
        batch_size=BATCH_SIZE,
    )


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        return parse_json_object(path.read_text(encoding="utf-8"))
    except (JsonFormatError, OSError) as error:
        raise OutcomeV2PreflightError("cannot read a valid outcome-v2 manifest") from error


def _read_training_records(path: Path) -> tuple[OutcomeTrainingRecord, ...]:
    try:
        return read_outcome_training_jsonl(path)
    except (JsonFormatError, OSError) as error:
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


def _require_full_readiness(manifest: dict[str, object]) -> None:
    evaluation = _object_field(manifest, "evaluation")
    ready = _boolean_field(evaluation, "full_outcome_v2_ready")
    if not ready:
        raise OutcomeV2PreflightError("full_outcome_v2_ready is false")
    missing = _string_tuple_field(evaluation, "full_outcome_v2_missing")
    if missing:
        raise OutcomeV2PreflightError("full_outcome_v2_missing must be empty")
    for field_name in (
        "historical_gate_passes_raw_elo",
        "historical_gate_passes_recalibrated_elo",
    ):
        if not _boolean_field(evaluation, field_name):
            raise OutcomeV2PreflightError(f"evaluation.{field_name} is false")
    untouched_seasons = _integer_tuple_field(evaluation, "untouched_evaluation_seasons")
    if len(untouched_seasons) < 2 or len(set(untouched_seasons)) != len(untouched_seasons):
        raise OutcomeV2PreflightError(
            "at least two unique untouched evaluation seasons are required"
        )


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
) -> dict[str, str]:
    paths = {
        TRAINING_FILENAME: training_path,
        FEATURE_ROWS_FILENAME: artifacts.feature_rows_path,
        SNAPSHOT_PACK_FILENAME: artifacts.snapshot_pack_path,
        EVIDENCE_BUNDLES_FILENAME: artifacts.evidence_bundles_path,
        ELO_STATES_FILENAME: artifacts.elo_states_path,
        SEASONS_FILENAME: artifacts.seasons_path,
        RESOLUTIONS_FILENAME: artifacts.resolutions_path,
        RIGHTS_LOCK_FILENAME: artifacts.rights_lock_path,
    }
    return {
        filename: _verify_output_hash(manifest, path, filename) for filename, path in paths.items()
    }


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
    records: tuple[OutcomeTrainingRecord, ...],
    feature_rows: tuple[NbaRichFeatureRow, ...],
    *,
    action_at: datetime,
) -> None:
    feature_ids = tuple(row.question_id for row in feature_rows)
    frozen_seasons = _read_frozen_seasons(artifacts.seasons_path, feature_ids)
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
            feature_rows,
            frozen_seasons,
            action_at=action_at,
        )
        elo_states = read_nba_elo_states_jsonl(artifacts.elo_states_path)
        validate_elo_states_against_feature_rows(
            elo_states,
            feature_rows,
            action_at=action_at,
        )
        resolutions = read_nba_resolutions_jsonl(
            artifacts.resolutions_path,
            snapshot_index=snapshot_index,
        )
        validate_outcome_training_labels(
            bundles,
            resolutions,
            records,
            snapshot_index=snapshot_index,
            action_at=action_at,
        )
    except (
        NbaEloStateError,
        NbaEvidenceError,
        NbaEvidenceIoError,
        NbaResolutionError,
        NbaRightsApprovalError,
        SnapshotPackError,
    ) as error:
        raise OutcomeV2PreflightError(f"NBA provenance validation failed: {error}") from error


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
    if row_count % BATCH_SIZE != 0:
        raise OutcomeV2PreflightError(
            f"training row count is not divisible by batch size {BATCH_SIZE}"
        )


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
    expected_content = _feature_row_user_content(row)
    if record["messages"][1]["content"] != expected_content:
        raise OutcomeV2PreflightError("training prompt content differs from its sealed feature row")


def _feature_row_user_content(row: NbaRichFeatureRow) -> str:
    prompt = {
        "evidence": [
            f"{_FEATURE_CARD_PREFIX}{canonical_json({name: value})}"
            for name, value in row.feature_items
        ],
        "outcomes": [TEAM_OUTCOME, OPPONENT_OUTCOME],
        "prior": {
            TEAM_OUTCOME: row.elo_team_win_probability,
            OPPONENT_OUTCOME: row.elo_opponent_win_probability,
        },
        "question": QUESTION_TEXT,
        "resolution_rule": RESOLUTION_RULE,
    }
    return canonical_json(prompt)


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
