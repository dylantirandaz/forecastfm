"""Offline readiness checks for an eventual outcome-v2 SFT run."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from forecastfm.integrity import canonical_json, canonical_sha256, file_sha256
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
from forecastfm.nba_rich import NBA_RICH_SCHEMA_SHA256, NBA_RICH_SCHEMA_VERSION
from forecastfm.outcome import OPPONENT_OUTCOME, OUTCOME_INPUT_SCHEMA_VERSION, TEAM_OUTCOME
from forecastfm.outcome_v2_config import (
    BATCH_SIZE,
    MANIFEST_SCHEMA_VERSION,
    TRAINING_FILENAME,
)
from forecastfm.tinker_data import OutcomeTrainingRecord, read_outcome_training_jsonl
from forecastfm.tinker_screening import (
    TinkerScreeningError,
    require_text_health_screen_passes,
)

_SHA256_LENGTH = 64
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_USER_PROMPT_KEYS = {"evidence", "outcomes", "prior", "question", "resolution_rule"}
_FEATURE_PREFIXES = ("Pregame numeric features: ", "Pregame numeric feature: ")


class OutcomeV2PreflightError(ValueError):
    """Raised when an outcome-v2 artifact is not safe to hand to a trainer."""


@dataclass(frozen=True, slots=True)
class OutcomeV2Preflight:
    """The exact local artifact accepted by the offline gate."""

    training_sha256: str
    row_count: int
    pair_count: int
    batch_size: int


def require_outcome_v2_sft_ready(
    manifest_path: Path,
    training_path: Path,
) -> OutcomeV2Preflight:
    """Validate the complete local SFT boundary without importing Tinker."""
    manifest = _read_manifest(manifest_path)
    _require_schema_versions(manifest)
    _require_full_readiness(manifest)
    _require_feature_schema(manifest)
    _require_upload_rights(manifest)

    expected_hash = _expected_training_hash(manifest, training_path)
    actual_hash = _training_hash(training_path)
    if actual_hash != expected_hash:
        raise OutcomeV2PreflightError("training file SHA-256 does not match the manifest")

    records = _read_training_records(training_path)
    _require_health_screen(records)
    original_ids = _require_exact_pairs(records)
    _require_row_contract(manifest, len(records), len(original_ids))
    _require_exact_original_id_order(manifest, original_ids)

    return OutcomeV2Preflight(
        training_sha256=actual_hash,
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


def _require_upload_rights(manifest: dict[str, object]) -> None:
    rights = _object_field(manifest, "upload_rights")
    for field_name in ("third_party_processing", "tinker_processing"):
        permission = _string_field(rights, field_name)
        if permission != "allowed":
            raise OutcomeV2PreflightError(f"upload_rights.{field_name} must equal allowed")
    if _boolean_field(rights, "player_health_included"):
        raise OutcomeV2PreflightError("standard Tinker training cannot include player health")


def _require_feature_schema(manifest: dict[str, object]) -> None:
    features = _object_field(manifest, "features")
    full_schema = _object_field(features, "full_schema")
    if _integer_field(full_schema, "version") != NBA_RICH_SCHEMA_VERSION:
        raise OutcomeV2PreflightError("outcome-v2 richer feature schema version differs")
    schema_hash = _string_field(full_schema, "sha256")
    _require_sha256(schema_hash, "features.full_schema.sha256")
    if schema_hash != NBA_RICH_SCHEMA_SHA256:
        raise OutcomeV2PreflightError("outcome-v2 richer feature schema hash differs")
    if not _boolean_field(full_schema, "current_artifact_contains_full_schema"):
        raise OutcomeV2PreflightError("current artifact does not contain the richer feature schema")


def _expected_training_hash(manifest: dict[str, object], training_path: Path) -> str:
    if training_path.name != TRAINING_FILENAME:
        raise OutcomeV2PreflightError(f"training file must be named {TRAINING_FILENAME}")
    outputs = _object_field(manifest, "outputs")
    expected_hash = _string_field(outputs, TRAINING_FILENAME)
    _require_sha256(expected_hash, f"outputs.{TRAINING_FILENAME}")
    return expected_hash


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
    _require_prompt_side_swap(original, swapped)
    return original_id, swapped_id


def _require_prompt_side_swap(
    original: OutcomeTrainingRecord,
    swapped: OutcomeTrainingRecord,
) -> None:
    if original["messages"][0] != swapped["messages"][0]:
        raise OutcomeV2PreflightError("side-swap system prompts differ")
    try:
        original_user = parse_json_object(original["messages"][1]["content"])
        swapped_user = parse_json_object(swapped["messages"][1]["content"])
        expected = _side_swap_user_prompt(original_user)
    except JsonFormatError as error:
        raise OutcomeV2PreflightError("side-swap user prompt is not valid numeric input") from error
    if canonical_json(_normalize_feature_texts(swapped_user)) != canonical_json(
        _normalize_feature_texts(expected)
    ):
        raise OutcomeV2PreflightError("side-swap user prompts are not exact complements")


def _side_swap_user_prompt(prompt: dict[str, object]) -> dict[str, object]:
    require_exact_keys(prompt, _USER_PROMPT_KEYS, "outcome-v2 user prompt")
    outcomes = require_list(required_field(prompt, "outcomes"), "outcomes")
    if outcomes != [TEAM_OUTCOME, OPPONENT_OUTCOME]:
        raise JsonFormatError("outcome-v2 outcomes differ from the canonical order")

    prior = require_object(required_field(prompt, "prior"), "prior")
    require_exact_keys(prior, {TEAM_OUTCOME, OPPONENT_OUTCOME}, "prior")
    team_probability = require_float(required_field(prior, TEAM_OUTCOME), TEAM_OUTCOME)
    opponent_probability = require_float(
        required_field(prior, OPPONENT_OUTCOME),
        OPPONENT_OUTCOME,
    )
    evidence = require_list(required_field(prompt, "evidence"), "evidence")
    swapped_evidence = [
        _side_swap_feature_text(require_string(value, "evidence")) for value in evidence
    ]
    return {
        **prompt,
        "prior": {
            TEAM_OUTCOME: opponent_probability,
            OPPONENT_OUTCOME: team_probability,
        },
        "evidence": swapped_evidence,
    }


def _side_swap_feature_text(text: str) -> str:
    for prefix in _FEATURE_PREFIXES:
        if text.startswith(prefix):
            features = parse_json_object(text.removeprefix(prefix))
            if not features:
                raise JsonFormatError("numeric evidence cannot be empty")
            swapped = {
                name: _negate(require_float(value, name)) for name, value in features.items()
            }
            return prefix + canonical_json(swapped)
    raise JsonFormatError("unsupported numeric evidence format")


def _normalize_feature_texts(prompt: dict[str, object]) -> dict[str, object]:
    evidence = require_list(required_field(prompt, "evidence"), "evidence")
    return {
        **prompt,
        "evidence": [
            _canonical_feature_text(require_string(value, "evidence")) for value in evidence
        ],
    }


def _canonical_feature_text(text: str) -> str:
    for prefix in _FEATURE_PREFIXES:
        if text.startswith(prefix):
            features = parse_json_object(text.removeprefix(prefix))
            for name, value in features.items():
                require_float(value, name)
            return prefix + canonical_json(features)
    raise JsonFormatError("unsupported numeric evidence format")


def _negate(value: float) -> float:
    if not isfinite(value):
        raise JsonFormatError("numeric evidence must be finite")
    if value == 0.0:
        return 0.0
    return -value


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


def _training_hash(path: Path) -> str:
    try:
        return file_sha256(path)
    except OSError as error:
        raise OutcomeV2PreflightError("cannot read the outcome-v2 training file") from error
