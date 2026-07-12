"""Create and verify immutable training and prospective experiment locks."""

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from forecastfm.integrity import canonical_sha256, file_sha256, text_sha256
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_object,
    require_string,
    required_field,
)
from forecastfm.prompting import MODEL_INPUT_SCHEMA_VERSION, SYSTEM_PROMPT
from forecastfm.run_config import (
    decoding_settings,
    model_settings,
    package_versions,
    training_settings,
)

TRAINING_LOCK_SCHEMA_VERSION = 1
EXPERIMENT_LOCK_SCHEMA_VERSION = 1

TRAINING_DATA_PATH = Path("data/processed/nba_elo_train_sft.jsonl")
DATA_MANIFEST_PATH = Path("data/processed/manifest.json")
LOCKED_CODE_PATHS = (
    Path("examples/train_tinker_sft.py"),
    Path("src/forecastfm/prompting.py"),
    Path("src/forecastfm/run_config.py"),
)

_TRAINING_LOCK_KEYS = {
    "schema_version",
    "kind",
    "status",
    "created_at",
    "code_revision",
    "code_files",
    "model",
    "model_reference_sha256",
    "prompt",
    "data",
    "training",
    "decoding",
    "packages",
    "uv_lock_sha256",
}
_EXPERIMENT_LOCK_KEYS = {
    "schema_version",
    "kind",
    "status",
    "created_at",
    "training_lock_sha256",
    "adapter_sampler_path",
    "checkpoint_metadata_sha256",
}
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40,64}")


class RunLockError(ValueError):
    """Raised when a run lock is incomplete, stale, or modified."""


def build_training_lock(
    project_root: Path,
    code_revision: str,
    created_at: datetime,
) -> dict[str, object]:
    """Build a lock for every input and setting used by training."""
    _require_revision(code_revision)
    manifest_path = project_root / DATA_MANIFEST_PATH
    training_path = project_root / TRAINING_DATA_PATH
    _verify_dataset_manifest(manifest_path, training_path)

    model = model_settings()
    prompt_path = project_root / "src/forecastfm/prompting.py"
    return {
        "schema_version": TRAINING_LOCK_SCHEMA_VERSION,
        "kind": "forecastfm_training_lock",
        "status": "awaiting_trained_sampler",
        "created_at": _utc_text(created_at, "created_at"),
        "code_revision": code_revision,
        "code_files": {str(path): file_sha256(project_root / path) for path in LOCKED_CODE_PATHS},
        "model": model,
        "model_reference_sha256": canonical_sha256(model),
        "prompt": {
            "model_input_schema_version": MODEL_INPUT_SCHEMA_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "system_prompt_sha256": text_sha256(SYSTEM_PROMPT),
            "prompting_module_sha256": file_sha256(prompt_path),
        },
        "data": {
            "manifest_path": str(DATA_MANIFEST_PATH),
            "manifest_sha256": file_sha256(manifest_path),
            "training_path": str(TRAINING_DATA_PATH),
            "training_sha256": file_sha256(training_path),
        },
        "training": training_settings(),
        "decoding": decoding_settings(),
        "packages": package_versions(),
        "uv_lock_sha256": file_sha256(project_root / "uv.lock"),
    }


def verify_training_lock(project_root: Path, path: Path) -> dict[str, object]:
    """Return a training lock only when it still matches the working files."""
    record = _read_lock(path, _TRAINING_LOCK_KEYS, "training lock")
    code_revision = require_string(required_field(record, "code_revision"), "code_revision")
    created_at = _parse_utc(required_field(record, "created_at"), "created_at")
    try:
        expected = build_training_lock(project_root, code_revision, created_at)
    except (FileNotFoundError, JsonFormatError, RunLockError) as error:
        raise RunLockError("training lock inputs cannot be verified") from error
    if canonical_sha256(record) != canonical_sha256(expected):
        raise RunLockError("training lock differs from the current code, data, or settings")
    return record


def build_experiment_lock(
    training_lock_path: Path,
    adapter_sampler_path: str,
    checkpoint_metadata: Mapping[str, object],
    created_at: datetime,
) -> dict[str, object]:
    """Bind a completed Tinker sampler checkpoint to its frozen training lock."""
    _require_sampler_path(adapter_sampler_path)
    return {
        "schema_version": EXPERIMENT_LOCK_SCHEMA_VERSION,
        "kind": "forecastfm_experiment_lock",
        "status": "ready_for_prospective_forecasts",
        "created_at": _utc_text(created_at, "created_at"),
        "training_lock_sha256": file_sha256(training_lock_path),
        "adapter_sampler_path": adapter_sampler_path,
        "checkpoint_metadata_sha256": canonical_sha256(dict(checkpoint_metadata)),
    }


def verify_experiment_lock(
    training_lock_path: Path,
    experiment_lock_path: Path,
) -> dict[str, object]:
    """Return a complete experiment lock bound to the exact training lock."""
    record = _read_lock(experiment_lock_path, _EXPERIMENT_LOCK_KEYS, "experiment lock")
    version = required_field(record, "schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise RunLockError("experiment lock schema must be an integer")
    if version != EXPERIMENT_LOCK_SCHEMA_VERSION:
        raise RunLockError(f"unsupported experiment lock schema: {version}")
    if required_field(record, "kind") != "forecastfm_experiment_lock":
        raise RunLockError("unexpected experiment lock kind")
    if required_field(record, "status") != "ready_for_prospective_forecasts":
        raise RunLockError("experiment is not ready for prospective forecasts")
    expected_hash = require_string(
        required_field(record, "training_lock_sha256"),
        "training_lock_sha256",
    )
    _require_hash(expected_hash, "training_lock_sha256")
    if expected_hash != file_sha256(training_lock_path):
        raise RunLockError("experiment references a different training lock")
    adapter_path = require_string(
        required_field(record, "adapter_sampler_path"),
        "adapter_sampler_path",
    )
    _require_sampler_path(adapter_path)
    checkpoint_hash = require_string(
        required_field(record, "checkpoint_metadata_sha256"),
        "checkpoint_metadata_sha256",
    )
    _require_hash(checkpoint_hash, "checkpoint_metadata_sha256")
    _parse_utc(required_field(record, "created_at"), "created_at")
    return record


def write_new_lock(path: Path, record: Mapping[str, object]) -> None:
    """Create a formatted lock file without permitting accidental replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(record, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")


def _verify_dataset_manifest(manifest_path: Path, training_path: Path) -> None:
    manifest = parse_json_object(manifest_path.read_text(encoding="utf-8"))
    version = required_field(manifest, "model_input_schema_version")
    if version != MODEL_INPUT_SCHEMA_VERSION:
        raise RunLockError("dataset manifest uses a stale model-input schema")
    outputs = require_object(required_field(manifest, "outputs"), "outputs")
    expected_hash = require_string(
        required_field(outputs, training_path.name),
        training_path.name,
    )
    if expected_hash != file_sha256(training_path):
        raise RunLockError("training data differs from the dataset manifest")


def _read_lock(path: Path, keys: set[str], field_name: str) -> dict[str, object]:
    try:
        record = parse_json_object(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RunLockError(f"{field_name} is missing: {path}") from error
    require_exact_keys(record, keys, field_name)
    return record


def _parse_utc(value: object, field_name: str) -> datetime:
    text = require_string(value, field_name)
    try:
        result = datetime.fromisoformat(text)
    except ValueError as error:
        raise RunLockError(f"{field_name} must be an ISO datetime") from error
    _utc_text(result, field_name)
    return result


def _utc_text(value: datetime, field_name: str) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise RunLockError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != UTC.utcoffset(value):
        raise RunLockError(f"{field_name} must use UTC")
    return value.isoformat()


def _require_hash(value: str, field_name: str) -> None:
    if _HASH_PATTERN.fullmatch(value) is None:
        raise RunLockError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_revision(value: str) -> None:
    if _REVISION_PATTERN.fullmatch(value) is None:
        raise RunLockError("code_revision must be a lowercase Git object ID")


def _require_sampler_path(value: str) -> None:
    if not value.startswith("tinker://"):
        raise RunLockError("adapter sampler path must be an immutable tinker:// path")
