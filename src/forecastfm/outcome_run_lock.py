"""Immutable lock for the realized-winner outcome training run."""

import re
from datetime import UTC, datetime
from pathlib import Path

from forecastfm.integrity import canonical_sha256, file_sha256, text_sha256
from forecastfm.json_utils import (
    parse_json_object,
    require_exact_keys,
    require_object,
    require_string,
    required_field,
)
from forecastfm.outcome import OUTCOME_INPUT_SCHEMA_VERSION, OUTCOME_SYSTEM_PROMPT
from forecastfm.outcome_config import (
    OUTCOME_RENDERER_NAME,
    outcome_inference_settings,
    outcome_training_settings,
)
from forecastfm.run_config import model_settings, package_versions
from forecastfm.run_lock import RunLockError

OUTCOME_TRAINING_LOCK_SCHEMA_VERSION = 1
OUTCOME_TRAINING_DATA_PATH = Path("data/processed/outcome_v1/nba_train_outcome.jsonl")
OUTCOME_DATA_MANIFEST_PATH = Path("data/processed/outcome_v1/manifest.json")
OUTCOME_LOCKED_CODE_PATHS = (
    Path("examples/build_outcome_dataset.py"),
    Path("examples/tinker_outcome_inference.py"),
    Path("examples/train_tinker_outcome_sft.py"),
    Path("src/forecastfm/nba_data.py"),
    Path("src/forecastfm/outcome.py"),
    Path("src/forecastfm/outcome_config.py"),
    Path("src/forecastfm/run_config.py"),
    Path("src/forecastfm/tinker_data.py"),
)

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40,64}")
_LOCK_KEYS = {
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


def build_outcome_training_lock(
    project_root: Path,
    code_revision: str,
    created_at: datetime,
) -> dict[str, object]:
    """Bind the exact outcome prompt, data, code, model, and settings."""
    if _REVISION_PATTERN.fullmatch(code_revision) is None:
        raise RunLockError("code_revision must be a lowercase Git object ID")
    created_at_text = _utc_text(created_at)
    manifest_path = project_root / OUTCOME_DATA_MANIFEST_PATH
    training_path = project_root / OUTCOME_TRAINING_DATA_PATH
    _verify_dataset_manifest(manifest_path, training_path)

    model = model_settings()
    model["renderer"] = OUTCOME_RENDERER_NAME
    prompt_path = project_root / "src/forecastfm/outcome.py"
    return {
        "schema_version": OUTCOME_TRAINING_LOCK_SCHEMA_VERSION,
        "kind": "forecastfm_outcome_training_lock",
        "status": "awaiting_trained_sampler",
        "created_at": created_at_text,
        "code_revision": code_revision,
        "code_files": {
            str(path): file_sha256(project_root / path) for path in OUTCOME_LOCKED_CODE_PATHS
        },
        "model": model,
        "model_reference_sha256": canonical_sha256(model),
        "prompt": {
            "outcome_input_schema_version": OUTCOME_INPUT_SCHEMA_VERSION,
            "system_prompt": OUTCOME_SYSTEM_PROMPT,
            "system_prompt_sha256": text_sha256(OUTCOME_SYSTEM_PROMPT),
            "outcome_module_sha256": file_sha256(prompt_path),
        },
        "data": {
            "manifest_path": str(OUTCOME_DATA_MANIFEST_PATH),
            "manifest_sha256": file_sha256(manifest_path),
            "training_path": str(OUTCOME_TRAINING_DATA_PATH),
            "training_sha256": file_sha256(training_path),
        },
        "training": outcome_training_settings(),
        "decoding": outcome_inference_settings(),
        "packages": package_versions(),
        "uv_lock_sha256": file_sha256(project_root / "uv.lock"),
    }


def verify_outcome_training_lock(project_root: Path, path: Path) -> dict[str, object]:
    """Return the outcome lock only when every bound input still matches."""
    try:
        record = parse_json_object(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RunLockError(f"outcome training lock is missing: {path}") from error
    require_exact_keys(record, _LOCK_KEYS, "outcome training lock")
    revision = require_string(required_field(record, "code_revision"), "code_revision")
    created_at = _parse_utc(required_field(record, "created_at"))
    expected = build_outcome_training_lock(project_root, revision, created_at)
    if canonical_sha256(record) != canonical_sha256(expected):
        raise RunLockError("outcome training lock differs from current code, data, or settings")
    return record


def _verify_dataset_manifest(manifest_path: Path, training_path: Path) -> None:
    manifest = parse_json_object(manifest_path.read_text(encoding="utf-8"))
    version = required_field(manifest, "outcome_input_schema_version")
    if version != OUTCOME_INPUT_SCHEMA_VERSION:
        raise RunLockError("outcome manifest uses a stale input schema")
    outputs = require_object(required_field(manifest, "outputs"), "outputs")
    expected_hash = require_string(
        required_field(outputs, training_path.name),
        training_path.name,
    )
    if expected_hash != file_sha256(training_path):
        raise RunLockError("outcome training data differs from its manifest")


def _parse_utc(value: object) -> datetime:
    text = require_string(value, "created_at")
    try:
        result = datetime.fromisoformat(text)
    except ValueError as error:
        raise RunLockError("created_at must be an ISO datetime") from error
    _utc_text(result)
    return result


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise RunLockError("created_at must be timezone-aware")
    if value.utcoffset() != UTC.utcoffset(value):
        raise RunLockError("created_at must use UTC")
    return value.isoformat()
