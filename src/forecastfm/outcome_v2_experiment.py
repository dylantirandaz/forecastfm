"""Immutable post-training seal for one outcome-v2 Tinker experiment."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from forecastfm.integrity import bytes_sha256, canonical_json
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_string,
    required_field,
)
from forecastfm.outcome_v2_run import OutcomeV2RunError, OutcomeV2RunLock

OUTCOME_V2_EXPERIMENT_LOCK_SCHEMA_VERSION = 1

_KIND = "forecastfm_outcome_v2_experiment_lock"
_STATUS = "ready_for_prospective_forecasts"
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_LOCK_KEYS = {
    "schema_version",
    "kind",
    "status",
    "created_at",
    "outcome_v2_run_lock_sha256",
    "state_path",
    "sampler_path",
}

type JsonObject = dict[str, object]


class OutcomeV2ExperimentError(ValueError):
    """Raised when an outcome-v2 experiment seal is invalid or stale."""


@dataclass(frozen=True, slots=True)
class OutcomeV2ExperimentLock:
    """One strict experiment lock retained as immutable canonical bytes."""

    canonical_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_immutable_bytes(self.canonical_bytes)
        _record_from_bytes(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        """Return the digest of the exact canonical lock bytes."""
        return bytes_sha256(self.canonical_bytes)

    def to_record(self) -> JsonObject:
        """Return a newly decoded copy of the experiment record."""
        return _record_from_bytes(self.canonical_bytes)


def build_outcome_v2_experiment_lock(
    run_lock_path: Path,
    state_path: str,
    sampler_path: str,
    created_at: datetime,
) -> OutcomeV2ExperimentLock:
    """Bind permanent trained paths to the exact valid outcome-v2 run lock."""
    _require_tinker_path(state_path, "state_path")
    _require_tinker_path(sampler_path, "sampler_path")
    if state_path == sampler_path:
        raise OutcomeV2ExperimentError("state_path and sampler_path must differ")
    run_lock_bytes = _read_valid_run_lock(run_lock_path)
    record: JsonObject = {
        "schema_version": OUTCOME_V2_EXPERIMENT_LOCK_SCHEMA_VERSION,
        "kind": _KIND,
        "status": _STATUS,
        "created_at": _utc_text(created_at, "created_at"),
        "outcome_v2_run_lock_sha256": bytes_sha256(run_lock_bytes),
        "state_path": state_path,
        "sampler_path": sampler_path,
    }
    return OutcomeV2ExperimentLock(canonical_json(record).encode("utf-8"))


def write_outcome_v2_experiment_lock(
    path: Path,
    lock: OutcomeV2ExperimentLock,
) -> str:
    """Create and fsync one canonical experiment lock without replacement."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(lock.canonical_bytes)
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError:
        raise
    except OSError as error:
        raise OutcomeV2ExperimentError("cannot write the outcome-v2 experiment lock") from error
    return lock.sha256


def read_outcome_v2_experiment_lock(path: Path) -> OutcomeV2ExperimentLock:
    """Read one strict canonical experiment lock from disk."""
    try:
        value = path.read_bytes()
    except OSError as error:
        raise OutcomeV2ExperimentError("cannot read the outcome-v2 experiment lock") from error
    return OutcomeV2ExperimentLock(value)


def verify_outcome_v2_experiment_lock(
    run_lock_path: Path,
    experiment_lock_path: Path,
) -> OutcomeV2ExperimentLock:
    """Verify the experiment seal and its exact referenced run-lock bytes."""
    lock = read_outcome_v2_experiment_lock(experiment_lock_path)
    record = lock.to_record()
    expected_hash = _string(record, "outcome_v2_run_lock_sha256")
    run_lock_bytes = _read_bytes(run_lock_path)
    if bytes_sha256(run_lock_bytes) != expected_hash:
        raise OutcomeV2ExperimentError("referenced outcome-v2 run lock bytes changed")
    _validate_run_lock_bytes(run_lock_bytes)
    return lock


def _read_valid_run_lock(path: Path) -> bytes:
    value = _read_bytes(path)
    _validate_run_lock_bytes(value)
    return value


def _require_immutable_bytes(value: object) -> None:
    if not isinstance(value, bytes):
        raise OutcomeV2ExperimentError("experiment lock requires immutable bytes")


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise OutcomeV2ExperimentError("cannot read the referenced outcome-v2 run lock") from error


def _validate_run_lock_bytes(value: bytes) -> None:
    try:
        OutcomeV2RunLock(value)
    except OutcomeV2RunError as error:
        raise OutcomeV2ExperimentError("referenced outcome-v2 run lock is invalid") from error


def _record_from_bytes(value: bytes) -> JsonObject:
    try:
        text = value.decode("utf-8")
        record = parse_json_object(text)
    except (JsonFormatError, UnicodeError) as error:
        raise OutcomeV2ExperimentError("experiment lock must be one UTF-8 JSON object") from error
    if text != canonical_json(record):
        raise OutcomeV2ExperimentError("experiment lock must use canonical JSON bytes")
    _validate_record(record)
    return record


def _validate_record(record: Mapping[str, object]) -> None:
    try:
        require_exact_keys(record, _LOCK_KEYS, "outcome-v2 experiment lock")
        if _integer(record, "schema_version") != OUTCOME_V2_EXPERIMENT_LOCK_SCHEMA_VERSION:
            raise OutcomeV2ExperimentError("unsupported outcome-v2 experiment lock schema")
        if _string(record, "kind") != _KIND:
            raise OutcomeV2ExperimentError("unexpected outcome-v2 experiment lock kind")
        if _string(record, "status") != _STATUS:
            raise OutcomeV2ExperimentError("experiment is not ready for prospective forecasts")
        _parse_utc(_string(record, "created_at"), "created_at")
        digest = _string(record, "outcome_v2_run_lock_sha256")
        _require_hash(digest, "outcome_v2_run_lock_sha256")
        state_path = _string(record, "state_path")
        sampler_path = _string(record, "sampler_path")
        _require_tinker_path(state_path, "state_path")
        _require_tinker_path(sampler_path, "sampler_path")
        if state_path == sampler_path:
            raise OutcomeV2ExperimentError("state_path and sampler_path must differ")
    except JsonFormatError as error:
        raise OutcomeV2ExperimentError("invalid outcome-v2 experiment lock structure") from error


def _string(record: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = required_field(record, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutcomeV2ExperimentError(f"{field_name} must be an integer")
    return value


def _require_hash(value: str, field_name: str) -> None:
    if _HASH_PATTERN.fullmatch(value) is None:
        raise OutcomeV2ExperimentError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_tinker_path(value: str, field_name: str) -> None:
    suffix = value.removeprefix("tinker://")
    has_control = any(character.isspace() or ord(character) < 32 for character in value)
    if not value.startswith("tinker://") or not suffix or has_control:
        raise OutcomeV2ExperimentError(f"{field_name} must be a permanent tinker:// path")
    if "?" in suffix or "#" in suffix:
        raise OutcomeV2ExperimentError(f"{field_name} must not contain query or fragment data")


def _parse_utc(value: str, field_name: str) -> datetime:
    if not value.endswith("Z"):
        raise OutcomeV2ExperimentError(f"{field_name} must use canonical UTC Z notation")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise OutcomeV2ExperimentError(f"{field_name} must be an ISO 8601 datetime") from error
    if _utc_text(parsed, field_name) != value:
        raise OutcomeV2ExperimentError(f"{field_name} must use canonical UTC notation")
    return parsed.astimezone(UTC)


def _utc_text(value: object, field_name: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise OutcomeV2ExperimentError(f"{field_name} must be a UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise OutcomeV2ExperimentError(f"{field_name} must be a UTC datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
