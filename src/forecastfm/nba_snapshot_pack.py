"""Immutable, provider-neutral snapshot packs for point-in-time NBA data."""

from __future__ import annotations

import base64
import binascii
import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from forecastfm.integrity import canonical_json, canonical_sha256
from forecastfm.json_utils import (
    parse_json_object,
    require_exact_keys,
    require_object,
    require_string,
    required_field,
)
from forecastfm.nba_evidence import (
    CaptureMethod,
    Permission,
    Sensitivity,
    SourceRights,
    SourceSnapshot,
)

SNAPSHOT_PACK_SCHEMA_VERSION = 2
_HASH_CHARACTERS = frozenset("0123456789abcdef")
_SENSITIVITIES = frozenset({"ordinary", "player_health"})
_RECORD_KEYS = {"schema_version", "metadata", "payload_base64"}
_METADATA_KEYS = {
    "source_id",
    "rights_scope",
    "source_url",
    "version",
    "effective_at",
    "provider_published_at",
    "retrieved_at",
    "available_at",
    "capture_method",
    "sensitivity",
    "payload_sha256",
    "archive_attestation_sha256",
    "rights",
}
_RIGHTS_KEYS = {
    "license_name",
    "terms_url",
    "terms_sha256",
    "rights_as_of",
    "local_processing",
    "third_party_processing",
    "tinker_processing",
    "redistribution",
}

type JsonObject = dict[str, object]


class SnapshotPackError(ValueError):
    """Raised when a snapshot pack violates its integrity or timing contract."""


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise SnapshotPackError(f"{field_name} must not be empty")


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in _HASH_CHARACTERS for character in value):
        raise SnapshotPackError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise SnapshotPackError(f"{field_name} must be in UTC")


def _utc_text(value: datetime) -> str:
    _require_utc(value, "datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class NbaSnapshotMetadata:
    """Canonical timing, rights, version, and integrity metadata for one payload."""

    source_id: str
    rights_scope: str
    source_url: str
    version: str
    effective_at: datetime
    provider_published_at: datetime
    retrieved_at: datetime
    available_at: datetime
    capture_method: CaptureMethod
    sensitivity: Sensitivity
    payload_sha256: str
    archive_attestation_sha256: str | None
    rights: SourceRights

    def __post_init__(self) -> None:
        _require_text(self.source_id, "source_id")
        _require_text(self.rights_scope, "rights_scope")
        _require_text(self.source_url, "source_url")
        _require_text(self.version, "version")
        _require_utc(self.effective_at, "effective_at")
        _require_utc(self.provider_published_at, "provider_published_at")
        _require_utc(self.retrieved_at, "retrieved_at")
        _require_utc(self.available_at, "available_at")
        _require_sha256(self.payload_sha256, "payload_sha256")
        if self.sensitivity not in _SENSITIVITIES:
            raise SnapshotPackError("unsupported source sensitivity")
        if self.provider_published_at > self.retrieved_at:
            raise SnapshotPackError("provider_published_at cannot be after retrieved_at")
        self._validate_availability()

    def _validate_availability(self) -> None:
        if self.capture_method == "live":
            if self.archive_attestation_sha256 is not None:
                raise SnapshotPackError("live snapshots cannot carry an archive attestation")
            if self.available_at != self.retrieved_at:
                raise SnapshotPackError("live snapshot available_at must equal retrieved_at")
            return
        if self.capture_method != "provider_versioned_archive":
            raise SnapshotPackError("unsupported capture_method")
        if self.archive_attestation_sha256 is None:
            raise SnapshotPackError("provider archives require an attestation digest")
        _require_sha256(self.archive_attestation_sha256, "archive_attestation_sha256")
        if self.available_at != self.provider_published_at:
            message = "provider archive available_at must equal provider_published_at"
            raise SnapshotPackError(message)


@dataclass(frozen=True, slots=True)
class NbaSnapshot:
    """One immutable metadata record bound to its exact raw payload bytes."""

    metadata: NbaSnapshotMetadata
    payload: bytes

    def __post_init__(self) -> None:
        actual_sha256 = hashlib.sha256(self.payload).hexdigest()
        if actual_sha256 != self.metadata.payload_sha256:
            raise SnapshotPackError("payload_sha256 does not match the exact payload bytes")

    def to_source_snapshot(self) -> SourceSnapshot:
        """Bridge this pack record into the existing evidence-lineage schema."""
        metadata = self.metadata
        is_archive = metadata.capture_method == "provider_versioned_archive"
        return SourceSnapshot(
            source_id=metadata.source_id,
            rights_scope=metadata.rights_scope,
            source_url=metadata.source_url,
            payload_sha256=metadata.payload_sha256,
            snapshot_metadata_sha256=snapshot_metadata_sha256(metadata),
            published_at=metadata.provider_published_at,
            retrieved_at=metadata.retrieved_at,
            capture_method=metadata.capture_method,
            sensitivity=metadata.sensitivity,
            rights=metadata.rights,
            archive_version_id=metadata.version if is_archive else None,
            archive_attestation_sha256=(
                metadata.archive_attestation_sha256 if is_archive else None
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class NbaSnapshotIndex:
    """A deterministic, version-unique index over validated snapshots."""

    snapshots: tuple[NbaSnapshot, ...]

    def __init__(self, snapshots: Iterable[NbaSnapshot]) -> None:
        """Validate versions and freeze snapshots in canonical storage order."""
        ordered = tuple(sorted(snapshots, key=_storage_order))
        if not ordered:
            raise SnapshotPackError("snapshot pack must contain at least one snapshot")
        _validate_versions(ordered)
        object.__setattr__(self, "snapshots", ordered)

    def __iter__(self) -> Iterator[NbaSnapshot]:
        """Iterate in canonical storage order."""
        return iter(self.snapshots)

    def latest_eligible(self, source_id: str, cutoff: datetime) -> NbaSnapshot | None:
        """Return the latest snapshot truly available by a UTC cutoff."""
        _require_text(source_id, "source_id")
        _require_utc(cutoff, "cutoff")
        eligible = tuple(
            snapshot
            for snapshot in self.snapshots
            if snapshot.metadata.source_id == source_id and snapshot.metadata.available_at <= cutoff
        )
        if not eligible:
            return None
        latest_times = max(_knowledge_times(snapshot) for snapshot in eligible)
        latest = tuple(
            snapshot for snapshot in eligible if _knowledge_times(snapshot) == latest_times
        )
        if len(latest) != 1:
            raise SnapshotPackError("ambiguous snapshots tie at the latest availability")
        return latest[0]


def snapshot_metadata_sha256(metadata: NbaSnapshotMetadata) -> str:
    """Hash the canonical metadata independently of JSONL field ordering."""
    return canonical_sha256(_metadata_to_dict(metadata))


def write_snapshot_pack(snapshots: Iterable[NbaSnapshot], path: Path) -> None:
    """Validate and create an immutable canonical, self-contained JSONL pack."""
    index = NbaSnapshotIndex(snapshots)
    lines = (canonical_json(_snapshot_to_dict(snapshot)) for snapshot in index)
    text = "".join(f"{line}\n" for line in lines)
    try:
        with path.open("x", encoding="utf-8") as file:
            file.write(text)
    except FileExistsError as error:
        raise SnapshotPackError(
            "snapshot pack already exists; immutable packs cannot be replaced"
        ) from error


def load_snapshot_pack(path: Path) -> NbaSnapshotIndex:
    """Load a local canonical JSONL pack without making network requests."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise SnapshotPackError("snapshot pack must be UTF-8") from error
    except OSError as error:
        raise SnapshotPackError("cannot read snapshot pack") from error
    if text and not text.endswith("\n"):
        raise SnapshotPackError("snapshot pack must end with a newline")
    snapshots = tuple(
        _snapshot_from_line(line, line_number)
        for line_number, line in enumerate(text.splitlines(), start=1)
    )
    return NbaSnapshotIndex(snapshots)


def _snapshot_from_line(line: str, line_number: int) -> NbaSnapshot:
    if not line:
        raise SnapshotPackError(f"blank snapshot record on line {line_number}")
    try:
        record = parse_json_object(line)
        snapshot = _snapshot_from_dict(record)
        if line != canonical_json(_snapshot_to_dict(snapshot)):
            raise SnapshotPackError("snapshot record is not canonical JSON")
        return snapshot
    except ValueError as error:
        raise SnapshotPackError(f"invalid snapshot record on line {line_number}") from error


def _snapshot_from_dict(record: Mapping[str, object]) -> NbaSnapshot:
    require_exact_keys(record, _RECORD_KEYS, "snapshot record")
    version = required_field(record, "schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise SnapshotPackError("schema_version must be an integer")
    if version != SNAPSHOT_PACK_SCHEMA_VERSION:
        raise SnapshotPackError(f"unsupported snapshot pack schema version: {version}")
    metadata_record = require_object(required_field(record, "metadata"), "metadata")
    payload_text = require_string(required_field(record, "payload_base64"), "payload_base64")
    return NbaSnapshot(
        metadata=_metadata_from_dict(metadata_record),
        payload=_decode_payload(payload_text),
    )


def _metadata_from_dict(record: Mapping[str, object]) -> NbaSnapshotMetadata:
    require_exact_keys(record, _METADATA_KEYS, "metadata")
    attestation_value = required_field(record, "archive_attestation_sha256")
    attestation = (
        None
        if attestation_value is None
        else require_string(attestation_value, "archive_attestation_sha256")
    )
    rights_record = require_object(required_field(record, "rights"), "rights")
    return NbaSnapshotMetadata(
        source_id=_string_field(record, "source_id"),
        rights_scope=_string_field(record, "rights_scope"),
        source_url=_string_field(record, "source_url"),
        version=_string_field(record, "version"),
        effective_at=_time_field(record, "effective_at"),
        provider_published_at=_time_field(record, "provider_published_at"),
        retrieved_at=_time_field(record, "retrieved_at"),
        available_at=_time_field(record, "available_at"),
        capture_method=_capture_method(required_field(record, "capture_method")),
        sensitivity=_sensitivity(required_field(record, "sensitivity")),
        payload_sha256=_string_field(record, "payload_sha256"),
        archive_attestation_sha256=attestation,
        rights=_rights_from_dict(rights_record),
    )


def _rights_from_dict(record: Mapping[str, object]) -> SourceRights:
    require_exact_keys(record, _RIGHTS_KEYS, "rights")
    return SourceRights(
        license_name=_string_field(record, "license_name"),
        terms_url=_string_field(record, "terms_url"),
        terms_sha256=_string_field(record, "terms_sha256"),
        rights_as_of=_time_field(record, "rights_as_of"),
        local_processing=_permission_field(record, "local_processing"),
        third_party_processing=_permission_field(record, "third_party_processing"),
        tinker_processing=_permission_field(record, "tinker_processing"),
        redistribution=_permission_field(record, "redistribution"),
    )


def _snapshot_to_dict(snapshot: NbaSnapshot) -> JsonObject:
    return {
        "schema_version": SNAPSHOT_PACK_SCHEMA_VERSION,
        "metadata": _metadata_to_dict(snapshot.metadata),
        "payload_base64": base64.b64encode(snapshot.payload).decode("ascii"),
    }


def _metadata_to_dict(metadata: NbaSnapshotMetadata) -> JsonObject:
    return {
        "source_id": metadata.source_id,
        "rights_scope": metadata.rights_scope,
        "source_url": metadata.source_url,
        "version": metadata.version,
        "effective_at": _utc_text(metadata.effective_at),
        "provider_published_at": _utc_text(metadata.provider_published_at),
        "retrieved_at": _utc_text(metadata.retrieved_at),
        "available_at": _utc_text(metadata.available_at),
        "capture_method": metadata.capture_method,
        "sensitivity": metadata.sensitivity,
        "payload_sha256": metadata.payload_sha256,
        "archive_attestation_sha256": metadata.archive_attestation_sha256,
        "rights": _rights_to_dict(metadata.rights),
    }


def _rights_to_dict(rights: SourceRights) -> JsonObject:
    return {
        "license_name": rights.license_name,
        "terms_url": rights.terms_url,
        "terms_sha256": rights.terms_sha256,
        "rights_as_of": _utc_text(rights.rights_as_of),
        "local_processing": rights.local_processing,
        "third_party_processing": rights.third_party_processing,
        "tinker_processing": rights.tinker_processing,
        "redistribution": rights.redistribution,
    }


def _parse_utc(value: object, field_name: str) -> datetime:
    text = require_string(value, field_name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise SnapshotPackError(f"{field_name} must be an ISO 8601 datetime") from error
    _require_utc(parsed, field_name)
    return parsed.astimezone(UTC)


def _decode_payload(value: str) -> bytes:
    try:
        encoded = value.encode("ascii")
        payload = base64.b64decode(encoded, validate=True)
    except (UnicodeEncodeError, binascii.Error) as error:
        raise SnapshotPackError("payload_base64 must be canonical base64") from error
    if base64.b64encode(payload) != encoded:
        raise SnapshotPackError("payload_base64 must be canonical base64")
    return payload


def _capture_method(value: object) -> CaptureMethod:
    text = require_string(value, "capture_method")
    match text:
        case "live" | "provider_versioned_archive":
            return text
        case _:
            raise SnapshotPackError("unsupported capture_method")


def _sensitivity(value: object) -> Sensitivity:
    text = require_string(value, "sensitivity")
    match text:
        case "ordinary" | "player_health":
            return text
        case _:
            raise SnapshotPackError("unsupported source sensitivity")


def _permission_field(record: Mapping[str, object], field_name: str) -> Permission:
    value = _string_field(record, field_name)
    match value:
        case "allowed" | "prohibited" | "unknown":
            return value
        case _:
            raise SnapshotPackError(f"{field_name} must be allowed, prohibited, or unknown")


def _string_field(record: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _time_field(record: Mapping[str, object], field_name: str) -> datetime:
    return _parse_utc(required_field(record, field_name), field_name)


def _storage_order(
    snapshot: NbaSnapshot,
) -> tuple[str, datetime, datetime, datetime, str, str]:
    metadata = snapshot.metadata
    return (
        metadata.source_id,
        metadata.effective_at,
        metadata.provider_published_at,
        metadata.available_at,
        metadata.version,
        metadata.payload_sha256,
    )


def _knowledge_times(snapshot: NbaSnapshot) -> tuple[datetime, datetime]:
    metadata = snapshot.metadata
    return (metadata.available_at, metadata.provider_published_at)


def _validate_versions(snapshots: Sequence[NbaSnapshot]) -> None:
    hashes: dict[tuple[str, str], str] = {}
    for snapshot in snapshots:
        metadata = snapshot.metadata
        key = (metadata.source_id, metadata.version)
        previous_hash = hashes.get(key)
        if previous_hash is None:
            hashes[key] = metadata.payload_sha256
        elif previous_hash != metadata.payload_sha256:
            raise SnapshotPackError("one source version cannot identify multiple payload hashes")
        else:
            raise SnapshotPackError("duplicate source version in snapshot pack")
