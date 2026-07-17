"""Bind NBA source permissions to an exact, human-reviewed agreement artifact.

This module proves only that a rights decision is bound to the exact agreement
bytes reviewed under a recorded decision reference. It does not authenticate the
agreement, verify signatures, or make a legal interpretation of its contents.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from forecastfm.integrity import canonical_json, file_sha256
from forecastfm.json_utils import (
    parse_json_object,
    require_exact_keys,
    require_list,
    require_string,
    required_field,
)
from forecastfm.nba_evidence import Permission, PermissionField, SourceRights
from forecastfm.nba_snapshot_pack import NbaSnapshotIndex

NBA_RIGHTS_LOCK_SCHEMA_VERSION = 2

_HASH_CHARACTERS = frozenset("0123456789abcdef")
_LOCK_KEYS = {
    "schema_version",
    "provider_id",
    "license_id",
    "agreement_reference",
    "agreement_sha256",
    "rights_as_of",
    "local_processing",
    "third_party_processing",
    "tinker_processing",
    "redistribution",
    "approved_rights_scopes",
    "review_decision_id",
}


class NbaRightsApprovalError(ValueError):
    """Raised when an NBA rights approval is invalid or insufficient."""


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise NbaRightsApprovalError(f"{field_name} must not be empty")


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in _HASH_CHARACTERS for character in value):
        raise NbaRightsApprovalError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise NbaRightsApprovalError(f"{field_name} must be in UTC")


@dataclass(frozen=True, slots=True)
class NbaRightsApprovalLock:
    """One reviewed permission decision bound to exact agreement bytes and feed scopes."""

    provider_id: str
    license_id: str
    agreement_reference: str
    agreement_sha256: str
    rights_as_of: datetime
    local_processing: Permission
    third_party_processing: Permission
    tinker_processing: Permission
    redistribution: Permission
    approved_rights_scopes: tuple[str, ...]
    review_decision_id: str

    def __post_init__(self) -> None:
        _require_text(self.provider_id, "provider_id")
        _require_text(self.license_id, "license_id")
        _require_text(self.agreement_reference, "agreement_reference")
        _require_sha256(self.agreement_sha256, "agreement_sha256")
        _require_utc(self.rights_as_of, "rights_as_of")
        _require_text(self.review_decision_id, "review_decision_id")
        _validate_permissions(self)
        _validate_rights_scopes(self.approved_rights_scopes)

    def to_source_rights(self) -> SourceRights:
        """Return the sole ``SourceRights`` value approved by this lock."""
        return SourceRights(
            license_name=f"{self.provider_id}/{self.license_id}",
            terms_url=self.agreement_reference,
            terms_sha256=self.agreement_sha256,
            rights_as_of=self.rights_as_of,
            local_processing=self.local_processing,
            third_party_processing=self.third_party_processing,
            tinker_processing=self.tinker_processing,
            redistribution=self.redistribution,
        )


def load_nba_rights_approval_lock(
    lock_path: Path,
    agreement_path: Path,
) -> NbaRightsApprovalLock:
    """Load canonical JSON and verify its digest against the exact agreement file."""
    try:
        text = lock_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise NbaRightsApprovalError("cannot read the rights approval lock") from error
    try:
        record = parse_json_object(text)
    except ValueError as error:
        raise NbaRightsApprovalError("rights approval lock must contain one JSON object") from error
    if text != canonical_json(record):
        raise NbaRightsApprovalError("rights approval lock must use canonical JSON encoding")
    try:
        approval = _lock_from_dict(record)
    except ValueError as error:
        raise NbaRightsApprovalError("invalid rights approval lock") from error
    try:
        agreement_sha256 = file_sha256(agreement_path)
    except OSError as error:
        raise NbaRightsApprovalError("cannot read the exact agreement file") from error
    if agreement_sha256 != approval.agreement_sha256:
        raise NbaRightsApprovalError("agreement_sha256 does not match the exact agreement bytes")
    return approval


def require_approved_action(
    approval: NbaRightsApprovalLock,
    action: PermissionField,
) -> None:
    """Require explicit approval for an action, including Tinker's third-party hop."""
    if action == "tinker_processing":
        _require_allowed(approval, "third_party_processing")
    _require_allowed(approval, action)


def require_snapshot_index_rights(
    index: NbaSnapshotIndex,
    approval: NbaRightsApprovalLock,
    *,
    action: PermissionField,
    action_at: datetime,
) -> None:
    """Require reviewed rights and snapshots already retained by the action time."""
    _require_utc(action_at, "action_at")
    require_approved_action(approval, action)
    if approval.rights_as_of > action_at:
        raise NbaRightsApprovalError("reviewed rights cannot postdate the protected action")
    expected_rights = approval.to_source_rights()
    approved_rights_scopes = frozenset(approval.approved_rights_scopes)
    for snapshot in index:
        source_id = snapshot.metadata.source_id
        rights_scope = snapshot.metadata.rights_scope
        if rights_scope not in approved_rights_scopes:
            message = f"snapshot rights scope is not reviewed: {rights_scope}"
            raise NbaRightsApprovalError(message)
        if snapshot.metadata.rights != expected_rights:
            message = f"snapshot rights do not match the reviewed lock: {source_id}"
            raise NbaRightsApprovalError(message)
        if snapshot.metadata.retrieved_at > action_at:
            message = f"snapshot retrieval postdates the protected action: {source_id}"
            raise NbaRightsApprovalError(message)


def _lock_from_dict(record: Mapping[str, object]) -> NbaRightsApprovalLock:
    require_exact_keys(record, _LOCK_KEYS, "rights approval lock")
    version = required_field(record, "schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise NbaRightsApprovalError("schema_version must be an integer")
    if version != NBA_RIGHTS_LOCK_SCHEMA_VERSION:
        raise NbaRightsApprovalError(f"unsupported rights lock schema version: {version}")
    return NbaRightsApprovalLock(
        provider_id=_string_field(record, "provider_id"),
        license_id=_string_field(record, "license_id"),
        agreement_reference=_string_field(record, "agreement_reference"),
        agreement_sha256=_string_field(record, "agreement_sha256"),
        rights_as_of=_time_field(record, "rights_as_of"),
        local_processing=_permission_field(record, "local_processing"),
        third_party_processing=_permission_field(record, "third_party_processing"),
        tinker_processing=_permission_field(record, "tinker_processing"),
        redistribution=_permission_field(record, "redistribution"),
        approved_rights_scopes=_rights_scopes_field(record),
        review_decision_id=_string_field(record, "review_decision_id"),
    )


def _validate_permissions(approval: NbaRightsApprovalLock) -> None:
    permissions = (
        approval.local_processing,
        approval.third_party_processing,
        approval.tinker_processing,
        approval.redistribution,
    )
    if any(permission not in {"allowed", "prohibited", "unknown"} for permission in permissions):
        raise NbaRightsApprovalError("permissions must be allowed, prohibited, or unknown")


def _validate_rights_scopes(rights_scopes: tuple[str, ...]) -> None:
    if not rights_scopes:
        raise NbaRightsApprovalError("approved_rights_scopes must not be empty")
    for rights_scope in rights_scopes:
        _require_text(rights_scope, "approved_rights_scopes item")
    if rights_scopes != tuple(sorted(set(rights_scopes))):
        raise NbaRightsApprovalError("approved_rights_scopes must be unique and sorted")


def _require_allowed(approval: NbaRightsApprovalLock, field_name: PermissionField) -> None:
    permission = _permission(approval, field_name)
    if permission != "allowed":
        raise NbaRightsApprovalError(f"{field_name} is {permission}; explicit allowed is required")


def _permission(approval: NbaRightsApprovalLock, field_name: PermissionField) -> Permission:
    match field_name:
        case "local_processing":
            return approval.local_processing
        case "third_party_processing":
            return approval.third_party_processing
        case "tinker_processing":
            return approval.tinker_processing
        case "redistribution":
            return approval.redistribution


def _rights_scopes_field(record: Mapping[str, object]) -> tuple[str, ...]:
    values = require_list(
        required_field(record, "approved_rights_scopes"),
        "approved_rights_scopes",
    )
    return tuple(require_string(value, "approved_rights_scopes item") for value in values)


def _permission_field(record: Mapping[str, object], field_name: str) -> Permission:
    value = _string_field(record, field_name)
    match value:
        case "allowed" | "prohibited" | "unknown":
            return value
        case _:
            raise NbaRightsApprovalError(f"{field_name} must be allowed, prohibited, or unknown")


def _string_field(record: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _time_field(record: Mapping[str, object], field_name: str) -> datetime:
    text = _string_field(record, field_name)
    if not text.endswith("Z"):
        raise NbaRightsApprovalError(f"{field_name} must use canonical UTC Z notation")
    try:
        parsed = datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise NbaRightsApprovalError(f"{field_name} must be an ISO 8601 datetime") from error
    _require_utc(parsed, field_name)
    canonical = parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if text != canonical:
        raise NbaRightsApprovalError(f"{field_name} must use canonical UTC notation")
    return parsed.astimezone(UTC)
