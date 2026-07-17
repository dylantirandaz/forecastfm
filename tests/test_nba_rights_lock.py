"""Tests for agreement-bound NBA rights approval locks."""

import hashlib
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from forecastfm.integrity import canonical_json
from forecastfm.nba_evidence import SourceRights
from forecastfm.nba_rights_lock import (
    NBA_RIGHTS_LOCK_SCHEMA_VERSION,
    NbaRightsApprovalError,
    NbaRightsApprovalLock,
    load_nba_rights_approval_lock,
    require_approved_action,
    require_snapshot_index_rights,
)
from forecastfm.nba_snapshot_pack import NbaSnapshot, NbaSnapshotIndex, NbaSnapshotMetadata

RIGHTS_AS_OF = datetime(2026, 7, 15, 18, 30, tzinfo=UTC)
ACTION_AT = RIGHTS_AS_OF + timedelta(days=2)
SOURCE_ID = "lineups:game-1"
RIGHTS_SCOPE = "provider-test:nba:lineups"


def lock_record(agreement: bytes) -> dict[str, object]:
    return {
        "schema_version": NBA_RIGHTS_LOCK_SCHEMA_VERSION,
        "provider_id": "provider-test",
        "license_id": "order-2026-42",
        "agreement_reference": "vault://agreements/order-2026-42.pdf",
        "agreement_sha256": hashlib.sha256(agreement).hexdigest(),
        "rights_as_of": "2026-07-15T18:30:00Z",
        "local_processing": "allowed",
        "third_party_processing": "allowed",
        "tinker_processing": "allowed",
        "redistribution": "prohibited",
        "approved_rights_scopes": [RIGHTS_SCOPE],
        "review_decision_id": "legal-review-1842",
    }


def load_lock(
    tmp_path: Path,
    *,
    record: dict[str, object] | None = None,
) -> NbaRightsApprovalLock:
    agreement = b"exact signed agreement bytes\n\x00\xff"
    agreement_path = tmp_path / "agreement.pdf"
    lock_path = tmp_path / "rights-lock.json"
    agreement_path.write_bytes(agreement)
    lock_path.write_text(canonical_json(record or lock_record(agreement)), encoding="utf-8")
    return load_nba_rights_approval_lock(lock_path, agreement_path)


def make_snapshot(
    rights: SourceRights,
    *,
    source_id: str = SOURCE_ID,
    rights_scope: str = RIGHTS_SCOPE,
) -> NbaSnapshot:
    payload = b'{"status":"projected"}'
    published_at = RIGHTS_AS_OF + timedelta(days=1)
    retrieved_at = published_at + timedelta(minutes=1)
    return NbaSnapshot(
        metadata=NbaSnapshotMetadata(
            source_id=source_id,
            rights_scope=rights_scope,
            source_url="https://provider.test/lineups/game-1",
            version="v1",
            effective_at=published_at,
            provider_published_at=published_at,
            retrieved_at=retrieved_at,
            available_at=retrieved_at,
            capture_method="live",
            sensitivity="ordinary",
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            archive_attestation_sha256=None,
            rights=rights,
        ),
        payload=payload,
    )


def test_loader_binds_exact_agreement_bytes_and_derives_source_rights(
    tmp_path: Path,
) -> None:
    approval = load_lock(tmp_path)

    assert approval.provider_id == "provider-test"
    assert approval.license_id == "order-2026-42"
    assert approval.review_decision_id == "legal-review-1842"
    assert approval.rights_as_of == RIGHTS_AS_OF
    assert approval.to_source_rights() == SourceRights(
        license_name="provider-test/order-2026-42",
        terms_url="vault://agreements/order-2026-42.pdf",
        terms_sha256=approval.agreement_sha256,
        rights_as_of=RIGHTS_AS_OF,
        local_processing="allowed",
        third_party_processing="allowed",
        tinker_processing="allowed",
        redistribution="prohibited",
    )
    with pytest.raises(FrozenInstanceError):
        approval.provider_id = "changed"  # pyright: ignore[reportAttributeAccessIssue]


def test_loader_rejects_changed_agreement_bytes(tmp_path: Path) -> None:
    agreement = b"reviewed bytes"
    agreement_path = tmp_path / "agreement.pdf"
    lock_path = tmp_path / "rights-lock.json"
    agreement_path.write_bytes(agreement + b" tampered")
    lock_path.write_text(canonical_json(lock_record(agreement)), encoding="utf-8")

    with pytest.raises(NbaRightsApprovalError, match="exact agreement bytes"):
        load_nba_rights_approval_lock(lock_path, agreement_path)


def test_loader_requires_strict_canonical_json_and_exact_fields(tmp_path: Path) -> None:
    agreement = b"exact signed agreement bytes\n\x00\xff"
    agreement_path = tmp_path / "agreement.pdf"
    lock_path = tmp_path / "rights-lock.json"
    agreement_path.write_bytes(agreement)
    record = lock_record(agreement)
    lock_path.write_text(canonical_json(record) + "\n", encoding="utf-8")
    with pytest.raises(NbaRightsApprovalError, match="canonical JSON"):
        load_nba_rights_approval_lock(lock_path, agreement_path)

    record["unreviewed_note"] = "not part of the schema"
    lock_path.write_text(canonical_json(record), encoding="utf-8")
    with pytest.raises(NbaRightsApprovalError, match="invalid rights approval lock"):
        load_nba_rights_approval_lock(lock_path, agreement_path)


def test_lock_requires_utc_review_reference_permissions_and_source_scope(
    tmp_path: Path,
) -> None:
    agreement = b"exact signed agreement bytes\n\x00\xff"
    base = lock_record(agreement)

    bad_fields: tuple[tuple[str, object], ...] = (
        ("rights_as_of", "2026-07-15T13:30:00-05:00"),
        ("review_decision_id", " "),
        ("local_processing", "assumed"),
        ("approved_rights_scopes", []),
    )
    for field_name, bad_value in bad_fields:
        record: dict[str, object] = {**base, field_name: bad_value}
        with pytest.raises(NbaRightsApprovalError, match="invalid rights approval lock"):
            load_lock(tmp_path, record=record)


def test_permission_checks_fail_closed_and_tinker_requires_third_party() -> None:
    approval = NbaRightsApprovalLock(
        provider_id="provider-test",
        license_id="order-1",
        agreement_reference="vault://agreements/order-1.pdf",
        agreement_sha256="a" * 64,
        rights_as_of=RIGHTS_AS_OF,
        local_processing="allowed",
        third_party_processing="unknown",
        tinker_processing="allowed",
        redistribution="prohibited",
        approved_rights_scopes=(RIGHTS_SCOPE,),
        review_decision_id="review-1",
    )

    require_approved_action(approval, "local_processing")
    with pytest.raises(NbaRightsApprovalError, match="third_party_processing is unknown"):
        require_approved_action(approval, "tinker_processing")
    with pytest.raises(NbaRightsApprovalError, match="redistribution is prohibited"):
        require_approved_action(approval, "redistribution")


def test_snapshot_index_requires_reviewed_scope_and_exact_lock_rights(tmp_path: Path) -> None:
    approval = load_lock(tmp_path)
    exact_snapshot = make_snapshot(approval.to_source_rights())
    require_snapshot_index_rights(
        NbaSnapshotIndex((exact_snapshot,)),
        approval,
        action="tinker_processing",
        action_at=ACTION_AT,
    )

    another_entity = make_snapshot(
        approval.to_source_rights(),
        source_id="lineups:game-2",
    )
    require_snapshot_index_rights(
        NbaSnapshotIndex((another_entity,)),
        approval,
        action="local_processing",
        action_at=ACTION_AT,
    )

    unreviewed = make_snapshot(
        approval.to_source_rights(),
        source_id="injuries:game-1",
        rights_scope="provider-test:nba:injuries",
    )
    with pytest.raises(NbaRightsApprovalError, match="rights scope is not reviewed"):
        require_snapshot_index_rights(
            NbaSnapshotIndex((unreviewed,)),
            approval,
            action="local_processing",
            action_at=ACTION_AT,
        )

    for field_name, permission in (
        ("terms_sha256", "b" * 64),
        ("tinker_processing", "unknown"),
    ):
        mismatched = replace(
            approval.to_source_rights(),
            **{field_name: permission},
        )
        with pytest.raises(NbaRightsApprovalError, match="do not match the reviewed lock"):
            require_snapshot_index_rights(
                NbaSnapshotIndex((make_snapshot(mismatched),)),
                approval,
                action="local_processing",
                action_at=ACTION_AT,
            )

    future_snapshot = replace(
        exact_snapshot,
        metadata=replace(
            exact_snapshot.metadata,
            retrieved_at=ACTION_AT + timedelta(seconds=1),
            available_at=ACTION_AT + timedelta(seconds=1),
        ),
    )
    with pytest.raises(NbaRightsApprovalError, match="retrieval postdates"):
        require_snapshot_index_rights(
            NbaSnapshotIndex((future_snapshot,)),
            approval,
            action="local_processing",
            action_at=ACTION_AT,
        )


def test_constructor_rejects_unsorted_or_duplicate_rights_scopes() -> None:
    for rights_scopes in (("z", "a"), ("a", "a")):
        with pytest.raises(NbaRightsApprovalError, match="unique and sorted"):
            NbaRightsApprovalLock(
                provider_id="provider-test",
                license_id="order-1",
                agreement_reference="vault://agreements/order-1.pdf",
                agreement_sha256="a" * 64,
                rights_as_of=RIGHTS_AS_OF,
                local_processing="allowed",
                third_party_processing="unknown",
                tinker_processing="unknown",
                redistribution="prohibited",
                approved_rights_scopes=rights_scopes,
                review_decision_id="review-1",
            )


def test_constructor_rejects_non_utc_rights_time() -> None:
    with pytest.raises(NbaRightsApprovalError, match="rights_as_of must be in UTC"):
        NbaRightsApprovalLock(
            provider_id="provider-test",
            license_id="order-1",
            agreement_reference="vault://agreements/order-1.pdf",
            agreement_sha256="a" * 64,
            rights_as_of=RIGHTS_AS_OF.replace(tzinfo=None),
            local_processing="allowed",
            third_party_processing="unknown",
            tinker_processing="unknown",
            redistribution="prohibited",
            approved_rights_scopes=(RIGHTS_SCOPE,),
            review_decision_id="review-1",
        )
