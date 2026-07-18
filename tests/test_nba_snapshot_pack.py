"""Tests for immutable, point-in-time NBA snapshot packs."""

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from forecastfm.nba_evidence import CaptureMethod, SourceRights
from forecastfm.nba_snapshot_pack import (
    NbaSnapshot,
    NbaSnapshotIndex,
    NbaSnapshotMetadata,
    SnapshotPackError,
    load_snapshot_pack,
    load_snapshot_pack_bytes,
    snapshot_metadata_sha256,
    write_snapshot_pack,
)

BASE_TIME = datetime(2026, 10, 1, 10, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class SnapshotTimes:
    effective_at: datetime
    provider_published_at: datetime
    retrieved_at: datetime


def make_rights() -> SourceRights:
    return SourceRights(
        license_name="Test data agreement",
        terms_url="https://provider.test/terms",
        terms_sha256="a" * 64,
        rights_as_of=BASE_TIME - timedelta(days=1),
        local_processing="allowed",
        third_party_processing="unknown",
        tinker_processing="unknown",
        redistribution="prohibited",
    )


def make_snapshot(
    version: str,
    payload: bytes,
    *,
    source_id: str = "lineups:game-1",
    times: SnapshotTimes | None = None,
    capture_method: CaptureMethod = "live",
) -> NbaSnapshot:
    snapshot_times = times or SnapshotTimes(
        effective_at=BASE_TIME,
        provider_published_at=BASE_TIME + timedelta(minutes=1),
        retrieved_at=BASE_TIME + timedelta(minutes=2),
    )
    is_archive = capture_method == "provider_versioned_archive"
    return NbaSnapshot(
        metadata=NbaSnapshotMetadata(
            source_id=source_id,
            rights_scope="provider-test:nba:lineups",
            source_url=f"https://provider.test/snapshots/{source_id}/{version}",
            version=version,
            effective_at=snapshot_times.effective_at,
            provider_published_at=snapshot_times.provider_published_at,
            retrieved_at=snapshot_times.retrieved_at,
            available_at=(
                snapshot_times.provider_published_at if is_archive else snapshot_times.retrieved_at
            ),
            capture_method=capture_method,
            sensitivity="ordinary",
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            archive_attestation_sha256="b" * 64 if is_archive else None,
            rights=make_rights(),
        ),
        payload=payload,
    )


def test_snapshot_pack_round_trip_preserves_exact_bytes_and_canonical_order(
    tmp_path: Path,
) -> None:
    first = make_snapshot("v1", b'{"status":"projected"}\n\x00\xff')
    second = make_snapshot(
        "v2",
        b'{"status":"confirmed"}\n',
        times=SnapshotTimes(
            effective_at=BASE_TIME + timedelta(minutes=20),
            provider_published_at=BASE_TIME + timedelta(minutes=21),
            retrieved_at=BASE_TIME + timedelta(minutes=22),
        ),
    )
    path = tmp_path / "snapshots.jsonl"

    write_snapshot_pack((second, first), path)
    loaded = load_snapshot_pack(path)

    assert loaded.snapshots == (first, second)
    assert load_snapshot_pack_bytes(path.read_bytes()) == loaded
    assert loaded.snapshots[0].payload == first.payload
    assert isinstance(loaded.snapshots[0].metadata.rights, SourceRights)
    assert loaded.snapshots[0].metadata.rights == first.metadata.rights
    assert loaded.snapshots[0].metadata.source_url == first.metadata.source_url
    assert loaded.snapshots[0].metadata.rights_scope == first.metadata.rights_scope
    assert loaded.snapshots[0].metadata.sensitivity == "ordinary"
    assert snapshot_metadata_sha256(loaded.snapshots[0].metadata) == snapshot_metadata_sha256(
        first.metadata
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    canonical_lines = (
        json.dumps(json.loads(line), sort_keys=True, separators=(",", ":")) for line in lines
    )
    assert all(line == canonical for line, canonical in zip(lines, canonical_lines, strict=True))


def test_snapshot_pack_bytes_reject_non_utf8() -> None:
    with pytest.raises(SnapshotPackError, match="must be UTF-8"):
        load_snapshot_pack_bytes(b"\xff")


def test_payload_hash_and_source_versions_are_fail_closed() -> None:
    first = make_snapshot("v1", b"first")
    bad_metadata = replace(first.metadata, payload_sha256="0" * 64)
    with pytest.raises(SnapshotPackError, match="exact payload bytes"):
        NbaSnapshot(metadata=bad_metadata, payload=first.payload)

    with pytest.raises(SnapshotPackError, match="duplicate source version"):
        NbaSnapshotIndex((first, first))

    conflicting = make_snapshot("v1", b"different")
    with pytest.raises(SnapshotPackError, match="multiple payload hashes"):
        NbaSnapshotIndex((first, conflicting))

    with pytest.raises(SnapshotPackError, match="at least one snapshot"):
        NbaSnapshotIndex(())


def test_capture_method_controls_when_snapshot_becomes_available() -> None:
    live = make_snapshot("live-v1", b"live")
    with pytest.raises(SnapshotPackError, match="available_at must equal retrieved_at"):
        replace(live.metadata, available_at=live.metadata.provider_published_at)
    with pytest.raises(SnapshotPackError, match="cannot carry an archive attestation"):
        replace(live.metadata, archive_attestation_sha256="c" * 64)

    archive = make_snapshot(
        "archive-v1",
        b"archive",
        capture_method="provider_versioned_archive",
    )
    with pytest.raises(SnapshotPackError, match="require an attestation"):
        replace(archive.metadata, archive_attestation_sha256=None)
    with pytest.raises(SnapshotPackError, match="available_at must equal provider_published_at"):
        replace(archive.metadata, available_at=archive.metadata.retrieved_at)
    assert live.to_source_snapshot().historical_available_at() == live.metadata.retrieved_at
    archive_source = archive.to_source_snapshot()
    assert archive_source.archive_version_id == archive.metadata.version
    assert archive_source.historical_available_at() == archive.metadata.provider_published_at


def test_source_url_and_sensitivity_survive_the_lineage_bridge() -> None:
    snapshot = make_snapshot("health-v1", b"availability status")
    health_metadata = replace(snapshot.metadata, sensitivity="player_health")
    health_snapshot = NbaSnapshot(metadata=health_metadata, payload=snapshot.payload)

    source = health_snapshot.to_source_snapshot()

    assert source.source_url == health_metadata.source_url
    assert source.rights_scope == health_metadata.rights_scope
    assert source.sensitivity == "player_health"
    with pytest.raises(SnapshotPackError, match="source_url must not be empty"):
        replace(snapshot.metadata, source_url=" ")
    with pytest.raises(SnapshotPackError, match="rights_scope must not be empty"):
        replace(snapshot.metadata, rights_scope=" ")


def test_live_snapshot_identity_binds_version_and_effective_time() -> None:
    original = make_snapshot("live-v1", b"same payload")
    changed_version = NbaSnapshot(
        metadata=replace(original.metadata, version="live-v2"),
        payload=original.payload,
    )
    changed_effective_at = NbaSnapshot(
        metadata=replace(
            original.metadata,
            effective_at=original.metadata.effective_at + timedelta(minutes=1),
        ),
        payload=original.payload,
    )

    source = original.to_source_snapshot()
    assert source.snapshot_metadata_sha256 == snapshot_metadata_sha256(original.metadata)
    assert changed_version.to_source_snapshot().snapshot_metadata_sha256 != (
        source.snapshot_metadata_sha256
    )
    assert changed_effective_at.to_source_snapshot().snapshot_metadata_sha256 != (
        source.snapshot_metadata_sha256
    )


def test_all_snapshot_times_must_be_utc_and_publication_precedes_retrieval() -> None:
    snapshot = make_snapshot("v1", b"payload")
    central = timezone(timedelta(hours=-5))

    with pytest.raises(SnapshotPackError, match="effective_at must be in UTC"):
        replace(snapshot.metadata, effective_at=BASE_TIME.astimezone(central))
    with pytest.raises(SnapshotPackError, match="cannot be after retrieved_at"):
        replace(
            snapshot.metadata,
            provider_published_at=snapshot.metadata.retrieved_at + timedelta(seconds=1),
        )


def test_latest_eligible_never_uses_a_future_live_backfill() -> None:
    cutoff = BASE_TIME + timedelta(hours=2)
    early_live = make_snapshot("v1", b"early")
    late_live = make_snapshot(
        "v2",
        b"late",
        times=SnapshotTimes(
            effective_at=BASE_TIME + timedelta(hours=1),
            provider_published_at=BASE_TIME + timedelta(hours=1, minutes=1),
            retrieved_at=cutoff + timedelta(minutes=1),
        ),
    )
    live_index = NbaSnapshotIndex((late_live, early_live))

    assert live_index.latest_eligible("lineups:game-1", cutoff) == early_live
    assert live_index.latest_eligible("missing-source", cutoff) is None

    attested_archive = make_snapshot(
        "archive-v3",
        b"attested",
        times=SnapshotTimes(
            effective_at=BASE_TIME + timedelta(hours=1, minutes=30),
            provider_published_at=cutoff - timedelta(minutes=10),
            retrieved_at=cutoff + timedelta(days=1),
        ),
        capture_method="provider_versioned_archive",
    )
    archive_index = NbaSnapshotIndex((late_live, early_live, attested_archive))
    assert archive_index.latest_eligible("lineups:game-1", cutoff) == attested_archive

    central = timezone(timedelta(hours=-5))
    with pytest.raises(SnapshotPackError, match="cutoff must be in UTC"):
        archive_index.latest_eligible("lineups:game-1", cutoff.astimezone(central))


def test_selection_allows_known_future_events_and_prioritizes_availability() -> None:
    cutoff = BASE_TIME + timedelta(hours=2)
    future_lineup = make_snapshot(
        "future-lineup",
        b"tomorrow's projected lineup",
        times=SnapshotTimes(
            effective_at=cutoff + timedelta(days=1),
            provider_published_at=cutoff - timedelta(minutes=10),
            retrieved_at=cutoff - timedelta(minutes=5),
        ),
    )
    assert (
        NbaSnapshotIndex((future_lineup,)).latest_eligible("lineups:game-1", cutoff)
        == future_lineup
    )

    older_availability = make_snapshot(
        "older-availability",
        b"first update",
        times=SnapshotTimes(
            effective_at=cutoff + timedelta(days=2),
            provider_published_at=BASE_TIME + timedelta(minutes=1),
            retrieved_at=BASE_TIME + timedelta(minutes=2),
        ),
    )
    newer_availability = make_snapshot(
        "newer-availability",
        b"correction to an older event",
        times=SnapshotTimes(
            effective_at=BASE_TIME - timedelta(days=1),
            provider_published_at=cutoff - timedelta(minutes=2),
            retrieved_at=cutoff - timedelta(minutes=1),
        ),
    )
    index = NbaSnapshotIndex((newer_availability, older_availability))
    assert index.latest_eligible("lineups:game-1", cutoff) == newer_availability


def test_opaque_versions_cannot_break_an_availability_tie() -> None:
    cutoff = BASE_TIME + timedelta(hours=1)
    version_9 = make_snapshot("v9", b"version nine")
    version_10 = make_snapshot("v10", b"version ten")
    index = NbaSnapshotIndex((version_9, version_10))

    with pytest.raises(SnapshotPackError, match="ambiguous snapshots"):
        index.latest_eligible("lineups:game-1", cutoff)


def test_loader_rejects_tampering_duplicates_and_noncanonical_json(tmp_path: Path) -> None:
    snapshot = make_snapshot("v1", b"payload")
    path = tmp_path / "snapshots.jsonl"
    write_snapshot_pack((snapshot,), path)
    canonical_line = path.read_text(encoding="utf-8")

    path.write_text(canonical_line.replace("cGF5bG9hZA==", "dGFtcGVyZWQ="), encoding="utf-8")
    with pytest.raises(SnapshotPackError, match="line 1"):
        load_snapshot_pack(path)

    path.write_text(canonical_line + canonical_line, encoding="utf-8")
    with pytest.raises(SnapshotPackError, match="duplicate source version"):
        load_snapshot_pack(path)

    decoded = json.loads(canonical_line)
    path.write_text(json.dumps(decoded) + "\n", encoding="utf-8")
    with pytest.raises(SnapshotPackError, match="line 1"):
        load_snapshot_pack(path)

    alternate_utc = canonical_line.replace(
        "2026-10-01T10:01:00Z",
        "2026-10-01T10:01:00+00:00",
    )
    assert alternate_utc != canonical_line
    path.write_text(alternate_utc, encoding="utf-8")
    with pytest.raises(SnapshotPackError, match="line 1"):
        load_snapshot_pack(path)


def test_writer_refuses_to_replace_an_existing_snapshot_pack(tmp_path: Path) -> None:
    path = tmp_path / "snapshots.jsonl"
    snapshot = make_snapshot("v1", b"payload")
    write_snapshot_pack((snapshot,), path)

    with pytest.raises(SnapshotPackError, match="cannot be replaced"):
        write_snapshot_pack((snapshot,), path)
