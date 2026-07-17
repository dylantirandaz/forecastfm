"""Tests for canonical, snapshot-bound NBA evidence-bundle I/O."""

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from forecastfm.integrity import canonical_json
from forecastfm.json_utils import parse_json_object, require_list, require_object, required_field
from forecastfm.ledger import CohortGame
from forecastfm.models import ForecastQuestion
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_evidence import (
    NbaEvidenceBundle,
    NbaEvidenceError,
    NbaEvidenceRecord,
    Sensitivity,
    SourceRights,
    evidence_bundle_sha256,
)
from forecastfm.nba_evidence_io import (
    NbaEvidenceIoError,
    read_nba_evidence_bundles_jsonl,
    validate_tinker_feature_rows_from_bundles,
    write_nba_evidence_bundles_jsonl,
)
from forecastfm.nba_feature_rows import (
    NbaEloPriorInput,
    build_local_rich_feature_row,
    build_tinker_rich_feature_row,
)
from forecastfm.nba_rich import NBA_RICH_FEATURE_SPECS
from forecastfm.nba_snapshot_pack import NbaSnapshot, NbaSnapshotIndex, NbaSnapshotMetadata

CUTOFF = datetime(2026, 10, 23, 0, tzinfo=UTC)
TIPOFF = CUTOFF + timedelta(minutes=60)
RETRIEVED = CUTOFF - timedelta(minutes=10)
ACTION_AT = CUTOFF
SEASON = 2027

TEAM_VALUES = {
    "rest_days": 4.0,
    "back_to_back": 1.0,
    "games_last_7": 3.0,
    "road_games_last_7": 2.0,
    "travel_miles": 800.0,
    "travel_time_zones": 1.0,
    "roster_continuity": 0.8,
    "expected_lineup_continuity": 0.6,
    "rolling_team_net_rating": 5.0,
    "rolling_player_value": 3.0,
    "schedule_strength": 1_525.0,
}
OPPONENT_VALUES = {
    "rest_days": 2.0,
    "back_to_back": 0.0,
    "games_last_7": 2.0,
    "road_games_last_7": 1.0,
    "travel_miles": 300.0,
    "travel_time_zones": 0.0,
    "roster_continuity": 0.7,
    "expected_lineup_continuity": 0.4,
    "rolling_team_net_rating": 1.0,
    "rolling_player_value": -1.0,
    "schedule_strength": 1_500.0,
}


def _rights() -> SourceRights:
    return SourceRights(
        license_name="Test data agreement",
        terms_url="https://provider.test/terms",
        terms_sha256="a" * 64,
        rights_as_of=RETRIEVED - timedelta(days=1),
        local_processing="allowed",
        third_party_processing="allowed",
        tinker_processing="allowed",
        redistribution="unknown",
    )


def _snapshot(
    version: str = "live-v1",
    *,
    retrieved_at: datetime = RETRIEVED,
    sensitivity: Sensitivity = "ordinary",
) -> NbaSnapshot:
    payload = f"opaque provider bytes:{version}".encode()
    metadata = NbaSnapshotMetadata(
        source_id="licensed-feed",
        rights_scope="provider-test:nba:metrics",
        source_url="https://provider.test/snapshot",
        version=version,
        effective_at=retrieved_at - timedelta(minutes=2),
        provider_published_at=retrieved_at - timedelta(minutes=1),
        retrieved_at=retrieved_at,
        available_at=retrieved_at,
        capture_method="live",
        sensitivity=sensitivity,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        archive_attestation_sha256=None,
        rights=_rights(),
    )
    return NbaSnapshot(metadata=metadata, payload=payload)


def _bundle(question_id: str, snapshot: NbaSnapshot | None = None) -> NbaEvidenceBundle:
    selected = _snapshot() if snapshot is None else snapshot
    records = tuple(
        NbaEvidenceRecord(
            record_id=f"feature-{index:02d}",
            kind=spec.kind,
            feature_name=spec.name,
            team_value=TEAM_VALUES[spec.name],
            opponent_value=OPPONENT_VALUES[spec.name],
            source_ids=(selected.metadata.source_id,),
            available_at=selected.metadata.available_at,
        )
        for index, spec in enumerate(NBA_RICH_FEATURE_SPECS)
    )
    game = CohortGame(
        question_id=question_id,
        source_game_id=f"provider-{question_id}",
        matchup="Team vs Opponent",
        outcomes=("team", "opponent"),
        forecast_deadline=CUTOFF,
        scheduled_tipoff=TIPOFF,
    )
    question = ForecastQuestion(
        question_id=question_id,
        text="Will the listed team win?",
        resolution_rule="Use the official final score after the game ends.",
        resolution_source="https://provider.test/official-score",
        outcomes=game.outcomes,
        forecast_at=CUTOFF,
        resolves_at=TIPOFF + timedelta(hours=3),
    )
    return NbaEvidenceBundle(
        game=game,
        question=question,
        sources=(selected.to_source_snapshot(),),
        records=records,
    )


def _elo() -> NbaEloPriorInput:
    return NbaEloPriorInput(
        team_win_probability=0.61,
        available_at=RETRIEVED - timedelta(minutes=1),
        state_sha256="d" * 64,
    )


def _payload(path: Path) -> dict[str, object]:
    return parse_json_object(path.read_text(encoding="utf-8").rstrip("\n"))


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_text(f"{canonical_json(payload)}\n", encoding="utf-8")


def test_bundle_jsonl_round_trip_preserves_frozen_nonlexicographic_order(tmp_path: Path) -> None:
    snapshot = _snapshot()
    index = NbaSnapshotIndex((snapshot,))
    bundles = (_bundle("nba-z", snapshot), _bundle("nba-a", snapshot))
    path = tmp_path / "evidence.jsonl"

    write_nba_evidence_bundles_jsonl(path, bundles, snapshot_index=index)
    loaded = read_nba_evidence_bundles_jsonl(path, snapshot_index=index)

    assert loaded == bundles
    assert tuple(bundle.game.question_id for bundle in loaded) == ("nba-z", "nba-a")
    assert path.read_text(encoding="utf-8").endswith("\n")
    for line, bundle in zip(path.read_text(encoding="utf-8").splitlines(), bundles, strict=True):
        assert line == canonical_json(parse_json_object(line))
        assert parse_json_object(line)["evidence_bundle_sha256"] == evidence_bundle_sha256(bundle)

    with pytest.raises(NbaEvidenceIoError, match="already exists"):
        write_nba_evidence_bundles_jsonl(path, bundles, snapshot_index=index)


def test_bundle_jsonl_requires_nonempty_unique_original_bundles(tmp_path: Path) -> None:
    snapshot = _snapshot()
    index = NbaSnapshotIndex((snapshot,))
    bundle = _bundle("nba-1", snapshot)

    with pytest.raises(NbaEvidenceIoError, match="must not be empty"):
        write_nba_evidence_bundles_jsonl(
            tmp_path / "empty.jsonl",
            (),
            snapshot_index=index,
        )
    with pytest.raises(NbaEvidenceIoError, match="duplicate question ID"):
        write_nba_evidence_bundles_jsonl(
            tmp_path / "duplicate.jsonl",
            (bundle, bundle),
            snapshot_index=index,
        )

    suffixed_id = f"nba-1{SIDE_SWAP_SUFFIX}"
    suffixed = replace(
        bundle,
        game=replace(bundle.game, question_id=suffixed_id),
        question=replace(bundle.question, question_id=suffixed_id),
    )
    with pytest.raises(NbaEvidenceIoError, match="only original bundles"):
        write_nba_evidence_bundles_jsonl(
            tmp_path / "swapped.jsonl",
            (suffixed,),
            snapshot_index=index,
        )


@pytest.mark.parametrize(
    "forbidden_field",
    ["realized_outcome", "score", "result", "final_stats", "target", "label"],
)
def test_reader_rejects_postgame_and_target_fields(
    tmp_path: Path,
    forbidden_field: str,
) -> None:
    snapshot = _snapshot()
    index = NbaSnapshotIndex((snapshot,))
    path = tmp_path / "evidence.jsonl"
    write_nba_evidence_bundles_jsonl(path, (_bundle("nba-1", snapshot),), snapshot_index=index)
    payload = _payload(path)
    payload[forbidden_field] = "forbidden"
    _write_payload(path, payload)

    with pytest.raises(NbaEvidenceIoError, match="invalid NBA evidence bundle"):
        read_nba_evidence_bundles_jsonl(path, snapshot_index=index)


def test_reader_requires_exact_types_timestamps_rights_and_digest(tmp_path: Path) -> None:
    snapshot = _snapshot()
    index = NbaSnapshotIndex((snapshot,))
    bundle = _bundle("nba-1", snapshot)

    schema_path = tmp_path / "schema.jsonl"
    write_nba_evidence_bundles_jsonl(schema_path, (bundle,), snapshot_index=index)
    schema_payload = _payload(schema_path)
    schema_payload["schema_version"] = True
    _write_payload(schema_path, schema_payload)
    with pytest.raises(NbaEvidenceIoError, match="invalid NBA evidence bundle"):
        read_nba_evidence_bundles_jsonl(schema_path, snapshot_index=index)

    rights_path = tmp_path / "rights.jsonl"
    write_nba_evidence_bundles_jsonl(rights_path, (bundle,), snapshot_index=index)
    rights_payload = _payload(rights_path)
    sources = require_list(required_field(rights_payload, "sources"), "sources")
    source = require_object(sources[0], "source")
    rights = require_object(required_field(source, "rights"), "rights")
    rights["unreviewed_permission"] = "allowed"
    source["rights"] = rights
    sources[0] = source
    rights_payload["sources"] = sources
    _write_payload(rights_path, rights_payload)
    with pytest.raises(NbaEvidenceIoError, match="invalid NBA evidence bundle"):
        read_nba_evidence_bundles_jsonl(rights_path, snapshot_index=index)

    digest_path = tmp_path / "digest.jsonl"
    write_nba_evidence_bundles_jsonl(digest_path, (bundle,), snapshot_index=index)
    digest_payload = _payload(digest_path)
    records = require_list(required_field(digest_payload, "records"), "records")
    first_record = require_object(records[0], "record")
    first_record["team_value"] = 5.0
    records[0] = first_record
    digest_payload["records"] = records
    _write_payload(digest_path, digest_payload)
    with pytest.raises(NbaEvidenceIoError, match="invalid NBA evidence bundle"):
        read_nba_evidence_bundles_jsonl(digest_path, snapshot_index=index)


def test_reader_rejects_noncanonical_number_and_timestamp_encodings(tmp_path: Path) -> None:
    snapshot = _snapshot()
    index = NbaSnapshotIndex((snapshot,))
    bundle = _bundle("nba-1", snapshot)

    number_path = tmp_path / "number.jsonl"
    write_nba_evidence_bundles_jsonl(number_path, (bundle,), snapshot_index=index)
    number_payload = _payload(number_path)
    records = require_list(required_field(number_payload, "records"), "records")
    first_record = require_object(records[0], "record")
    first_record["team_value"] = 4
    records[0] = first_record
    number_payload["records"] = records
    _write_payload(number_path, number_payload)
    with pytest.raises(NbaEvidenceIoError, match="canonical JSONL"):
        read_nba_evidence_bundles_jsonl(number_path, snapshot_index=index)

    time_path = tmp_path / "time.jsonl"
    write_nba_evidence_bundles_jsonl(time_path, (bundle,), snapshot_index=index)
    time_payload = _payload(time_path)
    game = require_object(required_field(time_payload, "game"), "game")
    question = require_object(required_field(time_payload, "question"), "question")
    game["forecast_deadline"] = "2026-10-23T00:00:00+00:00"
    question["forecast_at"] = "2026-10-23T00:00:00+00:00"
    time_payload["game"] = game
    time_payload["question"] = question
    _write_payload(time_path, time_payload)
    with pytest.raises(NbaEvidenceIoError, match="canonical JSONL"):
        read_nba_evidence_bundles_jsonl(time_path, snapshot_index=index)

    whitespace_path = tmp_path / "whitespace.jsonl"
    write_nba_evidence_bundles_jsonl(whitespace_path, (bundle,), snapshot_index=index)
    whitespace_payload = _payload(whitespace_path)
    whitespace_path.write_text(f"{json.dumps(whitespace_payload)}\n", encoding="utf-8")
    with pytest.raises(NbaEvidenceIoError, match="canonical JSONL"):
        read_nba_evidence_bundles_jsonl(whitespace_path, snapshot_index=index)


def test_bundle_source_must_equal_latest_eligible_snapshot(tmp_path: Path) -> None:
    earlier = _snapshot("live-v1")
    later = _snapshot("live-v2", retrieved_at=RETRIEVED + timedelta(minutes=1))
    index = NbaSnapshotIndex((earlier, later))
    stale_bundle = _bundle("nba-1", earlier)

    with pytest.raises(NbaEvidenceIoError, match="not the latest eligible snapshot"):
        write_nba_evidence_bundles_jsonl(
            tmp_path / "stale.jsonl",
            (stale_bundle,),
            snapshot_index=index,
        )

    future = _snapshot("live-future", retrieved_at=CUTOFF + timedelta(minutes=1))
    future_index = NbaSnapshotIndex((future,))
    with pytest.raises(NbaEvidenceIoError, match="no eligible snapshot"):
        write_nba_evidence_bundles_jsonl(
            tmp_path / "future.jsonl",
            (_bundle("nba-2", earlier),),
            snapshot_index=future_index,
        )


def test_bundle_identity_rejects_changed_live_version_or_effective_time(
    tmp_path: Path,
) -> None:
    original = _snapshot()
    bundle = _bundle("nba-1", original)
    changed_metadata = (
        replace(original.metadata, version="live-v2"),
        replace(
            original.metadata,
            effective_at=original.metadata.effective_at + timedelta(minutes=1),
        ),
    )

    for index, metadata in enumerate(changed_metadata):
        changed = NbaSnapshot(metadata=metadata, payload=original.payload)
        assert changed.to_source_snapshot().snapshot_metadata_sha256 != (
            original.to_source_snapshot().snapshot_metadata_sha256
        )
        with pytest.raises(NbaEvidenceIoError, match="not the latest eligible snapshot"):
            write_nba_evidence_bundles_jsonl(
                tmp_path / f"changed-{index}.jsonl",
                (bundle,),
                snapshot_index=NbaSnapshotIndex((changed,)),
            )


def test_tinker_feature_rows_recompute_and_validate_frozen_seasons() -> None:
    snapshot = _snapshot()
    bundles = (_bundle("nba-z", snapshot), _bundle("nba-a", snapshot))
    rows = tuple(
        build_tinker_rich_feature_row(
            bundle,
            season=SEASON,
            elo=_elo(),
            action_at=ACTION_AT,
        )
        for bundle in bundles
    )
    seasons = {bundle.game.question_id: SEASON for bundle in bundles}

    validate_tinker_feature_rows_from_bundles(
        bundles,
        rows,
        seasons,
        action_at=ACTION_AT,
    )

    with pytest.raises(NbaEvidenceIoError, match="IDs and order"):
        validate_tinker_feature_rows_from_bundles(
            bundles,
            tuple(reversed(rows)),
            seasons,
            action_at=ACTION_AT,
        )
    with pytest.raises(NbaEvidenceIoError, match="season differs"):
        validate_tinker_feature_rows_from_bundles(
            bundles,
            (replace(rows[0], season=2028), rows[1]),
            seasons,
            action_at=ACTION_AT,
        )
    with pytest.raises(NbaEvidenceIoError, match="exactly cover"):
        validate_tinker_feature_rows_from_bundles(
            bundles,
            rows,
            {"nba-z": SEASON},
            action_at=ACTION_AT,
        )


def test_tinker_feature_rows_bind_cutoff_tipoff_hash_vector_and_availability() -> None:
    snapshot = _snapshot()
    bundle = _bundle("nba-1", snapshot)
    row = build_tinker_rich_feature_row(
        bundle,
        season=SEASON,
        elo=_elo(),
        action_at=ACTION_AT,
    )
    seasons = {bundle.game.question_id: SEASON}

    shifted_game = replace(bundle.game, scheduled_tipoff=TIPOFF + timedelta(minutes=1))
    shifted_bundle = replace(bundle, game=shifted_game)
    with pytest.raises(NbaEvidenceIoError, match="tipoff differs"):
        validate_tinker_feature_rows_from_bundles(
            (shifted_bundle,),
            (row,),
            seasons,
            action_at=ACTION_AT,
        )

    wrong_hash = replace(row, evidence_bundle_sha256="e" * 64)
    with pytest.raises(NbaEvidenceIoError, match="evidence digest differs"):
        validate_tinker_feature_rows_from_bundles(
            (bundle,),
            (wrong_hash,),
            seasons,
            action_at=ACTION_AT,
        )

    changed_features = replace(row.rich_features, rest_days_difference=1.0)
    wrong_vector = replace(row, rich_features=changed_features)
    with pytest.raises(NbaEvidenceIoError, match="vector differs"):
        validate_tinker_feature_rows_from_bundles(
            (bundle,),
            (wrong_vector,),
            seasons,
            action_at=ACTION_AT,
        )

    wrong_availability = replace(
        row,
        input_available_at=row.input_available_at + timedelta(seconds=1),
    )
    with pytest.raises(NbaEvidenceIoError, match="input availability differs"):
        validate_tinker_feature_rows_from_bundles(
            (bundle,),
            (wrong_availability,),
            seasons,
            action_at=ACTION_AT,
        )


def test_tinker_feature_validation_rejects_health_lineage_and_late_action() -> None:
    health_snapshot = _snapshot(sensitivity="player_health")
    health_bundle = _bundle("nba-1", health_snapshot)
    local_row = build_local_rich_feature_row(
        health_bundle,
        season=SEASON,
        elo=_elo(),
        action_at=ACTION_AT,
    )

    with pytest.raises(NbaEvidenceError, match="player-health lineage"):
        validate_tinker_feature_rows_from_bundles(
            (health_bundle,),
            (local_row,),
            {"nba-1": SEASON},
            action_at=ACTION_AT,
        )

    ordinary_bundle = _bundle("nba-2")
    ordinary_row = build_tinker_rich_feature_row(
        ordinary_bundle,
        season=SEASON,
        elo=_elo(),
        action_at=ACTION_AT,
    )
    with pytest.raises(NbaEvidenceError, match="after the protected action"):
        validate_tinker_feature_rows_from_bundles(
            (ordinary_bundle,),
            (ordinary_row,),
            {"nba-2": SEASON},
            action_at=RETRIEVED - timedelta(seconds=1),
        )
