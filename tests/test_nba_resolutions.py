"""Tests for independently sealed NBA final-score resolutions."""

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from forecastfm.integrity import canonical_json
from forecastfm.ledger import CohortGame
from forecastfm.models import ForecastQuestion
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_evidence import (
    NbaEvidenceBundle,
    NbaEvidenceRecord,
    SourceRights,
)
from forecastfm.nba_resolutions import (
    NbaResolution,
    NbaResolutionError,
    read_nba_resolutions_jsonl,
    validate_outcome_training_labels,
    write_nba_resolutions_jsonl,
)
from forecastfm.nba_snapshot_pack import (
    NbaSnapshot,
    NbaSnapshotIndex,
    NbaSnapshotMetadata,
    snapshot_metadata_sha256,
)
from forecastfm.outcome import (
    NBA_OUTCOMES,
    OPPONENT_LABEL,
    OUTCOME_SYSTEM_PROMPT,
    TEAM_LABEL,
)
from forecastfm.prompting import ChatMessage
from forecastfm.tinker_data import OutcomeTrainingRecord

CUTOFF = datetime(2026, 10, 21, 22, tzinfo=UTC)
TIPOFF = CUTOFF + timedelta(hours=1)
RESOLVED_AT = TIPOFF + timedelta(hours=3)
ACTION_AT = RESOLVED_AT + timedelta(hours=1)


def _rights() -> SourceRights:
    return SourceRights(
        license_name="Test agreement",
        terms_url="https://provider.test/terms",
        terms_sha256="a" * 64,
        rights_as_of=CUTOFF - timedelta(days=1),
        local_processing="allowed",
        third_party_processing="allowed",
        tinker_processing="allowed",
        redistribution="unknown",
    )


def _snapshot(
    source_id: str,
    *,
    version: str = "v1",
    retrieved_at: datetime = RESOLVED_AT - timedelta(minutes=1),
) -> NbaSnapshot:
    payload = f"opaque final-score payload:{source_id}:{version}".encode()
    return NbaSnapshot(
        metadata=NbaSnapshotMetadata(
            source_id=source_id,
            rights_scope="provider-test:nba:final-scores",
            source_url=f"https://provider.test/finals/{source_id}/{version}",
            version=version,
            effective_at=retrieved_at - timedelta(minutes=1),
            provider_published_at=retrieved_at - timedelta(seconds=30),
            retrieved_at=retrieved_at,
            available_at=retrieved_at,
            capture_method="live",
            sensitivity="ordinary",
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            archive_attestation_sha256=None,
            rights=_rights(),
        ),
        payload=payload,
    )


def _resolution(
    snapshot: NbaSnapshot,
    question_id: str,
    *,
    team_score: int = 112,
    opponent_score: int = 105,
    resolved_at: datetime = RESOLVED_AT,
) -> NbaResolution:
    return NbaResolution(
        question_id=question_id,
        source_game_id=f"provider-{question_id}",
        team_score=team_score,
        opponent_score=opponent_score,
        resolved_at=resolved_at,
        source_id=snapshot.metadata.source_id,
        snapshot_metadata_sha256=snapshot_metadata_sha256(snapshot.metadata),
    )


def _bundle(question_id: str) -> NbaEvidenceBundle:
    source = _snapshot(
        "pregame-features",
        version="pregame-v1",
        retrieved_at=CUTOFF - timedelta(minutes=10),
    ).to_source_snapshot()
    game = CohortGame(
        question_id=question_id,
        source_game_id=f"provider-{question_id}",
        matchup="Team vs Opponent",
        outcomes=NBA_OUTCOMES,
        forecast_deadline=CUTOFF,
        scheduled_tipoff=TIPOFF,
    )
    question = ForecastQuestion(
        question_id=question_id,
        text="Will the listed team win?",
        resolution_rule="Use the official final score.",
        resolution_source="https://provider.test/finals",
        outcomes=NBA_OUTCOMES,
        forecast_at=CUTOFF,
        resolves_at=RESOLVED_AT,
    )
    record = NbaEvidenceRecord(
        record_id="pregame-feature",
        kind="team_metric",
        feature_name="pregame_value",
        team_value=1.0,
        opponent_value=0.0,
        source_ids=(source.source_id,),
        available_at=source.retrieved_at,
    )
    return NbaEvidenceBundle(
        game=game,
        question=question,
        sources=(source,),
        records=(record,),
    )


def _training_record(question_id: str, label: str) -> OutcomeTrainingRecord:
    messages = [
        ChatMessage(role="system", content=OUTCOME_SYSTEM_PROMPT),
        ChatMessage(role="user", content="{}"),
    ]
    return OutcomeTrainingRecord(question_id=question_id, messages=messages, label=label)


def test_resolution_jsonl_round_trip_is_canonical_create_only_and_ordered(
    tmp_path: Path,
) -> None:
    first_snapshot = _snapshot("final:nba-2")
    second_snapshot = _snapshot("final:nba-1")
    index = NbaSnapshotIndex((first_snapshot, second_snapshot))
    resolutions = (
        _resolution(first_snapshot, "nba-2"),
        _resolution(second_snapshot, "nba-1"),
    )
    path = tmp_path / "resolutions.jsonl"

    write_nba_resolutions_jsonl(path, resolutions, snapshot_index=index)

    assert read_nba_resolutions_jsonl(path, snapshot_index=index) == resolutions
    lines = path.read_text(encoding="utf-8").splitlines()
    assert all(line == canonical_json(json.loads(line)) for line in lines)
    assert [json.loads(line)["question_id"] for line in lines] == ["nba-2", "nba-1"]
    with pytest.raises(NbaResolutionError, match="already exists"):
        write_nba_resolutions_jsonl(path, resolutions, snapshot_index=index)


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("question_id", f"nba-1{SIDE_SWAP_SUFFIX}", "original game"),
        ("team_score", -1, "nonnegative integer"),
        ("team_score", True, "nonnegative integer"),
        ("snapshot_metadata_sha256", "not-a-digest", "SHA-256"),
    ],
)
def test_resolution_fields_fail_closed(
    field_name: str,
    bad_value: object,
    message: str,
) -> None:
    resolution = _resolution(_snapshot("final:nba-1"), "nba-1")

    with pytest.raises(NbaResolutionError, match=message):
        replace(resolution, **{field_name: bad_value})

    with pytest.raises(NbaResolutionError, match="cannot be tied"):
        replace(resolution, opponent_score=resolution.team_score)
    with pytest.raises(NbaResolutionError, match="must be in UTC"):
        replace(resolution, resolved_at=RESOLVED_AT.replace(tzinfo=None))
    with pytest.raises(FrozenInstanceError):
        resolution.team_score = 0  # pyright: ignore[reportAttributeAccessIssue]


def test_resolution_collection_requires_unique_original_game_identity(tmp_path: Path) -> None:
    first_snapshot = _snapshot("final:nba-1")
    second_snapshot = _snapshot("final:nba-2")
    index = NbaSnapshotIndex((first_snapshot, second_snapshot))
    first = _resolution(first_snapshot, "nba-1")
    path = tmp_path / "resolutions.jsonl"

    duplicate_question = replace(
        _resolution(second_snapshot, "nba-2"),
        question_id=first.question_id,
    )
    with pytest.raises(NbaResolutionError, match="duplicate question_id"):
        write_nba_resolutions_jsonl(
            path,
            (first, duplicate_question),
            snapshot_index=index,
        )

    duplicate_source_game = replace(
        _resolution(second_snapshot, "nba-2"),
        source_game_id=first.source_game_id,
    )
    with pytest.raises(NbaResolutionError, match="duplicate source_game_id"):
        write_nba_resolutions_jsonl(
            path,
            (first, duplicate_source_game),
            snapshot_index=index,
        )


def test_resolution_must_bind_the_latest_snapshot_available_when_resolved(
    tmp_path: Path,
) -> None:
    early = _snapshot(
        "final:nba-1",
        version="v1",
        retrieved_at=RESOLVED_AT - timedelta(minutes=20),
    )
    latest = _snapshot(
        "final:nba-1",
        version="v2",
        retrieved_at=RESOLVED_AT - timedelta(minutes=1),
    )
    index = NbaSnapshotIndex((early, latest))
    path = tmp_path / "resolutions.jsonl"

    with pytest.raises(NbaResolutionError, match="latest eligible snapshot"):
        write_nba_resolutions_jsonl(
            path,
            (_resolution(early, "nba-1"),),
            snapshot_index=index,
        )

    future = _snapshot(
        "final:nba-2",
        retrieved_at=RESOLVED_AT + timedelta(minutes=1),
    )
    with pytest.raises(NbaResolutionError, match="no source snapshot"):
        write_nba_resolutions_jsonl(
            path,
            (_resolution(future, "nba-2"),),
            snapshot_index=NbaSnapshotIndex((future,)),
        )


def test_resolution_loader_rejects_noncanonical_or_target_bearing_bytes(tmp_path: Path) -> None:
    snapshot = _snapshot("final:nba-1")
    index = NbaSnapshotIndex((snapshot,))
    path = tmp_path / "resolutions.jsonl"
    resolution = _resolution(snapshot, "nba-1")
    write_nba_resolutions_jsonl(path, (resolution,), snapshot_index=index)
    payload = json.loads(path.read_text(encoding="utf-8"))

    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(NbaResolutionError, match="canonical JSONL"):
        read_nba_resolutions_jsonl(path, snapshot_index=index)

    payload["label"] = TEAM_LABEL
    path.write_text(f"{canonical_json(payload)}\n", encoding="utf-8")
    with pytest.raises(NbaResolutionError, match="invalid NBA resolution"):
        read_nba_resolutions_jsonl(path, snapshot_index=index)


def test_validator_derives_both_original_and_swap_labels_from_scores() -> None:
    first_snapshot = _snapshot("final:nba-1")
    second_snapshot = _snapshot("final:nba-2")
    index = NbaSnapshotIndex((first_snapshot, second_snapshot))
    bundles = (_bundle("nba-1"), _bundle("nba-2"))
    resolutions = (
        _resolution(first_snapshot, "nba-1", team_score=115, opponent_score=109),
        _resolution(second_snapshot, "nba-2", team_score=98, opponent_score=104),
    )
    records = (
        _training_record("nba-1", TEAM_LABEL),
        _training_record(f"nba-1{SIDE_SWAP_SUFFIX}", OPPONENT_LABEL),
        _training_record("nba-2", OPPONENT_LABEL),
        _training_record(f"nba-2{SIDE_SWAP_SUFFIX}", TEAM_LABEL),
    )

    validate_outcome_training_labels(
        bundles,
        resolutions,
        records,
        snapshot_index=index,
        action_at=ACTION_AT,
    )

    wrong_labels = (
        _training_record("nba-1", OPPONENT_LABEL),
        *records[1:],
    )
    with pytest.raises(NbaResolutionError, match="sealed score winner"):
        validate_outcome_training_labels(
            bundles,
            resolutions,
            wrong_labels,
            snapshot_index=index,
            action_at=ACTION_AT,
        )


def test_validator_requires_exact_bundle_order_game_id_tipoff_and_complete_pairs() -> None:
    first_snapshot = _snapshot("final:nba-1")
    second_snapshot = _snapshot("final:nba-2")
    index = NbaSnapshotIndex((first_snapshot, second_snapshot))
    bundles = (_bundle("nba-1"), _bundle("nba-2"))
    first = _resolution(first_snapshot, "nba-1")
    second = _resolution(second_snapshot, "nba-2")
    records = (
        _training_record("nba-1", TEAM_LABEL),
        _training_record(f"nba-1{SIDE_SWAP_SUFFIX}", OPPONENT_LABEL),
        _training_record("nba-2", TEAM_LABEL),
        _training_record(f"nba-2{SIDE_SWAP_SUFFIX}", OPPONENT_LABEL),
    )

    with pytest.raises(NbaResolutionError, match="IDs or order"):
        validate_outcome_training_labels(
            bundles,
            (second, first),
            records,
            snapshot_index=index,
            action_at=ACTION_AT,
        )

    with pytest.raises(NbaResolutionError, match="source_game_id differs"):
        validate_outcome_training_labels(
            bundles,
            (replace(first, source_game_id="wrong-game"), second),
            records,
            snapshot_index=index,
            action_at=ACTION_AT,
        )

    tipoff_snapshot = _snapshot(
        "final:nba-1",
        version="tipoff-v1",
        retrieved_at=TIPOFF - timedelta(seconds=1),
    )
    tipoff_resolution = _resolution(
        tipoff_snapshot,
        "nba-1",
        resolved_at=TIPOFF,
    )
    with pytest.raises(NbaResolutionError, match="after the frozen scheduled tipoff"):
        validate_outcome_training_labels(
            (bundles[0],),
            (tipoff_resolution,),
            records[:2],
            snapshot_index=NbaSnapshotIndex((tipoff_snapshot,)),
            action_at=ACTION_AT,
        )

    with pytest.raises(NbaResolutionError, match="complete pair"):
        validate_outcome_training_labels(
            bundles,
            (first, second),
            records[:-1],
            snapshot_index=index,
            action_at=ACTION_AT,
        )

    with pytest.raises(NbaResolutionError, match="postdate the protected action"):
        validate_outcome_training_labels(
            (bundles[0],),
            (replace(first, resolved_at=ACTION_AT + timedelta(seconds=1)),),
            records[:2],
            snapshot_index=index,
            action_at=ACTION_AT,
        )
