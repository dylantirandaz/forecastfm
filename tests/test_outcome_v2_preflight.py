"""Tests for the complete offline outcome-v2 SFT readiness gate."""

import hashlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from forecastfm.integrity import canonical_json, canonical_sha256, file_sha256
from forecastfm.json_utils import parse_json_object, require_object, required_field
from forecastfm.ledger import CohortGame
from forecastfm.models import ForecastQuestion
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_elo_state import NbaEloState, write_nba_elo_states_jsonl
from forecastfm.nba_evidence import (
    NbaEvidenceBundle,
    NbaEvidenceRecord,
    SourceRights,
)
from forecastfm.nba_evidence_io import write_nba_evidence_bundles_jsonl
from forecastfm.nba_feature_rows import (
    NbaEloPriorInput,
    NbaRichFeatureRow,
    build_tinker_rich_feature_row,
    write_nba_feature_rows_jsonl,
)
from forecastfm.nba_resolutions import NbaResolution, write_nba_resolutions_jsonl
from forecastfm.nba_rich import (
    NBA_RICH_FEATURE_NAMES,
    NBA_RICH_FEATURE_SPECS,
    NBA_RICH_SCHEMA_SHA256,
    NBA_RICH_SCHEMA_VERSION,
)
from forecastfm.nba_rights_lock import NBA_RIGHTS_LOCK_SCHEMA_VERSION
from forecastfm.nba_snapshot_pack import (
    NbaSnapshot,
    NbaSnapshotIndex,
    NbaSnapshotMetadata,
    snapshot_metadata_sha256,
    write_snapshot_pack,
)
from forecastfm.outcome import (
    NBA_OUTCOMES,
    OPPONENT_LABEL,
    OPPONENT_OUTCOME,
    OUTCOME_SYSTEM_PROMPT,
    TEAM_LABEL,
    TEAM_OUTCOME,
)
from forecastfm.outcome_v2_config import (
    BATCH_SIZE,
    DROP_LAST,
    ELO_STATES_FILENAME,
    EVIDENCE_BUNDLES_FILENAME,
    FEATURE_ROWS_FILENAME,
    QUESTION_TEXT,
    RESOLUTION_RULE,
    RESOLUTIONS_FILENAME,
    RIGHTS_LOCK_FILENAME,
    SEASONS_FILENAME,
    SNAPSHOT_PACK_FILENAME,
    TRAINING_FILENAME,
    outcome_v2_sft_settings,
)
from forecastfm.outcome_v2_preflight import (
    OutcomeV2Artifacts,
    OutcomeV2Preflight,
    OutcomeV2PreflightError,
    require_outcome_v2_sft_ready,
)
from forecastfm.prompting import ChatMessage
from forecastfm.tinker_data import OutcomeTrainingRecord

PROJECT_ROOT = Path(__file__).parents[1]
CHECKED_IN_MANIFEST = PROJECT_ROOT / "data/processed/outcome_v2/manifest.json"

CUTOFF = datetime(2026, 6, 21, 22, tzinfo=UTC)
TIPOFF = CUTOFF + timedelta(minutes=60)
RESOLVED_AT = TIPOFF + timedelta(hours=3)
ACTION_AT = RESOLVED_AT + timedelta(days=1)
RIGHTS_AS_OF = datetime(2026, 1, 15, 18, 30, tzinfo=UTC)
ELO_AVAILABLE_AT = CUTOFF - timedelta(minutes=20)
PREGAME_AVAILABLE_AT = CUTOFF - timedelta(minutes=10)
FINAL_AVAILABLE_AT = TIPOFF + timedelta(hours=2)
SEASON = 2027

PROVIDER_ID = "fixture-provider"
LICENSE_ID = "order-2026-42"
AGREEMENT_REFERENCE = "vault://agreements/order-2026-42.pdf"
AGREEMENT_FILENAME = "reviewed-data-agreement.pdf"
PREGAME_SCOPE = "fixture-provider:nba:pregame"
FINAL_SCOPE = "fixture-provider:nba:final-scores"
APPROVED_SCOPES = (FINAL_SCOPE, PREGAME_SCOPE)
AGREEMENT_BYTES = b"exact reviewed agreement fixture bytes\n\x00\xff"


@dataclass(frozen=True, slots=True)
class _Fixture:
    manifest_path: Path
    training_path: Path
    artifacts: OutcomeV2Artifacts
    action_at: datetime
    original_ids: tuple[str, ...]


def _source_rights() -> SourceRights:
    return SourceRights(
        license_name=f"{PROVIDER_ID}/{LICENSE_ID}",
        terms_url=AGREEMENT_REFERENCE,
        terms_sha256=hashlib.sha256(AGREEMENT_BYTES).hexdigest(),
        rights_as_of=RIGHTS_AS_OF,
        local_processing="allowed",
        third_party_processing="allowed",
        tinker_processing="allowed",
        redistribution="prohibited",
    )


def _snapshot(
    source_id: str,
    rights_scope: str,
    payload: bytes,
    available_at: datetime,
) -> NbaSnapshot:
    return NbaSnapshot(
        metadata=NbaSnapshotMetadata(
            source_id=source_id,
            rights_scope=rights_scope,
            source_url=f"https://provider.test/{source_id}",
            version="v1",
            effective_at=available_at - timedelta(minutes=2),
            provider_published_at=available_at - timedelta(minutes=1),
            retrieved_at=available_at,
            available_at=available_at,
            capture_method="live",
            sensitivity="ordinary",
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            archive_attestation_sha256=None,
            rights=_source_rights(),
        ),
        payload=payload,
    )


def _evidence_bundle(question_id: str, snapshot: NbaSnapshot) -> NbaEvidenceBundle:
    source = snapshot.to_source_snapshot()
    records = tuple(
        NbaEvidenceRecord(
            record_id=f"{question_id}:feature-{index:02d}",
            kind=spec.kind,
            feature_name=spec.name,
            team_value=spec.minimum + 1.0,
            opponent_value=spec.minimum,
            source_ids=(source.source_id,),
            available_at=snapshot.metadata.available_at,
        )
        for index, spec in enumerate(NBA_RICH_FEATURE_SPECS)
    )
    game = CohortGame(
        question_id=question_id,
        source_game_id=f"provider-{question_id}",
        matchup="Listed team vs opponent",
        outcomes=NBA_OUTCOMES,
        forecast_deadline=CUTOFF,
        scheduled_tipoff=TIPOFF,
    )
    question = ForecastQuestion(
        question_id=question_id,
        text=QUESTION_TEXT,
        resolution_rule=RESOLUTION_RULE,
        resolution_source="https://provider.test/final-scores",
        outcomes=NBA_OUTCOMES,
        forecast_at=CUTOFF,
        resolves_at=RESOLVED_AT,
    )
    return NbaEvidenceBundle(
        game=game,
        question=question,
        sources=(source,),
        records=records,
    )


def _elo_state(question_id: str) -> NbaEloState:
    return NbaEloState(
        question_id=question_id,
        available_at=ELO_AVAILABLE_AT,
        team_rating=1_600.0,
        opponent_rating=1_500.0,
        home_advantage=0.0,
        rating_scale=400.0,
        recipe_sha256="e" * 64,
    )


def _resolution(
    question_id: str,
    index: int,
    snapshot: NbaSnapshot,
) -> NbaResolution:
    team_won = index % 2 == 0
    return NbaResolution(
        question_id=question_id,
        source_game_id=f"provider-{question_id}",
        team_score=110 if team_won else 100,
        opponent_score=100 if team_won else 110,
        resolved_at=RESOLVED_AT,
        source_id=snapshot.metadata.source_id,
        snapshot_metadata_sha256=snapshot_metadata_sha256(snapshot.metadata),
    )


def _user_content(row: NbaRichFeatureRow) -> str:
    return canonical_json(
        {
            "evidence": [
                f"Pregame numeric feature: {canonical_json({name: value})}"
                for name, value in row.feature_items
            ],
            "outcomes": [TEAM_OUTCOME, OPPONENT_OUTCOME],
            "prior": {
                TEAM_OUTCOME: row.elo_team_win_probability,
                OPPONENT_OUTCOME: row.elo_opponent_win_probability,
            },
            "question": QUESTION_TEXT,
            "resolution_rule": RESOLUTION_RULE,
        }
    )


def _training_record(row: NbaRichFeatureRow, label: str) -> OutcomeTrainingRecord:
    return OutcomeTrainingRecord(
        question_id=row.question_id,
        messages=[
            ChatMessage(role="system", content=OUTCOME_SYSTEM_PROMPT),
            ChatMessage(role="user", content=_user_content(row)),
        ],
        label=label,
    )


def _write_rights_files(directory: Path) -> tuple[Path, Path]:
    agreement_path = directory / AGREEMENT_FILENAME
    rights_lock_path = directory / RIGHTS_LOCK_FILENAME
    agreement_path.write_bytes(AGREEMENT_BYTES)
    rights_lock_path.write_text(
        canonical_json(
            {
                "schema_version": NBA_RIGHTS_LOCK_SCHEMA_VERSION,
                "provider_id": PROVIDER_ID,
                "license_id": LICENSE_ID,
                "agreement_reference": AGREEMENT_REFERENCE,
                "agreement_sha256": hashlib.sha256(AGREEMENT_BYTES).hexdigest(),
                "rights_as_of": "2026-01-15T18:30:00Z",
                "local_processing": "allowed",
                "third_party_processing": "allowed",
                "tinker_processing": "allowed",
                "redistribution": "prohibited",
                "approved_rights_scopes": list(APPROVED_SCOPES),
                "review_decision_id": "legal-review-1842",
            }
        ),
        encoding="utf-8",
    )
    return rights_lock_path, agreement_path


def _build_fixture(
    tmp_path: Path,
    *,
    pair_count: int = 7,
    ready: bool = True,
    row_rest_difference: float = 1.0,
    mislabel_first: bool = False,
) -> _Fixture:
    original_ids = tuple(f"game-{index}" for index in range(pair_count))
    pregame_snapshot = _snapshot(
        "pregame-feed",
        PREGAME_SCOPE,
        b"opaque pregame feature payload",
        PREGAME_AVAILABLE_AT,
    )
    final_snapshot = _snapshot(
        "final-score-feed",
        FINAL_SCOPE,
        b"opaque independently retained final-score payload",
        FINAL_AVAILABLE_AT,
    )
    snapshot_index = NbaSnapshotIndex((pregame_snapshot, final_snapshot))
    bundles = tuple(_evidence_bundle(question_id, pregame_snapshot) for question_id in original_ids)
    elo_states = tuple(_elo_state(question_id) for question_id in original_ids)

    feature_rows: list[NbaRichFeatureRow] = []
    for bundle, state in zip(bundles, elo_states, strict=True):
        row = build_tinker_rich_feature_row(
            bundle,
            season=SEASON,
            elo=NbaEloPriorInput(
                team_win_probability=state.team_win_probability,
                available_at=state.available_at,
                state_sha256=state.state_sha256,
            ),
            action_at=ACTION_AT,
        )
        if row_rest_difference != 1.0:
            changed = replace(
                row.rich_features,
                rest_days_difference=row_rest_difference,
            )
            row = replace(row, rich_features=changed)
        feature_rows.append(row)
    rows = tuple(feature_rows)
    resolutions = tuple(
        _resolution(question_id, index, final_snapshot)
        for index, question_id in enumerate(original_ids)
    )

    training_records: list[OutcomeTrainingRecord] = []
    for index, row in enumerate(rows):
        team_won = index % 2 == 0
        original_label = TEAM_LABEL if team_won else OPPONENT_LABEL
        swapped_label = OPPONENT_LABEL if team_won else TEAM_LABEL
        if mislabel_first and index == 0:
            original_label, swapped_label = swapped_label, original_label
        training_records.extend(
            (
                _training_record(row, original_label),
                _training_record(row.side_swap(), swapped_label),
            )
        )

    training_path = tmp_path / TRAINING_FILENAME
    feature_rows_path = tmp_path / FEATURE_ROWS_FILENAME
    snapshot_pack_path = tmp_path / SNAPSHOT_PACK_FILENAME
    evidence_bundles_path = tmp_path / EVIDENCE_BUNDLES_FILENAME
    elo_states_path = tmp_path / ELO_STATES_FILENAME
    seasons_path = tmp_path / SEASONS_FILENAME
    resolutions_path = tmp_path / RESOLUTIONS_FILENAME
    manifest_path = tmp_path / "manifest.json"

    training_path.write_text(
        "".join(f"{canonical_json(record)}\n" for record in training_records),
        encoding="utf-8",
    )
    write_nba_feature_rows_jsonl(feature_rows_path, rows)
    write_snapshot_pack(snapshot_index.snapshots, snapshot_pack_path)
    write_nba_evidence_bundles_jsonl(
        evidence_bundles_path,
        bundles,
        snapshot_index=snapshot_index,
    )
    write_nba_elo_states_jsonl(elo_states_path, elo_states)
    seasons_path.write_text(
        canonical_json(
            {
                "schema_version": 1,
                "seasons": [
                    {"question_id": question_id, "season": SEASON} for question_id in original_ids
                ],
            }
        ),
        encoding="utf-8",
    )
    write_nba_resolutions_jsonl(
        resolutions_path,
        resolutions,
        snapshot_index=snapshot_index,
    )
    rights_lock_path, agreement_path = _write_rights_files(tmp_path)

    output_paths = (
        training_path,
        feature_rows_path,
        snapshot_pack_path,
        evidence_bundles_path,
        elo_states_path,
        seasons_path,
        resolutions_path,
        rights_lock_path,
    )
    manifest_path.write_text(
        canonical_json(
            {
                "schema_version": 1,
                "outcome_input_schema_version": 2,
                "evaluation": {
                    "full_outcome_v2_ready": ready,
                    "full_outcome_v2_missing": [],
                    "historical_gate_passes_raw_elo": True,
                    "historical_gate_passes_recalibrated_elo": True,
                    "untouched_evaluation_seasons": [2024, 2025],
                },
                "features": {
                    "full_schema": {
                        "version": NBA_RICH_SCHEMA_VERSION,
                        "sha256": NBA_RICH_SCHEMA_SHA256,
                        "standard_names": list(NBA_RICH_FEATURE_NAMES),
                        "current_artifact_contains_full_schema": True,
                    }
                },
                "upload_rights": {
                    "third_party_processing": "allowed",
                    "tinker_processing": "allowed",
                    "player_health_included": False,
                },
                "outputs": {path.name: file_sha256(path) for path in output_paths},
                "splits": {
                    "train": {
                        "original_games": len(original_ids),
                        "side_swapped_training_rows": len(training_records),
                        "question_ids_sha256": canonical_sha256(list(original_ids)),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    artifacts = OutcomeV2Artifacts(
        feature_rows_path=feature_rows_path,
        snapshot_pack_path=snapshot_pack_path,
        evidence_bundles_path=evidence_bundles_path,
        elo_states_path=elo_states_path,
        seasons_path=seasons_path,
        resolutions_path=resolutions_path,
        rights_lock_path=rights_lock_path,
        agreement_path=agreement_path,
    )
    return _Fixture(
        manifest_path=manifest_path,
        training_path=training_path,
        artifacts=artifacts,
        action_at=ACTION_AT,
        original_ids=original_ids,
    )


def _require_ready(fixture: _Fixture) -> OutcomeV2Preflight:
    return require_outcome_v2_sft_ready(
        fixture.manifest_path,
        fixture.training_path,
        fixture.artifacts,
        action_at=fixture.action_at,
    )


def _set_output_hash(fixture: _Fixture, filename: str, sha256: str) -> None:
    manifest = parse_json_object(fixture.manifest_path.read_text(encoding="utf-8"))
    outputs = require_object(required_field(manifest, "outputs"), "outputs")
    outputs[filename] = sha256
    manifest["outputs"] = outputs
    fixture.manifest_path.write_text(canonical_json(manifest), encoding="utf-8")


def _rehash_output(fixture: _Fixture, path: Path) -> None:
    _set_output_hash(fixture, path.name, file_sha256(path))


def _set_evaluation_field(fixture: _Fixture, field_name: str, value: object) -> None:
    manifest = parse_json_object(fixture.manifest_path.read_text(encoding="utf-8"))
    evaluation = require_object(required_field(manifest, "evaluation"), "evaluation")
    evaluation[field_name] = value
    manifest["evaluation"] = evaluation
    fixture.manifest_path.write_text(canonical_json(manifest), encoding="utf-8")


def _artifact_path(fixture: _Fixture, filename: str) -> Path:
    paths = {
        TRAINING_FILENAME: fixture.training_path,
        FEATURE_ROWS_FILENAME: fixture.artifacts.feature_rows_path,
        SNAPSHOT_PACK_FILENAME: fixture.artifacts.snapshot_pack_path,
        EVIDENCE_BUNDLES_FILENAME: fixture.artifacts.evidence_bundles_path,
        ELO_STATES_FILENAME: fixture.artifacts.elo_states_path,
        SEASONS_FILENAME: fixture.artifacts.seasons_path,
        RESOLUTIONS_FILENAME: fixture.artifacts.resolutions_path,
        RIGHTS_LOCK_FILENAME: fixture.artifacts.rights_lock_path,
    }
    return paths[filename]


def test_config_names_every_required_sealed_artifact() -> None:
    settings = outcome_v2_sft_settings()

    assert BATCH_SIZE == 14
    assert DROP_LAST is False
    assert settings["batch_size"] == 14
    assert settings["drop_last"] is False
    assert settings["snapshot_pack_filename"] == SNAPSHOT_PACK_FILENAME
    assert settings["evidence_bundles_filename"] == EVIDENCE_BUNDLES_FILENAME
    assert settings["elo_states_filename"] == ELO_STATES_FILENAME
    assert settings["seasons_filename"] == SEASONS_FILENAME
    assert settings["resolutions_filename"] == RESOLUTIONS_FILENAME
    assert settings["rights_lock_filename"] == RIGHTS_LOCK_FILENAME


def test_checked_in_manifest_fails_before_missing_artifact_access() -> None:
    missing_training = PROJECT_ROOT / "does-not-exist" / TRAINING_FILENAME

    with pytest.raises(OutcomeV2PreflightError, match="full_outcome_v2_ready is false"):
        require_outcome_v2_sft_ready(CHECKED_IN_MANIFEST, missing_training)


def test_ready_manifest_requires_artifacts_and_action_time(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)

    with pytest.raises(OutcomeV2PreflightError, match="complete sealed NBA artifact set"):
        require_outcome_v2_sft_ready(fixture.manifest_path, fixture.training_path)
    with pytest.raises(OutcomeV2PreflightError, match="protected Tinker action time"):
        require_outcome_v2_sft_ready(
            fixture.manifest_path,
            fixture.training_path,
            fixture.artifacts,
        )

    future = replace(fixture, action_at=datetime.now(UTC) + timedelta(days=1))
    with pytest.raises(OutcomeV2PreflightError, match="cannot be in the future"):
        _require_ready(future)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("full_outcome_v2_missing", ["production connector"], "must be empty"),
        ("historical_gate_passes_raw_elo", False, "raw_elo is false"),
        (
            "historical_gate_passes_recalibrated_elo",
            False,
            "recalibrated_elo is false",
        ),
        ("untouched_evaluation_seasons", [2025], "two unique untouched"),
    ],
)
def test_readiness_boolean_cannot_bypass_evaluation_gates(
    tmp_path: Path,
    field_name: str,
    value: object,
    message: str,
) -> None:
    fixture = _build_fixture(tmp_path)
    _set_evaluation_field(fixture, field_name, value)

    with pytest.raises(OutcomeV2PreflightError, match=message):
        _require_ready(fixture)


def test_valid_seven_pair_artifact_graph_passes_without_tinker(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)

    result = _require_ready(fixture)

    assert result.manifest_sha256 == file_sha256(fixture.manifest_path)
    assert result.action_at == fixture.action_at
    assert result.training_sha256 == file_sha256(fixture.training_path)
    assert result.feature_rows_sha256 == file_sha256(fixture.artifacts.feature_rows_path)
    assert result.snapshot_pack_sha256 == file_sha256(fixture.artifacts.snapshot_pack_path)
    assert result.evidence_bundles_sha256 == file_sha256(fixture.artifacts.evidence_bundles_path)
    assert result.elo_states_sha256 == file_sha256(fixture.artifacts.elo_states_path)
    assert result.seasons_sha256 == file_sha256(fixture.artifacts.seasons_path)
    assert result.resolutions_sha256 == file_sha256(fixture.artifacts.resolutions_path)
    assert result.rights_lock_sha256 == file_sha256(fixture.artifacts.rights_lock_path)
    assert result.row_count == 14
    assert result.pair_count == 7
    assert result.batch_size == 14


@pytest.mark.parametrize(
    "filename",
    [
        SNAPSHOT_PACK_FILENAME,
        EVIDENCE_BUNDLES_FILENAME,
        ELO_STATES_FILENAME,
        SEASONS_FILENAME,
        RESOLUTIONS_FILENAME,
    ],
)
def test_provenance_artifact_manifest_hashes_fail_closed(
    tmp_path: Path,
    filename: str,
) -> None:
    fixture = _build_fixture(tmp_path)
    assert _artifact_path(fixture, filename).is_file()
    _set_output_hash(fixture, filename, "0" * 64)

    with pytest.raises(OutcomeV2PreflightError, match=rf"{filename} SHA-256"):
        _require_ready(fixture)


def test_training_feature_and_rights_hashes_remain_manifest_bound(tmp_path: Path) -> None:
    for filename in (TRAINING_FILENAME, FEATURE_ROWS_FILENAME, RIGHTS_LOCK_FILENAME):
        directory = tmp_path / filename.removesuffix(".jsonl").removesuffix(".json")
        directory.mkdir()
        fixture = _build_fixture(directory)
        _set_output_hash(fixture, filename, "0" * 64)
        with pytest.raises(OutcomeV2PreflightError, match=rf"{filename} SHA-256"):
            _require_ready(fixture)


def test_feature_claim_cannot_diverge_from_recomputed_evidence(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, row_rest_difference=2.0)

    with pytest.raises(OutcomeV2PreflightError, match="feature-row vector differs"):
        _require_ready(fixture)


def test_labels_must_match_independently_sealed_alternating_scores(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, mislabel_first=True)

    with pytest.raises(OutcomeV2PreflightError, match="labels do not match the sealed score"):
        _require_ready(fixture)


def test_prompt_content_cannot_diverge_from_sealed_feature_rows(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    text = fixture.training_path.read_text(encoding="utf-8")
    fixture.training_path.write_text(
        text.replace("rest_days", "arbitrary_signal"),
        encoding="utf-8",
    )
    _rehash_output(fixture, fixture.training_path)

    with pytest.raises(OutcomeV2PreflightError, match="differs from its sealed feature row"):
        _require_ready(fixture)


def test_upload_rights_must_match_reviewed_lock(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    manifest = parse_json_object(fixture.manifest_path.read_text(encoding="utf-8"))
    upload_rights = require_object(required_field(manifest, "upload_rights"), "upload_rights")
    upload_rights["tinker_processing"] = "unknown"
    manifest["upload_rights"] = upload_rights
    fixture.manifest_path.write_text(canonical_json(manifest), encoding="utf-8")

    with pytest.raises(OutcomeV2PreflightError, match="differs from the reviewed rights lock"):
        _require_ready(fixture)


def test_changed_agreement_bytes_fail_closed(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    fixture.artifacts.agreement_path.write_bytes(AGREEMENT_BYTES + b"tampered")

    with pytest.raises(OutcomeV2PreflightError, match="exact agreement bytes"):
        _require_ready(fixture)


def test_manifest_question_id_order_is_frozen(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    manifest = parse_json_object(fixture.manifest_path.read_text(encoding="utf-8"))
    splits = require_object(required_field(manifest, "splits"), "splits")
    train = require_object(required_field(splits, "train"), "train")
    train["question_ids_sha256"] = canonical_sha256(list(reversed(fixture.original_ids)))
    splits["train"] = train
    manifest["splits"] = splits
    fixture.manifest_path.write_text(canonical_json(manifest), encoding="utf-8")

    with pytest.raises(OutcomeV2PreflightError, match="IDs or their order"):
        _require_ready(fixture)


def test_batch_size_cannot_drop_training_rows(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, pair_count=6)

    with pytest.raises(OutcomeV2PreflightError, match="not divisible by batch size 14"):
        _require_ready(fixture)


def test_side_swap_rows_must_be_adjacent_and_exact(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    text = fixture.training_path.read_text(encoding="utf-8")
    fixture.training_path.write_text(
        text.replace(f"game-0{SIDE_SWAP_SUFFIX}", "game-0-wrong", 1),
        encoding="utf-8",
    )
    _rehash_output(fixture, fixture.training_path)

    with pytest.raises(OutcomeV2PreflightError, match="exact adjacent side-swap pairs"):
        _require_ready(fixture)
