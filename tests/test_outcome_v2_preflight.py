"""Tests for the complete offline outcome-v2 SFT readiness gate."""

import hashlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from forecastfm.integrity import canonical_json, canonical_sha256, file_sha256
from forecastfm.json_utils import (
    parse_json_object,
    require_float,
    require_list,
    require_object,
    required_field,
)
from forecastfm.ledger import CohortGame
from forecastfm.models import ForecastQuestion
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_elo_replay import (
    NbaEloReplayRow,
    replay_nba_elo_states,
    write_nba_elo_replay_rows_jsonl,
)
from forecastfm.nba_elo_state import NbaEloState, write_nba_elo_states_jsonl
from forecastfm.nba_evaluation_gate import (
    NbaEvaluationAnswer,
    NbaEvaluationCohortInput,
    NbaEvaluationGateArtifacts,
    NbaEvaluationGatePolicy,
    NbaRecalibrationRow,
    verify_untouched_nba_evaluation_gate,
    write_nba_evaluation_answers_jsonl,
    write_nba_evaluation_cohort_jsonl,
    write_nba_evaluation_forecasts_jsonl,
    write_nba_evaluation_gate_report,
    write_nba_recalibration_rows_jsonl,
)
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
    NbaRichFeatures,
)
from forecastfm.nba_rich_baseline import (
    build_nba_rich_baseline_forecast_lock,
    fit_nba_rich_baseline,
    predict_nba_rich_baseline,
    write_nba_rich_baseline_forecast_lock,
    write_nba_rich_baseline_model,
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
    TEAM_LABEL,
    TEAM_OUTCOME,
)
from forecastfm.outcome_v2_config import (
    BATCH_SIZE,
    DROP_LAST,
    ELO_REPLAY_FILENAME,
    ELO_STATES_FILENAME,
    EVALUATION_ANSWERS_FILENAME,
    EVALUATION_COHORT_FILENAME,
    EVALUATION_ELO_REPLAY_FILENAME,
    EVALUATION_ELO_STATES_FILENAME,
    EVALUATION_FEATURE_ROWS_FILENAME,
    EVALUATION_FORECASTS_FILENAME,
    EVALUATION_REPORT_FILENAME,
    EVALUATION_RESOLUTIONS_FILENAME,
    EVIDENCE_BUNDLES_FILENAME,
    FEATURE_ROWS_FILENAME,
    QUESTION_TEXT,
    RECALIBRATION_FILENAME,
    RESOLUTION_RULE,
    RESOLUTIONS_FILENAME,
    RICH_BASELINE_FORECAST_LOCK_FILENAME,
    RICH_BASELINE_MODEL_FILENAME,
    RIGHTS_LOCK_FILENAME,
    SEASONS_FILENAME,
    SNAPSHOT_PACK_FILENAME,
    TRAINING_FILENAME,
    outcome_v2_elo_recipe,
    outcome_v2_rich_baseline_fit_config,
    outcome_v2_sft_settings,
)
from forecastfm.outcome_v2_preflight import (
    OutcomeV2Artifacts,
    OutcomeV2Preflight,
    OutcomeV2PreflightError,
    PreparedOutcomeV2Run,
    prepare_outcome_v2_sft_run,
    require_outcome_v2_sft_ready,
)
from forecastfm.outcome_v2_prompt import OUTCOME_V2_SYSTEM_PROMPT
from forecastfm.prompting import ChatMessage
from forecastfm.tinker_data import (
    OutcomeTrainingRecord,
    read_outcome_training_jsonl_bytes,
)

PROJECT_ROOT = Path(__file__).parents[1]
CHECKED_IN_MANIFEST = PROJECT_ROOT / "data/processed/outcome_v2/manifest.json"

CUTOFF = datetime(2023, 10, 21, 22, tzinfo=UTC)
TIPOFF = CUTOFF + timedelta(minutes=60)
RESOLVED_AT = TIPOFF + timedelta(hours=3)
ACTION_AT = datetime(2026, 2, 1, tzinfo=UTC)
RIGHTS_AS_OF = datetime(2023, 1, 15, 18, 30, tzinfo=UTC)
ELO_AVAILABLE_AT = CUTOFF
PREGAME_AVAILABLE_AT = CUTOFF - timedelta(minutes=10)
FINAL_AVAILABLE_AT = TIPOFF + timedelta(hours=2)
SEASON = 2024
EVALUATION_SEASONS = (2025, 2026)
TEST_EVALUATION_POLICY = NbaEvaluationGatePolicy(
    minimum_games_per_season=1,
    minimum_calendar_blocks_per_season=1,
    recalibration_gradient_steps=2_000,
    recalibration_learning_rate=0.05,
    recalibration_initial_intercept=0.0,
    recalibration_initial_slope=1.0,
)

PROVIDER_ID = "fixture-provider"
LICENSE_ID = "order-2026-42"
AGREEMENT_REFERENCE = "vault://agreements/order-2026-42.pdf"
AGREEMENT_FILENAME = "reviewed-data-agreement.pdf"
PREGAME_SCOPE = "fixture-provider:nba:pregame"
FINAL_SCOPE = "fixture-provider:nba:final-scores"
APPROVED_SCOPES = (FINAL_SCOPE, PREGAME_SCOPE)
AGREEMENT_BYTES = b"exact reviewed agreement fixture bytes\n\x00\xff"


@pytest.fixture(autouse=True)
def use_test_preflight_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "forecastfm.outcome_v2_preflight.outcome_v2_evaluation_policy",
        lambda: TEST_EVALUATION_POLICY,
    )
    monkeypatch.setattr(
        "forecastfm.outcome_v2_preflight._require_reviewed_external_proofs",
        lambda: None,
    )


@dataclass(frozen=True, slots=True)
class _Fixture:
    manifest_path: Path
    training_path: Path
    artifacts: OutcomeV2Artifacts
    action_at: datetime
    original_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _EvaluationFixture:
    gate: NbaEvaluationGateArtifacts
    feature_rows_path: Path
    elo_replay_path: Path
    elo_states_path: Path
    resolutions_path: Path
    rich_baseline_model_path: Path
    rich_baseline_forecast_lock_path: Path


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


def _evidence_bundle(
    question_id: str,
    snapshot: NbaSnapshot,
    feature_difference: float,
) -> NbaEvidenceBundle:
    source = snapshot.to_source_snapshot()
    records = tuple(
        NbaEvidenceRecord(
            record_id=f"{question_id}:feature-{index:02d}",
            kind=spec.kind,
            feature_name=spec.name,
            team_value=spec.minimum + max(feature_difference, 0.0),
            opponent_value=spec.minimum + max(-feature_difference, 0.0),
            source_ids=(source.source_id,),
            available_at=snapshot.metadata.available_at,
        )
        for index, spec in enumerate(NBA_RICH_FEATURE_SPECS)
    )
    game = CohortGame(
        question_id=question_id,
        source_game_id=f"provider-{question_id}",
        team_id=f"team-{question_id}",
        opponent_id=f"opponent-{question_id}",
        site="neutral",
        matchup=f"team-{question_id} vs opponent-{question_id}",
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
    recipe = outcome_v2_elo_recipe()
    return NbaEloState(
        question_id=question_id,
        available_at=ELO_AVAILABLE_AT,
        team_rating=recipe.initial_rating,
        opponent_rating=recipe.initial_rating,
        home_advantage=0.0,
        rating_scale=recipe.rating_scale,
        recipe_sha256=recipe.recipe_sha256,
    )


def _elo_replay_row(question_id: str, index: int) -> NbaEloReplayRow:
    return NbaEloReplayRow(
        question_id=question_id,
        source_game_id=f"provider-{question_id}",
        season=SEASON,
        team_id=f"team-{question_id}",
        opponent_id=f"opponent-{question_id}",
        site="neutral",
        forecast_cutoff=CUTOFF,
        scheduled_tipoff=TIPOFF,
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
        team_id=f"team-{question_id}",
        opponent_id=f"opponent-{question_id}",
        site="neutral",
        team_score=110 if team_won else 100,
        opponent_score=100 if team_won else 110,
        resolved_at=RESOLVED_AT,
        source_id=snapshot.metadata.source_id,
        snapshot_metadata_sha256=snapshot_metadata_sha256(snapshot.metadata),
    )


def _write_evaluation_files(
    directory: Path,
    training_feature_rows: tuple[NbaRichFeatureRow, ...],
    elo_states: tuple[NbaEloState, ...],
    resolutions: tuple[NbaResolution, ...],
    snapshot_index: NbaSnapshotIndex,
) -> _EvaluationFixture:
    original_ids = tuple(row.question_id for row in training_feature_rows)
    cohort_path = directory / EVALUATION_COHORT_FILENAME
    feature_rows_path = directory / EVALUATION_FEATURE_ROWS_FILENAME
    answers_path = directory / EVALUATION_ANSWERS_FILENAME
    forecasts_path = directory / EVALUATION_FORECASTS_FILENAME
    calibration_path = directory / RECALIBRATION_FILENAME
    report_path = directory / EVALUATION_REPORT_FILENAME
    evaluation_replay_path = directory / EVALUATION_ELO_REPLAY_FILENAME
    evaluation_states_path = directory / EVALUATION_ELO_STATES_FILENAME
    evaluation_resolutions_path = directory / EVALUATION_RESOLUTIONS_FILENAME
    rich_baseline_model_path = directory / RICH_BASELINE_MODEL_FILENAME
    rich_baseline_forecast_lock_path = directory / RICH_BASELINE_FORECAST_LOCK_FILENAME

    replay_rows = tuple(
        NbaEloReplayRow(
            question_id=f"evaluation-{season}",
            source_game_id=f"provider-evaluation-{season}",
            season=season,
            team_id=f"evaluation-team-{season}",
            opponent_id=f"evaluation-opponent-{season}",
            site="neutral",
            forecast_cutoff=datetime(season, 1, 8, 18, tzinfo=UTC),
            scheduled_tipoff=datetime(season, 1, 8, 19, tzinfo=UTC),
        )
        for season in EVALUATION_SEASONS
    )
    evaluation_resolutions = tuple(
        NbaResolution(
            question_id=row.question_id,
            source_game_id=row.source_game_id,
            team_id=row.team_id,
            opponent_id=row.opponent_id,
            site=row.site,
            team_score=110 if index % 2 == 0 else 100,
            opponent_score=100 if index % 2 == 0 else 110,
            resolved_at=row.scheduled_tipoff + timedelta(hours=3),
            source_id=f"evaluation-final-{row.season}",
            snapshot_metadata_sha256=snapshot_metadata_sha256(
                _required_snapshot(
                    snapshot_index,
                    f"evaluation-final-{row.season}",
                    row.scheduled_tipoff + timedelta(hours=3),
                ).metadata
            ),
        )
        for index, row in enumerate(replay_rows)
    )
    evaluation_states = replay_nba_elo_states(
        replay_rows,
        evaluation_resolutions,
        outcome_v2_elo_recipe(),
    )
    evaluation_feature_rows = tuple(
        NbaRichFeatureRow(
            question_id=row.question_id,
            source_game_id=row.source_game_id,
            team_id=row.team_id,
            opponent_id=row.opponent_id,
            site=row.site,
            season=row.season,
            forecast_cutoff=row.forecast_cutoff,
            scheduled_tipoff=row.scheduled_tipoff,
            elo_team_win_probability=state.team_win_probability,
            elo_opponent_win_probability=1.0 - state.team_win_probability,
            elo_available_at=state.available_at,
            elo_state_sha256=state.state_sha256,
            rich_features=NbaRichFeatures.from_vector(
                (1.0 if resolution.team_won else -1.0,) * len(NBA_RICH_FEATURE_NAMES)
            ),
            evidence_bundle_sha256=hashlib.sha256(
                f"evaluation-evidence:{row.question_id}".encode()
            ).hexdigest(),
            input_available_at=state.available_at,
        )
        for row, state, resolution in zip(
            replay_rows,
            evaluation_states,
            evaluation_resolutions,
            strict=True,
        )
    )
    cohort = tuple(
        NbaEvaluationCohortInput(
            question_id=row.question_id,
            season=row.season,
            game_date=row.scheduled_tipoff.date(),
            raw_elo_team_probability=state.team_win_probability,
        )
        for row, state in zip(replay_rows, evaluation_states, strict=True)
    )
    answers = tuple(
        NbaEvaluationAnswer(resolution.question_id, resolution.team_won)
        for resolution in evaluation_resolutions
    )
    model = fit_nba_rich_baseline(
        training_feature_rows,
        resolutions,
        outcome_v2_rich_baseline_fit_config(),
    )
    forecasts = predict_nba_rich_baseline(model, evaluation_feature_rows)
    forecast_lock = build_nba_rich_baseline_forecast_lock(
        model,
        evaluation_feature_rows,
        forecasts,
    )
    calibration = tuple(
        NbaRecalibrationRow(
            question_id=question_id,
            season=SEASON,
            game_date=TIPOFF.date(),
            raw_elo_team_probability=state.team_win_probability,
            realized_team_win=resolution.team_won,
        )
        for question_id, state, resolution in zip(
            original_ids,
            elo_states,
            resolutions,
            strict=True,
        )
    )

    write_nba_evaluation_cohort_jsonl(cohort_path, cohort)
    write_nba_feature_rows_jsonl(feature_rows_path, evaluation_feature_rows)
    write_nba_evaluation_answers_jsonl(answers_path, answers)
    write_nba_evaluation_forecasts_jsonl(forecasts_path, forecasts)
    write_nba_rich_baseline_model(rich_baseline_model_path, model)
    write_nba_rich_baseline_forecast_lock(
        rich_baseline_forecast_lock_path,
        forecast_lock,
    )
    write_nba_recalibration_rows_jsonl(calibration_path, calibration)
    write_nba_elo_replay_rows_jsonl(evaluation_replay_path, replay_rows)
    write_nba_elo_states_jsonl(evaluation_states_path, evaluation_states)
    write_nba_resolutions_jsonl(
        evaluation_resolutions_path,
        evaluation_resolutions,
        snapshot_index=snapshot_index,
    )
    inputs = NbaEvaluationGateArtifacts(
        cohort_path=cohort_path,
        answers_path=answers_path,
        forecasts_path=forecasts_path,
        calibration_path=calibration_path,
    )
    report = verify_untouched_nba_evaluation_gate(
        inputs,
        policy=TEST_EVALUATION_POLICY,
    )
    write_nba_evaluation_gate_report(report_path, report)
    return _EvaluationFixture(
        gate=replace(inputs, supplied_report_path=report_path),
        feature_rows_path=feature_rows_path,
        elo_replay_path=evaluation_replay_path,
        elo_states_path=evaluation_states_path,
        resolutions_path=evaluation_resolutions_path,
        rich_baseline_model_path=rich_baseline_model_path,
        rich_baseline_forecast_lock_path=rich_baseline_forecast_lock_path,
    )


def _required_snapshot(
    snapshot_index: NbaSnapshotIndex,
    source_id: str,
    available_by: datetime,
) -> NbaSnapshot:
    snapshot = snapshot_index.latest_eligible(source_id, available_by)
    assert snapshot is not None
    return snapshot


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
            ChatMessage(role="system", content=OUTCOME_V2_SYSTEM_PROMPT),
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
                "rights_as_of": "2023-01-15T18:30:00Z",
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
    evaluation_final_snapshots = tuple(
        _snapshot(
            f"evaluation-final-{season}",
            FINAL_SCOPE,
            f"opaque evaluation final payload {season}".encode(),
            datetime(season, 1, 8, 21, tzinfo=UTC),
        )
        for season in EVALUATION_SEASONS
    )
    snapshot_index = NbaSnapshotIndex(
        (pregame_snapshot, final_snapshot, *evaluation_final_snapshots)
    )
    bundles = tuple(
        _evidence_bundle(
            question_id,
            pregame_snapshot,
            1.0 if index % 2 == 0 else -1.0,
        )
        for index, question_id in enumerate(original_ids)
    )
    elo_states = tuple(_elo_state(question_id) for question_id in original_ids)
    elo_replay_rows = tuple(
        _elo_replay_row(question_id, index) for index, question_id in enumerate(original_ids)
    )

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

    (
        training_path,
        feature_rows_path,
        snapshot_pack_path,
        evidence_bundles_path,
        elo_states_path,
        elo_replay_path,
        seasons_path,
        resolutions_path,
        manifest_path,
    ) = (
        tmp_path / TRAINING_FILENAME,
        tmp_path / FEATURE_ROWS_FILENAME,
        tmp_path / SNAPSHOT_PACK_FILENAME,
        tmp_path / EVIDENCE_BUNDLES_FILENAME,
        tmp_path / ELO_STATES_FILENAME,
        tmp_path / ELO_REPLAY_FILENAME,
        tmp_path / SEASONS_FILENAME,
        tmp_path / RESOLUTIONS_FILENAME,
        tmp_path / "manifest.json",
    )

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
    write_nba_elo_replay_rows_jsonl(elo_replay_path, elo_replay_rows)
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
    evaluation = _write_evaluation_files(
        tmp_path,
        rows,
        elo_states,
        resolutions,
        snapshot_index,
    )
    assert evaluation.gate.supplied_report_path is not None

    output_paths = (
        training_path,
        feature_rows_path,
        snapshot_pack_path,
        evidence_bundles_path,
        elo_states_path,
        elo_replay_path,
        seasons_path,
        resolutions_path,
        rights_lock_path,
        evaluation.gate.cohort_path,
        evaluation.feature_rows_path,
        evaluation.gate.answers_path,
        evaluation.gate.forecasts_path,
        evaluation.gate.calibration_path,
        evaluation.elo_replay_path,
        evaluation.elo_states_path,
        evaluation.resolutions_path,
        evaluation.rich_baseline_model_path,
        evaluation.rich_baseline_forecast_lock_path,
        evaluation.gate.supplied_report_path,
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
                    "untouched_evaluation_seasons": list(EVALUATION_SEASONS),
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
        elo_replay_path=elo_replay_path,
        seasons_path=seasons_path,
        resolutions_path=resolutions_path,
        rights_lock_path=rights_lock_path,
        agreement_path=agreement_path,
        evaluation=evaluation.gate,
        evaluation_feature_rows_path=evaluation.feature_rows_path,
        evaluation_elo_replay_path=evaluation.elo_replay_path,
        evaluation_elo_states_path=evaluation.elo_states_path,
        evaluation_resolutions_path=evaluation.resolutions_path,
        rich_baseline_model_path=evaluation.rich_baseline_model_path,
        rich_baseline_forecast_lock_path=evaluation.rich_baseline_forecast_lock_path,
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


def _replace_jsonl_field(
    path: Path,
    line_index: int,
    field_name: str,
    value: object,
) -> None:
    payloads = [parse_json_object(line) for line in path.read_text(encoding="utf-8").splitlines()]
    payloads[line_index][field_name] = value
    path.write_text(
        "".join(f"{canonical_json(payload)}\n" for payload in payloads),
        encoding="utf-8",
    )


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
        ELO_REPLAY_FILENAME: fixture.artifacts.elo_replay_path,
        SEASONS_FILENAME: fixture.artifacts.seasons_path,
        RESOLUTIONS_FILENAME: fixture.artifacts.resolutions_path,
        RIGHTS_LOCK_FILENAME: fixture.artifacts.rights_lock_path,
        EVALUATION_COHORT_FILENAME: fixture.artifacts.evaluation.cohort_path,
        EVALUATION_FEATURE_ROWS_FILENAME: fixture.artifacts.evaluation_feature_rows_path,
        EVALUATION_ANSWERS_FILENAME: fixture.artifacts.evaluation.answers_path,
        EVALUATION_FORECASTS_FILENAME: fixture.artifacts.evaluation.forecasts_path,
        EVALUATION_ELO_REPLAY_FILENAME: fixture.artifacts.evaluation_elo_replay_path,
        EVALUATION_ELO_STATES_FILENAME: fixture.artifacts.evaluation_elo_states_path,
        EVALUATION_RESOLUTIONS_FILENAME: fixture.artifacts.evaluation_resolutions_path,
        RECALIBRATION_FILENAME: fixture.artifacts.evaluation.calibration_path,
        RICH_BASELINE_MODEL_FILENAME: fixture.artifacts.rich_baseline_model_path,
        RICH_BASELINE_FORECAST_LOCK_FILENAME: (fixture.artifacts.rich_baseline_forecast_lock_path),
    }
    report_path = fixture.artifacts.evaluation.supplied_report_path
    if report_path is not None:
        paths[EVALUATION_REPORT_FILENAME] = report_path
    return paths[filename]


def test_config_names_every_required_sealed_artifact() -> None:
    settings = outcome_v2_sft_settings()

    assert BATCH_SIZE == 14
    assert DROP_LAST is False
    assert settings["batch_size"] == 14
    assert settings["drop_last"] is False
    assert settings["training_filename"] == TRAINING_FILENAME
    assert settings["feature_rows_filename"] == FEATURE_ROWS_FILENAME
    assert settings["snapshot_pack_filename"] == SNAPSHOT_PACK_FILENAME
    assert settings["evidence_bundles_filename"] == EVIDENCE_BUNDLES_FILENAME
    assert settings["elo_states_filename"] == ELO_STATES_FILENAME
    assert settings["elo_replay_filename"] == ELO_REPLAY_FILENAME
    assert settings["seasons_filename"] == SEASONS_FILENAME
    assert settings["resolutions_filename"] == RESOLUTIONS_FILENAME
    assert settings["rights_lock_filename"] == RIGHTS_LOCK_FILENAME
    assert settings["evaluation_cohort_filename"] == EVALUATION_COHORT_FILENAME
    assert settings["evaluation_feature_rows_filename"] == EVALUATION_FEATURE_ROWS_FILENAME
    assert settings["evaluation_answers_filename"] == EVALUATION_ANSWERS_FILENAME
    assert settings["evaluation_forecasts_filename"] == EVALUATION_FORECASTS_FILENAME
    assert settings["evaluation_elo_replay_filename"] == EVALUATION_ELO_REPLAY_FILENAME
    assert settings["evaluation_elo_states_filename"] == EVALUATION_ELO_STATES_FILENAME
    assert settings["evaluation_resolutions_filename"] == EVALUATION_RESOLUTIONS_FILENAME
    assert settings["recalibration_filename"] == RECALIBRATION_FILENAME
    assert settings["evaluation_report_filename"] == EVALUATION_REPORT_FILENAME
    assert settings["rich_baseline_model_filename"] == RICH_BASELINE_MODEL_FILENAME
    assert settings["rich_baseline_forecast_lock_filename"] == RICH_BASELINE_FORECAST_LOCK_FILENAME


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


def test_production_preflight_fails_without_external_proof_verifiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    monkeypatch.undo()
    monkeypatch.setattr(
        "forecastfm.outcome_v2_preflight.outcome_v2_evaluation_policy",
        lambda: TEST_EVALUATION_POLICY,
    )

    with pytest.raises(OutcomeV2PreflightError, match="no reviewed production connector"):
        _require_ready(fixture)


def test_prepared_run_retains_the_exact_validated_training_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    original_bytes = fixture.training_path.read_bytes()
    monkeypatch.setattr(
        "forecastfm.outcome_v2_preflight._utc_now",
        lambda: ACTION_AT,
    )
    offline_proof = _require_ready(fixture)

    prepared = prepare_outcome_v2_sft_run(
        fixture.manifest_path,
        fixture.training_path,
        fixture.artifacts,
    )
    fixture.training_path.write_bytes(b"changed after preparation")

    assert prepared.proof.action_at == ACTION_AT
    assert prepared.proof.action_time_source == "internal_paid_preparation"
    assert prepared.training_jsonl == original_bytes
    assert (
        len(
            read_outcome_training_jsonl_bytes(
                prepared.training_jsonl,
                expected_system_prompt=OUTCOME_V2_SYSTEM_PROMPT,
            )
        )
        == 14
    )
    with pytest.raises(OutcomeV2PreflightError, match="differ from the preflight proof"):
        PreparedOutcomeV2Run(prepared.proof, b"different bytes")
    with pytest.raises(OutcomeV2PreflightError, match="internally derived action time"):
        PreparedOutcomeV2Run(offline_proof, original_bytes)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("full_outcome_v2_missing", ["production connector"], "must be empty"),
        ("untouched_evaluation_seasons", [2025], "two unique untouched"),
    ],
)
def test_readiness_summary_cannot_bypass_required_evaluation_inputs(
    tmp_path: Path,
    field_name: str,
    value: object,
    message: str,
) -> None:
    fixture = _build_fixture(tmp_path)
    _set_evaluation_field(fixture, field_name, value)

    with pytest.raises(OutcomeV2PreflightError, match=message):
        _require_ready(fixture)


def test_historical_diagnostic_booleans_are_not_gate_authority(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    _set_evaluation_field(fixture, "historical_gate_passes_raw_elo", False)
    _set_evaluation_field(fixture, "historical_gate_passes_recalibrated_elo", False)

    result = _require_ready(fixture)

    assert result.untouched_evaluation_seasons == EVALUATION_SEASONS


def test_valid_seven_pair_artifact_graph_passes_without_tinker(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)

    result = _require_ready(fixture)

    assert result.manifest_sha256 == file_sha256(fixture.manifest_path)
    assert result.action_at == fixture.action_at
    assert result.action_time_source == "caller_supplied_offline_check"
    assert result.untouched_evaluation_seasons == EVALUATION_SEASONS
    assert result.training_sha256 == file_sha256(fixture.training_path)
    assert result.feature_rows_sha256 == file_sha256(fixture.artifacts.feature_rows_path)
    assert result.snapshot_pack_sha256 == file_sha256(fixture.artifacts.snapshot_pack_path)
    assert result.evidence_bundles_sha256 == file_sha256(fixture.artifacts.evidence_bundles_path)
    assert result.elo_states_sha256 == file_sha256(fixture.artifacts.elo_states_path)
    assert result.elo_replay_sha256 == file_sha256(fixture.artifacts.elo_replay_path)
    assert result.seasons_sha256 == file_sha256(fixture.artifacts.seasons_path)
    assert result.resolutions_sha256 == file_sha256(fixture.artifacts.resolutions_path)
    assert result.rights_lock_sha256 == file_sha256(fixture.artifacts.rights_lock_path)
    assert result.evaluation_feature_rows_sha256 == file_sha256(
        fixture.artifacts.evaluation_feature_rows_path
    )
    assert result.evaluation_elo_replay_sha256 == file_sha256(
        fixture.artifacts.evaluation_elo_replay_path
    )
    assert result.evaluation_elo_states_sha256 == file_sha256(
        fixture.artifacts.evaluation_elo_states_path
    )
    assert result.evaluation_resolutions_sha256 == file_sha256(
        fixture.artifacts.evaluation_resolutions_path
    )
    assert result.rich_baseline_model_sha256 == file_sha256(
        fixture.artifacts.rich_baseline_model_path
    )
    assert result.rich_baseline_forecast_lock_sha256 == file_sha256(
        fixture.artifacts.rich_baseline_forecast_lock_path
    )
    report_path = fixture.artifacts.evaluation.supplied_report_path
    assert report_path is not None
    assert result.evaluation_report_sha256 == file_sha256(report_path)
    assert result.row_count == 14
    assert result.pair_count == 7
    assert result.batch_size == 14


@pytest.mark.parametrize(
    "filename",
    [
        SNAPSHOT_PACK_FILENAME,
        EVIDENCE_BUNDLES_FILENAME,
        ELO_STATES_FILENAME,
        ELO_REPLAY_FILENAME,
        SEASONS_FILENAME,
        RESOLUTIONS_FILENAME,
        EVALUATION_COHORT_FILENAME,
        EVALUATION_FEATURE_ROWS_FILENAME,
        EVALUATION_ANSWERS_FILENAME,
        EVALUATION_FORECASTS_FILENAME,
        EVALUATION_ELO_REPLAY_FILENAME,
        EVALUATION_ELO_STATES_FILENAME,
        EVALUATION_RESOLUTIONS_FILENAME,
        RECALIBRATION_FILENAME,
        RICH_BASELINE_MODEL_FILENAME,
        RICH_BASELINE_FORECAST_LOCK_FILENAME,
        EVALUATION_REPORT_FILENAME,
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


def test_elo_replay_schedule_must_match_the_training_bundle(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    _replace_jsonl_field(
        fixture.artifacts.elo_replay_path,
        0,
        "forecast_cutoff",
        "2023-10-20T22:00:00Z",
    )
    _replace_jsonl_field(
        fixture.artifacts.elo_replay_path,
        0,
        "scheduled_tipoff",
        "2023-10-20T23:00:00Z",
    )
    _rehash_output(fixture, fixture.artifacts.elo_replay_path)

    with pytest.raises(OutcomeV2PreflightError, match="replay cutoff differs"):
        _require_ready(fixture)


def test_recalibration_must_be_the_exact_training_cohort(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    calibration_path = fixture.artifacts.evaluation.calibration_path
    _replace_jsonl_field(calibration_path, 0, "raw_elo_team_probability", 0.6)
    _rehash_output(fixture, calibration_path)

    with pytest.raises(OutcomeV2PreflightError, match="probability differs from training"):
        _require_ready(fixture)


def test_evaluation_gate_is_recomputed_instead_of_trusting_booleans(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    forecasts_path = fixture.artifacts.evaluation.forecasts_path
    _replace_jsonl_field(forecasts_path, 0, "team_probability", 0.1)
    _replace_jsonl_field(forecasts_path, 1, "team_probability", 0.9)
    _rehash_output(fixture, forecasts_path)

    with pytest.raises(OutcomeV2PreflightError, match="deterministic rich baseline inference"):
        _require_ready(fixture)


def test_rich_baseline_model_is_refit_instead_of_trusting_sealed_weights(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    model_path = fixture.artifacts.rich_baseline_model_path
    payload = parse_json_object(model_path.read_text(encoding="utf-8"))
    weights = require_list(required_field(payload, "weights"), "weights")
    first_weight = require_float(weights[0], "weights[0]")
    weights[0] = first_weight + 0.1
    payload["weights"] = weights
    model_path.write_text(canonical_json(payload), encoding="utf-8")
    _rehash_output(fixture, model_path)

    with pytest.raises(OutcomeV2PreflightError, match="deterministic training replay"):
        _require_ready(fixture)


def test_rich_baseline_forecast_lock_is_rebuilt_from_answer_free_rows(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    lock_path = fixture.artifacts.rich_baseline_forecast_lock_path
    payload = parse_json_object(lock_path.read_text(encoding="utf-8"))
    payload["model_sha256"] = "0" * 64
    lock_path.write_text(canonical_json(payload), encoding="utf-8")
    _rehash_output(fixture, lock_path)

    with pytest.raises(OutcomeV2PreflightError, match="forecast lock differs"):
        _require_ready(fixture)


def test_evaluation_feature_state_hash_is_bound_to_replayed_elo(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    feature_rows_path = fixture.artifacts.evaluation_feature_rows_path
    _replace_jsonl_field(feature_rows_path, 0, "elo_state_sha256", "f" * 64)
    _rehash_output(fixture, feature_rows_path)

    with pytest.raises(OutcomeV2PreflightError, match="Elo state digest differs"):
        _require_ready(fixture)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("question_id", "wrong-evaluation-id", "identical IDs and order"),
        ("season", 2027, "evaluation season differs"),
    ],
)
def test_evaluation_feature_identity_and_season_match_the_cohort(
    tmp_path: Path,
    field_name: str,
    value: object,
    message: str,
) -> None:
    fixture = _build_fixture(tmp_path)
    feature_rows_path = fixture.artifacts.evaluation_feature_rows_path
    _replace_jsonl_field(feature_rows_path, 0, field_name, value)
    _rehash_output(fixture, feature_rows_path)

    with pytest.raises(OutcomeV2PreflightError, match=message):
        _require_ready(fixture)


def test_evaluation_feature_date_and_cutoff_match_the_replay(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    feature_rows_path = fixture.artifacts.evaluation_feature_rows_path
    _replace_jsonl_field(feature_rows_path, 0, "forecast_cutoff", "2025-01-09T18:00:00Z")
    _replace_jsonl_field(feature_rows_path, 0, "scheduled_tipoff", "2025-01-09T19:00:00Z")
    _rehash_output(fixture, feature_rows_path)

    with pytest.raises(OutcomeV2PreflightError, match="evaluation date differs"):
        _require_ready(fixture)


def test_evaluation_feature_elo_probability_matches_the_replayed_state(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    feature_rows_path = fixture.artifacts.evaluation_feature_rows_path
    _replace_jsonl_field(feature_rows_path, 0, "elo_team_win_probability", 0.6)
    _replace_jsonl_field(feature_rows_path, 0, "elo_opponent_win_probability", 0.4)
    _rehash_output(fixture, feature_rows_path)

    with pytest.raises(OutcomeV2PreflightError, match="Elo team probability differs"):
        _require_ready(fixture)


def test_evaluation_raw_elo_is_bound_to_its_replayed_state(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    cohort_path = fixture.artifacts.evaluation.cohort_path
    _replace_jsonl_field(cohort_path, 0, "raw_elo_team_probability", 0.6)
    _rehash_output(fixture, cohort_path)

    with pytest.raises(OutcomeV2PreflightError, match="raw Elo differs"):
        _require_ready(fixture)


def test_evaluation_answer_is_bound_to_its_sealed_final_score(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    answers_path = fixture.artifacts.evaluation.answers_path
    _replace_jsonl_field(answers_path, 0, "realized_team_win", False)
    _rehash_output(fixture, answers_path)

    with pytest.raises(OutcomeV2PreflightError, match="answer differs from the sealed resolution"):
        _require_ready(fixture)


def test_manifest_seasons_must_equal_the_sealed_evaluation_cohort(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    _set_evaluation_field(fixture, "untouched_evaluation_seasons", [2025, 2027])

    with pytest.raises(OutcomeV2PreflightError, match="seasons differ from the manifest"):
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


def test_training_seasons_cannot_be_declared_untouched(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    _set_evaluation_field(fixture, "untouched_evaluation_seasons", [SEASON, 2025])

    with pytest.raises(OutcomeV2PreflightError, match="training seasons overlap"):
        _require_ready(fixture)


def test_final_partial_batch_is_retained(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, pair_count=6)

    result = _require_ready(fixture)

    assert result.row_count == 12
    assert result.pair_count == 6


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
