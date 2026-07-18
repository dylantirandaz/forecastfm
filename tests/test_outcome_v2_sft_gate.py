"""Tests for the separate answer-free and post-SFT evaluation chain."""

import base64
import sys
from dataclasses import dataclass, fields, replace
from datetime import UTC, date, datetime, timedelta
from math import log
from pathlib import Path

import pytest

import forecastfm.github_actions_receipt as github_receipt_module
import forecastfm.outcome_v2_run as outcome_v2_run_module
import forecastfm.outcome_v2_sft_gate as sft_gate_module
from forecastfm.github_actions_receipt import (
    GitHubActionsReceiptPolicy,
    GitHubActionsReceiptRequest,
    build_github_actions_receipt,
    write_github_actions_receipt,
)
from forecastfm.integrity import bytes_sha256, canonical_json, canonical_sha256, file_sha256
from forecastfm.json_utils import require_object, require_string, required_field
from forecastfm.nba_elo_replay import NbaEloReplayRow, write_nba_elo_replay_rows_jsonl
from forecastfm.nba_evaluation_gate import (
    NbaEvaluationAnswer,
    NbaEvaluationCohortInput,
    NbaEvaluationGateArtifacts,
    NbaEvaluationGatePolicy,
    NbaRecalibrationRow,
    nba_recalibration_rows_jsonl_bytes,
    read_nba_evaluation_forecasts_jsonl,
    verify_untouched_nba_evaluation_gate,
    write_nba_evaluation_answers_jsonl,
    write_nba_evaluation_cohort_jsonl,
    write_nba_evaluation_forecasts_jsonl,
    write_nba_evaluation_gate_report,
    write_nba_recalibration_rows_jsonl,
)
from forecastfm.nba_evidence import SourceRights
from forecastfm.nba_feature_rows import NbaRichFeatureRow, write_nba_feature_rows_jsonl
from forecastfm.nba_provider_conformance import NbaProviderConformanceReport
from forecastfm.nba_resolutions import NbaResolution, write_nba_resolutions_jsonl
from forecastfm.nba_rich import NbaRichFeatures
from forecastfm.nba_snapshot_pack import (
    NbaSnapshot,
    NbaSnapshotIndex,
    NbaSnapshotMetadata,
    snapshot_metadata_sha256,
    write_snapshot_pack,
)
from forecastfm.outcome_v2_aggregation import (
    OutcomeV2AggregationError,
    OutcomeV2RollingAggregate,
    OutcomeV2RollingAggregateFiles,
    OutcomeV2RollingAggregationArtifacts,
    build_outcome_v2_rolling_aggregate,
    verify_outcome_v2_rolling_aggregate,
    write_outcome_v2_rolling_aggregate,
)
from forecastfm.outcome_v2_coverage import (
    OutcomeV2CoverageError,
    OutcomeV2ScheduleCoverageArtifacts,
    OutcomeV2ScheduleCoverageConfig,
    OutcomeV2ScheduleCoverageReceiptArtifacts,
    build_outcome_v2_schedule_coverage_seal,
    verify_outcome_v2_schedule_coverage_receipt,
    verify_outcome_v2_schedule_coverage_seal,
    write_outcome_v2_schedule_coverage_seal,
)
from forecastfm.outcome_v2_experiment import (
    build_outcome_v2_experiment_lock,
    write_outcome_v2_experiment_lock,
)
from forecastfm.outcome_v2_inference import (
    InferenceRecord,
    OrientationScore,
    OutcomeV2GenerationArtifacts,
    OutcomeV2InferenceError,
    binary_forecasts_from_inference_records,
    build_orientation_score,
    build_outcome_v2_generation_lock,
    completed_inference_record,
    failed_inference_record,
    outcome_v2_prompt_pairs_jsonl_bytes,
    rendered_prompt_token_ids_sha256,
    write_outcome_v2_generation_lock,
    write_outcome_v2_inference_records,
)
from forecastfm.outcome_v2_metrics import BinaryForecast
from forecastfm.outcome_v2_preflight import OutcomeV2Preflight, PreparedOutcomeV2Run
from forecastfm.outcome_v2_prompt import OUTCOME_V2_SYSTEM_PROMPT
from forecastfm.outcome_v2_rolling import (
    OutcomeV2ProspectiveBatchArtifacts,
    OutcomeV2ProspectivePlan,
    OutcomeV2ProspectivePlanArtifacts,
    OutcomeV2ProspectivePlanConfig,
    OutcomeV2ProspectiveReceiptArtifacts,
    OutcomeV2RollingError,
    build_outcome_v2_prospective_batch_seal,
    build_outcome_v2_prospective_plan,
    verify_outcome_v2_prospective_batch_receipt,
    verify_outcome_v2_prospective_batch_seal,
    verify_outcome_v2_prospective_plan,
    write_outcome_v2_prospective_batch_seal,
    write_outcome_v2_prospective_plan,
)
from forecastfm.outcome_v2_rolling_gate import (
    OutcomeV2RollingGateArtifacts,
    verify_outcome_v2_claimed_rolling_gate,
    verify_outcome_v2_rolling_gate,
    write_outcome_v2_claimed_rolling_gate_report,
)
from forecastfm.outcome_v2_rolling_score import (
    OutcomeV2RollingResolutionArtifacts,
    OutcomeV2RollingScoreError,
    OutcomeV2RollingScoringArtifacts,
    OutcomeV2RollingScoringFiles,
    OutcomeV2RollingScoringInputs,
    OutcomeV2RollingScoringSeal,
    build_outcome_v2_rolling_scoring_inputs,
    build_outcome_v2_rolling_scoring_seal,
    verify_outcome_v2_rolling_scoring_seal,
    write_outcome_v2_rolling_scoring_inputs,
    write_outcome_v2_rolling_scoring_seal,
)
from forecastfm.outcome_v2_run import build_outcome_v2_run_lock
from forecastfm.outcome_v2_sft_gate import (
    OUTCOME_V2_SFT_CANDIDATE_ROLE,
    OutcomeV2PostSftGateArtifacts,
    OutcomeV2SftForecastArtifacts,
    OutcomeV2SftGateError,
    build_outcome_v2_sft_forecast_seal,
    read_outcome_v2_post_sft_gate_report,
    read_outcome_v2_sft_forecast_seal,
    verify_outcome_v2_post_sft_gate,
    verify_outcome_v2_sft_forecast_seal,
    write_outcome_v2_post_sft_gate_report,
    write_outcome_v2_sft_forecast_seal,
)
from forecastfm.tinker_data import read_outcome_forecast_jsonl

PROJECT_ROOT = Path(__file__).parents[1]
REVISION = "a" * 40
RUN_AT = datetime(2026, 7, 17, 18, tzinfo=UTC)
EXPERIMENT_AT = datetime(2026, 7, 17, 19, tzinfo=UTC)
FORECAST_AT = datetime(2026, 7, 17, 20, tzinfo=UTC)
GENERATION_AT = FORECAST_AT - timedelta(minutes=30)
TRAINING_BYTES = b'{"label":"TEAM","question_id":"game-1"}\n'
ORIGINAL_TOKENS_SHA256 = rendered_prompt_token_ids_sha256((11, 12, 13))
SWAPPED_TOKENS_SHA256 = rendered_prompt_token_ids_sha256((21, 22, 23))
POLICY = NbaEvaluationGatePolicy(
    minimum_games_per_season=2,
    minimum_calendar_blocks_per_season=1,
    recalibration_gradient_steps=50,
    recalibration_learning_rate=0.05,
    recalibration_initial_intercept=0.0,
    recalibration_initial_slope=1.0,
)


@pytest.fixture(autouse=True)
def use_fixture_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        outcome_v2_run_module,
        "outcome_v2_evaluation_policy",
        lambda: POLICY,
    )
    monkeypatch.setattr(
        sft_gate_module,
        "outcome_v2_evaluation_policy",
        lambda: POLICY,
    )


def _cohort(prefix: str, seasons: tuple[int, int]) -> tuple[NbaEvaluationCohortInput, ...]:
    rows: list[NbaEvaluationCohortInput] = []
    for season in seasons:
        year = season - 1
        rows.extend(
            (
                NbaEvaluationCohortInput(
                    f"{prefix}-{season}-a",
                    season,
                    date(year, 10, 2),
                    0.6,
                ),
                NbaEvaluationCohortInput(
                    f"{prefix}-{season}-b",
                    season,
                    date(year, 10, 9),
                    0.4,
                ),
            )
        )
    return tuple(rows)


def _calibration() -> tuple[NbaRecalibrationRow, ...]:
    return (
        NbaRecalibrationRow("cal-a", 2023, date(2022, 10, 3), 0.6, True),
        NbaRecalibrationRow("cal-b", 2023, date(2022, 10, 10), 0.6, False),
        NbaRecalibrationRow("cal-c", 2023, date(2022, 10, 17), 0.4, True),
        NbaRecalibrationRow("cal-d", 2023, date(2022, 10, 24), 0.4, False),
    )


def _write_gate_artifacts(
    directory: Path,
    cohort: tuple[NbaEvaluationCohortInput, ...],
    *,
    failed_sequences: frozenset[int] = frozenset(),
    forecasts: tuple[BinaryForecast, ...] | None = None,
) -> NbaEvaluationGateArtifacts:
    directory.mkdir(parents=True, exist_ok=True)
    answers = tuple(
        NbaEvaluationAnswer(row.question_id, index % 2 == 0) for index, row in enumerate(cohort)
    )
    candidate_forecasts = (
        _candidate_forecasts(cohort, failed_sequences) if forecasts is None else forecasts
    )
    artifacts = NbaEvaluationGateArtifacts(
        cohort_path=directory / "cohort.jsonl",
        answers_path=directory / "answers.jsonl",
        forecasts_path=directory / "forecasts.jsonl",
        calibration_path=directory / "calibration.jsonl",
    )
    write_nba_evaluation_cohort_jsonl(artifacts.cohort_path, cohort)
    write_nba_evaluation_answers_jsonl(artifacts.answers_path, answers)
    write_nba_evaluation_forecasts_jsonl(artifacts.forecasts_path, candidate_forecasts)
    write_nba_recalibration_rows_jsonl(artifacts.calibration_path, _calibration())
    return artifacts


def _candidate_forecasts(
    cohort: tuple[NbaEvaluationCohortInput, ...],
    failed_sequences: frozenset[int] = frozenset(),
) -> tuple[BinaryForecast, ...]:
    return tuple(
        BinaryForecast(
            row.question_id,
            None if index in failed_sequences else (0.9 if index % 2 == 0 else 0.1),
            "candidate_output_invalid" if index in failed_sequences else None,
        )
        for index, row in enumerate(cohort)
    )


def _write_prompts(path: Path, cohort: tuple[NbaEvaluationCohortInput, ...]) -> None:
    path.write_bytes(outcome_v2_prompt_pairs_jsonl_bytes(_feature_rows(cohort)))


def _feature_rows(
    cohort: tuple[NbaEvaluationCohortInput, ...],
) -> tuple[NbaRichFeatureRow, ...]:
    rows: list[NbaRichFeatureRow] = []
    for index, member in enumerate(cohort, start=1):
        tipoff = datetime.combine(member.game_date, datetime.min.time(), tzinfo=UTC) + timedelta(
            hours=20
        )
        cutoff = tipoff - timedelta(hours=1)
        rows.append(
            NbaRichFeatureRow(
                question_id=member.question_id,
                source_game_id=f"source-{member.question_id}",
                team_id=f"team-{member.question_id}",
                opponent_id=f"opponent-{member.question_id}",
                site="neutral",
                season=member.season,
                forecast_cutoff=cutoff,
                scheduled_tipoff=tipoff,
                elo_team_win_probability=member.raw_elo_team_probability,
                elo_opponent_win_probability=1.0 - member.raw_elo_team_probability,
                elo_available_at=cutoff - timedelta(minutes=2),
                elo_state_sha256="a" * 64,
                rich_features=NbaRichFeatures.from_vector((float(index), *([0.0] * 10))),
                evidence_bundle_sha256="b" * 64,
                input_available_at=cutoff - timedelta(minutes=1),
            )
        )
    return tuple(rows)


def _orientation_score(
    elo_probability: float,
    target_probability: float,
) -> OrientationScore:
    delta = log(target_probability / (1.0 - target_probability)) - log(
        elo_probability / (1.0 - elo_probability)
    )
    return build_orientation_score(elo_probability, -10.0 + delta, -10.0)


def _write_inference_chain(
    directory: Path,
    generation_artifacts: OutcomeV2GenerationArtifacts,
    rows: tuple[NbaRichFeatureRow, ...],
    requested_forecasts: tuple[BinaryForecast, ...],
    generation_at: datetime = GENERATION_AT,
) -> tuple[Path, Path, Path, tuple[BinaryForecast, ...]]:
    generation_lock = build_outcome_v2_generation_lock(
        generation_artifacts,
        (101, 202),
        generation_at,
    )
    generation_lock_path = directory / "generation-lock.json"
    write_outcome_v2_generation_lock(generation_lock_path, generation_lock)

    inference_records: list[InferenceRecord] = []
    for sequence, (row, forecast) in enumerate(zip(rows, requested_forecasts, strict=True)):
        if forecast.team_probability is None:
            inference_records.append(
                failed_inference_record(
                    generation_lock,
                    sequence,
                    row,
                    original_prompt_token_ids_sha256=ORIGINAL_TOKENS_SHA256,
                    swapped_prompt_token_ids_sha256=SWAPPED_TOKENS_SHA256,
                    failure_reason="candidate_output_invalid",
                )
            )
            continue
        inference_records.append(
            completed_inference_record(
                generation_lock,
                sequence,
                row,
                original_prompt_token_ids_sha256=ORIGINAL_TOKENS_SHA256,
                swapped_prompt_token_ids_sha256=SWAPPED_TOKENS_SHA256,
                original=_orientation_score(
                    row.elo_team_win_probability,
                    forecast.team_probability,
                ),
                swapped=_orientation_score(
                    row.elo_opponent_win_probability,
                    1.0 - forecast.team_probability,
                ),
            )
        )
    inference_records_path = directory / "inference-records.jsonl"
    write_outcome_v2_inference_records(
        inference_records_path,
        inference_records,
        generation_lock,
    )
    inference_journal_path = directory / "inference-journal.jsonl"
    inference_journal_path.write_text(
        f"{canonical_json({'generation_lock_sha256': generation_lock.sha256})}\n",
        encoding="utf-8",
    )
    derived_forecasts = binary_forecasts_from_inference_records(
        inference_records,
        generation_lock,
    )
    return (
        generation_lock_path,
        inference_journal_path,
        inference_records_path,
        derived_forecasts,
    )


def _prepared(tabular_report_sha256: str) -> PreparedOutcomeV2Run:
    proof = OutcomeV2Preflight(
        manifest_sha256="1" * 64,
        action_at=RUN_AT,
        action_time_source="internal_paid_preparation",
        untouched_evaluation_seasons=(2024, 2025),
        training_sha256=bytes_sha256(TRAINING_BYTES),
        feature_rows_sha256="2" * 64,
        snapshot_pack_sha256="3" * 64,
        evidence_bundles_sha256="4" * 64,
        elo_states_sha256="5" * 64,
        elo_replay_sha256="6" * 64,
        seasons_sha256="7" * 64,
        resolutions_sha256="8" * 64,
        rights_lock_sha256="9" * 64,
        evaluation_feature_rows_sha256="0" * 64,
        evaluation_elo_replay_sha256="a" * 64,
        evaluation_elo_states_sha256="b" * 64,
        evaluation_resolutions_sha256="c" * 64,
        calibration_sha256=bytes_sha256(nba_recalibration_rows_jsonl_bytes(_calibration())),
        rich_baseline_model_sha256="e" * 64,
        rich_baseline_forecast_lock_sha256="f" * 64,
        evaluation_report_sha256=tabular_report_sha256,
        row_count=14,
        pair_count=7,
        batch_size=14,
    )
    return PreparedOutcomeV2Run(proof, TRAINING_BYTES)


def _write_model_chain(
    directory: Path,
    tabular_report_sha256: str,
    *,
    sampler_path: str = "tinker://run/sampler/final",
) -> tuple[Path, Path]:
    run_path = directory / "training_lock.json"
    experiment_path = directory / "experiment.json"
    run_lock = build_outcome_v2_run_lock(
        PROJECT_ROOT,
        _prepared(tabular_report_sha256),
        REVISION,
    )
    run_path.write_bytes(run_lock.canonical_bytes)
    experiment = build_outcome_v2_experiment_lock(
        run_path,
        "tinker://run/state/final",
        sampler_path,
        EXPERIMENT_AT,
    )
    write_outcome_v2_experiment_lock(experiment_path, experiment)
    return run_path, experiment_path


def _build_artifacts(
    tmp_path: Path,
    *,
    tabular_cohort: tuple[NbaEvaluationCohortInput, ...] | None = None,
    sft_cohort: tuple[NbaEvaluationCohortInput, ...] | None = None,
    failed_sequences: frozenset[int] = frozenset(),
    inference_times: tuple[datetime, datetime] | None = None,
) -> OutcomeV2PostSftGateArtifacts:
    tabular_rows = _cohort("tabular", (2024, 2025)) if tabular_cohort is None else tabular_cohort
    sft_rows = _cohort("sft", (2026, 2027)) if sft_cohort is None else sft_cohort
    tabular = _write_gate_artifacts(tmp_path / "tabular", tabular_rows)
    tabular_report = verify_untouched_nba_evaluation_gate(tabular, policy=POLICY)
    tabular_report_path = tmp_path / "tabular" / "gate.json"
    write_nba_evaluation_gate_report(tabular_report_path, tabular_report)
    run_path, experiment_path = _write_model_chain(tmp_path, tabular_report.sha256)

    sft_directory = tmp_path / "sft"
    sft_directory.mkdir(parents=True)
    feature_rows = _feature_rows(sft_rows)
    feature_rows_path = sft_directory / "feature-rows.jsonl"
    write_nba_feature_rows_jsonl(feature_rows_path, feature_rows)
    prompts_path = sft_directory / "prompts.jsonl"
    _write_prompts(prompts_path, sft_rows)
    generation_artifacts = OutcomeV2GenerationArtifacts(
        project_root=PROJECT_ROOT,
        run_lock_path=run_path,
        experiment_lock_path=experiment_path,
        feature_rows_path=feature_rows_path,
    )
    generation_at, forecast_at = inference_times or (GENERATION_AT, FORECAST_AT)
    (
        generation_lock_path,
        inference_journal_path,
        inference_records_path,
        forecasts,
    ) = _write_inference_chain(
        sft_directory,
        generation_artifacts,
        feature_rows,
        _candidate_forecasts(sft_rows, failed_sequences),
        generation_at,
    )
    sft = _write_gate_artifacts(
        sft_directory,
        sft_rows,
        forecasts=forecasts,
    )
    forecast_artifacts = OutcomeV2SftForecastArtifacts(
        project_root=PROJECT_ROOT,
        cohort_path=sft.cohort_path,
        feature_rows_path=feature_rows_path,
        prompts_path=prompts_path,
        forecasts_path=sft.forecasts_path,
        run_lock_path=run_path,
        experiment_lock_path=experiment_path,
        generation_lock_path=generation_lock_path,
        inference_journal_path=inference_journal_path,
        inference_records_path=inference_records_path,
    )
    seal = build_outcome_v2_sft_forecast_seal(forecast_artifacts, forecast_at)
    seal_path = sft_directory / "forecast-seal.json"
    write_outcome_v2_sft_forecast_seal(seal_path, seal)
    return OutcomeV2PostSftGateArtifacts(
        tabular_cohort_path=tabular.cohort_path,
        tabular_evaluation_report_path=tabular_report_path,
        sft_forecast=forecast_artifacts,
        sft_forecast_seal_path=seal_path,
        sft_evaluation=sft,
    )


def _assert_post_sft_report(
    artifacts: OutcomeV2PostSftGateArtifacts,
    payload: dict[str, object],
) -> None:
    assert payload["kind"] == "forecastfm_nba_outcome_v2_post_sft_gate"
    assert payload["candidate_role"] == OUTCOME_V2_SFT_CANDIDATE_ROLE
    assert payload["status"] == "passed"
    cohorts = require_object(payload["cohorts"], "cohorts")
    assert cohorts["relation"] == "disjoint_and_strictly_later"
    assert cohorts["tabular_seasons"] == [2024, 2025]
    assert cohorts["sft_seasons"] == [2026, 2027]
    evaluation = require_object(payload["evaluation"], "evaluation")
    assert evaluation["mode"] == "retrospective_answer_held_holdout"
    assert evaluation["tabular_gate_role"] == "sft_training_prerequisite_only"
    assert evaluation["post_sft_gate_role"] == "sft_candidate_advancement"
    proof_scope = require_object(payload["proof_scope"], "proof_scope")
    assert proof_scope["pretraining_contamination"] == "possible_not_ruled_out"
    assert proof_scope["rolling_prospective_proof"] == "not_satisfied"
    assert proof_scope["durable_inference_journal"] == (
        "locally_sha256_bound_not_remote_execution_or_one_attempt_proof"
    )
    report_artifacts = require_object(payload["artifacts"], "artifacts")
    assert report_artifacts["sft_generation_lock_sha256"] == file_sha256(
        artifacts.sft_forecast.generation_lock_path
    )
    assert report_artifacts["sft_inference_journal_sha256"] == file_sha256(
        artifacts.sft_forecast.inference_journal_path
    )
    assert report_artifacts["sft_inference_records_sha256"] == file_sha256(
        artifacts.sft_forecast.inference_records_path
    )
    assert report_artifacts["sft_failed_record_count"] == 0


def test_answer_free_seal_and_distinct_post_sft_gate_round_trip(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path)
    seal = verify_outcome_v2_sft_forecast_seal(
        artifacts.sft_forecast,
        artifacts.sft_forecast_seal_path,
    )
    seal_record = seal.to_record()

    assert {item.name for item in fields(OutcomeV2SftForecastArtifacts)} == {
        "project_root",
        "cohort_path",
        "feature_rows_path",
        "prompts_path",
        "forecasts_path",
        "run_lock_path",
        "experiment_lock_path",
        "generation_lock_path",
        "inference_journal_path",
        "inference_records_path",
    }
    assert not any("answer" in key or "resolution" in key for key in seal_record)
    assert seal_record["kind"] == "forecastfm_outcome_v2_sft_forecast_seal"
    assert seal_record["candidate_role"] == OUTCOME_V2_SFT_CANDIDATE_ROLE
    assert seal_record["evaluation_cohort_sha256"] == file_sha256(
        artifacts.sft_forecast.cohort_path
    )
    assert seal_record["evaluation_prompts_sha256"] == file_sha256(
        artifacts.sft_forecast.prompts_path
    )
    assert seal_record["evaluation_feature_rows_sha256"] == file_sha256(
        artifacts.sft_forecast.feature_rows_path
    )
    assert seal_record["evaluation_generation_lock_sha256"] == file_sha256(
        artifacts.sft_forecast.generation_lock_path
    )
    assert seal_record["evaluation_inference_journal_sha256"] == file_sha256(
        artifacts.sft_forecast.inference_journal_path
    )
    assert seal_record["evaluation_inference_records_sha256"] == file_sha256(
        artifacts.sft_forecast.inference_records_path
    )
    assert seal_record["failed_record_count"] == 0
    assert read_outcome_v2_sft_forecast_seal(artifacts.sft_forecast_seal_path) == seal

    report = verify_outcome_v2_post_sft_gate(artifacts, policy=POLICY)
    _assert_post_sft_report(artifacts, report.payload)

    report_path = tmp_path / "post-sft-report.json"
    assert write_outcome_v2_post_sft_gate_report(report_path, report) == report.sha256
    assert read_outcome_v2_post_sft_gate_report(report_path) == report
    with pytest.raises(FileExistsError):
        write_outcome_v2_post_sft_gate_report(report_path, report)


def test_post_sft_gate_rejects_any_cohort_id_overlap(tmp_path: Path) -> None:
    tabular = _cohort("tabular", (2024, 2025))
    sft = list(_cohort("sft", (2026, 2027)))
    sft[0] = replace(sft[0], question_id=tabular[0].question_id)
    artifacts = _build_artifacts(
        tmp_path,
        tabular_cohort=tabular,
        sft_cohort=tuple(sft),
    )

    with pytest.raises(OutcomeV2SftGateError, match="IDs must be disjoint"):
        verify_outcome_v2_post_sft_gate(artifacts, policy=POLICY)


def test_post_sft_gate_requires_every_sft_season_to_be_later(tmp_path: Path) -> None:
    with pytest.raises(OutcomeV2InferenceError, match="later than every tabular season"):
        _build_artifacts(
            tmp_path,
            sft_cohort=_cohort("sft", (2025, 2026)),
        )


def test_forecast_seal_rejects_changed_forecast_bytes(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path)
    path = artifacts.sft_forecast.forecasts_path
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(OutcomeV2SftGateError, match="answer-free SFT forecast files"):
        verify_outcome_v2_sft_forecast_seal(
            artifacts.sft_forecast,
            artifacts.sft_forecast_seal_path,
        )


def test_forecast_seal_rejects_valid_forecast_not_derived_from_records(
    tmp_path: Path,
) -> None:
    artifacts = _build_artifacts(tmp_path)
    path = artifacts.sft_forecast.forecasts_path
    forecasts = read_nba_evaluation_forecasts_jsonl(path)
    changed = (replace(forecasts[0], team_probability=0.8), *forecasts[1:])
    path.unlink()
    write_nba_evaluation_forecasts_jsonl(path, changed)

    with pytest.raises(OutcomeV2SftGateError, match="terminal inference records"):
        build_outcome_v2_sft_forecast_seal(artifacts.sft_forecast, FORECAST_AT)


def test_forecast_seal_rejects_tampered_inference_records(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path)
    path = artifacts.sft_forecast.inference_records_path
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(OutcomeV2SftGateError, match="answer-free SFT forecast files"):
        verify_outcome_v2_sft_forecast_seal(
            artifacts.sft_forecast,
            artifacts.sft_forecast_seal_path,
        )


def test_forecast_seal_rejects_changed_inference_journal(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path)
    path = artifacts.sft_forecast.inference_journal_path
    path.write_bytes(path.read_bytes() + b'{"tampered":true}\n')

    with pytest.raises(OutcomeV2SftGateError, match="answer-free files"):
        verify_outcome_v2_sft_forecast_seal(
            artifacts.sft_forecast,
            artifacts.sft_forecast_seal_path,
        )


def test_forecast_seal_rejects_empty_inference_journal(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path)
    artifacts.sft_forecast.inference_journal_path.write_bytes(b"")

    with pytest.raises(OutcomeV2SftGateError, match="journal must not be empty"):
        build_outcome_v2_sft_forecast_seal(artifacts.sft_forecast, FORECAST_AT)


def test_failed_inference_is_retained_and_counted(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path, failed_sequences=frozenset({0}))
    seal = verify_outcome_v2_sft_forecast_seal(
        artifacts.sft_forecast,
        artifacts.sft_forecast_seal_path,
    )
    forecasts = read_nba_evaluation_forecasts_jsonl(artifacts.sft_forecast.forecasts_path)

    assert seal.to_record()["failed_record_count"] == 1
    assert forecasts[0].team_probability is None
    assert forecasts[0].failure_reason == "candidate_output_invalid"
    with pytest.raises(OutcomeV2SftGateError, match="generic evaluation gate failed"):
        verify_outcome_v2_post_sft_gate(artifacts, policy=POLICY)


def test_post_sft_gate_scores_the_same_bytes_verified_by_the_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _build_artifacts(tmp_path)
    source_path = artifacts.sft_forecast.cohort_path
    captured_bytes = source_path.read_bytes()
    original_reader = sft_gate_module.read_nba_evaluation_cohort_jsonl
    mutated = False

    def mutate_source_after_parse(path: Path) -> tuple[NbaEvaluationCohortInput, ...]:
        nonlocal mutated
        rows = original_reader(path)
        if not mutated:
            source_path.write_bytes(captured_bytes + b"\n")
            mutated = True
        return rows

    monkeypatch.setattr(
        sft_gate_module,
        "read_nba_evaluation_cohort_jsonl",
        mutate_source_after_parse,
    )

    report = verify_outcome_v2_post_sft_gate(artifacts, policy=POLICY)
    report_artifacts = require_object(report.payload["artifacts"], "artifacts")

    assert mutated
    assert report_artifacts["sft_cohort_sha256"] == bytes_sha256(captured_bytes)
    assert file_sha256(source_path) != bytes_sha256(captured_bytes)


def test_post_sft_gate_rejects_policy_not_committed_by_run_lock(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path)
    weaker_policy = replace(POLICY, minimum_games_per_season=1)

    with pytest.raises(OutcomeV2SftGateError, match="policy differs"):
        verify_outcome_v2_post_sft_gate(artifacts, policy=weaker_policy)


def test_post_sft_gate_reuses_the_tabular_calibration_comparator(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path)
    calibration_path = artifacts.sft_evaluation.calibration_path
    calibration_path.write_bytes(calibration_path.read_bytes() + b"\n")

    with pytest.raises(OutcomeV2SftGateError, match="comparator differs"):
        verify_outcome_v2_post_sft_gate(artifacts, policy=POLICY)


def test_forecast_seal_rejects_answer_injected_into_prompt(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path)
    path = artifacts.sft_forecast.prompts_path
    records = list(
        read_outcome_forecast_jsonl(
            path,
            expected_system_prompt=OUTCOME_V2_SYSTEM_PROMPT,
        )
    )
    records[0]["messages"][1]["content"] = canonical_json({"realized_team_win": True})
    path.write_text(
        "".join(f"{canonical_json(record)}\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(OutcomeV2SftGateError, match="generation lock"):
        build_outcome_v2_sft_forecast_seal(artifacts.sft_forecast, FORECAST_AT)


def test_forecast_seal_rejects_candidate_experiment_swapping(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path)
    other_experiment_path = tmp_path / "other-experiment.json"
    other = build_outcome_v2_experiment_lock(
        artifacts.sft_forecast.run_lock_path,
        "tinker://other/state/final",
        "tinker://other/sampler/final",
        EXPERIMENT_AT,
    )
    write_outcome_v2_experiment_lock(other_experiment_path, other)
    swapped = replace(
        artifacts.sft_forecast,
        experiment_lock_path=other_experiment_path,
    )

    with pytest.raises(OutcomeV2SftGateError, match="answer-free SFT forecast files"):
        verify_outcome_v2_sft_forecast_seal(swapped, artifacts.sft_forecast_seal_path)


def test_tabular_report_cannot_be_used_as_the_post_sft_report(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path)
    wrong_evaluation = replace(
        artifacts.sft_evaluation,
        supplied_report_path=artifacts.tabular_evaluation_report_path,
    )

    with pytest.raises(OutcomeV2SftGateError, match="generic evaluation gate failed"):
        verify_outcome_v2_post_sft_gate(
            replace(artifacts, sft_evaluation=wrong_evaluation),
            policy=POLICY,
        )


def test_post_sft_gate_rejects_tabular_report_not_bound_by_run_lock(tmp_path: Path) -> None:
    artifacts = _build_artifacts(tmp_path)
    replacement = verify_untouched_nba_evaluation_gate(
        artifacts.sft_evaluation,
        policy=POLICY,
    )
    artifacts.tabular_evaluation_report_path.write_bytes(replacement.canonical_text.encode("utf-8"))

    with pytest.raises(OutcomeV2SftGateError, match="run-lock prerequisite"):
        verify_outcome_v2_post_sft_gate(artifacts, policy=POLICY)


def test_tabular_seasons_must_equal_the_run_lock_prerequisite(tmp_path: Path) -> None:
    artifacts = _build_artifacts(
        tmp_path,
        tabular_cohort=_cohort("tabular", (2024, 2026)),
    )

    with pytest.raises(OutcomeV2SftGateError, match="seasons differ from the SFT run lock"):
        verify_outcome_v2_post_sft_gate(artifacts, policy=POLICY)


class _RollingGitHubApi:
    def __init__(self, responses: dict[str, dict[str, object]]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get_json(self, path: str) -> dict[str, object]:
        self.calls.append(path)
        try:
            return self.responses[path]
        except KeyError as error:
            raise AssertionError(f"unexpected rolling GitHub API path: {path}") from error


def _rolling_cohort(season: int = 2027) -> tuple[NbaEvaluationCohortInput, ...]:
    game_date = date(season - 1, 10, 2)
    return (
        NbaEvaluationCohortInput(f"rolling-{season}-a", season, game_date, 0.5),
        NbaEvaluationCohortInput(f"rolling-{season}-b", season, game_date, 0.5),
    )


def _write_rolling_batch_inputs(
    tmp_path: Path,
    generation_at: datetime | None = None,
    season: int = 2027,
) -> tuple[OutcomeV2ProspectiveBatchArtifacts, datetime, datetime, datetime]:
    cutoff = datetime(season - 1, 10, 2, 19, tzinfo=UTC)
    input_available_at = cutoff - timedelta(minutes=1)
    generation = generation_at or input_available_at + timedelta(seconds=10)
    forecast_at = input_available_at + timedelta(seconds=20)
    batch_at = input_available_at + timedelta(seconds=30)
    artifacts = _build_artifacts(
        tmp_path,
        sft_cohort=_rolling_cohort(season),
        inference_times=(generation, forecast_at),
    )
    plan_artifacts = OutcomeV2ProspectivePlanArtifacts(
        project_root=PROJECT_ROOT,
        run_lock_path=artifacts.sft_forecast.run_lock_path,
        experiment_lock_path=artifacts.sft_forecast.experiment_lock_path,
    )
    plan = build_outcome_v2_prospective_plan(
        plan_artifacts,
        OutcomeV2ProspectivePlanConfig(
            seasons=(2027, 2028),
            inclusion_rule="every provider-listed NBA game in the frozen seasons",
            created_at=EXPERIMENT_AT + timedelta(minutes=1),
            receipt_workflow_id=654321,
        ),
    )
    plan_path = tmp_path / "rolling" / "plan.json"
    write_outcome_v2_prospective_plan(plan_path, plan)
    batch = OutcomeV2ProspectiveBatchArtifacts(
        plan_path=plan_path,
        forecast=artifacts.sft_forecast,
        forecast_seal_path=artifacts.sft_forecast_seal_path,
    )
    return batch, batch_at, input_available_at, cutoff


def _receipt_policy(plan: OutcomeV2ProspectivePlan) -> GitHubActionsReceiptPolicy:
    payload = require_object(required_field(plan.to_record(), "receipt_policy"), "receipt_policy")
    workflow_id = required_field(payload, "workflow_id")
    if isinstance(workflow_id, bool) or not isinstance(workflow_id, int):
        raise AssertionError("fixture workflow_id must be an integer")
    return GitHubActionsReceiptPolicy(
        repository=require_string(required_field(payload, "repository"), "repository"),
        branch=require_string(required_field(payload, "branch"), "branch"),
        workflow_path=require_string(
            required_field(payload, "workflow_path"),
            "workflow_path",
        ),
        workflow_sha256=require_string(
            required_field(payload, "workflow_sha256"),
            "workflow_sha256",
        ),
        workflow_id=workflow_id,
        event=require_string(required_field(payload, "event"), "event"),
    )


def _github_content(path: str, value: bytes, blob_character: str) -> dict[str, object]:
    return {
        "type": "file",
        "path": path,
        "encoding": "base64",
        "content": base64.b64encode(value).decode("ascii"),
        "size": len(value),
        "sha": blob_character * 40,
    }


def _rolling_api(
    policy: GitHubActionsReceiptPolicy,
    bindings: tuple[tuple[int, str, datetime, datetime, str, bytes], ...],
) -> _RollingGitHubApi:
    responses: dict[str, dict[str, object]] = {}
    workflow_bytes = (PROJECT_ROOT / policy.workflow_path).read_bytes()
    artifact_blob_characters = "cdef0123456789ab"
    for index, (run_id, head_sha, created_at, updated_at, path, value) in enumerate(bindings):
        responses[f"/repos/{policy.repository}/actions/runs/{run_id}"] = {
            "id": run_id,
            "run_attempt": 1,
            "workflow_id": policy.workflow_id,
            "head_sha": head_sha,
            "head_branch": policy.branch,
            "event": policy.event,
            "path": policy.run_path,
            "status": "completed",
            "conclusion": "success",
            "created_at": created_at.isoformat().replace("+00:00", "Z"),
            "updated_at": updated_at.isoformat().replace("+00:00", "Z"),
            "html_url": f"https://github.com/{policy.repository}/actions/runs/{run_id}",
            "repository": {"full_name": policy.repository},
            "head_repository": {"full_name": policy.repository},
            "head_commit": {"id": head_sha},
        }
        workflow_endpoint = (
            f"/repos/{policy.repository}/contents/{policy.workflow_path}?ref={head_sha}"
        )
        artifact_endpoint = f"/repos/{policy.repository}/contents/{path}?ref={head_sha}"
        responses[workflow_endpoint] = _github_content(
            policy.workflow_path,
            workflow_bytes,
            chr(ord("a") + index),
        )
        responses[artifact_endpoint] = _github_content(
            path,
            value,
            artifact_blob_characters[index],
        )
    return _RollingGitHubApi(responses)


def _install_rolling_api(
    monkeypatch: pytest.MonkeyPatch,
    api: _RollingGitHubApi,
) -> None:
    def get_json(_client: object, path: str) -> dict[str, object]:
        return api.get_json(path)

    monkeypatch.setattr(github_receipt_module.GitHubRestClient, "get_json", get_json)


def test_rolling_plan_batch_and_live_receipts_close_the_timing_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, batch_at, input_available_at, cutoff = _write_rolling_batch_inputs(tmp_path)
    plan_artifacts = OutcomeV2ProspectivePlanArtifacts(
        project_root=PROJECT_ROOT,
        run_lock_path=batch.forecast.run_lock_path,
        experiment_lock_path=batch.forecast.experiment_lock_path,
    )
    plan = verify_outcome_v2_prospective_plan(plan_artifacts, batch.plan_path)
    seal = build_outcome_v2_prospective_batch_seal(batch, "2026-10-02", batch_at)
    seal_path = tmp_path / "rolling" / "batch-1" / "terminal-seal.json"
    write_outcome_v2_prospective_batch_seal(seal_path, seal)
    assert verify_outcome_v2_prospective_batch_seal(batch, seal_path) == seal

    policy = _receipt_policy(plan)
    plan_repository_path = "prospective/outcome_v2/rolling/plan.json"
    seal_repository_path = "prospective/outcome_v2/rolling/batch-1/terminal-seal.json"
    plan_run_at = EXPERIMENT_AT + timedelta(minutes=2)
    terminal_run_at = batch_at + timedelta(seconds=5)
    bindings = (
        (
            7001,
            "c" * 40,
            plan_run_at,
            plan_run_at + timedelta(seconds=5),
            plan_repository_path,
            plan.canonical_bytes,
        ),
        (
            7002,
            "d" * 40,
            terminal_run_at,
            cutoff + timedelta(minutes=1),
            seal_repository_path,
            seal.canonical_bytes,
        ),
    )
    api = _rolling_api(policy, bindings)
    _install_rolling_api(monkeypatch, api)

    plan_receipt = build_github_actions_receipt(
        policy,
        GitHubActionsReceiptRequest(
            run_id=7001,
            artifact_path=plan_repository_path,
            artifact_bytes=plan.canonical_bytes,
            not_before=EXPERIMENT_AT + timedelta(minutes=1),
            deadline=input_available_at,
        ),
    )
    plan_receipt_path = tmp_path / "rolling" / "plan-receipt.json"
    write_github_actions_receipt(plan_receipt_path, plan_receipt)
    terminal_receipt = build_github_actions_receipt(
        policy,
        GitHubActionsReceiptRequest(
            run_id=7002,
            artifact_path=seal_repository_path,
            artifact_bytes=seal.canonical_bytes,
            not_before=batch_at,
            deadline=cutoff,
        ),
    )
    terminal_receipt_path = tmp_path / "rolling" / "batch-1" / "terminal-receipt.json"
    write_github_actions_receipt(terminal_receipt_path, terminal_receipt)

    verified = verify_outcome_v2_prospective_batch_receipt(
        OutcomeV2ProspectiveReceiptArtifacts(
            batch=batch,
            batch_seal_path=seal_path,
            batch_seal_repository_path=seal_repository_path,
            plan_receipt_path=plan_receipt_path,
            plan_repository_path=plan_repository_path,
            terminal_receipt_path=terminal_receipt_path,
        )
    )

    assert verified.seal == seal
    assert verified.plan_receipt == plan_receipt
    assert verified.receipt == terminal_receipt
    assert verified.externally_committed_at == terminal_run_at


def test_rolling_batch_rejects_generation_before_its_inputs(tmp_path: Path) -> None:
    cutoff = datetime(2026, 10, 2, 19, tzinfo=UTC)
    batch, batch_at, input_available_at, _ = _write_rolling_batch_inputs(
        tmp_path,
        generation_at=cutoff - timedelta(minutes=1, seconds=1),
    )
    assert input_available_at == cutoff - timedelta(minutes=1)

    with pytest.raises(OutcomeV2RollingError, match="causal window"):
        build_outcome_v2_prospective_batch_seal(batch, "too-early", batch_at)


def test_rolling_plan_rejects_a_self_consistent_redirected_receipt_policy(
    tmp_path: Path,
) -> None:
    batch, _, _, _ = _write_rolling_batch_inputs(tmp_path)
    plan = OutcomeV2ProspectivePlan(batch.plan_path.read_bytes())
    record = plan.to_record()
    policy = require_object(required_field(record, "receipt_policy"), "receipt_policy")
    policy["repository"] = "attacker/redirect"
    record["receipt_policy"] = policy
    record["receipt_policy_sha256"] = canonical_sha256(policy)
    batch.plan_path.write_bytes(canonical_json(record).encode("utf-8"))
    artifacts = OutcomeV2ProspectivePlanArtifacts(
        project_root=PROJECT_ROOT,
        run_lock_path=batch.forecast.run_lock_path,
        experiment_lock_path=batch.forecast.experiment_lock_path,
    )

    with pytest.raises(OutcomeV2RollingError, match="differs from current model locks"):
        verify_outcome_v2_prospective_plan(artifacts, batch.plan_path)


def test_rolling_receipt_paths_must_match_the_workflow_trigger(tmp_path: Path) -> None:
    batch, _, _, _ = _write_rolling_batch_inputs(tmp_path)

    with pytest.raises(OutcomeV2RollingError, match="under prospective"):
        OutcomeV2ProspectiveReceiptArtifacts(
            batch=batch,
            batch_seal_path=tmp_path / "seal.json",
            batch_seal_repository_path="artifacts/seal.json",
            plan_receipt_path=tmp_path / "plan-receipt.json",
            plan_repository_path="prospective/outcome_v2/rolling/plan.json",
            terminal_receipt_path=tmp_path / "terminal-receipt.json",
        )


def _rolling_schedule_rows(season: int = 2027) -> tuple[NbaEloReplayRow, ...]:
    cutoff = datetime(season - 1, 10, 2, 19, tzinfo=UTC)
    return (
        NbaEloReplayRow(
            question_id=f"rolling-{season}-a",
            source_game_id=f"source-rolling-{season}-a",
            season=season,
            team_id=f"team-rolling-{season}-a",
            opponent_id=f"opponent-rolling-{season}-a",
            site="neutral",
            forecast_cutoff=cutoff,
            scheduled_tipoff=cutoff + timedelta(minutes=60),
        ),
        NbaEloReplayRow(
            question_id=f"rolling-{season}-b",
            source_game_id=f"source-rolling-{season}-b",
            season=season,
            team_id=f"team-rolling-{season}-b",
            opponent_id=f"opponent-rolling-{season}-b",
            site="neutral",
            forecast_cutoff=cutoff,
            scheduled_tipoff=cutoff + timedelta(minutes=60),
        ),
    )


def _write_rolling_schedule(
    path: Path,
    season: int = 2027,
) -> tuple[NbaEloReplayRow, ...]:
    rows = _rolling_schedule_rows(season)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_nba_elo_replay_rows_jsonl(path, rows)
    return rows


def _rolling_provider_report(
    game_count: int = 2,
    season: int = 2027,
) -> NbaProviderConformanceReport:
    return NbaProviderConformanceReport(
        inventory_sha256="1" * 64,
        connector_id="licensed-provider-v1",
        connector_sha256="2" * 64,
        snapshot_pack_sha256="3" * 64,
        cutoff_selection_sha256="4" * 64,
        schedule_derivation_sha256="5" * 64,
        replay_rows_sha256=canonical_sha256(
            [row.canonical_payload() for row in _rolling_schedule_rows(season)]
        ),
        schedule_season_types=("regular",),
        cohort_sha256=canonical_sha256({"season": season}),
        revision_count=5,
        schedule_game_count=game_count,
    )


def _rolling_coverage_artifacts(
    tmp_path: Path,
    batch: OutcomeV2ProspectiveBatchArtifacts,
    season: int = 2027,
) -> OutcomeV2ScheduleCoverageArtifacts:
    replay_path = tmp_path / "rolling" / str(season) / "complete-schedule.jsonl"
    _write_rolling_schedule(replay_path, season)
    return OutcomeV2ScheduleCoverageArtifacts(
        plan=OutcomeV2ProspectivePlanArtifacts(
            project_root=PROJECT_ROOT,
            run_lock_path=batch.forecast.run_lock_path,
            experiment_lock_path=batch.forecast.experiment_lock_path,
        ),
        plan_path=batch.plan_path,
        replay_rows_path=replay_path,
        provider_conformance_report=_rolling_provider_report(season=season),
    )


def test_schedule_coverage_seal_binds_the_exact_reviewed_season(
    tmp_path: Path,
) -> None:
    batch, _, input_available_at, _ = _write_rolling_batch_inputs(tmp_path)
    artifacts = _rolling_coverage_artifacts(tmp_path, batch)
    seal = build_outcome_v2_schedule_coverage_seal(
        artifacts,
        OutcomeV2ScheduleCoverageConfig(
            created_at=EXPERIMENT_AT + timedelta(minutes=2),
            commitment_deadline=input_available_at - timedelta(seconds=1),
        ),
    )
    path = tmp_path / "rolling" / "2027" / "coverage-seal.json"
    write_outcome_v2_schedule_coverage_seal(path, seal)

    assert verify_outcome_v2_schedule_coverage_seal(artifacts, path) == seal
    assert seal.to_record()["season"] == 2027
    assert seal.to_record()["game_count"] == 2
    assert seal.to_record()["provider_authenticity"] == "required_separately"

    artifacts.replay_rows_path.write_bytes(artifacts.replay_rows_path.read_bytes() + b"\n")
    with pytest.raises(OutcomeV2CoverageError, match="canonical schedule replay rows"):
        verify_outcome_v2_schedule_coverage_seal(artifacts, path)


def test_schedule_coverage_requires_one_matching_provider_report(tmp_path: Path) -> None:
    batch, _, input_available_at, _ = _write_rolling_batch_inputs(tmp_path)
    artifacts = _rolling_coverage_artifacts(tmp_path, batch)
    artifacts = replace(
        artifacts,
        provider_conformance_report=_rolling_provider_report(game_count=1),
    )

    with pytest.raises(OutcomeV2CoverageError, match="game count differs"):
        build_outcome_v2_schedule_coverage_seal(
            artifacts,
            OutcomeV2ScheduleCoverageConfig(
                created_at=EXPERIMENT_AT + timedelta(minutes=2),
                commitment_deadline=input_available_at - timedelta(seconds=1),
            ),
        )

    playoff_artifacts = replace(
        artifacts,
        provider_conformance_report=replace(
            _rolling_provider_report(),
            schedule_season_types=("playoffs",),
        ),
    )
    with pytest.raises(OutcomeV2CoverageError, match="regular-season policy"):
        build_outcome_v2_schedule_coverage_seal(
            playoff_artifacts,
            OutcomeV2ScheduleCoverageConfig(
                created_at=EXPERIMENT_AT + timedelta(minutes=2),
                commitment_deadline=input_available_at - timedelta(seconds=1),
            ),
        )

    exact_artifacts = _rolling_coverage_artifacts(tmp_path / "exact", batch)
    changed_rows = (
        replace(_rolling_schedule_rows()[0], team_id="MIA"),
        _rolling_schedule_rows()[1],
    )
    exact_artifacts.replay_rows_path.write_text(
        "".join(f"{canonical_json(row.canonical_payload())}\n" for row in changed_rows),
        encoding="utf-8",
    )
    with pytest.raises(OutcomeV2CoverageError, match="exact schedule rows"):
        build_outcome_v2_schedule_coverage_seal(
            exact_artifacts,
            OutcomeV2ScheduleCoverageConfig(
                created_at=EXPERIMENT_AT + timedelta(minutes=2),
                commitment_deadline=input_available_at - timedelta(seconds=1),
            ),
        )


def test_schedule_coverage_receipt_is_reverified_live(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, _, input_available_at, _ = _write_rolling_batch_inputs(tmp_path)
    artifacts = _rolling_coverage_artifacts(tmp_path, batch)
    created_at = EXPERIMENT_AT + timedelta(minutes=2)
    deadline = input_available_at - timedelta(seconds=1)
    seal = build_outcome_v2_schedule_coverage_seal(
        artifacts,
        OutcomeV2ScheduleCoverageConfig(created_at, deadline),
    )
    seal_path = tmp_path / "rolling" / "2027" / "coverage-seal.json"
    write_outcome_v2_schedule_coverage_seal(seal_path, seal)
    plan = verify_outcome_v2_prospective_plan(artifacts.plan, artifacts.plan_path)
    policy = _receipt_policy(plan)
    repository_path = "prospective/outcome_v2/rolling/2027/coverage-seal.json"
    run_at = created_at + timedelta(seconds=5)
    api = _rolling_api(
        policy,
        (
            (
                7101,
                "e" * 40,
                run_at,
                run_at + timedelta(seconds=5),
                repository_path,
                seal.canonical_bytes,
            ),
        ),
    )
    _install_rolling_api(monkeypatch, api)
    receipt = build_github_actions_receipt(
        policy,
        GitHubActionsReceiptRequest(
            run_id=7101,
            artifact_path=repository_path,
            artifact_bytes=seal.canonical_bytes,
            not_before=created_at,
            deadline=deadline,
        ),
    )
    receipt_path = tmp_path / "rolling" / "2027" / "coverage-receipt.json"
    write_github_actions_receipt(receipt_path, receipt)

    verified = verify_outcome_v2_schedule_coverage_receipt(
        OutcomeV2ScheduleCoverageReceiptArtifacts(
            coverage=artifacts,
            seal_path=seal_path,
            seal_repository_path=repository_path,
            receipt_path=receipt_path,
        )
    )

    assert verified.seal == seal
    assert tuple(row.question_id for row in verified.rows) == (
        "rolling-2027-a",
        "rolling-2027-b",
    )
    assert verified.receipt == receipt
    assert verified.externally_committed_at == run_at


type _RollingBatchFixture = tuple[
    OutcomeV2ProspectiveBatchArtifacts,
    datetime,
    datetime,
    datetime,
]
type _ApiBinding = tuple[int, str, datetime, datetime, str, bytes]


@dataclass(frozen=True, slots=True)
class _CoverageSealFixture:
    artifacts: OutcomeV2ScheduleCoverageArtifacts
    seal_path: Path
    repository_path: str
    created_at: datetime
    deadline: datetime


@dataclass(frozen=True, slots=True)
class _BatchSealFixture:
    artifacts: OutcomeV2ProspectiveBatchArtifacts
    seal_path: Path
    repository_path: str
    cutoff: datetime
    sealed_at: datetime


@dataclass(frozen=True, slots=True)
class _AggregationLocalFixture:
    batches: tuple[_RollingBatchFixture, ...]
    plans: tuple[OutcomeV2ProspectivePlan, ...]
    coverages: tuple[_CoverageSealFixture, ...]
    batch_seals: tuple[_BatchSealFixture, ...]
    policy: GitHubActionsReceiptPolicy
    bindings: tuple[_ApiBinding, ...]
    plan_repository_path: str


def _aggregation_coverage_fixture(
    tmp_path: Path,
    season: int,
    index: int,
    batch: _RollingBatchFixture,
) -> tuple[_CoverageSealFixture, _ApiBinding]:
    batch_artifacts, _, input_at, _ = batch
    artifacts = _rolling_coverage_artifacts(tmp_path, batch_artifacts, season)
    created_at = datetime(season - 1, 7, 17, 19, 2, tzinfo=UTC)
    deadline = input_at - timedelta(seconds=1)
    seal = build_outcome_v2_schedule_coverage_seal(
        artifacts,
        OutcomeV2ScheduleCoverageConfig(created_at, deadline),
    )
    seal_path = tmp_path / "rolling" / str(season) / "coverage-seal.json"
    write_outcome_v2_schedule_coverage_seal(seal_path, seal)
    repository_path = f"prospective/outcome_v2/rolling/{season}/coverage-seal.json"
    run_at = created_at + timedelta(seconds=5)
    binding = (
        7200 + index,
        chr(ord("a") + index) * 40,
        run_at,
        run_at + timedelta(seconds=5),
        repository_path,
        seal.canonical_bytes,
    )
    fixture = _CoverageSealFixture(
        artifacts,
        seal_path,
        repository_path,
        created_at,
        deadline,
    )
    return fixture, binding


def _aggregation_batch_fixture(
    tmp_path: Path,
    season: int,
    index: int,
    batch: _RollingBatchFixture,
) -> tuple[_BatchSealFixture, _ApiBinding]:
    artifacts, sealed_at, _, cutoff = batch
    seal = build_outcome_v2_prospective_batch_seal(
        artifacts,
        f"{season}-opening-slate",
        sealed_at,
    )
    seal_path = tmp_path / "rolling" / "batch" / "terminal-seal.json"
    write_outcome_v2_prospective_batch_seal(seal_path, seal)
    repository_path = f"prospective/outcome_v2/rolling/{season}/batch/terminal-seal.json"
    run_at = sealed_at + timedelta(seconds=5)
    binding = (
        7300 + index,
        chr(ord("c") + index) * 40,
        run_at,
        run_at + timedelta(seconds=5),
        repository_path,
        seal.canonical_bytes,
    )
    return _BatchSealFixture(artifacts, seal_path, repository_path, cutoff, sealed_at), binding


def _build_aggregation_local_fixture(tmp_path: Path) -> _AggregationLocalFixture:
    seasons = (2027, 2028)
    batches = tuple(
        _write_rolling_batch_inputs(tmp_path / f"season-{season}", season=season)
        for season in seasons
    )
    plans = tuple(
        verify_outcome_v2_prospective_plan(
            OutcomeV2ProspectivePlanArtifacts(
                project_root=PROJECT_ROOT,
                run_lock_path=batch.forecast.run_lock_path,
                experiment_lock_path=batch.forecast.experiment_lock_path,
            ),
            batch.plan_path,
        )
        for batch, _, _, _ in batches
    )
    assert plans[0].canonical_bytes == plans[1].canonical_bytes
    plan_repository_path = "prospective/outcome_v2/rolling/plan.json"
    plan_run_at = EXPERIMENT_AT + timedelta(minutes=1, seconds=30)
    bindings: list[_ApiBinding] = [
        (
            7200,
            "a" * 40,
            plan_run_at,
            plan_run_at + timedelta(seconds=5),
            plan_repository_path,
            plans[0].canonical_bytes,
        )
    ]
    coverages: list[_CoverageSealFixture] = []
    batch_seals: list[_BatchSealFixture] = []
    for index, (season, batch) in enumerate(zip(seasons, batches, strict=True), start=1):
        coverage, coverage_binding = _aggregation_coverage_fixture(
            tmp_path / f"season-{season}", season, index, batch
        )
        batch_seal, batch_binding = _aggregation_batch_fixture(
            tmp_path / f"season-{season}", season, index, batch
        )
        coverages.append(coverage)
        batch_seals.append(batch_seal)
        bindings.extend((coverage_binding, batch_binding))
    return _AggregationLocalFixture(
        batches,
        plans,
        tuple(coverages),
        tuple(batch_seals),
        _receipt_policy(plans[0]),
        tuple(bindings),
        plan_repository_path,
    )


def _build_aggregation_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[OutcomeV2RollingAggregationArtifacts, datetime]:
    local = _build_aggregation_local_fixture(tmp_path)
    _install_rolling_api(monkeypatch, _rolling_api(local.policy, local.bindings))
    plan_receipt = build_github_actions_receipt(
        local.policy,
        GitHubActionsReceiptRequest(
            run_id=7200,
            artifact_path=local.plan_repository_path,
            artifact_bytes=local.plans[0].canonical_bytes,
            not_before=EXPERIMENT_AT + timedelta(minutes=1),
            deadline=local.batches[0][2],
        ),
    )
    plan_receipt_path = tmp_path / "plan-receipt.json"
    write_github_actions_receipt(plan_receipt_path, plan_receipt)

    coverage_receipts: list[OutcomeV2ScheduleCoverageReceiptArtifacts] = []
    for index, coverage in enumerate(local.coverages, start=1):
        receipt = build_github_actions_receipt(
            local.policy,
            GitHubActionsReceiptRequest(
                run_id=7200 + index,
                artifact_path=coverage.repository_path,
                artifact_bytes=coverage.seal_path.read_bytes(),
                not_before=coverage.created_at,
                deadline=coverage.deadline,
            ),
        )
        receipt_path = coverage.seal_path.with_name("coverage-receipt.json")
        write_github_actions_receipt(receipt_path, receipt)
        coverage_receipts.append(
            OutcomeV2ScheduleCoverageReceiptArtifacts(
                coverage=coverage.artifacts,
                seal_path=coverage.seal_path,
                seal_repository_path=coverage.repository_path,
                receipt_path=receipt_path,
            )
        )

    batch_receipts: list[OutcomeV2ProspectiveReceiptArtifacts] = []
    for index, batch in enumerate(local.batch_seals, start=1):
        receipt = build_github_actions_receipt(
            local.policy,
            GitHubActionsReceiptRequest(
                run_id=7300 + index,
                artifact_path=batch.repository_path,
                artifact_bytes=batch.seal_path.read_bytes(),
                not_before=batch.sealed_at,
                deadline=batch.cutoff,
            ),
        )
        receipt_path = batch.seal_path.with_name("terminal-receipt.json")
        write_github_actions_receipt(receipt_path, receipt)
        batch_receipts.append(
            OutcomeV2ProspectiveReceiptArtifacts(
                batch=batch.artifacts,
                batch_seal_path=batch.seal_path,
                batch_seal_repository_path=batch.repository_path,
                plan_receipt_path=plan_receipt_path,
                plan_repository_path=local.plan_repository_path,
                terminal_receipt_path=receipt_path,
            )
        )

    aggregate_at = local.batches[-1][3] + timedelta(days=1)
    return (
        OutcomeV2RollingAggregationArtifacts(
            plan=local.coverages[0].artifacts.plan,
            plan_path=local.batches[0][0].plan_path,
            coverages=tuple(coverage_receipts),
            batches=tuple(batch_receipts),
        ),
        aggregate_at,
    )


def test_rolling_aggregate_requires_the_exact_multi_season_schedule_union(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts, aggregate_at = _build_aggregation_artifacts(tmp_path, monkeypatch)
    aggregate = build_outcome_v2_rolling_aggregate(artifacts, aggregate_at)
    record = aggregate.to_record()

    assert tuple(row.question_id for row in aggregate.schedule) == (
        "rolling-2027-a",
        "rolling-2027-b",
        "rolling-2028-a",
        "rolling-2028-b",
    )
    assert record["seasons"] == [2027, 2028]
    assert record["forecast_count"] == 4
    assert record["failed_record_count"] == 0
    assert record["provider_authenticity"] == "required_separately"
    assert record["remote_execution_attestation"] == "required_separately"
    assert record["canonical_elo_replay"] == "required_separately"
    assert "evaluation_cohort_sha256" not in record

    paths = OutcomeV2RollingAggregateFiles(
        schedule_path=tmp_path / "aggregate" / "schedule.jsonl",
        forecasts_path=tmp_path / "aggregate" / "forecasts.jsonl",
        seal_path=tmp_path / "aggregate" / "seal.json",
    )
    write_outcome_v2_rolling_aggregate(paths, aggregate)
    assert verify_outcome_v2_rolling_aggregate(artifacts, paths) == aggregate

    incomplete = replace(artifacts, batches=artifacts.batches[:1])
    with pytest.raises(OutcomeV2AggregationError, match=r"missing=2, extra=0"):
        build_outcome_v2_rolling_aggregate(incomplete, aggregate_at)


def test_rolling_aggregate_rejects_relabelled_game_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_schedule = _rolling_schedule_rows

    def relabelled_schedule(season: int = 2027) -> tuple[NbaEloReplayRow, ...]:
        rows = original_schedule(season)
        if season != 2027:
            return rows
        return (replace(rows[0], team_id="relabelled-team"), rows[1])

    monkeypatch.setattr(
        sys.modules[__name__],
        "_rolling_schedule_rows",
        relabelled_schedule,
    )
    artifacts, aggregate_at = _build_aggregation_artifacts(tmp_path, monkeypatch)

    with pytest.raises(OutcomeV2AggregationError, match="identity differs"):
        build_outcome_v2_rolling_aggregate(artifacts, aggregate_at)


@dataclass(frozen=True, slots=True)
class _RollingResolutionFixture:
    artifacts: OutcomeV2RollingResolutionArtifacts
    rows: tuple[NbaResolution, ...]
    scored_at: datetime


def _write_rolling_resolution_artifacts(
    directory: Path,
    schedule: tuple[NbaEloReplayRow, ...],
) -> _RollingResolutionFixture:
    rights = SourceRights(
        license_name="fixture final-score agreement",
        terms_url="https://provider.test/terms",
        terms_sha256="f" * 64,
        rights_as_of=min(row.forecast_cutoff for row in schedule) - timedelta(days=1),
        local_processing="allowed",
        third_party_processing="allowed",
        tinker_processing="allowed",
        redistribution="prohibited",
    )
    snapshots: list[NbaSnapshot] = []
    resolutions: list[NbaResolution] = []
    for index, row in enumerate(schedule):
        team_score = 110 if index % 2 == 0 else 100
        opponent_score = 100 if index % 2 == 0 else 110
        available_at = row.scheduled_tipoff + timedelta(hours=3)
        source_id = f"final-{row.source_game_id}"
        payload = canonical_json(
            {
                "final": True,
                "opponent_score": opponent_score,
                "source_game_id": row.source_game_id,
                "team_score": team_score,
            }
        ).encode("utf-8")
        snapshot = NbaSnapshot(
            metadata=NbaSnapshotMetadata(
                source_id=source_id,
                rights_scope="provider-test:nba:final-scores",
                source_url=f"https://provider.test/finals/{row.source_game_id}",
                version="final-v1",
                effective_at=available_at,
                provider_published_at=available_at,
                retrieved_at=available_at,
                available_at=available_at,
                capture_method="live",
                sensitivity="ordinary",
                payload_sha256=bytes_sha256(payload),
                archive_attestation_sha256=None,
                rights=rights,
            ),
            payload=payload,
        )
        snapshots.append(snapshot)
        resolutions.append(
            NbaResolution(
                question_id=row.question_id,
                source_game_id=row.source_game_id,
                team_id=row.team_id,
                opponent_id=row.opponent_id,
                site=row.site,
                team_score=team_score,
                opponent_score=opponent_score,
                resolved_at=available_at,
                source_id=source_id,
                snapshot_metadata_sha256=snapshot_metadata_sha256(snapshot.metadata),
            )
        )

    directory.mkdir(parents=True, exist_ok=True)
    snapshot_path = directory / "final-snapshots.jsonl"
    resolution_path = directory / "resolutions.jsonl"
    index = NbaSnapshotIndex(snapshots)
    write_snapshot_pack(index.snapshots, snapshot_path)
    rows = tuple(resolutions)
    write_nba_resolutions_jsonl(resolution_path, rows, snapshot_index=index)
    return _RollingResolutionFixture(
        OutcomeV2RollingResolutionArtifacts(snapshot_path, resolution_path),
        rows,
        max(row.resolved_at for row in rows) + timedelta(hours=1),
    )


@dataclass(frozen=True, slots=True)
class _RollingScoringFixture:
    aggregate: OutcomeV2RollingAggregate
    scoring: OutcomeV2RollingScoringArtifacts
    resolutions: _RollingResolutionFixture
    files: OutcomeV2RollingScoringFiles
    inputs: OutcomeV2RollingScoringInputs
    seal: OutcomeV2RollingScoringSeal


def _build_rolling_scoring_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _RollingScoringFixture:
    artifacts, aggregate_at = _build_aggregation_artifacts(tmp_path, monkeypatch)
    aggregate = build_outcome_v2_rolling_aggregate(artifacts, aggregate_at)
    aggregate_files = OutcomeV2RollingAggregateFiles(
        schedule_path=tmp_path / "aggregate" / "schedule.jsonl",
        forecasts_path=tmp_path / "aggregate" / "forecasts.jsonl",
        seal_path=tmp_path / "aggregate" / "seal.json",
    )
    write_outcome_v2_rolling_aggregate(aggregate_files, aggregate)
    resolutions = _write_rolling_resolution_artifacts(
        tmp_path / "resolutions",
        aggregate.schedule,
    )
    calibration_path = tmp_path / "scoring" / "calibration.jsonl"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    write_nba_recalibration_rows_jsonl(calibration_path, _calibration())
    scoring = OutcomeV2RollingScoringArtifacts(
        artifacts,
        aggregate_files,
        resolutions.artifacts,
        calibration_path,
    )
    inputs = build_outcome_v2_rolling_scoring_inputs(
        scoring,
        resolutions.scored_at,
    )
    scoring_files = OutcomeV2RollingScoringFiles(
        cohort_path=tmp_path / "scoring" / "cohort.jsonl",
        answers_path=tmp_path / "scoring" / "answers.jsonl",
        seal_path=tmp_path / "scoring" / "seal.json",
    )
    write_outcome_v2_rolling_scoring_inputs(scoring_files, inputs)
    seal = build_outcome_v2_rolling_scoring_seal(
        scoring,
        scoring_files,
        resolutions.scored_at,
    )
    write_outcome_v2_rolling_scoring_seal(scoring_files.seal_path, seal)
    return _RollingScoringFixture(
        aggregate,
        scoring,
        resolutions,
        scoring_files,
        inputs,
        seal,
    )


def test_rolling_scoring_replays_canonical_elo_after_forecast_aggregation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_rolling_scoring_fixture(tmp_path, monkeypatch)

    with pytest.raises(OutcomeV2RollingScoreError, match="cannot precede"):
        build_outcome_v2_rolling_scoring_inputs(
            fixture.scoring,
            fixture.resolutions.rows[-1].resolved_at - timedelta(seconds=1),
        )

    assert tuple(row.question_id for row in fixture.inputs.cohort) == tuple(
        row.question_id for row in fixture.aggregate.schedule
    )
    assert tuple(row.raw_elo_team_probability for row in fixture.inputs.cohort) == (0.5,) * 4
    assert tuple(row.realized_team_win for row in fixture.inputs.answers) == (
        True,
        False,
        True,
        False,
    )
    assert fixture.inputs.forecasts == fixture.aggregate.forecasts
    assert verify_outcome_v2_rolling_scoring_seal(fixture.scoring, fixture.files) == fixture.seal
    record = fixture.seal.to_record()
    assert record["provider_resolution_authenticity"] == "required_separately"
    assert record["provider_score_derivation"] == "required_separately"
    assert record["game_date_rule"] == {
        "source": "scheduled_tipoff",
        "timezone": "America/New_York",
    }


def test_claimed_rolling_gate_binds_policy_and_denies_production_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_rolling_scoring_fixture(tmp_path, monkeypatch)
    gate_artifacts = OutcomeV2RollingGateArtifacts(fixture.scoring, fixture.files)

    with pytest.raises(OutcomeV2RollingScoreError, match="no reviewed production"):
        verify_outcome_v2_rolling_gate(gate_artifacts, POLICY)

    with pytest.raises(OutcomeV2RollingScoreError, match="policy differs"):
        verify_outcome_v2_claimed_rolling_gate(
            gate_artifacts,
            replace(POLICY, minimum_games_per_season=1),
        )

    report = verify_outcome_v2_claimed_rolling_gate(gate_artifacts, POLICY)
    report_record = report.to_record()
    assert report_record["status"] == "structural_claim_only"
    assert report_record["prospective_win_authorization"] == "denied"
    assert report_record["rl_authorization"] == "denied"
    assert require_object(report_record["generic_report"], "generic_report")["status"] == "passed"
    rolling_report_path = tmp_path / "scoring" / "rolling-gate-report.json"
    write_outcome_v2_claimed_rolling_gate_report(rolling_report_path, report)
    assert (
        verify_outcome_v2_claimed_rolling_gate(
            replace(
                gate_artifacts,
                supplied_rolling_report_path=rolling_report_path,
            ),
            POLICY,
        )
        == report
    )


def test_rolling_scoring_seal_rejects_calibration_and_cohort_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_rolling_scoring_fixture(tmp_path, monkeypatch)
    calibration_path = fixture.scoring.calibration_path
    calibration_bytes = calibration_path.read_bytes()
    calibration_path.write_bytes(calibration_bytes + b"\n")
    with pytest.raises(OutcomeV2RollingScoreError, match="calibration differs"):
        verify_outcome_v2_rolling_scoring_seal(fixture.scoring, fixture.files)
    calibration_path.write_bytes(calibration_bytes)

    fixture.files.cohort_path.write_bytes(fixture.files.cohort_path.read_bytes() + b"\n")
    with pytest.raises(OutcomeV2RollingScoreError, match="cohort differs"):
        verify_outcome_v2_rolling_scoring_seal(fixture.scoring, fixture.files)


def test_rolling_scoring_rejects_noncanonical_feature_elo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def altered_elo_cohort(
        season: int = 2027,
    ) -> tuple[NbaEvaluationCohortInput, ...]:
        game_date = date(season - 1, 10, 2)
        return (
            NbaEvaluationCohortInput(f"rolling-{season}-a", season, game_date, 0.6),
            NbaEvaluationCohortInput(f"rolling-{season}-b", season, game_date, 0.4),
        )

    monkeypatch.setattr(sys.modules[__name__], "_rolling_cohort", altered_elo_cohort)
    artifacts, aggregate_at = _build_aggregation_artifacts(tmp_path, monkeypatch)
    aggregate = build_outcome_v2_rolling_aggregate(artifacts, aggregate_at)
    aggregate_files = OutcomeV2RollingAggregateFiles(
        schedule_path=tmp_path / "aggregate" / "schedule.jsonl",
        forecasts_path=tmp_path / "aggregate" / "forecasts.jsonl",
        seal_path=tmp_path / "aggregate" / "seal.json",
    )
    write_outcome_v2_rolling_aggregate(aggregate_files, aggregate)
    resolutions = _write_rolling_resolution_artifacts(
        tmp_path / "resolutions",
        aggregate.schedule,
    )
    calibration_path = tmp_path / "scoring" / "calibration.jsonl"
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    write_nba_recalibration_rows_jsonl(calibration_path, _calibration())

    with pytest.raises(OutcomeV2RollingScoreError, match="canonical Elo probability"):
        build_outcome_v2_rolling_scoring_inputs(
            OutcomeV2RollingScoringArtifacts(
                artifacts,
                aggregate_files,
                resolutions.artifacts,
                calibration_path,
            ),
            resolutions.scored_at,
        )
