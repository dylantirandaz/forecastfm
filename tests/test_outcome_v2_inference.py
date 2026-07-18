"""Tests for target-free outcome-v2 inference commitments and records."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import log
from pathlib import Path

import pytest

from forecastfm.integrity import bytes_sha256, canonical_json, canonical_sha256
from forecastfm.json_utils import require_object, required_field
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_feature_rows import NbaRichFeatureRow, write_nba_feature_rows_jsonl
from forecastfm.nba_rich import NbaRichFeatures
from forecastfm.outcome import OPPONENT_LABEL, TEAM_LABEL
from forecastfm.outcome_v2_experiment import (
    build_outcome_v2_experiment_lock,
    write_outcome_v2_experiment_lock,
)
from forecastfm.outcome_v2_inference import (
    InferenceRecord,
    OutcomeV2GenerationArtifacts,
    OutcomeV2GenerationLock,
    OutcomeV2InferenceError,
    binary_forecasts_from_inference_records,
    build_orientation_score,
    build_outcome_v2_generation_lock,
    build_outcome_v2_prompt_records,
    completed_inference_record,
    failed_inference_record,
    outcome_v2_prompt_pairs_jsonl_bytes,
    read_outcome_v2_generation_lock,
    read_outcome_v2_inference_records,
    rendered_prompt_token_ids_sha256,
    sanitize_inference_failure,
    verify_outcome_v2_generation_lock,
    write_outcome_v2_generation_lock,
    write_outcome_v2_inference_records,
)
from forecastfm.outcome_v2_preflight import OutcomeV2Preflight, PreparedOutcomeV2Run
from forecastfm.outcome_v2_run import build_outcome_v2_run_lock, write_outcome_v2_run_lock

PROJECT_ROOT = Path(__file__).parents[1]
REVISION = "a" * 40
ACTION_AT = datetime(2026, 7, 17, 18, tzinfo=UTC)
EXPERIMENT_AT = ACTION_AT + timedelta(minutes=30)
GENERATION_AT = EXPERIMENT_AT + timedelta(minutes=1)
TRAINING_BYTES = b'{"label":"TEAM","question_id":"training-game"}\n'
ORIGINAL_TOKENS_SHA256 = rendered_prompt_token_ids_sha256((11, 12, 13))
SWAPPED_TOKENS_SHA256 = rendered_prompt_token_ids_sha256((21, 22, 23))


def _prepared(evaluation_feature_rows_sha256: str = "0" * 64) -> PreparedOutcomeV2Run:
    proof = OutcomeV2Preflight(
        manifest_sha256="1" * 64,
        action_at=ACTION_AT,
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
        evaluation_feature_rows_sha256=evaluation_feature_rows_sha256,
        evaluation_elo_replay_sha256="a" * 64,
        evaluation_elo_states_sha256="b" * 64,
        evaluation_resolutions_sha256="c" * 64,
        calibration_sha256="0" * 64,
        rich_baseline_model_sha256="d" * 64,
        rich_baseline_forecast_lock_sha256="e" * 64,
        evaluation_report_sha256="f" * 64,
        row_count=14,
        pair_count=7,
        batch_size=14,
    )
    return PreparedOutcomeV2Run(proof, TRAINING_BYTES)


def _row(question_id: str, season: int) -> NbaRichFeatureRow:
    cutoff = datetime(season, 10, 21, 22, tzinfo=UTC)
    return NbaRichFeatureRow(
        question_id=question_id,
        source_game_id=f"source-{question_id}",
        team_id=f"team-{question_id}",
        opponent_id=f"opponent-{question_id}",
        site="neutral",
        season=season,
        forecast_cutoff=cutoff,
        scheduled_tipoff=cutoff + timedelta(hours=1),
        elo_team_win_probability=0.6,
        elo_opponent_win_probability=0.4,
        elo_available_at=cutoff - timedelta(hours=1),
        elo_state_sha256="a" * 64,
        rich_features=NbaRichFeatures.from_vector(
            (1.0, -1.0, 2.0, -2.0, 300.0, -1.0, 0.2, -0.2, 4.0, -4.0, 25.0)
        ),
        evidence_bundle_sha256="b" * 64,
        input_available_at=cutoff - timedelta(minutes=30),
    )


def _artifacts(
    tmp_path: Path,
    rows: tuple[NbaRichFeatureRow, ...] | None = None,
    *,
    reuse_feature_rows: bool = False,
) -> tuple[OutcomeV2GenerationArtifacts, tuple[NbaRichFeatureRow, ...]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    feature_rows = rows or (_row("new-2026", 2026), _row("new-2027", 2027))
    feature_path = tmp_path / "features.jsonl"
    write_nba_feature_rows_jsonl(feature_path, feature_rows)
    feature_sha256 = bytes_sha256(feature_path.read_bytes())
    prepared = _prepared(feature_sha256 if reuse_feature_rows else "0" * 64)

    run_path = tmp_path / "run.json"
    run_lock = build_outcome_v2_run_lock(PROJECT_ROOT, prepared, REVISION)
    write_outcome_v2_run_lock(run_path, run_lock)
    experiment_path = tmp_path / "experiment.json"
    experiment_lock = build_outcome_v2_experiment_lock(
        run_path,
        "tinker://run/state/final",
        "tinker://run/sampler/final",
        EXPERIMENT_AT,
    )
    write_outcome_v2_experiment_lock(experiment_path, experiment_lock)
    return (
        OutcomeV2GenerationArtifacts(PROJECT_ROOT, run_path, experiment_path, feature_path),
        feature_rows,
    )


def _lock(
    tmp_path: Path,
) -> tuple[OutcomeV2GenerationArtifacts, tuple[NbaRichFeatureRow, ...], OutcomeV2GenerationLock]:
    artifacts, rows = _artifacts(tmp_path)
    return (
        artifacts,
        rows,
        build_outcome_v2_generation_lock(
            artifacts,
            (101, 202),
            GENERATION_AT,
        ),
    )


def _completed(
    lock: OutcomeV2GenerationLock,
    sequence: int,
    row: NbaRichFeatureRow,
) -> InferenceRecord:
    original = build_orientation_score(0.6, log(0.4), log(0.2))
    swapped = build_orientation_score(0.4, log(0.2), log(0.4))
    return completed_inference_record(
        lock,
        sequence,
        row,
        original_prompt_token_ids_sha256=ORIGINAL_TOKENS_SHA256,
        swapped_prompt_token_ids_sha256=SWAPPED_TOKENS_SHA256,
        original=original,
        swapped=swapped,
    )


def test_generation_lock_binds_exact_target_free_inputs_and_call_policy(
    tmp_path: Path,
) -> None:
    artifacts, rows, lock = _lock(tmp_path)
    record = lock.to_record()
    labels = require_object(required_field(record, "label_token_ids"), "label_token_ids")
    policy = require_object(required_field(record, "call_policy"), "call_policy")

    assert record["candidate_role"] == "forecastfm_outcome_v2_sft_adapter"
    assert record["sampler_path"] == "tinker://run/sampler/final"
    assert record["feature_rows_sha256"] == bytes_sha256(artifacts.feature_rows_path.read_bytes())
    assert record["feature_row_sha256s"] == [row.row_sha256 for row in rows]
    assert record["prompt_pairs_sha256"] == bytes_sha256(outcome_v2_prompt_pairs_jsonl_bytes(rows))
    assert record["question_ids"] == ["new-2026", "new-2027"]
    assert record["question_ids_sha256"] == canonical_sha256(record["question_ids"])
    assert record["tabular_seasons"] == [2024, 2025]
    assert record["evaluation_seasons"] == [2026, 2027]
    assert labels == {TEAM_LABEL: 101, OPPONENT_LABEL: 202}
    assert policy == {
        "logical_calls_per_game": 4,
        "expected_logical_calls": 8,
        "application_attempts_per_game": 1,
        "application_retries": 0,
        "sdk_retry_logic_enabled": False,
        "sdk_internal_retransmission_window_seconds": 300,
        "generated_text_used": False,
        "sdk_internal_unused_generated_tokens_per_call": 1,
        "transport_retry_note": (
            "Tinker 0.22.7 may internally retransmit one logical request ID after connection or "
            "timeout errors and HTTP 408, 409, 429, or 5xx responses for up to five minutes."
        ),
    }
    lock_text = lock.canonical_bytes.decode("utf-8").lower()
    assert all(word not in lock_text for word in ("answer", "resolution", "cohort"))


def test_prompt_pairs_are_adjacent_target_free_original_and_swap() -> None:
    rows = (_row("first", 2026), _row("second", 2027))
    records = build_outcome_v2_prompt_records(rows)

    assert [record["question_id"] for record in records] == [
        "first",
        f"first{SIDE_SWAP_SUFFIX}",
        "second",
        f"second{SIDE_SWAP_SUFFIX}",
    ]
    assert all("label" not in canonical_json(record) for record in records)


def test_generation_lock_round_trip_is_create_only_and_verifies_inputs(tmp_path: Path) -> None:
    artifacts, _rows, lock = _lock(tmp_path)
    path = tmp_path / "generation.json"

    assert write_outcome_v2_generation_lock(path, lock) == lock.sha256
    assert read_outcome_v2_generation_lock(path) == lock
    assert verify_outcome_v2_generation_lock(artifacts, path) == lock
    with pytest.raises(FileExistsError):
        write_outcome_v2_generation_lock(path, lock)

    artifacts.feature_rows_path.write_bytes(artifacts.feature_rows_path.read_bytes() + b"\n")
    with pytest.raises(OutcomeV2InferenceError, match="feature rows"):
        verify_outcome_v2_generation_lock(artifacts, path)


def test_generation_requires_new_strictly_later_feature_rows(tmp_path: Path) -> None:
    old_artifacts, _ = _artifacts(
        tmp_path / "old",
        (_row("old-2025", 2025), _row("new-2026", 2026)),
    )
    with pytest.raises(OutcomeV2InferenceError, match="later than every tabular season"):
        build_outcome_v2_generation_lock(old_artifacts, (101, 202), GENERATION_AT)

    reused_artifacts, _ = _artifacts(tmp_path / "reused", reuse_feature_rows=True)
    with pytest.raises(OutcomeV2InferenceError, match="requires new feature rows"):
        build_outcome_v2_generation_lock(reused_artifacts, (101, 202), GENERATION_AT)


def test_generation_lock_allows_one_rolling_slate_season(tmp_path: Path) -> None:
    artifacts, _ = _artifacts(tmp_path, (_row("one-slate-game", 2026),))

    lock = build_outcome_v2_generation_lock(artifacts, (101, 202), GENERATION_AT)

    assert lock.to_record()["evaluation_seasons"] == [2026]


def test_orientation_and_terminal_records_have_one_scoring_mapping(tmp_path: Path) -> None:
    _artifacts_value, rows, lock = _lock(tmp_path)
    completed = _completed(lock, 0, rows[0])
    secret = "provider said api_key=secret"
    reason = sanitize_inference_failure(RuntimeError(secret))
    failed = failed_inference_record(
        lock,
        1,
        rows[1],
        original_prompt_token_ids_sha256="3" * 64,
        swapped_prompt_token_ids_sha256="4" * 64,
        failure_reason=reason,
    )
    forecasts = binary_forecasts_from_inference_records((completed, failed), lock)

    assert completed.original is not None
    assert completed.original.valid_label_mass == pytest.approx(0.6)
    assert completed.original.team_probability == pytest.approx(0.75)
    assert completed.team_probability == pytest.approx(0.75)
    assert completed.pre_average_side_swap_gap == pytest.approx(0.0)
    assert reason == "candidate_call_exception:RuntimeError"
    assert secret not in reason
    with pytest.raises(OutcomeV2InferenceError, match="sanitized reason"):
        replace(failed, failure_reason="provider_secret")
    assert forecasts[0].team_probability == pytest.approx(0.75)
    assert forecasts[1].team_probability is None
    assert forecasts[1].failure_reason == reason


def test_scores_fail_closed_on_nonfinite_overflow_and_boundary_probabilities() -> None:
    assert rendered_prompt_token_ids_sha256((1, 2)) == canonical_sha256([1, 2])
    with pytest.raises(OutcomeV2InferenceError, match="must not be empty"):
        rendered_prompt_token_ids_sha256(())
    with pytest.raises(OutcomeV2InferenceError, match="cannot derive"):
        build_orientation_score(0.5, 1_000.0, -1.0)
    with pytest.raises(OutcomeV2InferenceError, match="cannot derive"):
        build_orientation_score(0.5, float("nan"), -1.0)
    with pytest.raises(OutcomeV2InferenceError, match=r"differs|interior"):
        build_orientation_score(0.5, -1_000.0, 0.0)


def test_inference_jsonl_is_canonical_create_only_and_exactly_ordered(tmp_path: Path) -> None:
    _artifacts_value, rows, lock = _lock(tmp_path)
    first = _completed(lock, 0, rows[0])
    second = failed_inference_record(
        lock,
        1,
        rows[1],
        original_prompt_token_ids_sha256="3" * 64,
        swapped_prompt_token_ids_sha256="4" * 64,
        failure_reason="candidate_call_exception:TimeoutError",
    )
    path = tmp_path / "records.jsonl"

    digest = write_outcome_v2_inference_records(path, (first, second), lock)

    assert digest == bytes_sha256(path.read_bytes())
    assert read_outcome_v2_inference_records(path, lock) == (first, second)
    with pytest.raises(FileExistsError):
        write_outcome_v2_inference_records(path, (first, second), lock)
    with pytest.raises(OutcomeV2InferenceError, match="order"):
        write_outcome_v2_inference_records(tmp_path / "wrong.jsonl", (second, first), lock)

    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(OutcomeV2InferenceError, match="canonical JSONL bytes"):
        read_outcome_v2_inference_records(path, lock)


def test_strict_lock_and_record_validation_rejects_tampering(tmp_path: Path) -> None:
    _artifacts_value, rows, lock = _lock(tmp_path)
    wrong_role = lock.to_record()
    wrong_role["candidate_role"] = "base_model"
    with pytest.raises(OutcomeV2InferenceError, match="candidate role"):
        OutcomeV2GenerationLock(canonical_json(wrong_role).encode("utf-8"))

    completed = _completed(lock, 0, rows[0])
    different_row = replace(rows[0], evidence_bundle_sha256="c" * 64)
    with pytest.raises(OutcomeV2InferenceError, match="locked feature row"):
        _completed(lock, 0, different_row)
    with pytest.raises(OutcomeV2InferenceError, match="lowercase SHA-256"):
        replace(completed, original_prompt_token_ids_sha256="not-a-hash")
    with pytest.raises(OutcomeV2InferenceError, match="raw scores"):
        replace(completed, original=None)
