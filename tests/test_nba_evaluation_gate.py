"""Tests for the independent sealed untouched-evaluation gate."""

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from forecastfm.integrity import canonical_json, canonical_sha256
from forecastfm.json_utils import parse_json_object, require_object
from forecastfm.nba_evaluation_gate import (
    NbaEvaluationAnswer,
    NbaEvaluationCohortInput,
    NbaEvaluationGateArtifacts,
    NbaEvaluationGateError,
    NbaEvaluationGatePolicy,
    NbaRecalibrationRow,
    fit_training_only_logit_recalibrator,
    read_nba_evaluation_answers_jsonl,
    read_nba_evaluation_cohort_jsonl,
    read_nba_evaluation_forecasts_jsonl,
    read_nba_recalibration_rows_jsonl,
    verify_untouched_nba_evaluation_gate,
    write_nba_evaluation_answers_jsonl,
    write_nba_evaluation_cohort_jsonl,
    write_nba_evaluation_forecasts_jsonl,
    write_nba_evaluation_gate_report,
    write_nba_recalibration_rows_jsonl,
)
from forecastfm.outcome_v2_metrics import BinaryForecast

_TEST_POLICY = NbaEvaluationGatePolicy(
    minimum_games_per_season=2,
    minimum_calendar_blocks_per_season=2,
    recalibration_gradient_steps=50,
    recalibration_learning_rate=0.05,
    recalibration_initial_intercept=0.0,
    recalibration_initial_slope=1.0,
)


def _evaluation_rows() -> tuple[NbaEvaluationCohortInput, ...]:
    return (
        NbaEvaluationCohortInput("eval-2024-a", 2024, date(2023, 10, 2), 0.6),
        NbaEvaluationCohortInput("eval-2024-b", 2024, date(2023, 10, 9), 0.4),
        NbaEvaluationCohortInput("eval-2025-a", 2025, date(2024, 10, 7), 0.6),
        NbaEvaluationCohortInput("eval-2025-b", 2025, date(2024, 10, 14), 0.4),
    )


def _answers() -> tuple[NbaEvaluationAnswer, ...]:
    return (
        NbaEvaluationAnswer("eval-2024-a", True),
        NbaEvaluationAnswer("eval-2024-b", False),
        NbaEvaluationAnswer("eval-2025-a", True),
        NbaEvaluationAnswer("eval-2025-b", False),
    )


def _forecasts() -> tuple[BinaryForecast, ...]:
    return (
        BinaryForecast("eval-2024-a", 0.9),
        BinaryForecast("eval-2024-b", 0.1),
        BinaryForecast("eval-2025-a", 0.9),
        BinaryForecast("eval-2025-b", 0.1),
    )


def _calibration_rows(*, season: int = 2023) -> tuple[NbaRecalibrationRow, ...]:
    year = season - 1
    return (
        NbaRecalibrationRow("cal-a", season, date(year, 10, 3), 0.6, True),
        NbaRecalibrationRow("cal-b", season, date(year, 10, 10), 0.6, False),
        NbaRecalibrationRow("cal-c", season, date(year, 10, 17), 0.4, True),
        NbaRecalibrationRow("cal-d", season, date(year, 10, 24), 0.4, False),
    )


def _write_artifacts(
    directory: Path,
    *,
    calibration: tuple[NbaRecalibrationRow, ...] | None = None,
    cohort: tuple[NbaEvaluationCohortInput, ...] | None = None,
    answers: tuple[NbaEvaluationAnswer, ...] | None = None,
    forecasts: tuple[BinaryForecast, ...] | None = None,
) -> NbaEvaluationGateArtifacts:
    directory.mkdir(parents=True, exist_ok=True)
    cohort_path = directory / "cohort.jsonl"
    answers_path = directory / "answers.jsonl"
    forecasts_path = directory / "forecasts.jsonl"
    calibration_path = directory / "calibration.jsonl"
    write_nba_evaluation_cohort_jsonl(
        cohort_path,
        _evaluation_rows() if cohort is None else cohort,
    )
    write_nba_evaluation_answers_jsonl(
        answers_path,
        _answers() if answers is None else answers,
    )
    write_nba_evaluation_forecasts_jsonl(
        forecasts_path,
        _forecasts() if forecasts is None else forecasts,
    )
    write_nba_recalibration_rows_jsonl(
        calibration_path,
        _calibration_rows() if calibration is None else calibration,
    )
    return NbaEvaluationGateArtifacts(
        cohort_path=cohort_path,
        answers_path=answers_path,
        forecasts_path=forecasts_path,
        calibration_path=calibration_path,
    )


def test_valid_gate_round_trips_and_recomputes_both_season_gates(tmp_path: Path) -> None:
    artifacts = _write_artifacts(tmp_path)

    first = verify_untouched_nba_evaluation_gate(artifacts, policy=_TEST_POLICY)
    second = verify_untouched_nba_evaluation_gate(artifacts, policy=_TEST_POLICY)

    assert first == second
    assert first.sha256 == canonical_sha256(first.payload)
    assert first.payload["status"] == "passed"
    proof_scope = require_object(first.payload["proof_scope"], "proof_scope")
    assert proof_scope == {
        "candidate_model_and_run_provenance": "required_separately",
        "external_precommit_and_timestamp": "required_separately",
        "raw_provider_derivation": "required_separately",
    }
    evaluation = require_object(first.payload["evaluation"], "evaluation")
    assert evaluation["seasons"] == [2024, 2025]
    raw_gate = require_object(
        evaluation["candidate_forecasts_vs_raw_elo"],
        "candidate_forecasts_vs_raw_elo",
    )
    recalibrated_gate = require_object(
        evaluation["candidate_forecasts_vs_recalibrated_elo"],
        "candidate_forecasts_vs_recalibrated_elo",
    )
    assert raw_gate["passes"] is True
    assert recalibrated_gate["passes"] is True
    recalibration = require_object(first.payload["recalibration"], "recalibration")
    assert recalibration["training_ids_sha256"] == canonical_sha256(
        [row.question_id for row in _calibration_rows()]
    )
    assert isinstance(recalibration["model_sha256"], str)
    policy = require_object(first.payload["policy"], "policy")
    assert policy["config"] == _TEST_POLICY.canonical_payload()
    assert policy["policy_sha256"] == _TEST_POLICY.policy_sha256

    assert read_nba_evaluation_cohort_jsonl(artifacts.cohort_path) == _evaluation_rows()
    assert read_nba_evaluation_answers_jsonl(artifacts.answers_path) == _answers()
    assert read_nba_evaluation_forecasts_jsonl(artifacts.forecasts_path) == _forecasts()
    assert read_nba_recalibration_rows_jsonl(artifacts.calibration_path) == _calibration_rows()


def test_supplied_report_must_equal_the_recomputed_canonical_report(tmp_path: Path) -> None:
    artifacts = _write_artifacts(tmp_path)
    report = verify_untouched_nba_evaluation_gate(artifacts, policy=_TEST_POLICY)
    report_path = tmp_path / "report.json"
    write_nba_evaluation_gate_report(report_path, report)

    verified = verify_untouched_nba_evaluation_gate(
        replace(artifacts, supplied_report_path=report_path),
        policy=_TEST_POLICY,
    )
    assert verified == report
    with pytest.raises(NbaEvaluationGateError, match="already exists"):
        write_nba_evaluation_gate_report(report_path, report)

    payload = parse_json_object(report_path.read_text(encoding="utf-8"))
    payload["status"] = "claimed"
    report_path.write_text(canonical_json(payload), encoding="utf-8")
    with pytest.raises(NbaEvaluationGateError, match="differs from recomputed"):
        verify_untouched_nba_evaluation_gate(
            replace(artifacts, supplied_report_path=report_path),
            policy=_TEST_POLICY,
        )


def test_forecast_jsonl_retains_explicit_failures_and_is_create_only(tmp_path: Path) -> None:
    path = tmp_path / "forecasts.jsonl"
    rows = (
        BinaryForecast("valid", 0.75),
        BinaryForecast("failed", None, "malformed model response"),
    )

    write_nba_evaluation_forecasts_jsonl(path, rows)

    assert read_nba_evaluation_forecasts_jsonl(path) == rows
    with pytest.raises(NbaEvaluationGateError, match="already exists"):
        write_nba_evaluation_forecasts_jsonl(path, rows)


@pytest.mark.parametrize("target", ["answers", "forecasts"])
def test_answers_and_forecasts_require_exact_cohort_ids_and_order(
    tmp_path: Path,
    target: str,
) -> None:
    artifacts = _write_artifacts(tmp_path)
    path = artifacts.answers_path if target == "answers" else artifacts.forecasts_path
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(reversed(lines)), encoding="utf-8")

    with pytest.raises(NbaEvaluationGateError, match="IDs, order, or coverage"):
        verify_untouched_nba_evaluation_gate(artifacts, policy=_TEST_POLICY)


def test_calibration_ids_are_disjoint_and_all_seasons_are_strictly_earlier(
    tmp_path: Path,
) -> None:
    late = _write_artifacts(tmp_path / "late", calibration=_calibration_rows(season=2024))
    with pytest.raises(NbaEvaluationGateError, match="strictly before"):
        verify_untouched_nba_evaluation_gate(late, policy=_TEST_POLICY)

    overlapping_rows = (
        replace(_calibration_rows()[0], question_id="eval-2024-a"),
        *_calibration_rows()[1:],
    )
    overlap = _write_artifacts(tmp_path / "overlap", calibration=overlapping_rows)
    with pytest.raises(NbaEvaluationGateError, match="question IDs must be disjoint"):
        verify_untouched_nba_evaluation_gate(overlap, policy=_TEST_POLICY)


def test_recalibration_fit_is_deterministic_and_binds_training_id_order() -> None:
    rows = _calibration_rows()

    first = fit_training_only_logit_recalibrator(rows, policy=_TEST_POLICY)
    second = fit_training_only_logit_recalibrator(rows, policy=_TEST_POLICY)
    reversed_model = fit_training_only_logit_recalibrator(
        tuple(reversed(rows)),
        policy=_TEST_POLICY,
    )

    assert first == second
    assert first.model_sha256 == second.model_sha256
    assert first.training_ids_sha256 == canonical_sha256([row.question_id for row in rows])
    assert reversed_model.training_ids_sha256 != first.training_ids_sha256
    assert 0.0 < first.team_probability(0.6) < 1.0


def test_cohort_cannot_contain_outcomes_or_noncanonical_json(tmp_path: Path) -> None:
    path = tmp_path / "cohort.jsonl"
    write_nba_evaluation_cohort_jsonl(path, _evaluation_rows())
    lines = path.read_text(encoding="utf-8").splitlines()
    first = parse_json_object(lines[0])
    first["realized_team_win"] = True
    path.write_text(
        "\n".join((canonical_json(first), *lines[1:])) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(NbaEvaluationGateError, match="invalid NBA evaluation cohort"):
        read_nba_evaluation_cohort_jsonl(path)

    path.write_text(" " + "\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(NbaEvaluationGateError, match="canonical JSONL"):
        read_nba_evaluation_cohort_jsonl(path)


def test_strict_runtime_types_dates_and_signed_zero_fail_closed() -> None:
    with pytest.raises(NbaEvaluationGateError, match="finite float"):
        NbaEvaluationCohortInput("integer", 2024, date(2023, 10, 2), 1)
    with pytest.raises(NbaEvaluationGateError, match="negative zero"):
        NbaEvaluationCohortInput("negative-zero", 2024, date(2023, 10, 2), -0.0)
    with pytest.raises(NbaEvaluationGateError, match="game_date must be a date"):
        NbaEvaluationCohortInput(
            "datetime",
            2024,
            datetime(2023, 10, 2, tzinfo=UTC),
            0.6,
        )
    with pytest.raises(NbaEvaluationGateError, match="does not match game_date"):
        NbaEvaluationCohortInput("wrong-season", 2025, date(2023, 10, 2), 0.6)


def test_policy_requires_strict_positive_integers_and_finite_floats() -> None:
    with pytest.raises(NbaEvaluationGateError, match="positive integer"):
        replace(_TEST_POLICY, minimum_games_per_season=0)
    with pytest.raises(NbaEvaluationGateError, match="positive integer"):
        replace(_TEST_POLICY, recalibration_gradient_steps=True)
    with pytest.raises(NbaEvaluationGateError, match="finite float"):
        replace(_TEST_POLICY, recalibration_learning_rate=1)
    with pytest.raises(NbaEvaluationGateError, match="negative zero"):
        replace(_TEST_POLICY, recalibration_initial_slope=-0.0)


def test_every_season_must_meet_the_minimum_game_count(tmp_path: Path) -> None:
    artifacts = _write_artifacts(
        tmp_path,
        cohort=_evaluation_rows()[::2],
        answers=_answers()[::2],
        forecasts=_forecasts()[::2],
    )
    policy = replace(
        _TEST_POLICY,
        minimum_calendar_blocks_per_season=1,
    )

    with pytest.raises(NbaEvaluationGateError, match="minimum_games_per_season"):
        verify_untouched_nba_evaluation_gate(artifacts, policy=policy)


def test_every_season_must_meet_the_minimum_calendar_block_count(tmp_path: Path) -> None:
    rows = _evaluation_rows()
    one_block_per_season = (
        rows[0],
        replace(rows[1], game_date=date(2023, 10, 3)),
        rows[2],
        replace(rows[3], game_date=date(2024, 10, 8)),
    )
    artifacts = _write_artifacts(tmp_path, cohort=one_block_per_season)

    with pytest.raises(NbaEvaluationGateError, match="minimum_calendar_blocks_per_season"):
        verify_untouched_nba_evaluation_gate(artifacts, policy=_TEST_POLICY)


def test_report_and_model_hashes_bind_the_complete_policy(tmp_path: Path) -> None:
    artifacts = _write_artifacts(tmp_path)
    changed_policy = replace(_TEST_POLICY, recalibration_learning_rate=0.025)

    original = verify_untouched_nba_evaluation_gate(artifacts, policy=_TEST_POLICY)
    changed = verify_untouched_nba_evaluation_gate(artifacts, policy=changed_policy)

    assert original.sha256 != changed.sha256
    original_policy = require_object(original.payload["policy"], "policy")
    changed_policy_payload = require_object(changed.payload["policy"], "policy")
    assert original_policy["policy_sha256"] != changed_policy_payload["policy_sha256"]
    original_model = require_object(original.payload["recalibration"], "recalibration")
    changed_model = require_object(changed.payload["recalibration"], "recalibration")
    assert original_model["model_sha256"] != changed_model["model_sha256"]


def test_model_must_pass_both_recomputed_gates(tmp_path: Path) -> None:
    artifacts = _write_artifacts(tmp_path)
    losing = tuple(
        BinaryForecast(answer.question_id, 0.1 if answer.realized_team_win else 0.9)
        for answer in _answers()
    )
    artifacts.forecasts_path.unlink()
    write_nba_evaluation_forecasts_jsonl(artifacts.forecasts_path, losing)

    with pytest.raises(NbaEvaluationGateError, match="must beat raw"):
        verify_untouched_nba_evaluation_gate(artifacts, policy=_TEST_POLICY)
