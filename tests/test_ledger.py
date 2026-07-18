"""Tests for the prospective hash-chained ledger."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from forecastfm.integrity import canonical_json, text_sha256
from forecastfm.json_utils import parse_json_object, require_list, require_object, required_field
from forecastfm.ledger import (
    Cohort,
    CohortGame,
    ForecastSubmission,
    LedgerValidationError,
    ResolutionSubmission,
    append_forecast_batch,
    append_resolution_batch,
    audit_ledger,
    load_cohort,
)
from forecastfm.models import Distribution, ForecastPrediction

SCHEDULE_RETRIEVED = datetime(2026, 10, 1, 12, tzinfo=UTC)
INPUT_AS_OF = datetime(2026, 10, 1, 13, tzinfo=UTC)
GENERATED_AT = datetime(2026, 10, 1, 14, tzinfo=UTC)
FORECAST_RECORDED_AT = datetime(2026, 10, 1, 15, tzinfo=UTC)
FORECAST_DEADLINE = datetime(2026, 10, 1, 16, tzinfo=UTC)
TIPOFF = datetime(2026, 10, 1, 17, tzinfo=UTC)
RESOLVED_AT = datetime(2026, 10, 1, 20, tzinfo=UTC)
RESOLUTION_RECORDED_AT = datetime(2026, 10, 1, 21, tzinfo=UTC)


def make_cohort() -> Cohort:
    games = tuple(
        CohortGame(
            question_id=f"question-{index}",
            source_game_id=f"source-{index}",
            team_id=f"Team {index}",
            opponent_id=f"Opponent {index}",
            site="neutral",
            matchup=f"Team {index} vs Opponent {index}",
            outcomes=("listed_team_wins", "opponent_wins"),
            forecast_deadline=FORECAST_DEADLINE,
            scheduled_tipoff=TIPOFF + timedelta(hours=index),
        )
        for index in (1, 2)
    )
    return Cohort(
        cohort_id="nba-2026-10-01",
        experiment_sha256="a" * 64,
        schedule_source="https://example.test/schedule",
        schedule_snapshot_sha256="b" * 64,
        schedule_retrieved=SCHEDULE_RETRIEVED,
        inclusion_rule="Every scheduled NBA game in the October 1 slate.",
        games=games,
    )


def make_forecast(question_id: str, probability: float = 0.6) -> ForecastSubmission:
    prompt = f'{{"question_id":"{question_id}"}}'
    raw_response = canonical_json(
        {
            "probabilities": {
                "listed_team_wins": probability,
                "opponent_wins": 1.0 - probability,
            }
        }
    )
    prediction = ForecastPrediction(
        distribution=Distribution(
            outcomes=("listed_team_wins", "opponent_wins"),
            probabilities=(probability, 1.0 - probability),
        )
    )
    return ForecastSubmission(
        question_id=question_id,
        input_as_of=INPUT_AS_OF,
        generated_at=GENERATED_AT,
        evidence_bundle_sha256="c" * 64,
        prompt=prompt,
        prompt_sha256=text_sha256(prompt),
        raw_response=raw_response,
        prediction=prediction,
        provider_request_id=f"request-{question_id}",
    )


def make_forecasts(cohort: Cohort) -> tuple[ForecastSubmission, ...]:
    return tuple(make_forecast(game.question_id) for game in cohort.games)


def make_resolutions(cohort: Cohort) -> tuple[ResolutionSubmission, ...]:
    return tuple(
        ResolutionSubmission(
            question_id=game.question_id,
            realized_outcome="listed_team_wins",
            resolved_at=RESOLVED_AT,
            resolution_source="https://example.test/results",
            resolution_source_sha256="d" * 64,
        )
        for game in cohort.games
    )


def test_complete_batches_are_hash_chained_and_auditable(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    cohort = make_cohort()
    forecast_head = append_forecast_batch(
        path, cohort, make_forecasts(cohort), FORECAST_RECORDED_AT
    )

    forecast_audit = audit_ledger(path, expected_head=forecast_head)
    resolution_head = append_resolution_batch(
        path,
        cohort,
        make_resolutions(cohort),
        RESOLUTION_RECORDED_AT,
        expected_head=forecast_head,
    )
    audit = audit_ledger(
        path,
        expected_head=resolution_head,
        expected_experiment_sha256=cohort.experiment_sha256,
    )

    assert forecast_audit.unresolved_cohort_ids == (cohort.cohort_id,)
    assert audit.event_count == 2
    assert audit.cohort_count == 1
    assert audit.resolution_count == 1
    assert audit.unresolved_cohort_ids == ()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_load_cohort_reads_the_strict_embedded_schema(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    cohort_path = tmp_path / "cohort.json"
    cohort = make_cohort()
    append_forecast_batch(ledger_path, cohort, make_forecasts(cohort), FORECAST_RECORDED_AT)
    event = parse_json_object(ledger_path.read_text(encoding="utf-8"))
    payload = require_object(required_field(event, "payload"), "payload")
    cohort_record = require_object(required_field(payload, "cohort"), "cohort")
    cohort_path.write_text(canonical_json(cohort_record), encoding="utf-8")

    assert load_cohort(cohort_path) == cohort


def test_raw_response_must_equal_stored_prediction() -> None:
    stored = make_forecast("question-1", probability=0.7)
    raw_for_different_prediction = make_forecast("question-1", probability=0.6).raw_response

    with pytest.raises(LedgerValidationError, match="does not match"):
        replace(stored, raw_response=raw_for_different_prediction)


def test_forecast_requires_an_evidence_bundle_digest() -> None:
    with pytest.raises(LedgerValidationError, match="evidence_bundle_sha256"):
        replace(make_forecast("question-1"), evidence_bundle_sha256="not-a-digest")


def test_forecast_batch_rejects_missing_game_and_late_generation(tmp_path: Path) -> None:
    cohort = make_cohort()
    submissions = make_forecasts(cohort)

    with pytest.raises(LedgerValidationError, match="exactly match"):
        append_forecast_batch(
            tmp_path / "missing.jsonl",
            cohort,
            submissions[:-1],
            FORECAST_RECORDED_AT,
        )

    late = replace(submissions[0], generated_at=FORECAST_RECORDED_AT + timedelta(seconds=1))
    with pytest.raises(LedgerValidationError, match="prospective ordering"):
        append_forecast_batch(
            tmp_path / "late.jsonl",
            cohort,
            (late, submissions[1]),
            FORECAST_RECORDED_AT,
        )

    resolution_path = tmp_path / "missing-resolution.jsonl"
    append_forecast_batch(resolution_path, cohort, submissions, FORECAST_RECORDED_AT)
    with pytest.raises(LedgerValidationError, match="exactly match"):
        append_resolution_batch(
            resolution_path,
            cohort,
            make_resolutions(cohort)[:-1],
            RESOLUTION_RECORDED_AT,
        )


def test_dataclasses_reject_non_utc_timestamps() -> None:
    central = timezone(timedelta(hours=-6))

    with pytest.raises(LedgerValidationError, match="UTC"):
        replace(make_forecast("question-1"), generated_at=GENERATED_AT.astimezone(central))


def test_duplicate_and_out_of_order_batches_are_rejected(tmp_path: Path) -> None:
    cohort = make_cohort()
    forecasts = make_forecasts(cohort)
    resolutions = make_resolutions(cohort)
    path = tmp_path / "ledger.jsonl"

    with pytest.raises(LedgerValidationError, match="must follow"):
        append_resolution_batch(path, cohort, resolutions, RESOLUTION_RECORDED_AT)

    append_forecast_batch(path, cohort, forecasts, FORECAST_RECORDED_AT)
    with pytest.raises(LedgerValidationError, match="duplicate forecast"):
        append_forecast_batch(path, cohort, forecasts, FORECAST_RECORDED_AT)

    append_resolution_batch(path, cohort, resolutions, RESOLUTION_RECORDED_AT)
    with pytest.raises(LedgerValidationError, match="duplicate resolution"):
        append_resolution_batch(path, cohort, resolutions, RESOLUTION_RECORDED_AT)


def test_tampering_and_record_reordering_are_detected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    cohort = make_cohort()
    append_forecast_batch(path, cohort, make_forecasts(cohort), FORECAST_RECORDED_AT)
    append_resolution_batch(path, cohort, make_resolutions(cohort), RESOLUTION_RECORDED_AT)
    original_lines = path.read_text(encoding="utf-8").splitlines()

    first = parse_json_object(original_lines[0])
    payload = require_object(required_field(first, "payload"), "payload")
    submissions = require_list(required_field(payload, "submissions"), "submissions")
    first_submission = require_object(submissions[0], "submission")
    first_submission["prompt"] = "tampered"
    submissions[0] = first_submission
    payload["submissions"] = submissions
    first["payload"] = payload
    path.write_text(f"{canonical_json(first)}\n{original_lines[1]}\n", encoding="utf-8")
    with pytest.raises(LedgerValidationError, match="event_hash"):
        audit_ledger(path)

    path.write_text(f"{original_lines[1]}\n{original_lines[0]}\n", encoding="utf-8")
    with pytest.raises(LedgerValidationError, match="sequence"):
        audit_ledger(path)


def test_expected_head_detects_valid_prefix_truncation(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    cohort = make_cohort()
    append_forecast_batch(path, cohort, make_forecasts(cohort), FORECAST_RECORDED_AT)
    final_head = append_resolution_batch(
        path, cohort, make_resolutions(cohort), RESOLUTION_RECORDED_AT
    )
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(f"{first_line}\n", encoding="utf-8")

    with pytest.raises(LedgerValidationError, match="truncated"):
        audit_ledger(path, expected_head=final_head)


def test_expected_experiment_detects_substitution(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    cohort = make_cohort()
    append_forecast_batch(path, cohort, make_forecasts(cohort), FORECAST_RECORDED_AT)

    with pytest.raises(LedgerValidationError, match="different experiment"):
        audit_ledger(path, expected_experiment_sha256="c" * 64)
