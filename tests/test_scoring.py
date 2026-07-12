"""Tests for proper scoring and calibration diagnostics."""

from datetime import UTC, datetime, timedelta
from math import log

import pytest

from forecastfm.calibration import expected_calibration_error, reliability_bins
from forecastfm.models import Distribution, ForecastValidationError, ResolvedForecast
from forecastfm.scoring import (
    brier_score,
    log_loss,
    summarize_complete_cohort,
    summarize_scores,
)

FORECAST_AT = datetime(2026, 1, 1, tzinfo=UTC)


def resolved_forecast(
    question_id: str,
    probabilities: tuple[float, ...],
    realized_outcome: str,
    outcomes: tuple[str, ...] = ("yes", "no"),
) -> ResolvedForecast:
    return ResolvedForecast(
        question_id=question_id,
        forecast_at=FORECAST_AT,
        distribution=Distribution(outcomes=outcomes, probabilities=probabilities),
        realized_outcome=realized_outcome,
    )


def test_binary_brier_score_uses_common_normalization() -> None:
    forecast = resolved_forecast("question-1", (0.8, 0.2), "yes")

    assert brier_score(forecast) == pytest.approx(0.04)
    assert log_loss(forecast) == pytest.approx(-log(0.8))


def test_score_summary_reports_accuracy() -> None:
    forecasts = (
        resolved_forecast("question-1", (0.8, 0.2), "yes"),
        resolved_forecast("question-2", (0.4, 0.6), "yes"),
    )

    summary = summarize_scores(forecasts)

    assert summary.count == 2
    assert summary.outcome_count == 2
    assert summary.accuracy == 0.5


def test_multiclass_brier_score_uses_classwise_sum() -> None:
    forecast = resolved_forecast(
        "question-1",
        (0.6, 0.3, 0.1),
        "a",
        outcomes=("a", "b", "c"),
    )

    assert brier_score(forecast) == pytest.approx(0.26)


def test_score_summary_rejects_duplicate_identity() -> None:
    first = resolved_forecast("question-1", (0.8, 0.2), "yes")
    duplicate = ResolvedForecast(
        question_id=first.question_id,
        forecast_at=first.forecast_at,
        distribution=Distribution(outcomes=("yes", "no"), probabilities=(0.7, 0.3)),
        realized_outcome="yes",
    )

    with pytest.raises(ForecastValidationError, match="duplicate"):
        summarize_scores((first, duplicate))


def test_complete_cohort_rejects_dropped_forecast() -> None:
    forecast = resolved_forecast("question-1", (0.8, 0.2), "yes")
    expected = {
        ("question-1", FORECAST_AT): "yes",
        ("question-2", FORECAST_AT): "no",
    }

    with pytest.raises(ForecastValidationError, match="frozen cohort"):
        summarize_complete_cohort((forecast,), expected)


def test_complete_cohort_rejects_changed_answer() -> None:
    forecast = resolved_forecast("question-1", (0.8, 0.2), "yes")
    expected = {("question-1", FORECAST_AT): "no"}

    with pytest.raises(ForecastValidationError, match="answer key"):
        summarize_complete_cohort((forecast,), expected)


def test_reliability_bins_and_calibration_error() -> None:
    forecasts = (
        resolved_forecast("question-1", (0.2, 0.8), "no"),
        ResolvedForecast(
            question_id="question-2",
            forecast_at=FORECAST_AT + timedelta(seconds=1),
            distribution=Distribution(outcomes=("yes", "no"), probabilities=(0.8, 0.2)),
            realized_outcome="yes",
        ),
    )

    bins = reliability_bins(forecasts, positive_outcome="yes", bin_count=5)

    assert len(bins) == 2
    assert expected_calibration_error(bins) == pytest.approx(0.2)
