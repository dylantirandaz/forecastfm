"""Proper scoring rules and aggregate evaluation."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from math import log

from forecastfm.models import ForecastValidationError, ResolvedForecast

MINIMUM_LOG_PROBABILITY = 1e-15


def brier_score(forecast: ResolvedForecast) -> float:
    """Return the common binary or multiclass Brier score; lower is better."""
    realized_probability = forecast.distribution.probability_for(forecast.realized_outcome)
    if len(forecast.distribution.outcomes) == 2:
        return (1.0 - realized_probability) ** 2

    squared_errors = tuple(
        (probability - float(outcome == forecast.realized_outcome)) ** 2
        for outcome, probability in zip(
            forecast.distribution.outcomes,
            forecast.distribution.probabilities,
            strict=True,
        )
    )
    return sum(squared_errors)


def log_loss(forecast: ResolvedForecast) -> float:
    """Return negative log probability assigned to the realized outcome."""
    probability = forecast.distribution.probability_for(forecast.realized_outcome)
    return -log(max(probability, MINIMUM_LOG_PROBABILITY))


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    """Aggregate proper scores for a resolved forecast collection."""

    count: int
    outcome_count: int
    mean_brier: float
    mean_log_loss: float
    accuracy: float


def summarize_scores(forecasts: Sequence[ResolvedForecast]) -> ScoreSummary:
    """Calculate aggregate scores after rejecting duplicate or mixed-shape rows."""
    if not forecasts:
        raise ForecastValidationError("at least one resolved forecast is required")

    identities = {(forecast.question_id, forecast.forecast_at) for forecast in forecasts}
    if len(identities) != len(forecasts):
        raise ForecastValidationError("duplicate resolved forecast identity")
    outcome_counts = {len(forecast.distribution.outcomes) for forecast in forecasts}
    if len(outcome_counts) != 1:
        raise ForecastValidationError("score summaries cannot mix outcome counts")

    count = len(forecasts)
    outcome_count = outcome_counts.pop()
    correct = sum(
        forecast.distribution.predicted_outcome() == forecast.realized_outcome
        for forecast in forecasts
    )
    return ScoreSummary(
        count=count,
        outcome_count=outcome_count,
        mean_brier=sum(brier_score(forecast) for forecast in forecasts) / count,
        mean_log_loss=sum(log_loss(forecast) for forecast in forecasts) / count,
        accuracy=correct / count,
    )


def summarize_complete_cohort(
    forecasts: Sequence[ResolvedForecast],
    expected_outcomes: Mapping[tuple[str, datetime], str],
) -> ScoreSummary:
    """Score only when every frozen identity and outcome is present exactly once."""
    identities = [(forecast.question_id, forecast.forecast_at) for forecast in forecasts]
    if len(set(identities)) != len(identities):
        raise ForecastValidationError("duplicate resolved forecast identity")
    if set(identities) != set(expected_outcomes):
        raise ForecastValidationError("forecast identities do not match the frozen cohort")
    for forecast, identity in zip(forecasts, identities, strict=True):
        if forecast.realized_outcome != expected_outcomes[identity]:
            raise ForecastValidationError("realized outcome differs from the frozen answer key")
    return summarize_scores(forecasts)
