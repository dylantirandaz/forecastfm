"""Evaluation helpers for resolved NBA outcome forecasts."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, sqrt

from forecastfm.calibration import (
    ReliabilityBin,
    expected_calibration_error,
    reliability_bins,
)
from forecastfm.models import (
    Distribution,
    ForecastValidationError,
    ResolvedForecast,
    TrainingExample,
)
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.outcome import NBA_OUTCOMES, TEAM_OUTCOME
from forecastfm.scoring import brier_score, log_loss, summarize_scores

RELIABILITY_BIN_COUNT = 10
NORMAL_95_Z_SCORE = 1.959963984540054
HARD_CONFIDENCE_LIMIT = 0.60
EASY_CONFIDENCE_LIMIT = 0.75


@dataclass(frozen=True, slots=True)
class OutcomeMetrics:
    """Aggregate and per-game diagnostics for one aligned model forecast."""

    count: int
    mean_brier: float
    mean_log_loss: float
    accuracy: float
    expected_calibration_error: float
    reliability_bins: tuple[ReliabilityBin, ...]
    per_game_brier_scores: tuple[float, ...]
    per_game_log_losses: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PairedMeanDelta:
    """Candidate-minus-baseline mean delta and normal-approximation interval."""

    count: int
    mean_delta: float
    standard_error: float
    lower_95: float
    upper_95: float


@dataclass(frozen=True, slots=True)
class DifficultySubsets:
    """Aligned game indices partitioned by venue-adjusted Elo confidence."""

    hard: tuple[int, ...]
    medium: tuple[int, ...]
    easy: tuple[int, ...]


def summarize_outcome_metrics(
    examples: Sequence[TrainingExample],
    team_win_probabilities: Sequence[float],
) -> OutcomeMetrics:
    """Score aligned team-win probabilities against realized NBA winners."""
    forecasts = _resolved_forecasts(examples, team_win_probabilities)
    summary = summarize_scores(forecasts)
    bins = reliability_bins(
        forecasts,
        positive_outcome=TEAM_OUTCOME,
        bin_count=RELIABILITY_BIN_COUNT,
    )
    return OutcomeMetrics(
        count=summary.count,
        mean_brier=summary.mean_brier,
        mean_log_loss=summary.mean_log_loss,
        accuracy=summary.accuracy,
        expected_calibration_error=expected_calibration_error(bins),
        reliability_bins=bins,
        per_game_brier_scores=tuple(brier_score(forecast) for forecast in forecasts),
        per_game_log_losses=tuple(log_loss(forecast) for forecast in forecasts),
    )


def paired_mean_delta(
    baseline_values: Sequence[float],
    candidate_values: Sequence[float],
) -> PairedMeanDelta:
    """Return candidate-minus-baseline mean delta and a normal 95% interval."""
    if len(baseline_values) != len(candidate_values):
        raise ForecastValidationError("paired metric vectors must have equal lengths")
    if len(baseline_values) < 2:
        raise ForecastValidationError("at least two paired metric values are required")
    if not all(isfinite(value) for value in (*baseline_values, *candidate_values)):
        raise ForecastValidationError("paired metric values must be finite")

    deltas = tuple(
        candidate - baseline
        for baseline, candidate in zip(baseline_values, candidate_values, strict=True)
    )
    if not all(isfinite(delta) for delta in deltas):
        raise ForecastValidationError("paired metric deltas must be finite")

    count = len(deltas)
    mean_delta = sum(deltas) / count
    sample_variance = sum((delta - mean_delta) ** 2 for delta in deltas) / (count - 1)
    standard_error = sqrt(sample_variance / count)
    margin = NORMAL_95_Z_SCORE * standard_error
    return PairedMeanDelta(
        count=count,
        mean_delta=mean_delta,
        standard_error=standard_error,
        lower_95=mean_delta - margin,
        upper_95=mean_delta + margin,
    )


def difficulty_subsets(examples: Sequence[TrainingExample]) -> DifficultySubsets:
    """Partition aligned indices by the Elo target's winning-side confidence."""
    _require_original_resolved_examples(examples)
    hard: list[int] = []
    medium: list[int] = []
    easy: list[int] = []
    for index, example in enumerate(examples):
        confidence = max(example.target.distribution.probabilities)
        if confidence < HARD_CONFIDENCE_LIMIT:
            hard.append(index)
        elif confidence < EASY_CONFIDENCE_LIMIT:
            medium.append(index)
        else:
            easy.append(index)
    return DifficultySubsets(hard=tuple(hard), medium=tuple(medium), easy=tuple(easy))


def _resolved_forecasts(
    examples: Sequence[TrainingExample],
    team_win_probabilities: Sequence[float],
) -> tuple[ResolvedForecast, ...]:
    _require_original_resolved_examples(examples)
    if len(examples) != len(team_win_probabilities):
        raise ForecastValidationError("each example must have one team-win probability")

    forecasts: list[ResolvedForecast] = []
    for example, probability in zip(examples, team_win_probabilities, strict=True):
        realized_outcome = example.realized_outcome
        if realized_outcome is None:
            raise ForecastValidationError("outcome metrics require resolved examples")
        forecasts.append(
            ResolvedForecast(
                question_id=example.case.question.question_id,
                forecast_at=example.case.question.forecast_at,
                distribution=Distribution(
                    outcomes=NBA_OUTCOMES,
                    probabilities=(probability, 1.0 - probability),
                ),
                realized_outcome=realized_outcome,
            )
        )
    return tuple(forecasts)


def _require_original_resolved_examples(examples: Sequence[TrainingExample]) -> None:
    if not examples:
        raise ForecastValidationError("at least one resolved NBA example is required")
    identities = {
        (example.case.question.question_id, example.case.question.forecast_at)
        for example in examples
    }
    if len(identities) != len(examples):
        raise ForecastValidationError("duplicate resolved NBA example identity")
    for example in examples:
        if example.case.question.outcomes != NBA_OUTCOMES:
            raise ForecastValidationError("outcome metrics require canonical NBA outcomes")
        if example.realized_outcome is None:
            raise ForecastValidationError("outcome metrics require resolved examples")
        if example.case.question.question_id.endswith(SIDE_SWAP_SUFFIX):
            raise ForecastValidationError("outcome metrics require original game orientations")
