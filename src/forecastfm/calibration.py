"""Readable calibration diagnostics for binary forecasts."""

from collections.abc import Sequence
from dataclasses import dataclass

from forecastfm.models import ForecastValidationError, ResolvedForecast


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """Observed and predicted frequencies within one probability interval."""

    lower_bound: float
    upper_bound: float
    count: int
    mean_probability: float
    observed_rate: float


def reliability_bins(
    forecasts: Sequence[ResolvedForecast],
    positive_outcome: str,
    bin_count: int = 10,
) -> tuple[ReliabilityBin, ...]:
    """Group binary forecasts into equal-width reliability bins."""
    if bin_count < 2:
        raise ForecastValidationError("bin_count must be at least two")
    if not forecasts:
        raise ForecastValidationError("at least one resolved forecast is required")

    probabilities: list[list[float]] = [[] for _ in range(bin_count)]
    outcomes: list[list[float]] = [[] for _ in range(bin_count)]
    for forecast in forecasts:
        if len(forecast.distribution.outcomes) != 2:
            raise ForecastValidationError("reliability bins currently require binary forecasts")
        probability = forecast.distribution.probability_for(positive_outcome)
        index = min(int(probability * bin_count), bin_count - 1)
        probabilities[index].append(probability)
        outcomes[index].append(float(forecast.realized_outcome == positive_outcome))

    result: list[ReliabilityBin] = []
    for index, values in enumerate(probabilities):
        if not values:
            continue
        count = len(values)
        result.append(
            ReliabilityBin(
                lower_bound=index / bin_count,
                upper_bound=(index + 1) / bin_count,
                count=count,
                mean_probability=sum(values) / count,
                observed_rate=sum(outcomes[index]) / count,
            )
        )
    return tuple(result)


def expected_calibration_error(bins: Sequence[ReliabilityBin]) -> float:
    """Return count-weighted absolute calibration error."""
    total_count = sum(bin_.count for bin_ in bins)
    if total_count == 0:
        raise ForecastValidationError("at least one non-empty reliability bin is required")
    return (
        sum(bin_.count * abs(bin_.mean_probability - bin_.observed_rate) for bin_ in bins)
        / total_count
    )
