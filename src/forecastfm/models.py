"""Validated domain objects for probabilistic forecasting."""

from dataclasses import dataclass
from datetime import datetime
from math import isclose, isfinite

PROBABILITY_TOLERANCE = 1e-6


class ForecastValidationError(ValueError):
    """Raised when forecast data violates the core schema."""


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ForecastValidationError(f"{field_name} must not be empty")


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ForecastValidationError(f"{field_name} must be timezone-aware")


def _validate_outcomes(outcomes: tuple[str, ...]) -> None:
    if len(outcomes) < 2:
        raise ForecastValidationError("a forecast requires at least two outcomes")
    if len(set(outcomes)) != len(outcomes):
        raise ForecastValidationError("outcomes must be unique")
    for outcome in outcomes:
        _require_text(outcome, "outcome")


@dataclass(frozen=True, slots=True)
class Distribution:
    """A categorical probability distribution with stable outcome ordering."""

    outcomes: tuple[str, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        _validate_outcomes(self.outcomes)
        if len(self.probabilities) != len(self.outcomes):
            raise ForecastValidationError("each outcome must have one probability")
        if not all(isfinite(probability) for probability in self.probabilities):
            raise ForecastValidationError("probabilities must be finite")
        if not all(0.0 <= probability <= 1.0 for probability in self.probabilities):
            raise ForecastValidationError("probabilities must be between zero and one")
        if not isclose(sum(self.probabilities), 1.0, abs_tol=PROBABILITY_TOLERANCE):
            raise ForecastValidationError("probabilities must sum to one")

    def probability_for(self, outcome: str) -> float:
        """Return the probability assigned to an outcome."""
        try:
            index = self.outcomes.index(outcome)
        except ValueError as error:
            raise ForecastValidationError(f"unknown outcome: {outcome}") from error
        return self.probabilities[index]

    def predicted_outcome(self) -> str:
        """Return the first outcome with the greatest probability."""
        index = max(range(len(self.probabilities)), key=self.probabilities.__getitem__)
        return self.outcomes[index]

    def as_dict(self) -> dict[str, float]:
        """Return an insertion-ordered outcome-to-probability mapping."""
        return dict(zip(self.outcomes, self.probabilities, strict=True))


@dataclass(frozen=True, slots=True)
class ForecastQuestion:
    """A resolvable question as it existed at a forecast cutoff."""

    question_id: str
    text: str
    resolution_rule: str
    resolution_source: str
    outcomes: tuple[str, ...]
    forecast_at: datetime
    resolves_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.question_id, "question_id")
        _require_text(self.text, "text")
        _require_text(self.resolution_rule, "resolution_rule")
        _require_text(self.resolution_source, "resolution_source")
        _validate_outcomes(self.outcomes)
        _require_aware(self.forecast_at, "forecast_at")
        _require_aware(self.resolves_at, "resolves_at")
        if self.resolves_at <= self.forecast_at:
            raise ForecastValidationError("resolves_at must be after forecast_at")


@dataclass(frozen=True, slots=True)
class EvidenceCard:
    """A single fact that was available before the forecast cutoff."""

    text: str
    source: str
    available_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.text, "evidence text")
        _require_text(self.source, "evidence source")
        _require_aware(self.available_at, "available_at")


@dataclass(frozen=True, slots=True)
class ForecastCase:
    """The complete point-in-time input to a forecasting model."""

    question: ForecastQuestion
    prior: Distribution
    prior_source: str
    prior_as_of: datetime
    evidence: tuple[EvidenceCard, ...] = ()

    def __post_init__(self) -> None:
        if self.prior.outcomes != self.question.outcomes:
            raise ForecastValidationError("prior outcomes must match question outcomes")
        _require_text(self.prior_source, "prior_source")
        _require_aware(self.prior_as_of, "prior_as_of")
        if self.prior_as_of > self.question.forecast_at:
            raise ForecastValidationError("prior cannot be newer than the forecast cutoff")
        previous_time: datetime | None = None
        for card in self.evidence:
            if card.available_at > self.question.forecast_at:
                raise ForecastValidationError("evidence cannot be newer than the forecast cutoff")
            if previous_time is not None and card.available_at < previous_time:
                raise ForecastValidationError("evidence cards must be ordered by available_at")
            previous_time = card.available_at


@dataclass(frozen=True, slots=True)
class ForecastPrediction:
    """A model's probability distribution."""

    distribution: Distribution


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """A supervised example with an optional realized outcome."""

    case: ForecastCase
    target: ForecastPrediction
    target_information_cutoff: datetime
    target_method: str
    realized_outcome: str | None = None

    def __post_init__(self) -> None:
        if self.target.distribution.outcomes != self.case.question.outcomes:
            raise ForecastValidationError("target outcomes must match question outcomes")
        if self.realized_outcome is not None:
            self.case.prior.probability_for(self.realized_outcome)
        _require_aware(self.target_information_cutoff, "target_information_cutoff")
        if self.target_information_cutoff > self.case.question.forecast_at:
            raise ForecastValidationError(
                "target information cannot be newer than the forecast cutoff"
            )
        latest_input_time = max(
            (card.available_at for card in self.case.evidence),
            default=self.case.prior_as_of,
        )
        latest_input_time = max(latest_input_time, self.case.prior_as_of)
        if self.target_information_cutoff < latest_input_time:
            raise ForecastValidationError(
                "target information cutoff cannot predate its prior or evidence"
            )
        _require_text(self.target_method, "target_method")


@dataclass(frozen=True, slots=True)
class ResolvedForecast:
    """A probability distribution paired with its realized outcome."""

    question_id: str
    forecast_at: datetime
    distribution: Distribution
    realized_outcome: str

    def __post_init__(self) -> None:
        _require_text(self.question_id, "question_id")
        _require_aware(self.forecast_at, "forecast_at")
        self.distribution.probability_for(self.realized_outcome)
