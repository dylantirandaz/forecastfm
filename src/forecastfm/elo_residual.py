"""Dependency-free Elo-offset logistic regression."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite, log


class EloResidualError(ValueError):
    """Raised when Elo-residual data or settings are invalid."""


@dataclass(frozen=True, slots=True)
class EloResidualRow:
    """One resolved game with an Elo prior and fixed numeric features."""

    question_id: str
    elo_probability: float
    features: tuple[float, ...]
    outcome: int

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise EloResidualError("question_id must not be empty")
        _require_probability(self.elo_probability)
        if not self.features:
            raise EloResidualError("at least one feature is required")
        if not all(isfinite(value) for value in self.features):
            raise EloResidualError("features must be finite")
        if isinstance(self.outcome, bool) or self.outcome not in {0, 1}:
            raise EloResidualError("outcome must be zero or one")


@dataclass(frozen=True, slots=True)
class EloResidualFitConfig:
    """Deterministic full-batch gradient-descent settings."""

    steps: int = 1_000
    learning_rate: float = 0.1
    l2_penalty: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or self.steps <= 0:
            raise EloResidualError("steps must be a positive integer")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise EloResidualError("learning_rate must be positive and finite")
        if not isfinite(self.l2_penalty) or self.l2_penalty < 0.0:
            raise EloResidualError("l2_penalty must be non-negative and finite")


DEFAULT_FIT_CONFIG = EloResidualFitConfig()


@dataclass(frozen=True, slots=True)
class EloResidualModel:
    """An immutable logistic model that adds feature evidence to Elo log-odds."""

    feature_names: tuple[str, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_feature_names(self.feature_names)
        if len(self.weights) != len(self.feature_names):
            raise EloResidualError("each feature name must have one weight")
        if not all(isfinite(weight) for weight in self.weights):
            raise EloResidualError("weights must be finite")

    def predict_probability(
        self,
        elo_probability: float,
        features: tuple[float, ...],
    ) -> float:
        """Predict a team win from its Elo prior and aligned feature values."""
        _require_probability(elo_probability)
        if len(features) != len(self.weights):
            raise EloResidualError("prediction feature count differs from the model")
        if not all(isfinite(value) for value in features):
            raise EloResidualError("prediction features must be finite")
        residual = sum(weight * value for weight, value in zip(self.weights, features, strict=True))
        return probability_from_logit(probability_logit(elo_probability) + residual)

    def predict_row(self, row: EloResidualRow) -> float:
        """Predict one resolved row without using its outcome."""
        return self.predict_probability(row.elo_probability, row.features)


def fit_elo_residual(
    rows: Sequence[EloResidualRow],
    feature_names: tuple[str, ...],
    config: EloResidualFitConfig = DEFAULT_FIT_CONFIG,
) -> EloResidualModel:
    """Fit an Elo-offset logistic model with mean NLL and L2 regularization."""
    _require_training_rows(rows, feature_names)
    weights = [0.0] * len(feature_names)
    row_count = len(rows)
    row_scale = 1.0 / row_count
    prepared_rows = tuple(
        (probability_logit(row.elo_probability), row.features, row.outcome) for row in rows
    )

    for _ in range(config.steps):
        gradients = [config.l2_penalty * weight for weight in weights]
        for prior_logit, features, outcome in prepared_rows:
            residual = sum(weight * value for weight, value in zip(weights, features, strict=True))
            prediction = probability_from_logit(prior_logit + residual)
            scaled_error = (prediction - outcome) * row_scale
            for index, value in enumerate(features):
                gradients[index] += scaled_error * value

        weights = [
            weight - config.learning_rate * gradient
            for weight, gradient in zip(weights, gradients, strict=True)
        ]
        if not all(isfinite(weight) for weight in weights):
            raise EloResidualError("training produced non-finite weights")

    return EloResidualModel(feature_names=feature_names, weights=tuple(weights))


def probability_logit(probability: float) -> float:
    """Convert a strict probability to log-odds."""
    _require_probability(probability)
    return log(probability) - log(1.0 - probability)


def probability_from_logit(value: float) -> float:
    """Convert finite log-odds to a probability without overflow."""
    if not isfinite(value):
        raise EloResidualError("logit must be finite")
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    odds = exp(value)
    return odds / (1.0 + odds)


def _require_training_rows(
    rows: Sequence[EloResidualRow],
    feature_names: tuple[str, ...],
) -> None:
    _require_feature_names(feature_names)
    if not rows:
        raise EloResidualError("at least one training row is required")
    question_ids = {row.question_id for row in rows}
    if len(question_ids) != len(rows):
        raise EloResidualError("training question IDs must be unique")
    if any(len(row.features) != len(feature_names) for row in rows):
        raise EloResidualError("training feature count differs from feature names")


def _require_feature_names(feature_names: tuple[str, ...]) -> None:
    if not feature_names:
        raise EloResidualError("at least one feature name is required")
    if any(not name.strip() or name != name.strip() for name in feature_names):
        raise EloResidualError("feature names must be non-empty and trimmed")
    if len(set(feature_names)) != len(feature_names):
        raise EloResidualError("feature names must be unique")


def _require_probability(probability: float) -> None:
    if not isfinite(probability) or not 0.0 < probability < 1.0:
        raise EloResidualError("Elo probability must be finite and strictly between zero and one")
