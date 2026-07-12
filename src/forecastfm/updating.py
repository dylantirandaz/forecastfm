"""Small, explicit probability-update functions."""

from collections.abc import Sequence
from math import exp, inf, isnan, log

from forecastfm.models import Distribution, ForecastValidationError


def update_distribution(
    prior: Distribution,
    log_likelihoods: Sequence[float],
) -> Distribution:
    """Apply one log-likelihood value per outcome and normalize the result."""
    if len(log_likelihoods) != len(prior.outcomes):
        raise ForecastValidationError("each outcome must have one log-likelihood")
    if any(isnan(value) or value == inf for value in log_likelihoods):
        raise ForecastValidationError("log-likelihoods cannot contain NaN or positive infinity")

    log_weights = tuple(
        float("-inf") if probability == 0.0 else log(probability) + likelihood
        for probability, likelihood in zip(
            prior.probabilities,
            log_likelihoods,
            strict=True,
        )
    )
    largest_weight = max(log_weights)
    if largest_weight == -inf:
        raise ForecastValidationError("evidence cannot eliminate every possible outcome")
    weights = tuple(exp(value - largest_weight) for value in log_weights)
    total_weight = sum(weights)
    probabilities = tuple(weight / total_weight for weight in weights)
    return Distribution(outcomes=prior.outcomes, probabilities=probabilities)


def update_binary(
    prior: Distribution,
    positive_outcome: str,
    log_likelihood_ratio: float,
) -> Distribution:
    """Update binary odds by an evidence log-likelihood ratio."""
    if len(prior.outcomes) != 2:
        raise ForecastValidationError("binary updates require exactly two outcomes")
    positive_probability = prior.probability_for(positive_outcome)
    if isnan(log_likelihood_ratio):
        raise ForecastValidationError("log-likelihood ratio cannot be NaN")
    if log_likelihood_ratio == inf:
        if positive_probability == 0.0:
            raise ForecastValidationError("infinite evidence conflicts with a zero prior")
        probabilities = tuple(
            1.0 if outcome == positive_outcome else 0.0 for outcome in prior.outcomes
        )
        return Distribution(outcomes=prior.outcomes, probabilities=probabilities)
    log_likelihoods = tuple(
        log_likelihood_ratio if outcome == positive_outcome else 0.0 for outcome in prior.outcomes
    )
    return update_distribution(prior, log_likelihoods)
