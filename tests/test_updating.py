"""Tests for Bayesian probability updates."""

from math import inf, log

import pytest

from forecastfm.models import Distribution, ForecastValidationError
from forecastfm.updating import update_binary, update_distribution


def test_binary_update_matches_bayes_rule() -> None:
    prior = Distribution(outcomes=("yes", "no"), probabilities=(0.5, 0.5))

    posterior = update_binary(prior, "yes", log(0.75 / 0.25))

    assert posterior.probability_for("yes") == pytest.approx(0.75)


def test_zero_probability_remains_zero() -> None:
    prior = Distribution(outcomes=("yes", "no"), probabilities=(0.0, 1.0))

    posterior = update_distribution(prior, (10.0, 0.0))

    assert posterior.probabilities == (0.0, 1.0)


def test_negative_infinity_eliminates_an_outcome() -> None:
    prior = Distribution(outcomes=("yes", "no"), probabilities=(0.5, 0.5))

    posterior = update_distribution(prior, (-inf, 0.0))

    assert posterior.probabilities == (0.0, 1.0)


def test_positive_infinity_selects_the_positive_outcome() -> None:
    prior = Distribution(outcomes=("yes", "no"), probabilities=(0.5, 0.5))

    posterior = update_binary(prior, "yes", inf)

    assert posterior.probabilities == (1.0, 0.0)


def test_evidence_cannot_eliminate_every_outcome() -> None:
    prior = Distribution(outcomes=("yes", "no"), probabilities=(0.5, 0.5))

    with pytest.raises(ForecastValidationError, match="every possible outcome"):
        update_distribution(prior, (-inf, -inf))
