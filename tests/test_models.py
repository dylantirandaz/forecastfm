"""Tests for forecast schema validation."""

from datetime import UTC, datetime, timedelta

import pytest

from forecastfm.models import (
    Distribution,
    EvidenceCard,
    ForecastCase,
    ForecastQuestion,
    ForecastValidationError,
)
from tests.helpers import make_case


def test_distribution_preserves_outcome_order() -> None:
    distribution = Distribution(outcomes=("yes", "no"), probabilities=(0.7, 0.3))

    assert distribution.as_dict() == {"yes": 0.7, "no": 0.3}
    assert distribution.probability_for("no") == 0.3


def test_distribution_rejects_invalid_total() -> None:
    with pytest.raises(ForecastValidationError, match="sum to one"):
        Distribution(outcomes=("yes", "no"), probabilities=(0.7, 0.4))


def test_question_rejects_naive_timestamp() -> None:
    with pytest.raises(ForecastValidationError, match="timezone-aware"):
        ForecastQuestion(
            question_id="question-1",
            text="Will the event occur?",
            resolution_rule="Resolve yes if the event occurs.",
            resolution_source="test-source",
            outcomes=("yes", "no"),
            forecast_at=datetime(2026, 1, 1),  # noqa: DTZ001 - Intentional invalid input.
            resolves_at=datetime(2026, 1, 2, tzinfo=UTC),
        )


def test_case_rejects_future_evidence() -> None:
    case = make_case()
    future_evidence = EvidenceCard(
        text="This fact arrived too late.",
        source="test",
        available_at=case.question.forecast_at + timedelta(seconds=1),
    )

    with pytest.raises(ForecastValidationError, match="newer than"):
        ForecastCase(
            question=case.question,
            prior=case.prior,
            prior_source=case.prior_source,
            prior_as_of=case.prior_as_of,
            evidence=(future_evidence,),
        )


def test_case_rejects_future_prior() -> None:
    case = make_case()

    with pytest.raises(ForecastValidationError, match="prior cannot be newer"):
        ForecastCase(
            question=case.question,
            prior=case.prior,
            prior_source=case.prior_source,
            prior_as_of=case.question.forecast_at + timedelta(seconds=1),
        )


def test_case_rejects_evidence_out_of_order() -> None:
    forecast_at = datetime(2026, 1, 2, tzinfo=UTC)
    newer = EvidenceCard(text="Newer", source="test", available_at=forecast_at)
    older = EvidenceCard(
        text="Older",
        source="test",
        available_at=forecast_at - timedelta(days=1),
    )

    with pytest.raises(ForecastValidationError, match="ordered"):
        make_case(evidence=(newer, older))
