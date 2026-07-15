"""Small test-data factories."""

from datetime import UTC, datetime, timedelta

from forecastfm.models import (
    Distribution,
    EvidenceCard,
    ForecastCase,
    ForecastPrediction,
    ForecastQuestion,
    TrainingExample,
)


def make_case(evidence: tuple[EvidenceCard, ...] = ()) -> ForecastCase:
    """Create a valid binary forecast case."""
    forecast_at = datetime(2026, 1, 2, tzinfo=UTC)
    return ForecastCase(
        question=ForecastQuestion(
            question_id="question-1",
            text="Will the event occur?",
            resolution_rule="Resolve yes if the event occurs by the deadline.",
            resolution_source="test-source",
            outcomes=("yes", "no"),
            forecast_at=forecast_at,
            resolves_at=forecast_at + timedelta(days=1),
        ),
        prior=Distribution(outcomes=("yes", "no"), probabilities=(0.4, 0.6)),
        prior_source="test-prior",
        prior_as_of=forecast_at,
        evidence=evidence,
    )


def make_training_example() -> TrainingExample:
    """Create a valid binary training example."""
    case = make_case()
    return TrainingExample(
        case=case,
        target=ForecastPrediction(
            distribution=Distribution(
                outcomes=case.question.outcomes,
                probabilities=(0.7, 0.3),
            )
        ),
        target_information_cutoff=case.question.forecast_at,
        target_method="test fixture",
        realized_outcome="yes",
    )


def make_nba_training_example(
    realized_outcome: str | None = "team_wins",
) -> TrainingExample:
    """Create a canonical anonymous NBA outcome-training example."""
    forecast_at = datetime(2007, 1, 2, tzinfo=UTC)
    outcomes = ("team_wins", "opponent_wins")
    case = ForecastCase(
        question=ForecastQuestion(
            question_id="nba-example",
            text="Will the listed team defeat its opponent in this NBA game?",
            resolution_rule="Resolve to the team with the higher final score.",
            resolution_source="source",
            outcomes=outcomes,
            forecast_at=forecast_at,
            resolves_at=forecast_at + timedelta(days=2),
        ),
        prior=Distribution(outcomes=outcomes, probabilities=(0.4, 0.6)),
        prior_source="pregame Elo",
        prior_as_of=forecast_at,
        evidence=(
            EvidenceCard(
                text="Venue for the listed team: home.",
                source="source",
                available_at=forecast_at,
            ),
        ),
    )
    return TrainingExample(
        case=case,
        target=ForecastPrediction(
            distribution=Distribution(outcomes=outcomes, probabilities=(0.2, 0.8))
        ),
        target_information_cutoff=forecast_at,
        target_method="pregame Elo teacher",
        realized_outcome=realized_outcome,
    )
