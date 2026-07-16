"""Tests for resolved NBA outcome metrics."""

from dataclasses import replace
from math import log, sqrt

import pytest

from forecastfm.models import (
    Distribution,
    ForecastPrediction,
    ForecastValidationError,
    TrainingExample,
)
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.outcome_metrics import (
    NORMAL_95_Z_SCORE,
    difficulty_subsets,
    paired_mean_delta,
    summarize_outcome_metrics,
)
from tests.helpers import make_nba_training_example


def _example(
    index: int,
    *,
    target_team_probability: float = 0.6,
    realized_outcome: str | None = "team_wins",
) -> TrainingExample:
    example = make_nba_training_example(realized_outcome)
    question = replace(example.case.question, question_id=f"nba-{index}")
    target = ForecastPrediction(
        distribution=Distribution(
            outcomes=example.case.question.outcomes,
            probabilities=(target_team_probability, 1.0 - target_team_probability),
        )
    )
    return replace(example, case=replace(example.case, question=question), target=target)


def test_outcome_metrics_report_scores_calibration_and_per_game_values() -> None:
    examples = (
        _example(0, realized_outcome="team_wins"),
        _example(1, realized_outcome="opponent_wins"),
        _example(2, realized_outcome="team_wins"),
        _example(3, realized_outcome="opponent_wins"),
    )
    probabilities = (0.9, 0.8, 0.4, 0.1)

    metrics = summarize_outcome_metrics(examples, probabilities)

    expected_losses = (-log(0.9), -log(0.2), -log(0.4), -log(0.9))
    assert metrics.count == 4
    assert metrics.mean_brier == pytest.approx(0.255)
    assert metrics.mean_log_loss == pytest.approx(sum(expected_losses) / 4)
    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.expected_calibration_error == pytest.approx(0.4)
    assert metrics.per_game_brier_scores == pytest.approx((0.01, 0.64, 0.36, 0.01))
    assert metrics.per_game_log_losses == pytest.approx(expected_losses)
    assert tuple(bin_.count for bin_ in metrics.reliability_bins) == (1, 1, 1, 1)


def test_outcome_metrics_validate_examples_and_alignment() -> None:
    resolved = _example(0)

    with pytest.raises(ForecastValidationError, match="at least one"):
        summarize_outcome_metrics((), ())
    with pytest.raises(ForecastValidationError, match="one team-win probability"):
        summarize_outcome_metrics((resolved,), ())
    with pytest.raises(ForecastValidationError, match="resolved"):
        summarize_outcome_metrics((_example(1, realized_outcome=None),), (0.5,))
    with pytest.raises(ForecastValidationError, match="between zero and one"):
        summarize_outcome_metrics((resolved,), (1.1,))


def test_outcome_metrics_reject_side_swapped_or_duplicate_games() -> None:
    original = _example(0)
    swapped_question = replace(
        original.case.question,
        question_id=f"{original.case.question.question_id}{SIDE_SWAP_SUFFIX}",
    )
    swapped = replace(original, case=replace(original.case, question=swapped_question))

    with pytest.raises(ForecastValidationError, match="original game orientations"):
        summarize_outcome_metrics((swapped,), (0.5,))
    with pytest.raises(ForecastValidationError, match="duplicate"):
        summarize_outcome_metrics((original, original), (0.5, 0.5))


def test_paired_mean_delta_uses_candidate_minus_baseline_and_sample_variance() -> None:
    result = paired_mean_delta((0.0, 0.0, 0.0), (1.0, 2.0, 3.0))
    expected_error = 1.0 / sqrt(3.0)

    assert result.count == 3
    assert result.mean_delta == pytest.approx(2.0)
    assert result.standard_error == pytest.approx(expected_error)
    assert result.lower_95 == pytest.approx(2.0 - NORMAL_95_Z_SCORE * expected_error)
    assert result.upper_95 == pytest.approx(2.0 + NORMAL_95_Z_SCORE * expected_error)


@pytest.mark.parametrize(
    ("baseline", "candidate", "message"),
    [
        ((), (), "at least two"),
        ((1.0,), (1.0,), "at least two"),
        ((1.0, 2.0), (1.0,), "equal lengths"),
        ((1.0, float("inf")), (1.0, 2.0), "finite"),
    ],
)
def test_paired_mean_delta_validates_inputs(
    baseline: tuple[float, ...],
    candidate: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ForecastValidationError, match=message):
        paired_mean_delta(baseline, candidate)


def test_difficulty_subsets_use_venue_adjusted_elo_target_confidence() -> None:
    examples = (
        _example(0, target_team_probability=0.59),
        _example(1, target_team_probability=0.60),
        _example(2, target_team_probability=0.749999),
        _example(3, target_team_probability=0.75),
        _example(4, target_team_probability=0.10),
    )

    subsets = difficulty_subsets(examples)

    assert subsets.hard == (0,)
    assert subsets.medium == (1, 2)
    assert subsets.easy == (3, 4)


def test_difficulty_subsets_require_nonempty_resolved_originals() -> None:
    with pytest.raises(ForecastValidationError, match="at least one"):
        difficulty_subsets(())
    with pytest.raises(ForecastValidationError, match="resolved"):
        difficulty_subsets((_example(0, realized_outcome=None),))
