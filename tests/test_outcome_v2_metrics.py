"""Tests for outcome-v2 frozen-cohort evaluation metrics."""

from dataclasses import replace
from datetime import date
from math import log

import pytest

from forecastfm.outcome_v2_metrics import (
    BOOTSTRAP_BLOCK_DAYS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FAILURE_REALIZED_PROBABILITY,
    ONE_SIDED_ALPHA,
    BinaryForecast,
    DatedBinaryCohortMember,
    MultiSeasonEvaluation,
    OutcomeV2MetricsError,
    evaluate_multi_season,
    failure_team_probability,
)


def _member(
    question_id: str,
    season: int,
    game_date: date,
    *,
    team_won: bool = True,
) -> DatedBinaryCohortMember:
    return DatedBinaryCohortMember(
        question_id=question_id,
        season=season,
        game_date=game_date,
        realized_team_win=team_won,
        baseline_team_probability=0.6,
    )


def _evaluate(
    rows: tuple[tuple[DatedBinaryCohortMember, float], ...],
    seasons: tuple[int, ...],
) -> MultiSeasonEvaluation:
    cohort = tuple(member for member, _ in rows)
    forecasts = tuple(
        BinaryForecast(member.question_id, probability) for member, probability in rows
    )
    return evaluate_multi_season(forecasts, cohort, seasons)


def test_bootstrap_and_proper_scores_are_deterministic() -> None:
    rows = (
        (_member("a", 2025, date(2024, 10, 21)), 0.8),
        (_member("b", 2025, date(2024, 10, 22)), 0.7),
        (_member("c", 2025, date(2024, 10, 29)), 0.75),
        (_member("d", 2025, date(2024, 11, 5)), 0.85),
    )

    first = _evaluate(rows, (2025,))
    second = _evaluate(rows, (2025,))

    assert first == second
    assert first.bootstrap_block_days == BOOTSTRAP_BLOCK_DAYS
    assert first.bootstrap_resamples == BOOTSTRAP_RESAMPLES
    assert first.bootstrap_seed == BOOTSTRAP_SEED
    assert first.one_sided_alpha == ONE_SIDED_ALPHA
    season = first.seasons[0]
    assert season.calendar_block_count == 3
    assert season.model.mean_log_loss == pytest.approx(
        sum(-log(probability) for _, probability in rows) / len(rows)
    )
    assert season.model.mean_brier == pytest.approx(
        sum((1.0 - probability) ** 2 for _, probability in rows) / len(rows)
    )
    assert season.baseline.mean_log_loss == pytest.approx(-log(0.6))
    assert season.baseline.mean_brier == pytest.approx(0.16)


def test_pooled_win_does_not_hide_a_losing_season() -> None:
    rows = (
        (_member("losing", 2025, date(2024, 10, 21)), 0.4),
        (_member("win-1", 2026, date(2025, 10, 20)), 0.9),
        (_member("win-2", 2026, date(2025, 10, 27)), 0.9),
        (_member("win-3", 2026, date(2025, 11, 3)), 0.9),
    )

    report = _evaluate(rows, (2025, 2026))

    assert report.pooled_baseline_relative_log_score > 0.0
    assert report.seasons[0].passes is False
    assert report.seasons[1].passes is True
    assert report.passes is False


def test_opponent_win_scores_use_complementary_probabilities() -> None:
    member = _member(
        "opponent-win",
        2025,
        date(2024, 10, 21),
        team_won=False,
    )

    season = _evaluate(((member, 0.2),), (2025,)).seasons[0]

    assert season.model.mean_log_loss == pytest.approx(-log(0.8))
    assert season.model.mean_brier == pytest.approx(0.04)
    assert season.baseline.mean_log_loss == pytest.approx(-log(0.4))
    assert season.baseline.mean_brier == pytest.approx(0.36)


def test_conjunction_passes_when_every_declared_season_wins() -> None:
    rows = (
        (_member("2025-a", 2025, date(2024, 10, 21)), 0.8),
        (_member("2025-b", 2025, date(2024, 10, 28)), 0.8),
        (_member("2026-a", 2026, date(2025, 10, 20)), 0.75),
        (_member("2026-b", 2026, date(2025, 10, 27)), 0.75),
    )

    report = _evaluate(rows, (2025, 2026))

    assert all(season.mean_baseline_relative_log_score > 0.0 for season in report.seasons)
    assert all(season.lower_one_sided_95 > 0.0 for season in report.seasons)
    assert report.passes is True


def test_duplicate_and_extra_rows_are_rejected_while_missing_rows_are_penalized() -> None:
    first = _member("first", 2025, date(2024, 10, 21))
    second = _member("second", 2025, date(2024, 10, 22))
    forecast = BinaryForecast("first", 0.8)

    with pytest.raises(OutcomeV2MetricsError, match="cohort question IDs"):
        evaluate_multi_season((forecast,), (first, first), (2025,))
    report = evaluate_multi_season((forecast,), (first, second), (2025,))
    assert report.game_count == 2
    assert report.seasons[0].model.mean_log_loss == pytest.approx(
        (-log(0.8) - log(FAILURE_REALIZED_PROBABILITY)) / 2.0
    )
    with pytest.raises(OutcomeV2MetricsError, match="outside the frozen cohort"):
        evaluate_multi_season(
            (forecast, BinaryForecast("extra", 0.8)),
            (first,),
            (2025,),
        )


def test_frozen_metadata_cannot_be_rewritten_by_a_prediction() -> None:
    member = _member("frozen", 2025, date(2024, 10, 21))
    forecast = BinaryForecast("frozen", 0.8)

    report = evaluate_multi_season((forecast,), (member,), (2025,))

    assert report.seasons[0].baseline.mean_log_loss == pytest.approx(-log(0.6))
    with pytest.raises(OutcomeV2MetricsError, match="does not match"):
        replace(member, season=2024)


def test_invalid_probabilities_are_rejected_without_silent_clipping() -> None:
    with pytest.raises(OutcomeV2MetricsError, match="strictly between"):
        BinaryForecast("zero", 0.0)
    with pytest.raises(OutcomeV2MetricsError, match="strictly between"):
        BinaryForecast("one", 1.0)
    with pytest.raises(OutcomeV2MetricsError, match="strictly between"):
        DatedBinaryCohortMember("baseline", 2025, date(2024, 10, 21), True, 0.0)
    with pytest.raises(OutcomeV2MetricsError, match="requires a reason"):
        BinaryForecast("failed", None)


def test_failed_forecast_penalty_is_explicit_and_interior() -> None:
    assert failure_team_probability(True) == FAILURE_REALIZED_PROBABILITY
    assert failure_team_probability(False) == 1.0 - FAILURE_REALIZED_PROBABILITY
    assert 0.0 < failure_team_probability(True) < 1.0
    assert 0.0 < failure_team_probability(False) < 1.0

    member = _member("malformed", 2025, date(2024, 10, 21))
    failed = BinaryForecast("malformed", None, "malformed model output")
    report = evaluate_multi_season((failed,), (member,), (2025,))
    assert report.seasons[0].model.mean_log_loss == pytest.approx(
        -log(FAILURE_REALIZED_PROBABILITY)
    )


def test_declared_season_coverage_is_exact() -> None:
    row = (_member("valid", 2025, date(2024, 10, 21)), 0.8)
    extra = (_member("extra", 2026, date(2025, 10, 20)), 0.8)

    with pytest.raises(OutcomeV2MetricsError, match="missing a declared season"):
        _evaluate((row,), (2025, 2026))
    with pytest.raises(OutcomeV2MetricsError, match="undeclared season"):
        _evaluate((row, extra), (2025,))
