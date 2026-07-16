"""Tests for outcome-v2 multi-season evaluation metrics."""

from dataclasses import replace
from datetime import date
from math import log

import pytest

from forecastfm.outcome_v2_metrics import (
    BOOTSTRAP_BLOCK_DAYS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    ONE_SIDED_ALPHA,
    DatedBinaryForecast,
    OutcomeV2MetricsError,
    evaluate_multi_season,
)


def _row(
    question_id: str,
    season: int,
    game_date: date,
    model_probability: float,
    *,
    team_won: bool = True,
) -> DatedBinaryForecast:
    return DatedBinaryForecast(
        question_id=question_id,
        season=season,
        game_date=game_date,
        realized_team_win=team_won,
        model_team_probability=model_probability,
        elo_team_probability=0.6,
    )


def test_bootstrap_and_proper_scores_are_deterministic() -> None:
    rows = (
        _row("a", 2025, date(2024, 10, 21), 0.8),
        _row("b", 2025, date(2024, 10, 22), 0.7),
        _row("c", 2025, date(2024, 10, 29), 0.75),
        _row("d", 2025, date(2024, 11, 5), 0.85),
    )

    first = evaluate_multi_season(rows, (2025,))
    second = evaluate_multi_season(rows, (2025,))

    assert first == second
    assert first.bootstrap_block_days == BOOTSTRAP_BLOCK_DAYS
    assert first.bootstrap_resamples == BOOTSTRAP_RESAMPLES
    assert first.bootstrap_seed == BOOTSTRAP_SEED
    assert first.one_sided_alpha == ONE_SIDED_ALPHA
    season = first.seasons[0]
    assert season.calendar_block_count == 3
    assert season.model.mean_log_loss == pytest.approx(
        sum(-log(row.model_team_probability) for row in rows) / len(rows)
    )
    assert season.model.mean_brier == pytest.approx(
        sum((1.0 - row.model_team_probability) ** 2 for row in rows) / len(rows)
    )
    assert season.elo.mean_log_loss == pytest.approx(-log(0.6))
    assert season.elo.mean_brier == pytest.approx(0.16)


def test_pooled_win_does_not_hide_a_losing_season() -> None:
    rows = (
        _row("losing", 2025, date(2024, 10, 21), 0.4),
        _row("win-1", 2026, date(2025, 10, 20), 0.9),
        _row("win-2", 2026, date(2025, 10, 27), 0.9),
        _row("win-3", 2026, date(2025, 11, 3), 0.9),
    )

    report = evaluate_multi_season(rows, (2025, 2026))

    assert report.pooled_elo_relative_log_score > 0.0
    assert report.seasons[0].passes is False
    assert report.seasons[1].passes is True
    assert report.passes is False


def test_opponent_win_scores_use_complementary_probabilities() -> None:
    row = _row(
        "opponent-win",
        2025,
        date(2024, 10, 21),
        0.2,
        team_won=False,
    )

    season = evaluate_multi_season((row,), (2025,)).seasons[0]

    assert season.model.mean_log_loss == pytest.approx(-log(0.8))
    assert season.model.mean_brier == pytest.approx(0.04)
    assert season.elo.mean_log_loss == pytest.approx(-log(0.4))
    assert season.elo.mean_brier == pytest.approx(0.36)


def test_conjunction_passes_when_every_declared_season_wins() -> None:
    rows = (
        _row("2025-a", 2025, date(2024, 10, 21), 0.8),
        _row("2025-b", 2025, date(2024, 10, 28), 0.8),
        _row("2026-a", 2026, date(2025, 10, 20), 0.75),
        _row("2026-b", 2026, date(2025, 10, 27), 0.75),
    )

    report = evaluate_multi_season(rows, (2025, 2026))

    assert all(season.mean_elo_relative_log_score > 0.0 for season in report.seasons)
    assert all(season.lower_one_sided_95 > 0.0 for season in report.seasons)
    assert report.passes is True


def test_duplicate_and_missing_season_coverage_are_rejected() -> None:
    row = _row("duplicate", 2025, date(2024, 10, 21), 0.8)

    with pytest.raises(OutcomeV2MetricsError, match="duplicate"):
        evaluate_multi_season((row, row), (2025,))
    with pytest.raises(OutcomeV2MetricsError, match="missing declared season"):
        evaluate_multi_season((row,), (2025, 2026))


def test_invalid_probability_and_mismatched_season_are_rejected() -> None:
    row = _row("valid", 2025, date(2024, 10, 21), 0.8)
    extra = _row("extra", 2026, date(2025, 10, 20), 0.8)

    with pytest.raises(OutcomeV2MetricsError, match="between zero and one"):
        replace(row, model_team_probability=1.1)
    with pytest.raises(OutcomeV2MetricsError, match="does not match"):
        replace(row, season=2024)
    with pytest.raises(OutcomeV2MetricsError, match="undeclared"):
        evaluate_multi_season((row, extra), (2025,))
