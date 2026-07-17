"""Tests for deterministic open-modern validation models."""

from datetime import date
from math import log, sqrt

import pytest

from forecastfm.open_modern_model import (
    FORECAST_SPEC,
    MAXIMUM_SIDE_SWAP_GAP,
    OPEN_MODERN_MODEL_CONTRACT_SHA256,
    RECALIBRATION_SPEC,
    OpenModernModelError,
    OpenModernResolvedRow,
    fit_open_modern_validation,
    fit_train_rms_scaler,
)


def _row(season: int, index: int, outcome: int) -> OpenModernResolvedRow:
    probability = 0.4 + 0.02 * index
    source_log_odds = log(probability / (1.0 - probability))
    return OpenModernResolvedRow(
        question_id=f"game-{season}-{index}",
        season=season,
        game_date=date(season, 1, index + 1),
        source_probability=probability,
        features=(
            source_log_odds,
            float(index + 1),
            float(index % 2 * 2 - 1),
            float(index + 2),
            float(index - 2) / 10.0,
            float(index + 3),
            float(index - 1),
        ),
        outcome=outcome,
    )


def _experiment_rows() -> tuple[OpenModernResolvedRow, ...]:
    rows: list[OpenModernResolvedRow] = []
    for season_index, season in enumerate(range(2016, 2021)):
        rows.append(_row(season, season_index + 1, season_index % 2))
        rows.append(_row(season, season_index + 2, (season_index + 1) % 2))
    return tuple(rows)


def test_predeclared_models_and_contract_are_frozen() -> None:
    assert RECALIBRATION_SPEC.candidate_id == "recalibration-l2-0"
    assert FORECAST_SPEC.candidate_id == "full-l2-0.01"
    assert OPEN_MODERN_MODEL_CONTRACT_SHA256 == (
        "f6a363ecf1d0b16a425eea7c55ae026153f514a6584ed18be8a7131b737a528d"
    )


def test_rms_scaler_uses_training_rows_without_centering() -> None:
    training = tuple(row for row in _experiment_rows() if row.season < 2020)

    scaler = fit_train_rms_scaler(training)

    expected_first = sqrt(sum(row.features[0] ** 2 for row in training) / len(training))
    assert scaler.scales[0] == pytest.approx(expected_first)
    transformed = scaler.transform(training[0].features)
    assert transformed[0] == pytest.approx(training[0].features[0] / expected_first)


def test_rms_scaler_rejects_validation_rows() -> None:
    with pytest.raises(OpenModernModelError, match="training seasons only"):
        fit_train_rms_scaler((_row(2020, 1, 1),))


def test_full_validation_fit_is_deterministic_and_symmetric() -> None:
    rows = _experiment_rows()

    first = fit_open_modern_validation(rows)
    second = fit_open_modern_validation(tuple(reversed(rows)))

    assert first == second
    assert first.raw_source_metrics.count == 2
    assert first.recalibration.metrics.count == 2
    assert first.forecast.spec == FORECAST_SPEC
    assert first.forecast.metrics.count == 2
    assert first.maximum_side_swap_gap <= MAXIMUM_SIDE_SWAP_GAP
    assert first.forecast_vs_raw_source.game_count == 2
    assert first.forecast_vs_recalibration.game_count == 2


def test_resolved_row_requires_actual_binary_winner_and_matching_log_odds() -> None:
    row = _row(2020, 1, 1)

    with pytest.raises(OpenModernModelError, match="outcome must be zero or one"):
        OpenModernResolvedRow(
            question_id=row.question_id,
            season=row.season,
            game_date=row.game_date,
            source_probability=row.source_probability,
            features=row.features,
            outcome=2,
        )
    with pytest.raises(OpenModernModelError, match="log odds do not match"):
        OpenModernResolvedRow(
            question_id=row.question_id,
            season=row.season,
            game_date=row.game_date,
            source_probability=row.source_probability,
            features=(0.0, *row.features[1:]),
            outcome=row.outcome,
        )


def test_2020_bubble_date_is_valid_for_source_season() -> None:
    probability = 0.6
    row = OpenModernResolvedRow(
        question_id="bubble-game",
        season=2020,
        game_date=date(2020, 10, 1),
        source_probability=probability,
        features=(log(probability / (1.0 - probability)), 1.0, 1.0, 1.0, 0.1, 1.0, 1.0),
        outcome=1,
    )

    assert row.season == 2020
