"""Tests for the dependency-free Elo-residual baseline."""

from math import log

import pytest

from forecastfm.elo_residual import (
    EloResidualError,
    EloResidualFitConfig,
    EloResidualModel,
    EloResidualRow,
    fit_elo_residual,
)


def _signal_rows() -> tuple[EloResidualRow, ...]:
    rows: list[EloResidualRow] = []
    for index in range(40):
        feature = 1.0 if index % 2 == 0 else -1.0
        rows.append(
            EloResidualRow(
                question_id=f"game-{index}",
                elo_probability=0.5,
                features=(feature,),
                outcome=int(feature > 0.0),
            )
        )
    return tuple(rows)


def _mean_log_loss(model: EloResidualModel, rows: tuple[EloResidualRow, ...]) -> float:
    losses: list[float] = []
    for row in rows:
        probability = model.predict_row(row)
        realized_probability = probability if row.outcome == 1 else 1.0 - probability
        losses.append(-log(realized_probability))
    return sum(losses) / len(losses)


def test_fit_learns_a_residual_signal_and_improves_log_loss() -> None:
    rows = _signal_rows()
    baseline = EloResidualModel(feature_names=("signal",), weights=(0.0,))

    model = fit_elo_residual(
        rows,
        ("signal",),
        EloResidualFitConfig(steps=300, learning_rate=0.2, l2_penalty=0.01),
    )

    assert model.weights[0] > 0.0
    assert _mean_log_loss(model, rows) < _mean_log_loss(baseline, rows)


def test_side_swap_predicts_the_exact_complement() -> None:
    model = EloResidualModel(
        feature_names=("rest_delta", "travel_delta"),
        weights=(0.4, -0.2),
    )

    original = model.predict_probability(0.7, (2.0, -3.0))
    swapped = model.predict_probability(0.3, (-2.0, 3.0))

    assert swapped == pytest.approx(1.0 - original)


def test_training_is_deterministic() -> None:
    rows = _signal_rows()
    config = EloResidualFitConfig(steps=20, learning_rate=0.1, l2_penalty=0.1)

    first = fit_elo_residual(rows, ("signal",), config)
    second = fit_elo_residual(rows, ("signal",), config)

    assert first == second


def test_rows_reject_malformed_values() -> None:
    with pytest.raises(EloResidualError, match="question_id"):
        EloResidualRow(" ", 0.5, (1.0,), 1)
    with pytest.raises(EloResidualError, match="strictly between"):
        EloResidualRow("game", 1.0, (1.0,), 1)
    with pytest.raises(EloResidualError, match="features must be finite"):
        EloResidualRow("game", 0.5, (float("nan"),), 1)
    with pytest.raises(EloResidualError, match="zero or one"):
        EloResidualRow("game", 0.5, (1.0,), 2)
    with pytest.raises(EloResidualError, match="zero or one"):
        EloResidualRow("game", 0.5, (1.0,), True)


def test_fit_rejects_empty_duplicate_or_misaligned_data() -> None:
    with pytest.raises(EloResidualError, match="at least one training row"):
        fit_elo_residual((), ("signal",))

    row = EloResidualRow("game", 0.5, (1.0,), 1)
    with pytest.raises(EloResidualError, match="unique"):
        fit_elo_residual((row, row), ("signal",))
    with pytest.raises(EloResidualError, match="feature count"):
        fit_elo_residual((row,), ("first", "second"))


def test_model_and_config_reject_malformed_settings() -> None:
    with pytest.raises(EloResidualError, match="feature names must be unique"):
        EloResidualModel(("same", "same"), (0.0, 0.0))
    with pytest.raises(EloResidualError, match="one weight"):
        EloResidualModel(("first", "second"), (0.0,))
    with pytest.raises(EloResidualError, match="steps"):
        EloResidualFitConfig(steps=0)
    with pytest.raises(EloResidualError, match="l2_penalty"):
        EloResidualFitConfig(l2_penalty=-0.1)
