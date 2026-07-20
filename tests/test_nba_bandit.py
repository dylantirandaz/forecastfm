"""Tests for the dependency-free NBA contextual bandit mixture selector."""

from math import log

import pytest

from forecastfm.nba_bandit import (
    BANDIT_CONTEXT_NAMES,
    BanditFitConfig,
    BanditGame,
    BanditSelector,
    evaluate_bandit,
    fit_bandit_selector,
    selection_weight_distribution,
    unavailable_minutes_bucket,
)

ARM_NAMES = ("strong", "weak")


def _game(
    question_id: str,
    context: tuple[float, ...],
    arm_probabilities: tuple[float, ...],
    outcome: int,
) -> BanditGame:
    return BanditGame(
        question_id=question_id,
        season=2025,
        context=context,
        arm_probabilities=arm_probabilities,
        outcome=outcome,
    )


def _context_signal_games() -> tuple[BanditGame, ...]:
    """Arm 0 is right when the signal is 0; arm 1 is right when the signal is 1."""
    games: list[BanditGame] = []
    for index in range(60):
        signal = float(index % 2)
        context = (1.0, signal, 0.5, 0.5, 0.5, 0.5)
        outcome = int(signal == 0.0)
        games.append(_game(f"game-{index}", context, (0.8, 0.2), outcome))
    return tuple(games)


def _mean_mixture_log_loss(selector: BanditSelector, games: tuple[BanditGame, ...]) -> float:
    losses = []
    for game in games:
        probability = selector.forecast(game.context, game.arm_probabilities)
        realized = probability if game.outcome == 1 else 1.0 - probability
        losses.append(-log(realized))
    return sum(losses) / len(games)


def test_zero_theta_recovers_the_uniform_mixture() -> None:
    selector = BanditSelector(
        context_names=BANDIT_CONTEXT_NAMES,
        arm_names=("a", "b", "c", "d"),
        scales=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        theta=((0.0,) * 4,) * 6,
    )

    weights = selector.mixture_weights((1.0, 0.7, -3.0, 900.0, 12.0, -0.4))

    assert weights == pytest.approx((0.25, 0.25, 0.25, 0.25))


def test_softmax_is_stable_for_extreme_theta() -> None:
    selector = BanditSelector(
        context_names=BANDIT_CONTEXT_NAMES,
        arm_names=ARM_NAMES,
        scales=(1.0,) * 6,
        theta=((1e6, -1e6),) * 6,
    )

    weights = selector.mixture_weights((1.0, 1.0, 1.0, 1.0, 1.0, 1.0))

    assert weights == pytest.approx((1.0, 0.0))
    assert sum(weights) == pytest.approx(1.0)


def test_gradient_descent_improves_training_log_loss_with_context_signal() -> None:
    games = _context_signal_games()
    uniform = BanditSelector(
        context_names=BANDIT_CONTEXT_NAMES,
        arm_names=ARM_NAMES,
        scales=(1.0,) * 6,
        theta=((0.0, 0.0),) * 6,
    )

    selector = fit_bandit_selector(
        games, ARM_NAMES, BanditFitConfig(steps=400, learning_rate=0.2, l2_penalty=0.001)
    )

    assert _mean_mixture_log_loss(selector, games) < _mean_mixture_log_loss(uniform, games)
    signal_row = selector.theta[1]
    assert signal_row[0] < 0.0 < signal_row[1]


def test_oracle_matches_or_beats_every_arm() -> None:
    games = (
        _game("g1", (1.0, 0.1, 0.0, 0.0, 0.0, 0.0), (0.9, 0.4), 1),
        _game("g2", (1.0, 0.2, 0.0, 0.0, 0.0, 0.0), (0.6, 0.8), 1),
        _game("g3", (1.0, 0.3, 0.0, 0.0, 0.0, 0.0), (0.55, 0.45), 0),
    )
    selector = fit_bandit_selector(
        games, ARM_NAMES, BanditFitConfig(steps=10, learning_rate=0.05, l2_penalty=0.0)
    )

    evaluation = evaluate_bandit(selector, games)

    assert evaluation.oracle_log_loss <= min(loss for _, loss in evaluation.arm_log_losses)


def test_refit_is_deterministic() -> None:
    games = _context_signal_games()
    config = BanditFitConfig(steps=200, learning_rate=0.1, l2_penalty=0.01)

    first = fit_bandit_selector(games, ARM_NAMES, config)
    second = fit_bandit_selector(games, ARM_NAMES, config)

    assert first == second


def test_selection_weight_distribution_buckets_games() -> None:
    games = (
        _game("g1", (1.0, 0.1, 0.0, 0.0, 0.0, 0.0), (0.6, 0.4), 1),
        _game("g2", (1.0, 0.1, 0.0, 0.0, 10.0, 0.0), (0.6, 0.4), 1),
        _game("g3", (1.0, 0.1, 0.0, 0.0, 30.0, 0.0), (0.6, 0.4), 1),
    )
    selector = fit_bandit_selector(
        games, ARM_NAMES, BanditFitConfig(steps=10, learning_rate=0.05, l2_penalty=0.0)
    )

    buckets = selection_weight_distribution(
        selector, games, lambda game: unavailable_minutes_bucket(game.context[4])
    )

    assert [bucket.bucket for bucket in buckets] == ["(0,15]", "0", ">15"]  # sorted labels
    for bucket in buckets:
        assert bucket.game_count == 1
        assert sum(weight for _, weight in bucket.mean_weights) == pytest.approx(1.0)
