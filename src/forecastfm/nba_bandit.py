"""Dependency-free offline contextual bandit over frozen NBA forecast arms.

This is horizon-1 reinforcement learning over already-resolved games: each game supplies a
context vector, the softmax selector mixes the frozen arms' home-win probabilities, and the
realized log score is the reward. The selector is a soft mixture, never a hard argmax, so the
forecast stays a strict probability and the training objective (mean log loss) is smooth in
the mixture parameters.

Missing-context policy (disclosed): when a game has no pre-cutoff injury report, its health
and projected-rotation context components are filled with ``0.0``. The zero bucket of the
unavailable-minutes diagnostic therefore mixes genuine zero differences with no-report games;
both states mean "no availability signal" for the selector.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log

from forecastfm.elo_residual import probability_logit

BANDIT_CONTEXT_NAMES = (
    "intercept",
    "abs_elo_logit",
    "abs_rest_days_difference",
    "abs_travel_miles_difference",
    "abs_unavailable_rotation_minutes_difference",
    "abs_projected_rotation_difference",
)
UNAVAILABLE_MINUTES_CONTEXT_INDEX = BANDIT_CONTEXT_NAMES.index(
    "abs_unavailable_rotation_minutes_difference"
)
UNAVAILABLE_MINUTES_BUCKETS = ("0", "(0,15]", ">15")
MISSING_CONTEXT_POLICY = (
    "Games without a pre-cutoff injury report contribute 0.0 to the unavailable-minutes and "
    "projected-rotation context components; the zero unavailable-minutes bucket mixes genuine "
    "zero differences with no-report games."
)
VERDICT_MARGIN = 0.002


class NbaBanditError(ValueError):
    """Raised when bandit data or settings are invalid."""


@dataclass(frozen=True, slots=True)
class BanditGame:
    """One resolved game with a context, frozen arm probabilities, and an answer."""

    question_id: str
    season: int
    context: tuple[float, ...]
    arm_probabilities: tuple[float, ...]
    outcome: int

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise NbaBanditError("question_id must not be empty")
        if isinstance(self.season, bool) or self.season <= 0:
            raise NbaBanditError("season must be a positive integer")
        if not self.context or not all(isfinite(value) for value in self.context):
            raise NbaBanditError("context must be non-empty and finite")
        if len(self.arm_probabilities) < 2:
            raise NbaBanditError("at least two arms are required")
        if not all(0.0 < probability < 1.0 for probability in self.arm_probabilities):
            raise NbaBanditError("arm probabilities must be strictly between zero and one")
        if isinstance(self.outcome, bool) or self.outcome not in {0, 1}:
            raise NbaBanditError("outcome must be zero or one")


@dataclass(frozen=True, slots=True)
class BanditFitConfig:
    """Deterministic full-batch gradient-descent settings for the mixture parameters."""

    steps: int = 2_000
    learning_rate: float = 0.05
    l2_penalty: float = 0.01

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or self.steps <= 0:
            raise NbaBanditError("steps must be a positive integer")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise NbaBanditError("learning_rate must be positive and finite")
        if not isfinite(self.l2_penalty) or self.l2_penalty < 0.0:
            raise NbaBanditError("l2_penalty must be non-negative and finite")


DEFAULT_BANDIT_FIT_CONFIG = BanditFitConfig()


@dataclass(frozen=True, slots=True)
class BanditSelector:
    """An immutable softmax mixture with frozen training-only context scales.

    ``theta`` is a context_dim x arm_count matrix; the frozen ``scales`` are uncentered RMS
    context scales computed on training games only, mirroring the Elo-residual recipe.
    """

    context_names: tuple[str, ...]
    arm_names: tuple[str, ...]
    scales: tuple[float, ...]
    theta: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.context_names or len(set(self.context_names)) != len(self.context_names):
            raise NbaBanditError("context names must be unique")
        if len(self.arm_names) < 2 or len(set(self.arm_names)) != len(self.arm_names):
            raise NbaBanditError("at least two unique arm names are required")
        if len(self.scales) != len(self.context_names):
            raise NbaBanditError("each context name must have one scale")
        if not all(scale > 0.0 and isfinite(scale) for scale in self.scales):
            raise NbaBanditError("scales must be positive and finite")
        if len(self.theta) != len(self.context_names):
            raise NbaBanditError("theta must have one row per context component")
        if any(len(row) != len(self.arm_names) for row in self.theta):
            raise NbaBanditError("theta rows must have one weight per arm")
        if not all(isfinite(weight) for row in self.theta for weight in row):
            raise NbaBanditError("theta weights must be finite")

    def mixture_weights(self, context: tuple[float, ...]) -> tuple[float, ...]:
        """Return softmax arm weights for one context (uniform when theta is zero)."""
        scaled = _scale_context(context, self.scales)
        logits = [
            sum(row[arm] * value for row, value in zip(self.theta, scaled, strict=True))
            for arm in range(len(self.arm_names))
        ]
        return _softmax(logits)

    def forecast(self, context: tuple[float, ...], arm_probabilities: tuple[float, ...]) -> float:
        """Return the mixture home-win probability for one game."""
        if len(arm_probabilities) != len(self.arm_names):
            raise NbaBanditError("arm probability count differs from the selector")
        weights = self.mixture_weights(context)
        return sum(
            weight * probability
            for weight, probability in zip(weights, arm_probabilities, strict=True)
        )


@dataclass(frozen=True, slots=True)
class BanditEvaluation:
    """Mean log losses for every arm, the per-game oracle, and the mixture."""

    game_count: int
    arm_log_losses: tuple[tuple[str, float], ...]
    oracle_log_loss: float
    mixture_log_loss: float


@dataclass(frozen=True, slots=True)
class SelectionWeightBucket:
    """Mean mixture weights per arm over the games in one context bucket."""

    bucket: str
    game_count: int
    mean_weights: tuple[tuple[str, float], ...]


def bandit_context(
    *,
    elo_probability: float,
    rest_days_difference: float,
    travel_miles_difference: float,
    unavailable_rotation_minutes_difference: float | None,
    projected_rotation_difference: float | None,
) -> tuple[float, ...]:
    """Build the six-component context; ``None`` components fill with zero (see policy)."""
    values = (
        1.0,
        abs(probability_logit(elo_probability)),
        abs(rest_days_difference),
        abs(travel_miles_difference),
        abs(unavailable_rotation_minutes_difference or 0.0),
        abs(projected_rotation_difference or 0.0),
    )
    if not all(isfinite(value) for value in values):
        raise NbaBanditError("context components must be finite")
    return values


def unavailable_minutes_bucket(abs_difference: float) -> str:
    """Bucket one absolute unavailable-minutes difference as 0 / (0,15] / >15."""
    if not isfinite(abs_difference) or abs_difference < 0.0:
        raise NbaBanditError("absolute difference must be finite and non-negative")
    if abs_difference == 0.0:
        return UNAVAILABLE_MINUTES_BUCKETS[0]
    if abs_difference <= 15.0:
        return UNAVAILABLE_MINUTES_BUCKETS[1]
    return UNAVAILABLE_MINUTES_BUCKETS[2]


def fit_bandit_selector(
    games: Sequence[BanditGame],
    arm_names: tuple[str, ...],
    config: BanditFitConfig = DEFAULT_BANDIT_FIT_CONFIG,
) -> BanditSelector:
    """Fit the softmax mixture on mean log loss with L2 regularization from theta = 0."""
    _require_games(games, arm_names)
    context_dim = len(BANDIT_CONTEXT_NAMES)
    if len(games[0].context) != context_dim:
        raise NbaBanditError("context width differs from the bandit context schema")
    arm_count = len(arm_names)
    scales = _fit_context_scales(games, context_dim)
    prepared = tuple(
        (_scale_context(game.context, scales), game.arm_probabilities, game.outcome)
        for game in games
    )
    theta = [[0.0] * arm_count for _ in range(context_dim)]
    game_scale = 1.0 / len(prepared)
    for _ in range(config.steps):
        gradients = [[config.l2_penalty * weight for weight in row] for row in theta]
        for context, arm_probabilities, outcome in prepared:
            weights = _softmax(
                [
                    sum(row[arm] * value for row, value in zip(theta, context, strict=True))
                    for arm in range(arm_count)
                ]
            )
            mixture = sum(
                weight * probability
                for weight, probability in zip(weights, arm_probabilities, strict=True)
            )
            # d(-log realized p_mix)/d(p_mix) = (p_mix - y) / (p_mix * (1 - p_mix))
            direction = (mixture - outcome) / (mixture * (1.0 - mixture))
            for arm in range(arm_count):
                coefficient = (
                    direction * weights[arm] * (arm_probabilities[arm] - mixture) * game_scale
                )
                for index, value in enumerate(context):
                    gradients[index][arm] += coefficient * value
        theta = [
            [
                weight - config.learning_rate * gradient
                for weight, gradient in zip(row, gradient_row, strict=True)
            ]
            for row, gradient_row in zip(theta, gradients, strict=True)
        ]
        if not all(isfinite(weight) for row in theta for weight in row):
            raise NbaBanditError("training produced non-finite theta")
    return BanditSelector(
        context_names=BANDIT_CONTEXT_NAMES,
        arm_names=arm_names,
        scales=scales,
        theta=tuple(tuple(row) for row in theta),
    )


def evaluate_bandit(selector: BanditSelector, games: Sequence[BanditGame]) -> BanditEvaluation:
    """Score every frozen arm, the cheating per-game oracle, and the mixture."""
    _require_games(games, selector.arm_names)
    arm_totals = [0.0] * len(selector.arm_names)
    oracle_total = 0.0
    mixture_total = 0.0
    for game in games:
        arm_losses = [
            _log_loss(probability, game.outcome) for probability in game.arm_probabilities
        ]
        for arm, loss in enumerate(arm_losses):
            arm_totals[arm] += loss
        oracle_total += min(arm_losses)
        mixture_total += _log_loss(
            selector.forecast(game.context, game.arm_probabilities), game.outcome
        )
    game_count = len(games)
    return BanditEvaluation(
        game_count=game_count,
        arm_log_losses=tuple(
            (name, total / game_count)
            for name, total in zip(selector.arm_names, arm_totals, strict=True)
        ),
        oracle_log_loss=oracle_total / game_count,
        mixture_log_loss=mixture_total / game_count,
    )


def selection_weight_distribution(
    selector: BanditSelector,
    games: Sequence[BanditGame],
    bucket_of: Callable[[BanditGame], str],
) -> tuple[SelectionWeightBucket, ...]:
    """Return mean mixture weights per arm for each non-empty context bucket."""
    _require_games(games, selector.arm_names)
    grouped: dict[str, list[BanditGame]] = {}
    for game in games:
        grouped.setdefault(bucket_of(game), []).append(game)
    return tuple(
        _bucket_payload(selector, bucket, bucket_games)
        for bucket, bucket_games in sorted(grouped.items())
    )


def _bucket_payload(
    selector: BanditSelector,
    bucket: str,
    games: list[BanditGame],
) -> SelectionWeightBucket:
    totals = [0.0] * len(selector.arm_names)
    for game in games:
        for arm, weight in enumerate(selector.mixture_weights(game.context)):
            totals[arm] += weight
    return SelectionWeightBucket(
        bucket=bucket,
        game_count=len(games),
        mean_weights=tuple(
            (name, total / len(games))
            for name, total in zip(selector.arm_names, totals, strict=True)
        ),
    )


def _log_loss(probability: float, outcome: int) -> float:
    realized = probability if outcome == 1 else 1.0 - probability
    return -log(realized)


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    if not logits:
        raise NbaBanditError("softmax requires at least one logit")
    peak = max(logits)
    weights = [exp(logit - peak) for logit in logits]
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def _scale_context(context: tuple[float, ...], scales: tuple[float, ...]) -> tuple[float, ...]:
    if len(context) != len(scales):
        raise NbaBanditError("context width differs from the selector scales")
    return tuple(value / scale for value, scale in zip(context, scales, strict=True))


def _fit_context_scales(games: Sequence[BanditGame], context_dim: int) -> tuple[float, ...]:
    scales: list[float] = []
    for index in range(context_dim):
        mean_square = sum(game.context[index] ** 2 for game in games) / len(games)
        scales.append(mean_square**0.5 if mean_square > 0.0 else 1.0)
    return tuple(scales)


def _require_games(games: Sequence[BanditGame], arm_names: tuple[str, ...]) -> None:
    if not games:
        raise NbaBanditError("at least one game is required")
    context_dim = len(games[0].context)
    for game in games:
        if len(game.context) != context_dim:
            raise NbaBanditError("context widths differ across games")
        if len(game.arm_probabilities) != len(arm_names):
            raise NbaBanditError("arm probability count differs from arm names")
