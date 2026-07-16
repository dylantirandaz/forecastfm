"""Strict multi-season metrics for a frozen NBA forecast cohort."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from math import ceil, isfinite, log
from random import Random

BOOTSTRAP_BLOCK_DAYS = 7
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_716
ONE_SIDED_ALPHA = 0.05
FAILURE_REALIZED_PROBABILITY = 1e-15


class OutcomeV2MetricsError(ValueError):
    """Raised when a multi-season outcome evaluation is invalid."""


@dataclass(frozen=True, slots=True)
class BinaryForecast:
    """One candidate probability or explicit failure keyed by frozen identity."""

    question_id: str
    team_probability: float | None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        _require_question_id(self.question_id)
        if self.team_probability is None:
            if self.failure_reason is None or not self.failure_reason.strip():
                raise OutcomeV2MetricsError("failed forecast requires a reason")
            return
        if self.failure_reason is not None:
            raise OutcomeV2MetricsError("valid forecast cannot include a failure reason")
        _require_probability(self.team_probability, "team_probability")


@dataclass(frozen=True, slots=True)
class DatedBinaryCohortMember:
    """Frozen scoring metadata that a candidate forecast cannot alter."""

    question_id: str
    season: int
    game_date: date
    realized_team_win: bool
    baseline_team_probability: float

    def __post_init__(self) -> None:
        _require_question_id(self.question_id)
        if isinstance(self.season, bool) or self.season <= 0:
            raise OutcomeV2MetricsError("season must be positive")
        if self.season != _nba_season(self.game_date):
            raise OutcomeV2MetricsError("cohort season does not match its game date")
        _require_probability(self.baseline_team_probability, "baseline_team_probability")


@dataclass(frozen=True, slots=True)
class BinaryProperScores:
    """Mean binary proper scores for one model on one cohort."""

    mean_log_loss: float
    mean_brier: float


@dataclass(frozen=True, slots=True)
class SeasonEvaluation:
    """Proper scores and the predeclared superiority gate for one season."""

    season: int
    game_count: int
    calendar_block_count: int
    model: BinaryProperScores
    baseline: BinaryProperScores
    mean_baseline_relative_log_score: float
    lower_one_sided_95: float
    passes: bool


@dataclass(frozen=True, slots=True)
class MultiSeasonEvaluation:
    """A conjunction decision across an exact frozen cohort."""

    declared_seasons: tuple[int, ...]
    bootstrap_block_days: int
    bootstrap_resamples: int
    bootstrap_seed: int
    one_sided_alpha: float
    seasons: tuple[SeasonEvaluation, ...]
    game_count: int
    pooled_baseline_relative_log_score: float
    passes: bool


type _AlignedForecast = tuple[BinaryForecast, DatedBinaryCohortMember]


def evaluate_multi_season(
    forecasts: Sequence[BinaryForecast],
    cohort: Sequence[DatedBinaryCohortMember],
    declared_seasons: Sequence[int],
) -> MultiSeasonEvaluation:
    """Join predictions to a frozen cohort and require every season to win."""
    seasons = _validated_seasons(declared_seasons)
    aligned = _align_forecasts(forecasts, cohort)
    _validate_cohort_seasons(cohort, seasons)
    evaluations = tuple(
        _evaluate_season(
            season,
            tuple(pair for pair in aligned if pair[1].season == season),
        )
        for season in seasons
    )
    relative_scores = tuple(_baseline_relative_log_score(pair) for pair in aligned)
    return MultiSeasonEvaluation(
        declared_seasons=seasons,
        bootstrap_block_days=BOOTSTRAP_BLOCK_DAYS,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        one_sided_alpha=ONE_SIDED_ALPHA,
        seasons=evaluations,
        game_count=len(aligned),
        pooled_baseline_relative_log_score=_mean(relative_scores),
        passes=all(evaluation.passes for evaluation in evaluations),
    )


def failure_team_probability(realized_team_win: bool) -> float:
    """Return the frozen worst-case interior probability for a failed forecast."""
    if realized_team_win:
        return FAILURE_REALIZED_PROBABILITY
    return 1.0 - FAILURE_REALIZED_PROBABILITY


def _evaluate_season(
    season: int,
    rows: tuple[_AlignedForecast, ...],
) -> SeasonEvaluation:
    model_log_losses = tuple(
        _log_loss(_scored_probability(forecast, member), member.realized_team_win)
        for forecast, member in rows
    )
    baseline_log_losses = tuple(
        _log_loss(member.baseline_team_probability, member.realized_team_win) for _, member in rows
    )
    model_brier_scores = tuple(
        _brier(_scored_probability(forecast, member), member.realized_team_win)
        for forecast, member in rows
    )
    baseline_brier_scores = tuple(
        _brier(member.baseline_team_probability, member.realized_team_win) for _, member in rows
    )
    relative_scores = tuple(_baseline_relative_log_score(pair) for pair in rows)
    blocks = _calendar_blocks(rows, relative_scores)
    mean_improvement = _mean(relative_scores)
    lower_bound = _bootstrap_lower_bound(blocks, season)
    return SeasonEvaluation(
        season=season,
        game_count=len(rows),
        calendar_block_count=len(blocks),
        model=BinaryProperScores(
            mean_log_loss=_mean(model_log_losses),
            mean_brier=_mean(model_brier_scores),
        ),
        baseline=BinaryProperScores(
            mean_log_loss=_mean(baseline_log_losses),
            mean_brier=_mean(baseline_brier_scores),
        ),
        mean_baseline_relative_log_score=mean_improvement,
        lower_one_sided_95=lower_bound,
        passes=mean_improvement > 0.0 and lower_bound > 0.0,
    )


def _align_forecasts(
    forecasts: Sequence[BinaryForecast],
    cohort: Sequence[DatedBinaryCohortMember],
) -> tuple[_AlignedForecast, ...]:
    if not cohort:
        raise OutcomeV2MetricsError("frozen cohort must not be empty")
    members = {member.question_id: member for member in cohort}
    if len(members) != len(cohort):
        raise OutcomeV2MetricsError("frozen cohort question IDs must be unique")
    predictions = {forecast.question_id: forecast for forecast in forecasts}
    if len(predictions) != len(forecasts):
        raise OutcomeV2MetricsError("forecast question IDs must be unique")
    if set(predictions) - set(members):
        raise OutcomeV2MetricsError("forecast contains an ID outside the frozen cohort")
    return tuple((_forecast_or_failure(predictions, member), member) for member in cohort)


def _forecast_or_failure(
    predictions: dict[str, BinaryForecast],
    member: DatedBinaryCohortMember,
) -> BinaryForecast:
    return predictions.get(
        member.question_id,
        BinaryForecast(
            question_id=member.question_id,
            team_probability=None,
            failure_reason="missing forecast",
        ),
    )


def _validated_seasons(declared_seasons: Sequence[int]) -> tuple[int, ...]:
    seasons = tuple(declared_seasons)
    if not seasons:
        raise OutcomeV2MetricsError("at least one declared season is required")
    if len(set(seasons)) != len(seasons):
        raise OutcomeV2MetricsError("declared seasons must be unique")
    if seasons != tuple(sorted(seasons)):
        raise OutcomeV2MetricsError("declared seasons must be in increasing order")
    if any(season <= 0 for season in seasons):
        raise OutcomeV2MetricsError("declared seasons must be positive")
    return seasons


def _validate_cohort_seasons(
    cohort: Sequence[DatedBinaryCohortMember],
    seasons: tuple[int, ...],
) -> None:
    actual_seasons = {member.season for member in cohort}
    if set(seasons) - actual_seasons:
        raise OutcomeV2MetricsError("frozen cohort is missing a declared season")
    if actual_seasons - set(seasons):
        raise OutcomeV2MetricsError("frozen cohort contains an undeclared season")


def _calendar_blocks(
    rows: tuple[_AlignedForecast, ...],
    relative_scores: tuple[float, ...],
) -> tuple[tuple[float, int], ...]:
    grouped: dict[date, list[float]] = {}
    for (_, member), score in zip(rows, relative_scores, strict=True):
        block_start = member.game_date - timedelta(days=member.game_date.weekday())
        grouped.setdefault(block_start, []).append(score)
    return tuple(
        (sum(grouped[block_start]), len(grouped[block_start])) for block_start in sorted(grouped)
    )


def _bootstrap_lower_bound(blocks: tuple[tuple[float, int], ...], season: int) -> float:
    random = Random(BOOTSTRAP_SEED + season)
    block_count = len(blocks)
    means: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        total = 0.0
        game_count = 0
        for _ in range(block_count):
            block_total, block_games = blocks[random.randrange(block_count)]
            total += block_total
            game_count += block_games
        means.append(total / game_count)
    means.sort()
    lower_index = max(0, ceil(ONE_SIDED_ALPHA * BOOTSTRAP_RESAMPLES) - 1)
    return means[lower_index]


def _baseline_relative_log_score(row: _AlignedForecast) -> float:
    forecast, member = row
    model_probability = _realized_probability(
        _scored_probability(forecast, member),
        member.realized_team_win,
    )
    baseline_probability = _realized_probability(
        member.baseline_team_probability,
        member.realized_team_win,
    )
    return log(model_probability) - log(baseline_probability)


def _scored_probability(
    forecast: BinaryForecast,
    member: DatedBinaryCohortMember,
) -> float:
    if forecast.team_probability is None:
        return failure_team_probability(member.realized_team_win)
    return forecast.team_probability


def _log_loss(team_probability: float, realized_team_win: bool) -> float:
    return -log(_realized_probability(team_probability, realized_team_win))


def _brier(team_probability: float, realized_team_win: bool) -> float:
    probability = _realized_probability(team_probability, realized_team_win)
    return (1.0 - probability) ** 2


def _realized_probability(team_probability: float, realized_team_win: bool) -> float:
    if realized_team_win:
        return team_probability
    return 1.0 - team_probability


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise OutcomeV2MetricsError("cannot average an empty sequence")
    return sum(values) / len(values)


def _nba_season(game_date: date) -> int:
    return game_date.year + 1 if game_date.month >= 7 else game_date.year


def _require_question_id(question_id: str) -> None:
    if not question_id.strip():
        raise OutcomeV2MetricsError("question_id must not be empty")


def _require_probability(value: float, field_name: str) -> None:
    if not isfinite(value) or not 0.0 < value < 1.0:
        raise OutcomeV2MetricsError(
            f"{field_name} must be finite and strictly between zero and one"
        )
