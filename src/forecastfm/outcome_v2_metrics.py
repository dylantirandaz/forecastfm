"""Leakage-safe multi-season metrics for historical NBA outcome forecasts."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from math import ceil, isfinite, log
from random import Random

BOOTSTRAP_BLOCK_DAYS = 7
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_716
ONE_SIDED_ALPHA = 0.05
MINIMUM_LOG_PROBABILITY = 1e-15


class OutcomeV2MetricsError(ValueError):
    """Raised when a multi-season outcome evaluation is invalid."""


@dataclass(frozen=True, slots=True)
class DatedBinaryForecast:
    """One model and Elo forecast paired with a dated realized outcome."""

    question_id: str
    season: int
    game_date: date
    realized_team_win: bool
    model_team_probability: float
    elo_team_probability: float

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise OutcomeV2MetricsError("question_id must not be empty")
        if self.season <= 0:
            raise OutcomeV2MetricsError("season must be positive")
        if self.season != _nba_season(self.game_date):
            raise OutcomeV2MetricsError("forecast season does not match its game date")
        _require_probability(self.model_team_probability, "model_team_probability")
        _require_probability(self.elo_team_probability, "elo_team_probability")


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
    elo: BinaryProperScores
    mean_elo_relative_log_score: float
    lower_one_sided_95: float
    passes: bool


@dataclass(frozen=True, slots=True)
class MultiSeasonEvaluation:
    """A conjunction decision across an exact set of declared seasons."""

    declared_seasons: tuple[int, ...]
    bootstrap_block_days: int
    bootstrap_resamples: int
    bootstrap_seed: int
    one_sided_alpha: float
    seasons: tuple[SeasonEvaluation, ...]
    game_count: int
    pooled_elo_relative_log_score: float
    passes: bool


def evaluate_multi_season(
    rows: Sequence[DatedBinaryForecast],
    declared_seasons: Sequence[int],
) -> MultiSeasonEvaluation:
    """Evaluate exact season coverage and require every season to beat Elo."""
    seasons = _validate_coverage(rows, declared_seasons)
    evaluations = tuple(
        _evaluate_season(
            season,
            tuple(row for row in rows if row.season == season),
        )
        for season in seasons
    )
    relative_scores = tuple(_elo_relative_log_score(row) for row in rows)
    return MultiSeasonEvaluation(
        declared_seasons=seasons,
        bootstrap_block_days=BOOTSTRAP_BLOCK_DAYS,
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        one_sided_alpha=ONE_SIDED_ALPHA,
        seasons=evaluations,
        game_count=len(rows),
        pooled_elo_relative_log_score=_mean(relative_scores),
        passes=all(evaluation.passes for evaluation in evaluations),
    )


def _evaluate_season(
    season: int,
    rows: tuple[DatedBinaryForecast, ...],
) -> SeasonEvaluation:
    model_log_losses = tuple(
        _log_loss(row.model_team_probability, row.realized_team_win) for row in rows
    )
    elo_log_losses = tuple(
        _log_loss(row.elo_team_probability, row.realized_team_win) for row in rows
    )
    model_brier_scores = tuple(
        _brier(row.model_team_probability, row.realized_team_win) for row in rows
    )
    elo_brier_scores = tuple(
        _brier(row.elo_team_probability, row.realized_team_win) for row in rows
    )
    relative_scores = tuple(_elo_relative_log_score(row) for row in rows)
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
        elo=BinaryProperScores(
            mean_log_loss=_mean(elo_log_losses),
            mean_brier=_mean(elo_brier_scores),
        ),
        mean_elo_relative_log_score=mean_improvement,
        lower_one_sided_95=lower_bound,
        passes=mean_improvement > 0.0 and lower_bound > 0.0,
    )


def _validate_coverage(
    rows: Sequence[DatedBinaryForecast],
    declared_seasons: Sequence[int],
) -> tuple[int, ...]:
    seasons = tuple(declared_seasons)
    if not seasons:
        raise OutcomeV2MetricsError("at least one declared season is required")
    if len(set(seasons)) != len(seasons):
        raise OutcomeV2MetricsError("declared seasons must be unique")
    if seasons != tuple(sorted(seasons)):
        raise OutcomeV2MetricsError("declared seasons must be in increasing order")
    if any(season <= 0 for season in seasons):
        raise OutcomeV2MetricsError("declared seasons must be positive")

    question_ids = tuple(row.question_id for row in rows)
    if len(set(question_ids)) != len(question_ids):
        raise OutcomeV2MetricsError("duplicate forecast question_id")
    actual_seasons = {row.season for row in rows}
    missing = set(seasons) - actual_seasons
    if missing:
        raise OutcomeV2MetricsError("missing declared season")
    if actual_seasons - set(seasons):
        raise OutcomeV2MetricsError("forecast row belongs to an undeclared season")
    return seasons


def _calendar_blocks(
    rows: tuple[DatedBinaryForecast, ...],
    relative_scores: tuple[float, ...],
) -> tuple[tuple[float, int], ...]:
    grouped: dict[date, list[float]] = {}
    for row, score in zip(rows, relative_scores, strict=True):
        block_start = row.game_date - timedelta(days=row.game_date.weekday())
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


def _elo_relative_log_score(row: DatedBinaryForecast) -> float:
    model_probability = _realized_probability(
        row.model_team_probability,
        row.realized_team_win,
    )
    elo_probability = _realized_probability(
        row.elo_team_probability,
        row.realized_team_win,
    )
    return log(max(model_probability, MINIMUM_LOG_PROBABILITY)) - log(
        max(elo_probability, MINIMUM_LOG_PROBABILITY)
    )


def _log_loss(team_probability: float, realized_team_win: bool) -> float:
    probability = _realized_probability(team_probability, realized_team_win)
    return -log(max(probability, MINIMUM_LOG_PROBABILITY))


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


def _require_probability(value: float, field_name: str) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise OutcomeV2MetricsError(f"{field_name} must be finite and between zero and one")
