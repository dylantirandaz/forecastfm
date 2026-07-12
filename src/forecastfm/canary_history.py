"""Answer-gated historical diagnostics for an already sealed canary run."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log
from pathlib import Path

from forecastfm.canary import (
    CANARY_SIZE,
    INVALID_BRIER,
    INVALID_LOG_LOSS,
    INVALID_MAE,
    MINIMUM_LOG_PROBABILITY,
    CanaryValidationError,
    GenerationRecord,
    SealedGenerations,
    parsed_team_probability,
)
from forecastfm.integrity import file_sha256
from forecastfm.models import TrainingExample
from forecastfm.serialization import read_jsonl

VALIDATION_ANSWERS_LABEL = "data/processed/nba_elo_validation_answers.jsonl"


@dataclass(frozen=True, slots=True)
class HistoricalModelMetrics:
    """Historical diagnostics for one model on the 64 original games."""

    game_count: int
    valid_count: int
    teacher_mae: float
    valid_only_teacher_mae: float | None
    mean_brier: float
    valid_only_mean_brier: float | None
    mean_log_loss: float
    valid_only_mean_log_loss: float | None


@dataclass(frozen=True, slots=True)
class HistoricalComparison:
    """Paired historical diagnostics and adapter-minus-base deltas."""

    base: HistoricalModelMetrics
    adapter: HistoricalModelMetrics
    adapter_minus_base_teacher_mae: float
    adapter_minus_base_brier: float
    adapter_minus_base_log_loss: float


def score_historical(
    generations: SealedGenerations,
    answers_path: Path,
) -> HistoricalComparison:
    """Open the committed answers only after receiving sealed generations."""
    if answers_path.name != Path(VALIDATION_ANSWERS_LABEL).name:
        raise CanaryValidationError("historical scorer accepts validation answers only")
    if file_sha256(answers_path) != generations.manifest.source_answer_sha256:
        raise CanaryValidationError("validation answers differ from the frozen commitment")
    answers = {item.case.question.question_id: item for item in read_jsonl(answers_path)}
    expected = set(generations.manifest.question_ids)
    if not expected.issubset(answers):
        raise CanaryValidationError("validation answers do not cover the frozen canary")
    base = _model_metrics(generations, generations.base, answers)
    adapter = _model_metrics(generations, generations.adapter, answers)
    return HistoricalComparison(
        base=base,
        adapter=adapter,
        adapter_minus_base_teacher_mae=adapter.teacher_mae - base.teacher_mae,
        adapter_minus_base_brier=adapter.mean_brier - base.mean_brier,
        adapter_minus_base_log_loss=adapter.mean_log_loss - base.mean_log_loss,
    )


def _model_metrics(
    generations: SealedGenerations,
    records: Sequence[GenerationRecord],
    answers: Mapping[str, TrainingExample],
) -> HistoricalModelMetrics:
    teacher_errors: list[float] = []
    briers: list[float] = []
    losses: list[float] = []
    valid_teacher: list[float] = []
    valid_briers: list[float] = []
    valid_losses: list[float] = []
    for index, question_id in enumerate(generations.manifest.question_ids):
        answer = answers[question_id]
        probability = parsed_team_probability(records[index * 2])
        if probability is None:
            teacher_errors.append(INVALID_MAE)
            briers.append(INVALID_BRIER)
            losses.append(INVALID_LOG_LOSS)
            continue
        teacher, brier, loss = _valid_scores(probability, answer)
        teacher_errors.append(teacher)
        briers.append(brier)
        losses.append(loss)
        valid_teacher.append(teacher)
        valid_briers.append(brier)
        valid_losses.append(loss)
    return HistoricalModelMetrics(
        game_count=CANARY_SIZE,
        valid_count=len(valid_teacher),
        teacher_mae=_mean(teacher_errors),
        valid_only_teacher_mae=_optional_mean(valid_teacher),
        mean_brier=_mean(briers),
        valid_only_mean_brier=_optional_mean(valid_briers),
        mean_log_loss=_mean(losses),
        valid_only_mean_log_loss=_optional_mean(valid_losses),
    )


def _valid_scores(probability: float, answer: TrainingExample) -> tuple[float, float, float]:
    realized = answer.realized_outcome
    if realized is None:
        raise CanaryValidationError("historical answer is unresolved")
    target = answer.target.distribution.probability_for("team_wins")
    outcome_probability = probability if realized == "team_wins" else 1.0 - probability
    return (
        abs(probability - target),
        (1.0 - outcome_probability) ** 2,
        -log(max(outcome_probability, MINIMUM_LOG_PROBABILITY)),
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise CanaryValidationError("cannot average an empty metric")
    return sum(values) / len(values)


def _optional_mean(values: Sequence[float]) -> float | None:
    return None if not values else _mean(values)
