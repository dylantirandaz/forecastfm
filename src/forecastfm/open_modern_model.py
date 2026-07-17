"""Deterministic validation-only models for the open-modern NBA cohort."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from math import ceil, isclose, isfinite, log, sqrt
from random import Random

from forecastfm.elo_residual import (
    EloResidualFitConfig,
    EloResidualModel,
    EloResidualRow,
    fit_elo_residual,
)
from forecastfm.integrity import canonical_sha256
from forecastfm.open_modern import TRAIN_SEASONS, VALIDATION_SEASONS
from forecastfm.open_modern_features import OPEN_MODERN_CAUSAL_FEATURE_NAMES

FIT_STEPS = 1_000
FIT_LEARNING_RATE = 0.1
FORECAST_L2_PENALTY = 0.01

BOOTSTRAP_BLOCK_DAYS = 7
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_716
ONE_SIDED_ALPHA = 0.05
ECE_BIN_COUNT = 10
MAXIMUM_SIDE_SWAP_GAP = 1e-12

_SOURCE_LOG_ODDS = OPEN_MODERN_CAUSAL_FEATURE_NAMES[0]
_FULL_FEATURES = OPEN_MODERN_CAUSAL_FEATURE_NAMES
_FEATURE_INDEX = {name: index for index, name in enumerate(OPEN_MODERN_CAUSAL_FEATURE_NAMES)}


class OpenModernModelError(ValueError):
    """Raised when an open-modern validation experiment is invalid."""


@dataclass(frozen=True, slots=True)
class OpenModernResolvedRow:
    """One labeled game with its pregame baseline and causal features."""

    question_id: str
    season: int
    game_date: date
    source_probability: float
    features: tuple[float, ...]
    outcome: int

    def __post_init__(self) -> None:
        if not self.question_id or self.question_id != self.question_id.strip():
            raise OpenModernModelError("question ID must be present and trimmed")
        if isinstance(self.season, bool) or self.season <= 0:
            raise OpenModernModelError("season must be a positive integer")
        if self.game_date.year not in {self.season - 1, self.season}:
            raise OpenModernModelError("game date and season disagree")
        _require_probability(self.source_probability, "source_probability")
        if len(self.features) != len(OPEN_MODERN_CAUSAL_FEATURE_NAMES):
            raise OpenModernModelError("resolved row has the wrong feature count")
        if not all(isfinite(value) for value in self.features):
            raise OpenModernModelError("resolved row features must be finite")
        expected_log_odds = log(self.source_probability / (1.0 - self.source_probability))
        if not isclose(self.features[0], expected_log_odds, abs_tol=1e-9):
            raise OpenModernModelError("source log odds do not match the baseline")
        if isinstance(self.outcome, bool) or self.outcome not in {0, 1}:
            raise OpenModernModelError("outcome must be zero or one")


@dataclass(frozen=True, slots=True)
class OpenModernRmsScaler:
    """Uncentered feature RMS values fitted on training seasons only."""

    feature_names: tuple[str, ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.feature_names != OPEN_MODERN_CAUSAL_FEATURE_NAMES:
            raise OpenModernModelError("RMS scaler feature order is invalid")
        if len(self.scales) != len(self.feature_names):
            raise OpenModernModelError("RMS scaler has the wrong scale count")
        if not all(isfinite(scale) and scale > 0.0 for scale in self.scales):
            raise OpenModernModelError("RMS scales must be positive and finite")

    def transform(self, features: tuple[float, ...]) -> tuple[float, ...]:
        """Scale one full feature vector without mean centering."""
        if len(features) != len(self.scales):
            raise OpenModernModelError("feature vector and RMS scaler differ in length")
        if not all(isfinite(value) for value in features):
            raise OpenModernModelError("features to scale must be finite")
        return tuple(value / scale for value, scale in zip(features, self.scales, strict=True))


@dataclass(frozen=True, slots=True)
class OpenModernCandidateSpec:
    """One fixed feature set and regularization choice."""

    name: str
    feature_names: tuple[str, ...]
    l2_penalty: float

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise OpenModernModelError("candidate name must be present and trimmed")
        expected = tuple(
            name for name in OPEN_MODERN_CAUSAL_FEATURE_NAMES if name in self.feature_names
        )
        if not self.feature_names or self.feature_names != expected:
            raise OpenModernModelError("candidate features must follow the causal feature order")
        if not isfinite(self.l2_penalty) or self.l2_penalty < 0.0:
            raise OpenModernModelError("candidate L2 penalty must be non-negative and finite")

    @property
    def candidate_id(self) -> str:
        """Return a stable readable identifier."""
        return f"{self.name}-l2-{self.l2_penalty:g}"


RECALIBRATION_SPEC = OpenModernCandidateSpec(
    name="recalibration",
    feature_names=(_SOURCE_LOG_ODDS,),
    l2_penalty=0.0,
)
FORECAST_SPEC = OpenModernCandidateSpec(
    name="full",
    feature_names=_FULL_FEATURES,
    l2_penalty=FORECAST_L2_PENALTY,
)
OPEN_MODERN_MODEL_CONTRACT_SHA256 = canonical_sha256(
    {
        "schema_version": 2,
        "train_seasons": list(TRAIN_SEASONS),
        "validation_seasons": list(VALIDATION_SEASONS),
        "row_order": "season, date, question_id before fit, prediction, and scoring",
        "scaling": "uncentered RMS fitted on original training rows only",
        "intercept": False,
        "recalibration": {
            "name": RECALIBRATION_SPEC.name,
            "feature_names": list(RECALIBRATION_SPEC.feature_names),
            "l2_penalty": RECALIBRATION_SPEC.l2_penalty,
        },
        "forecast": {
            "name": FORECAST_SPEC.name,
            "feature_names": list(FORECAST_SPEC.feature_names),
            "l2_penalty": FORECAST_SPEC.l2_penalty,
            "selection": "predeclared before validation labels are scored",
        },
        "optimizer": {
            "loss": "binary cross entropy with source log-odds offset",
            "steps": FIT_STEPS,
            "learning_rate": FIT_LEARNING_RATE,
        },
        "validation_use": "gate only; no candidate or hyperparameter selection",
        "metrics": {
            "primary": "mean log loss",
            "secondary": ["Brier score", "10-bin equal-width ECE"],
            "bootstrap_block_days": BOOTSTRAP_BLOCK_DAYS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "one_sided_alpha": ONE_SIDED_ALPHA,
            "calendar_blocks": "Monday-anchored seven-day blocks",
            "resampling": (
                "sample blocks equally with replacement; compute game-weighted replicate means"
            ),
            "quantile_index": "ceil(alpha * resamples) - 1 after ascending sort",
            "random_stream": "restart the same fixed seed for each paired comparison",
        },
        "side_swap": "complement source probability and exactly negate selected features",
        "side_swap_training_rows": (
            "one exact swapped copy per training row, with complemented prior and outcome "
            "and negated scaled features"
        ),
        "advance": (
            "positive mean and one-sided lower bound versus raw source and fixed "
            "recalibration, plus maximum side-swap gap at most 1e-12"
        ),
    }
)


@dataclass(frozen=True, slots=True)
class OpenModernValidationMetrics:
    """Proper scores and calibration on one exact validation cohort."""

    count: int
    mean_log_loss: float
    mean_brier: float
    expected_calibration_error: float


@dataclass(frozen=True, slots=True)
class OpenModernCandidateFit:
    """One fitted candidate and its ordered validation forecasts."""

    spec: OpenModernCandidateSpec
    model: EloResidualModel
    validation_probabilities: tuple[float, ...]
    metrics: OpenModernValidationMetrics


@dataclass(frozen=True, slots=True)
class OpenModernBaselineComparison:
    """Calendar-block uncertainty for candidate improvement over a baseline."""

    baseline_name: str
    game_count: int
    calendar_block_count: int
    mean_baseline_relative_log_score: float
    lower_one_sided_95: float


@dataclass(frozen=True, slots=True)
class OpenModernValidationResult:
    """Complete deterministic result of training-only fitting and a validation gate."""

    scaler: OpenModernRmsScaler
    raw_source_metrics: OpenModernValidationMetrics
    recalibration: OpenModernCandidateFit
    forecast: OpenModernCandidateFit
    forecast_vs_raw_source: OpenModernBaselineComparison
    forecast_vs_recalibration: OpenModernBaselineComparison
    maximum_side_swap_gap: float
    advances_to_holdout: bool


def fit_train_rms_scaler(rows: Sequence[OpenModernResolvedRow]) -> OpenModernRmsScaler:
    """Fit uncentered RMS values using training-season rows and no validation data."""
    if not rows:
        raise OpenModernModelError("RMS scaling requires training rows")
    if any(row.season not in TRAIN_SEASONS for row in rows):
        raise OpenModernModelError("RMS scaling accepts training seasons only")
    sums = [0.0] * len(OPEN_MODERN_CAUSAL_FEATURE_NAMES)
    for row in rows:
        for index, value in enumerate(row.features):
            sums[index] += value * value
    scales = tuple(sqrt(total / len(rows)) for total in sums)
    return OpenModernRmsScaler(OPEN_MODERN_CAUSAL_FEATURE_NAMES, scales)


def fit_open_modern_validation(
    rows: Sequence[OpenModernResolvedRow],
) -> OpenModernValidationResult:
    """Fit two predeclared models on 2016-2019 and apply the 2020 gate."""
    ordered = _validated_ordered_rows(rows)
    training = tuple(row for row in ordered if row.season in TRAIN_SEASONS)
    validation = tuple(row for row in ordered if row.season in VALIDATION_SEASONS)
    scaler = fit_train_rms_scaler(training)
    recalibration = _fit_candidate(RECALIBRATION_SPEC, training, validation, scaler)
    forecast = _fit_candidate(FORECAST_SPEC, training, validation, scaler)
    raw_probabilities = tuple(row.source_probability for row in validation)
    versus_raw = _compare(
        "raw_source_probability",
        forecast.validation_probabilities,
        raw_probabilities,
        validation,
    )
    versus_recalibration = _compare(
        "fixed_training_recalibration",
        forecast.validation_probabilities,
        recalibration.validation_probabilities,
        validation,
    )
    side_swap_gap = _maximum_side_swap_gap(forecast, validation, scaler)
    advances = _advances(versus_raw, versus_recalibration, side_swap_gap)
    return OpenModernValidationResult(
        scaler=scaler,
        raw_source_metrics=_metrics(raw_probabilities, validation),
        recalibration=recalibration,
        forecast=forecast,
        forecast_vs_raw_source=versus_raw,
        forecast_vs_recalibration=versus_recalibration,
        maximum_side_swap_gap=side_swap_gap,
        advances_to_holdout=advances,
    )


def _validated_ordered_rows(
    rows: Sequence[OpenModernResolvedRow],
) -> tuple[OpenModernResolvedRow, ...]:
    if not rows:
        raise OpenModernModelError("validation experiment requires resolved rows")
    if len({row.question_id for row in rows}) != len(rows):
        raise OpenModernModelError("resolved question IDs must be unique")
    declared = {*TRAIN_SEASONS, *VALIDATION_SEASONS}
    if {row.season for row in rows} != declared:
        raise OpenModernModelError("resolved rows must cover exactly the declared seasons")
    return tuple(sorted(rows, key=lambda row: (row.season, row.game_date, row.question_id)))


def _fit_candidate(
    spec: OpenModernCandidateSpec,
    training: Sequence[OpenModernResolvedRow],
    validation: Sequence[OpenModernResolvedRow],
    scaler: OpenModernRmsScaler,
) -> OpenModernCandidateFit:
    training_rows = tuple(
        model_row for row in training for model_row in _model_row_and_side_swap(row, spec, scaler)
    )
    model = fit_elo_residual(
        training_rows,
        spec.feature_names,
        EloResidualFitConfig(FIT_STEPS, FIT_LEARNING_RATE, spec.l2_penalty),
    )
    probabilities = tuple(_predict(model, spec, scaler, row) for row in validation)
    return OpenModernCandidateFit(spec, model, probabilities, _metrics(probabilities, validation))


def _model_row_and_side_swap(
    row: OpenModernResolvedRow,
    spec: OpenModernCandidateSpec,
    scaler: OpenModernRmsScaler,
) -> tuple[EloResidualRow, EloResidualRow]:
    features = _selected_features(scaler.transform(row.features), spec)
    return (
        EloResidualRow(
            question_id=row.question_id,
            elo_probability=row.source_probability,
            features=features,
            outcome=row.outcome,
        ),
        EloResidualRow(
            question_id=f"{row.question_id}:side-swap",
            elo_probability=1.0 - row.source_probability,
            features=tuple(-value for value in features),
            outcome=1 - row.outcome,
        ),
    )


def _predict(
    model: EloResidualModel,
    spec: OpenModernCandidateSpec,
    scaler: OpenModernRmsScaler,
    row: OpenModernResolvedRow,
) -> float:
    features = _selected_features(scaler.transform(row.features), spec)
    return model.predict_probability(row.source_probability, features)


def _selected_features(
    scaled: tuple[float, ...],
    spec: OpenModernCandidateSpec,
) -> tuple[float, ...]:
    return tuple(scaled[_FEATURE_INDEX[name]] for name in spec.feature_names)


def _metrics(
    probabilities: Sequence[float],
    rows: Sequence[OpenModernResolvedRow],
) -> OpenModernValidationMetrics:
    _require_aligned_probabilities(probabilities, rows)
    losses = tuple(
        -log(_realized_probability(probability, row.outcome))
        for probability, row in zip(probabilities, rows, strict=True)
    )
    briers = tuple(
        (probability - row.outcome) ** 2
        for probability, row in zip(probabilities, rows, strict=True)
    )
    return OpenModernValidationMetrics(
        count=len(rows),
        mean_log_loss=sum(losses) / len(losses),
        mean_brier=sum(briers) / len(briers),
        expected_calibration_error=_expected_calibration_error(probabilities, rows),
    )


def _expected_calibration_error(
    probabilities: Sequence[float],
    rows: Sequence[OpenModernResolvedRow],
) -> float:
    bins: list[list[tuple[float, int]]] = [[] for _ in range(ECE_BIN_COUNT)]
    for probability, row in zip(probabilities, rows, strict=True):
        index = min(int(probability * ECE_BIN_COUNT), ECE_BIN_COUNT - 1)
        bins[index].append((probability, row.outcome))
    error = 0.0
    for values in bins:
        if values:
            mean_probability = sum(value[0] for value in values) / len(values)
            observed_rate = sum(value[1] for value in values) / len(values)
            error += len(values) * abs(mean_probability - observed_rate)
    return error / len(rows)


def _compare(
    baseline_name: str,
    candidate: Sequence[float],
    baseline: Sequence[float],
    rows: Sequence[OpenModernResolvedRow],
) -> OpenModernBaselineComparison:
    _require_aligned_probabilities(candidate, rows)
    _require_aligned_probabilities(baseline, rows)
    scores = tuple(
        log(_realized_probability(model_probability, row.outcome))
        - log(_realized_probability(baseline_probability, row.outcome))
        for model_probability, baseline_probability, row in zip(
            candidate, baseline, rows, strict=True
        )
    )
    blocks = _calendar_blocks(rows, scores)
    return OpenModernBaselineComparison(
        baseline_name=baseline_name,
        game_count=len(rows),
        calendar_block_count=len(blocks),
        mean_baseline_relative_log_score=sum(scores) / len(scores),
        lower_one_sided_95=_bootstrap_lower_bound(blocks),
    )


def _calendar_blocks(
    rows: Sequence[OpenModernResolvedRow],
    scores: Sequence[float],
) -> tuple[tuple[float, int], ...]:
    grouped: dict[date, list[float]] = {}
    for row, score in zip(rows, scores, strict=True):
        monday = row.game_date - timedelta(days=row.game_date.weekday())
        grouped.setdefault(monday, []).append(score)
    return tuple((sum(grouped[key]), len(grouped[key])) for key in sorted(grouped))


def _bootstrap_lower_bound(blocks: Sequence[tuple[float, int]]) -> float:
    random = Random(BOOTSTRAP_SEED)
    means: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        total = 0.0
        game_count = 0
        for _ in range(len(blocks)):
            block_total, block_games = blocks[random.randrange(len(blocks))]
            total += block_total
            game_count += block_games
        means.append(total / game_count)
    means.sort()
    index = max(0, ceil(ONE_SIDED_ALPHA * BOOTSTRAP_RESAMPLES) - 1)
    return means[index]


def _maximum_side_swap_gap(
    candidate: OpenModernCandidateFit,
    rows: Sequence[OpenModernResolvedRow],
    scaler: OpenModernRmsScaler,
) -> float:
    gaps: list[float] = []
    for row, probability in zip(rows, candidate.validation_probabilities, strict=True):
        scaled = scaler.transform(row.features)
        swapped_features = tuple(-value for value in _selected_features(scaled, candidate.spec))
        swapped_probability = candidate.model.predict_probability(
            1.0 - row.source_probability,
            swapped_features,
        )
        gaps.append(abs(swapped_probability - (1.0 - probability)))
    return max(gaps)


def _advances(
    versus_raw: OpenModernBaselineComparison,
    versus_recalibration: OpenModernBaselineComparison,
    side_swap_gap: float,
) -> bool:
    comparisons = (versus_raw, versus_recalibration)
    return (
        all(comparison.mean_baseline_relative_log_score > 0.0 for comparison in comparisons)
        and all(comparison.lower_one_sided_95 > 0.0 for comparison in comparisons)
        and side_swap_gap <= MAXIMUM_SIDE_SWAP_GAP
    )


def _require_aligned_probabilities(
    probabilities: Sequence[float],
    rows: Sequence[OpenModernResolvedRow],
) -> None:
    if not rows or len(probabilities) != len(rows):
        raise OpenModernModelError("each validation row must have one probability")
    for probability in probabilities:
        _require_probability(probability, "forecast probability")


def _realized_probability(probability: float, outcome: int) -> float:
    return probability if outcome == 1 else 1.0 - probability


def _require_probability(value: float, field_name: str) -> None:
    if not isfinite(value) or not 0.0 < value < 1.0:
        raise OpenModernModelError(f"{field_name} must be finite and strictly interior")
