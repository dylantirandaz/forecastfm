"""Sealed, dependency-free rich NBA baseline fitting and prediction."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from math import fsum, isfinite, sqrt
from pathlib import Path

from forecastfm.elo_residual import (
    DEFAULT_FIT_CONFIG,
    EloResidualFitConfig,
    EloResidualModel,
    EloResidualRow,
    fit_elo_residual,
)
from forecastfm.integrity import bytes_sha256, canonical_json, canonical_sha256
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_float,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_evaluation_gate import NBA_EVALUATION_FORECAST_SCHEMA_VERSION
from forecastfm.nba_feature_rows import NbaRichFeatureRow
from forecastfm.nba_resolutions import NBA_RESOLUTION_SCHEMA_VERSION, NbaResolution
from forecastfm.nba_rich import NBA_RICH_FEATURE_NAMES, NBA_RICH_SCHEMA_SHA256
from forecastfm.outcome_v2_metrics import BinaryForecast

NBA_RICH_BASELINE_SCHEMA_VERSION = 1
NBA_RICH_BASELINE_FORECAST_LOCK_SCHEMA_VERSION = 1

_ALGORITHM = "no_intercept_elo_offset_logistic_regression"
_LOSS = "mean_binary_cross_entropy"
_SCALING = "uncentered_rms_on_original_training_rows_only"
_SIDE_SWAP = (
    "derive_one swapped fitting row per original; at prediction reject gaps above 1e-12 "
    "and average original with swapped complement"
)
_FORECAST_LOCK_KIND = "forecastfm_nba_rich_baseline_forecast_lock"
_ZERO_RMS_SCALE = 1.0
MAXIMUM_SIDE_SWAP_GAP = 1e-12
_HASH_CHARACTERS = frozenset("0123456789abcdef")
_MODEL_KEYS = {
    "algorithm",
    "feature_names",
    "feature_schema_sha256",
    "fit_config",
    "intercept",
    "loss",
    "rms_scaling",
    "schema_version",
    "side_swap",
    "training",
    "weights",
}
_FIT_KEYS = {"l2_penalty", "learning_rate", "steps"}
_SCALING_KEYS = {
    "method",
    "scales",
    "zero_rms_feature_names",
    "zero_rms_scale",
}
_TRAINING_KEYS = {
    "feature_rows_jsonl_sha256",
    "question_id_seasons_sha256",
    "question_ids_sha256",
    "resolutions_jsonl_sha256",
    "row_count",
    "seasons",
}
_FORECAST_LOCK_KEYS = {
    "evaluation_feature_rows_jsonl_sha256",
    "evaluation_question_ids_sha256",
    "evaluation_seasons",
    "forecast_count",
    "forecast_jsonl_sha256",
    "kind",
    "model_sha256",
    "schema_version",
    "training_seasons",
}

type JsonObject = dict[str, object]


class NbaRichBaselineError(ValueError):
    """Raised when a rich baseline artifact or input violates its contract."""


@dataclass(frozen=True, slots=True)
class NbaRichBaselineModel:
    """Immutable model parameters and exact fitting provenance."""

    feature_names: tuple[str, ...]
    fit_config: EloResidualFitConfig
    rms_scales: tuple[float, ...]
    zero_rms_feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    training_feature_rows_jsonl_sha256: str
    training_resolutions_jsonl_sha256: str
    training_row_count: int
    training_question_ids_sha256: str
    training_question_id_seasons_sha256: str
    training_seasons: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.feature_names != NBA_RICH_FEATURE_NAMES:
            raise NbaRichBaselineError("model feature order differs from the rich schema")
        _require_model_numbers(
            self.rms_scales,
            self.zero_rms_feature_names,
            self.weights,
        )
        _require_sha256(
            self.training_feature_rows_jsonl_sha256,
            "training_feature_rows_jsonl_sha256",
        )
        _require_sha256(
            self.training_resolutions_jsonl_sha256,
            "training_resolutions_jsonl_sha256",
        )
        _require_positive_integer(self.training_row_count, "training_row_count")
        _require_sha256(self.training_question_ids_sha256, "training_question_ids_sha256")
        _require_sha256(
            self.training_question_id_seasons_sha256,
            "training_question_id_seasons_sha256",
        )
        _require_seasons(self.training_seasons, "training")

    @property
    def model_sha256(self) -> str:
        """Hash the complete canonical model and provenance payload."""
        return bytes_sha256(self.canonical_bytes)

    @property
    def canonical_bytes(self) -> bytes:
        """Return the exact canonical JSON representation."""
        return canonical_json(self.canonical_payload()).encode("utf-8")

    def canonical_payload(self) -> JsonObject:
        """Return the complete readable model and provenance record."""
        return {
            "schema_version": NBA_RICH_BASELINE_SCHEMA_VERSION,
            "algorithm": _ALGORITHM,
            "loss": _LOSS,
            "intercept": False,
            "feature_schema_sha256": NBA_RICH_SCHEMA_SHA256,
            "feature_names": list(self.feature_names),
            "fit_config": {
                "steps": self.fit_config.steps,
                "learning_rate": self.fit_config.learning_rate,
                "l2_penalty": self.fit_config.l2_penalty,
            },
            "rms_scaling": {
                "method": _SCALING,
                "zero_rms_scale": _ZERO_RMS_SCALE,
                "scales": list(self.rms_scales),
                "zero_rms_feature_names": list(self.zero_rms_feature_names),
            },
            "side_swap": _SIDE_SWAP,
            "weights": list(self.weights),
            "training": {
                "feature_rows_jsonl_sha256": (self.training_feature_rows_jsonl_sha256),
                "resolutions_jsonl_sha256": self.training_resolutions_jsonl_sha256,
                "row_count": self.training_row_count,
                "question_ids_sha256": self.training_question_ids_sha256,
                "question_id_seasons_sha256": (self.training_question_id_seasons_sha256),
                "seasons": list(self.training_seasons),
            },
        }

    def predict_probability(self, row: NbaRichFeatureRow) -> float:
        """Predict from one target-free row without accepting an answer object."""
        scaled = _scale_features(row.rich_features.vector, self.rms_scales)
        model = EloResidualModel(self.feature_names, self.weights)
        return model.predict_probability(row.elo_team_win_probability, scaled)


@dataclass(frozen=True, slots=True)
class NbaRichBaselineForecastLock:
    """Answer-free binding from one frozen model to its evaluation forecasts."""

    model_sha256: str
    evaluation_feature_rows_jsonl_sha256: str
    evaluation_question_ids_sha256: str
    training_seasons: tuple[int, ...]
    evaluation_seasons: tuple[int, ...]
    forecast_jsonl_sha256: str
    forecast_count: int

    def __post_init__(self) -> None:
        _require_sha256(self.model_sha256, "model_sha256")
        _require_sha256(
            self.evaluation_feature_rows_jsonl_sha256,
            "evaluation_feature_rows_jsonl_sha256",
        )
        _require_sha256(
            self.evaluation_question_ids_sha256,
            "evaluation_question_ids_sha256",
        )
        _require_seasons(self.training_seasons, "training")
        _require_seasons(self.evaluation_seasons, "evaluation")
        _require_strictly_later_seasons(self.training_seasons, self.evaluation_seasons)
        _require_sha256(self.forecast_jsonl_sha256, "forecast_jsonl_sha256")
        _require_positive_integer(self.forecast_count, "forecast_count")

    @property
    def lock_sha256(self) -> str:
        """Hash the complete answer-free forecast binding."""
        return bytes_sha256(self.canonical_bytes)

    @property
    def canonical_bytes(self) -> bytes:
        """Return exact canonical forecast-lock JSON bytes."""
        return canonical_json(self.canonical_payload()).encode("utf-8")

    def canonical_payload(self) -> JsonObject:
        """Return the complete readable forecast-lock record."""
        return {
            "schema_version": NBA_RICH_BASELINE_FORECAST_LOCK_SCHEMA_VERSION,
            "kind": _FORECAST_LOCK_KIND,
            "model_sha256": self.model_sha256,
            "evaluation_feature_rows_jsonl_sha256": (self.evaluation_feature_rows_jsonl_sha256),
            "evaluation_question_ids_sha256": self.evaluation_question_ids_sha256,
            "training_seasons": list(self.training_seasons),
            "evaluation_seasons": list(self.evaluation_seasons),
            "forecast_jsonl_sha256": self.forecast_jsonl_sha256,
            "forecast_count": self.forecast_count,
        }


def fit_nba_rich_baseline(
    feature_rows: Sequence[NbaRichFeatureRow],
    resolutions: Sequence[NbaResolution],
    config: EloResidualFitConfig = DEFAULT_FIT_CONFIG,
) -> NbaRichBaselineModel:
    """Fit cross-entropy only after exact causal feature/answer alignment."""
    rows = _require_feature_rows(tuple(feature_rows), "training")
    answers = tuple(resolutions)
    _require_resolution_alignment(rows, answers)
    scales, zero_rms_feature_names = _fit_rms_scales(rows)
    model_rows = tuple(
        model_row
        for row, answer in zip(rows, answers, strict=True)
        for model_row in _training_pair(row, answer, scales)
    )
    fitted = fit_elo_residual(model_rows, NBA_RICH_FEATURE_NAMES, config)
    return NbaRichBaselineModel(
        feature_names=NBA_RICH_FEATURE_NAMES,
        fit_config=config,
        rms_scales=scales,
        zero_rms_feature_names=zero_rms_feature_names,
        weights=fitted.weights,
        training_feature_rows_jsonl_sha256=_feature_rows_sha256(rows),
        training_resolutions_jsonl_sha256=_resolutions_sha256(answers),
        training_row_count=len(rows),
        training_question_ids_sha256=_question_ids_sha256(rows),
        training_question_id_seasons_sha256=_question_id_seasons_sha256(rows),
        training_seasons=tuple(sorted({row.season for row in rows})),
    )


def predict_nba_rich_baseline(
    model: NbaRichBaselineModel,
    evaluation_rows: Sequence[NbaRichFeatureRow],
) -> tuple[BinaryForecast, ...]:
    """Forecast answer-free rows in their sealed order, with no answer parameter."""
    rows = _require_feature_rows(tuple(evaluation_rows), "evaluation")
    evaluation_seasons = tuple(sorted({row.season for row in rows}))
    _require_strictly_later_seasons(model.training_seasons, evaluation_seasons)
    return tuple(_forecast(model, row) for row in rows)


def build_nba_rich_baseline_forecast_lock(
    model: NbaRichBaselineModel,
    evaluation_rows: Sequence[NbaRichFeatureRow],
    forecasts: Sequence[BinaryForecast],
) -> NbaRichBaselineForecastLock:
    """Bind answer-free input rows and exact forecasts to one frozen model."""
    rows = _require_feature_rows(tuple(evaluation_rows), "evaluation")
    predictions = tuple(forecasts)
    _require_forecast_alignment(rows, predictions)
    if predictions != predict_nba_rich_baseline(model, rows):
        raise NbaRichBaselineError("forecasts differ from deterministic model inference")
    seasons = tuple(sorted({row.season for row in rows}))
    _require_strictly_later_seasons(model.training_seasons, seasons)
    return NbaRichBaselineForecastLock(
        model_sha256=model.model_sha256,
        evaluation_feature_rows_jsonl_sha256=_feature_rows_sha256(rows),
        evaluation_question_ids_sha256=_question_ids_sha256(rows),
        training_seasons=model.training_seasons,
        evaluation_seasons=seasons,
        forecast_jsonl_sha256=_forecasts_sha256(predictions),
        forecast_count=len(predictions),
    )


def write_nba_rich_baseline_model(path: Path, model: NbaRichBaselineModel) -> str:
    """Create and fsync one canonical model artifact without replacement."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(model.canonical_bytes)
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError as error:
        raise NbaRichBaselineError("NBA rich baseline model already exists") from error
    except OSError as error:
        raise NbaRichBaselineError("cannot write NBA rich baseline model") from error
    return model.model_sha256


def read_nba_rich_baseline_model(
    path: Path,
    expected_sha256: str | None = None,
) -> NbaRichBaselineModel:
    """Read strict canonical bytes and optionally verify their expected digest."""
    try:
        value = path.read_bytes()
    except OSError as error:
        raise NbaRichBaselineError("cannot read NBA rich baseline model") from error
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "expected_sha256")
        if bytes_sha256(value) != expected_sha256:
            raise NbaRichBaselineError("NBA rich baseline model digest changed")
    try:
        text = value.decode("utf-8")
        payload = parse_json_object(text)
        model = _model_from_payload(payload)
    except (JsonFormatError, UnicodeError, ValueError) as error:
        raise NbaRichBaselineError("invalid NBA rich baseline model") from error
    if value != model.canonical_bytes:
        raise NbaRichBaselineError("NBA rich baseline model must use canonical JSON bytes")
    return model


def write_nba_rich_baseline_forecast_lock(
    path: Path,
    lock: NbaRichBaselineForecastLock,
) -> str:
    """Create and fsync one canonical forecast lock without replacement."""
    _write_canonical_bytes(path, lock.canonical_bytes, "NBA rich baseline forecast lock")
    return lock.lock_sha256


def read_nba_rich_baseline_forecast_lock(
    path: Path,
    expected_sha256: str | None = None,
) -> NbaRichBaselineForecastLock:
    """Read a strict forecast lock and optionally verify its expected digest."""
    value = _read_bytes(path, "NBA rich baseline forecast lock")
    _require_expected_sha256(value, expected_sha256, "forecast lock")
    try:
        text = value.decode("utf-8")
        payload = parse_json_object(text)
        lock = _forecast_lock_from_payload(payload)
    except (JsonFormatError, UnicodeError, ValueError) as error:
        raise NbaRichBaselineError("invalid NBA rich baseline forecast lock") from error
    if value != lock.canonical_bytes:
        raise NbaRichBaselineError("NBA rich baseline forecast lock must use canonical JSON bytes")
    return lock


def _forecast(model: NbaRichBaselineModel, row: NbaRichFeatureRow) -> BinaryForecast:
    original = model.predict_probability(row)
    complemented_swap = 1.0 - model.predict_probability(row.side_swap())
    if abs(original - complemented_swap) > MAXIMUM_SIDE_SWAP_GAP:
        raise NbaRichBaselineError("rich baseline side-swap gap exceeds the frozen limit")
    probability = fsum((original, complemented_swap)) / 2.0
    if not 0.0 < probability < 1.0:
        return BinaryForecast(
            question_id=row.question_id,
            team_probability=None,
            failure_reason="rich baseline produced a non-interior probability",
        )
    return BinaryForecast(row.question_id, probability)


def _fit_rms_scales(
    rows: Sequence[NbaRichFeatureRow],
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    row_count = len(rows)
    scales: list[float] = []
    zero_rms_names: list[str] = []
    for index, name in enumerate(NBA_RICH_FEATURE_NAMES):
        squared_sum = fsum(row.rich_features.vector[index] ** 2 for row in rows)
        rms = sqrt(squared_sum / row_count)
        if rms == 0.0:
            zero_rms_names.append(name)
        scales.append(_ZERO_RMS_SCALE if rms == 0.0 else rms)
    return tuple(scales), tuple(zero_rms_names)


def _scale_features(
    features: tuple[float, ...],
    scales: tuple[float, ...],
) -> tuple[float, ...]:
    if len(features) != len(scales):
        raise NbaRichBaselineError("feature and RMS scale counts differ")
    return tuple(value / scale for value, scale in zip(features, scales, strict=True))


def _training_pair(
    row: NbaRichFeatureRow,
    resolution: NbaResolution,
    scales: tuple[float, ...],
) -> tuple[EloResidualRow, EloResidualRow]:
    original_features = _scale_features(row.rich_features.vector, scales)
    swapped = row.side_swap()
    swapped_features = _scale_features(swapped.rich_features.vector, scales)
    outcome = int(resolution.team_won)
    return (
        EloResidualRow(
            row.question_id,
            row.elo_team_win_probability,
            original_features,
            outcome,
        ),
        EloResidualRow(
            swapped.question_id,
            swapped.elo_team_win_probability,
            swapped_features,
            1 - outcome,
        ),
    )


def _require_feature_rows(
    rows: tuple[NbaRichFeatureRow, ...],
    purpose: str,
) -> tuple[NbaRichFeatureRow, ...]:
    if not rows:
        raise NbaRichBaselineError(f"{purpose} feature rows must not be empty")
    question_ids = tuple(row.question_id for row in rows)
    if any(question_id.endswith(SIDE_SWAP_SUFFIX) for question_id in question_ids):
        raise NbaRichBaselineError(f"{purpose} feature rows must contain original games only")
    if len(set(question_ids)) != len(question_ids):
        raise NbaRichBaselineError(f"{purpose} question IDs must be unique")
    if rows != tuple(sorted(rows, key=lambda row: (row.forecast_cutoff, row.question_id))):
        raise NbaRichBaselineError(f"{purpose} feature rows must be in chronological order")
    if any(_nba_season(row.forecast_cutoff) != row.season for row in rows):
        raise NbaRichBaselineError(f"{purpose} row season disagrees with its forecast cutoff")
    return rows


def _require_resolution_alignment(
    rows: tuple[NbaRichFeatureRow, ...],
    resolutions: tuple[NbaResolution, ...],
) -> None:
    if len(resolutions) != len(rows):
        raise NbaRichBaselineError("each training feature row requires one resolution")
    row_ids = tuple(row.question_id for row in rows)
    resolution_ids = tuple(resolution.question_id for resolution in resolutions)
    if resolution_ids != row_ids:
        raise NbaRichBaselineError("resolution IDs or order differ from training feature rows")
    for row, resolution in zip(rows, resolutions, strict=True):
        if (
            resolution.source_game_id,
            resolution.team_id,
            resolution.opponent_id,
            resolution.site,
        ) != (
            row.source_game_id,
            row.team_id,
            row.opponent_id,
            row.site,
        ):
            raise NbaRichBaselineError("resolution identity differs from its training feature row")
        if resolution.resolved_at <= row.scheduled_tipoff:
            raise NbaRichBaselineError("resolution must postdate its frozen scheduled tipoff")


def _feature_rows_sha256(rows: Sequence[NbaRichFeatureRow]) -> str:
    value = "".join(f"{canonical_json(row.canonical_payload())}\n" for row in rows)
    return bytes_sha256(value.encode("utf-8"))


def _question_ids_sha256(rows: Sequence[NbaRichFeatureRow]) -> str:
    return canonical_sha256([row.question_id for row in rows])


def _question_id_seasons_sha256(rows: Sequence[NbaRichFeatureRow]) -> str:
    return canonical_sha256(
        [{"question_id": row.question_id, "season": row.season} for row in rows]
    )


def _resolutions_sha256(resolutions: Sequence[NbaResolution]) -> str:
    value = "".join(f"{canonical_json(_resolution_payload(row))}\n" for row in resolutions)
    return bytes_sha256(value.encode("utf-8"))


def _forecasts_sha256(forecasts: Sequence[BinaryForecast]) -> str:
    value = "".join(f"{canonical_json(_forecast_payload(row))}\n" for row in forecasts)
    return bytes_sha256(value.encode("utf-8"))


def _forecast_payload(forecast: BinaryForecast) -> JsonObject:
    return {
        "schema_version": NBA_EVALUATION_FORECAST_SCHEMA_VERSION,
        "question_id": forecast.question_id,
        "team_probability": forecast.team_probability,
        "failure_reason": forecast.failure_reason,
    }


def _resolution_payload(resolution: NbaResolution) -> JsonObject:
    return {
        "schema_version": NBA_RESOLUTION_SCHEMA_VERSION,
        "question_id": resolution.question_id,
        "source_game_id": resolution.source_game_id,
        "team_id": resolution.team_id,
        "opponent_id": resolution.opponent_id,
        "site": resolution.site,
        "team_score": resolution.team_score,
        "opponent_score": resolution.opponent_score,
        "resolved_at": _utc_text(resolution.resolved_at),
        "source_id": resolution.source_id,
        "snapshot_metadata_sha256": resolution.snapshot_metadata_sha256,
    }


def _model_from_payload(payload: Mapping[str, object]) -> NbaRichBaselineModel:
    require_exact_keys(payload, _MODEL_KEYS, "NBA rich baseline model")
    _require_contract_constants(payload)
    fit = require_object(required_field(payload, "fit_config"), "fit_config")
    scaling = require_object(required_field(payload, "rms_scaling"), "rms_scaling")
    training = require_object(required_field(payload, "training"), "training")
    require_exact_keys(fit, _FIT_KEYS, "fit_config")
    require_exact_keys(scaling, _SCALING_KEYS, "rms_scaling")
    require_exact_keys(training, _TRAINING_KEYS, "training")
    _require_string_value(scaling, "method", _SCALING)
    if require_float(required_field(scaling, "zero_rms_scale"), "zero_rms_scale") != 1.0:
        raise JsonFormatError("zero_rms_scale differs from the frozen policy")
    return NbaRichBaselineModel(
        feature_names=_string_tuple(payload, "feature_names"),
        fit_config=_fit_config_from_payload(fit),
        rms_scales=_float_tuple(scaling, "scales"),
        zero_rms_feature_names=_string_tuple(scaling, "zero_rms_feature_names"),
        weights=_float_tuple(payload, "weights"),
        training_feature_rows_jsonl_sha256=_string_field(
            training,
            "feature_rows_jsonl_sha256",
        ),
        training_resolutions_jsonl_sha256=_string_field(
            training,
            "resolutions_jsonl_sha256",
        ),
        training_row_count=_integer_field(training, "row_count"),
        training_question_ids_sha256=_string_field(training, "question_ids_sha256"),
        training_question_id_seasons_sha256=_string_field(
            training,
            "question_id_seasons_sha256",
        ),
        training_seasons=_integer_tuple(training, "seasons"),
    )


def _forecast_lock_from_payload(
    payload: Mapping[str, object],
) -> NbaRichBaselineForecastLock:
    require_exact_keys(payload, _FORECAST_LOCK_KEYS, "NBA rich baseline forecast lock")
    version = _integer_field(payload, "schema_version")
    if version != NBA_RICH_BASELINE_FORECAST_LOCK_SCHEMA_VERSION:
        raise JsonFormatError("unsupported NBA rich baseline forecast-lock schema")
    _require_string_value(payload, "kind", _FORECAST_LOCK_KIND)
    return NbaRichBaselineForecastLock(
        model_sha256=_string_field(payload, "model_sha256"),
        evaluation_feature_rows_jsonl_sha256=_string_field(
            payload,
            "evaluation_feature_rows_jsonl_sha256",
        ),
        evaluation_question_ids_sha256=_string_field(
            payload,
            "evaluation_question_ids_sha256",
        ),
        training_seasons=_integer_tuple(payload, "training_seasons"),
        evaluation_seasons=_integer_tuple(payload, "evaluation_seasons"),
        forecast_jsonl_sha256=_string_field(payload, "forecast_jsonl_sha256"),
        forecast_count=_integer_field(payload, "forecast_count"),
    )


def _require_contract_constants(payload: Mapping[str, object]) -> None:
    if _integer_field(payload, "schema_version") != NBA_RICH_BASELINE_SCHEMA_VERSION:
        raise JsonFormatError("unsupported NBA rich baseline schema version")
    _require_string_value(payload, "algorithm", _ALGORITHM)
    _require_string_value(payload, "loss", _LOSS)
    _require_string_value(payload, "feature_schema_sha256", NBA_RICH_SCHEMA_SHA256)
    _require_string_value(payload, "side_swap", _SIDE_SWAP)
    if required_field(payload, "intercept") is not False:
        raise JsonFormatError("NBA rich baseline must not have an intercept")


def _fit_config_from_payload(payload: Mapping[str, object]) -> EloResidualFitConfig:
    return EloResidualFitConfig(
        steps=_integer_field(payload, "steps"),
        learning_rate=require_float(required_field(payload, "learning_rate"), "learning_rate"),
        l2_penalty=require_float(required_field(payload, "l2_penalty"), "l2_penalty"),
    )


def _require_model_numbers(
    scales: tuple[float, ...],
    zero_rms_feature_names: tuple[str, ...],
    weights: tuple[float, ...],
) -> None:
    if len(scales) != len(NBA_RICH_FEATURE_NAMES):
        raise NbaRichBaselineError("model must contain one RMS scale per rich feature")
    if not all(isfinite(scale) and scale > 0.0 for scale in scales):
        raise NbaRichBaselineError("RMS scales must be positive and finite")
    expected_zero_names = tuple(
        name for name in NBA_RICH_FEATURE_NAMES if name in zero_rms_feature_names
    )
    if zero_rms_feature_names != expected_zero_names:
        raise NbaRichBaselineError("zero-RMS feature names must follow the rich schema")
    scale_by_name = dict(zip(NBA_RICH_FEATURE_NAMES, scales, strict=True))
    if any(scale_by_name[name] != _ZERO_RMS_SCALE for name in zero_rms_feature_names):
        raise NbaRichBaselineError("zero-RMS features must use the explicit fallback scale")
    if len(weights) != len(NBA_RICH_FEATURE_NAMES):
        raise NbaRichBaselineError("model must contain one weight per rich feature")
    if not all(isfinite(weight) for weight in weights):
        raise NbaRichBaselineError("model weights must be finite")


def _require_forecast_alignment(
    rows: Sequence[NbaRichFeatureRow],
    forecasts: Sequence[BinaryForecast],
) -> None:
    row_ids = tuple(row.question_id for row in rows)
    forecast_ids = tuple(forecast.question_id for forecast in forecasts)
    if forecast_ids != row_ids:
        raise NbaRichBaselineError("forecast IDs or order differ from evaluation feature rows")


def _require_seasons(seasons: tuple[int, ...], purpose: str) -> None:
    if not seasons or seasons != tuple(sorted(set(seasons))):
        raise NbaRichBaselineError(f"{purpose} seasons must be unique and increasing")
    if any(isinstance(season, bool) or season <= 0 for season in seasons):
        raise NbaRichBaselineError(f"{purpose} seasons must be positive integers")


def _require_strictly_later_seasons(
    training_seasons: tuple[int, ...],
    evaluation_seasons: tuple[int, ...],
) -> None:
    _require_seasons(training_seasons, "training")
    _require_seasons(evaluation_seasons, "evaluation")
    if min(evaluation_seasons) <= max(training_seasons):
        raise NbaRichBaselineError(
            "every evaluation season must be strictly later than every training season"
        )


def _require_positive_integer(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NbaRichBaselineError(f"{field_name} must be a positive integer")


def _nba_season(value: datetime) -> int:
    return value.year + 1 if value.month >= 7 else value.year


def _string_tuple(payload: Mapping[str, object], field_name: str) -> tuple[str, ...]:
    values = require_list(required_field(payload, field_name), field_name)
    return tuple(require_string(value, field_name) for value in values)


def _float_tuple(payload: Mapping[str, object], field_name: str) -> tuple[float, ...]:
    values = require_list(required_field(payload, field_name), field_name)
    return tuple(require_float(value, field_name) for value in values)


def _integer_tuple(payload: Mapping[str, object], field_name: str) -> tuple[int, ...]:
    values = require_list(required_field(payload, field_name), field_name)
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise JsonFormatError(f"{field_name} must contain integers")
        result.append(value)
    return tuple(result)


def _string_field(payload: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(payload, field_name), field_name)


def _integer_field(payload: Mapping[str, object], field_name: str) -> int:
    value = required_field(payload, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise JsonFormatError(f"{field_name} must be an integer")
    return value


def _require_string_value(
    payload: Mapping[str, object],
    field_name: str,
    expected: str,
) -> None:
    if _string_field(payload, field_name) != expected:
        raise JsonFormatError(f"{field_name} differs from the frozen contract")


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in _HASH_CHARACTERS for character in value):
        raise NbaRichBaselineError(f"{field_name} must be a lowercase SHA-256 digest")


def _write_canonical_bytes(path: Path, value: bytes, artifact_name: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError as error:
        raise NbaRichBaselineError(f"{artifact_name} already exists") from error
    except OSError as error:
        raise NbaRichBaselineError(f"cannot write {artifact_name}") from error


def _read_bytes(path: Path, artifact_name: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise NbaRichBaselineError(f"cannot read {artifact_name}") from error


def _require_expected_sha256(
    value: bytes,
    expected_sha256: str | None,
    artifact_name: str,
) -> None:
    if expected_sha256 is None:
        return
    _require_sha256(expected_sha256, "expected_sha256")
    if bytes_sha256(value) != expected_sha256:
        raise NbaRichBaselineError(f"NBA rich baseline {artifact_name} digest changed")


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
