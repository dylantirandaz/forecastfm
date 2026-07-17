"""Fit the frozen open-modern models and write one immutable validation lock."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import cast

from forecastfm.integrity import canonical_sha256, file_sha256
from forecastfm.open_modern import (
    DEVELOPMENT_COLUMNS,
    OPEN_MODERN_DEVELOPMENT_IDS_SHA256,
    OPEN_MODERN_DEVELOPMENT_SHA256,
    OPEN_MODERN_EXPOSURE_SHA256,
    OPEN_MODERN_PROTOCOL_SHA256,
    OPEN_MODERN_SOURCE_SEAL_SHA256,
    OPEN_MODERN_TEST_COUNT,
    OPEN_MODERN_TEST_IDS_SHA256,
    OPEN_MODERN_TEST_INPUTS_SHA256,
    OPEN_MODERN_TRAIN_COUNT,
    OPEN_MODERN_TRAIN_IDS_SHA256,
    OPEN_MODERN_VALIDATION_COUNT,
    OPEN_MODERN_VALIDATION_IDS_SHA256,
    TRAIN_SEASONS,
    VALIDATION_SEASONS,
    require_open_modern_development,
)
from forecastfm.open_modern_features import (
    OPEN_MODERN_CAUSAL_FEATURE_CONTRACT_SHA256,
    OPEN_MODERN_CAUSAL_FEATURE_NAMES,
    RAPTOR_SOURCE_COMMIT,
    RAPTOR_SOURCE_GIT_BLOB,
    RAPTOR_SOURCE_SHA256,
    OpenModernFeatureRow,
    OpenModernInputGame,
    build_open_modern_features,
    load_open_modern_feature_inputs,
    load_prior_season_raptor,
)
from forecastfm.open_modern_model import (
    BOOTSTRAP_BLOCK_DAYS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    ECE_BIN_COUNT,
    FIT_LEARNING_RATE,
    FIT_STEPS,
    FORECAST_L2_PENALTY,
    FORECAST_SPEC,
    MAXIMUM_SIDE_SWAP_GAP,
    ONE_SIDED_ALPHA,
    OPEN_MODERN_MODEL_CONTRACT_SHA256,
    RECALIBRATION_SPEC,
    OpenModernBaselineComparison,
    OpenModernCandidateFit,
    OpenModernCandidateSpec,
    OpenModernResolvedRow,
    OpenModernValidationMetrics,
    OpenModernValidationResult,
    fit_open_modern_validation,
)
from forecastfm.publication import require_paths_at_head, require_published_head

PROJECT_ROOT = Path(__file__).parents[1]
DEVELOPMENT_PATH = PROJECT_ROOT / "data/processed/outcome_v2_open_modern/development.csv"
RAPTOR_PATH = PROJECT_ROOT / "data/raw/outcome_v2_open_modern/modern_RAPTOR_by_team.csv"
PROTOCOL_PATH = PROJECT_ROOT / "evaluation/outcome_v2_open_modern/protocol.json"
EXPOSURE_PATH = PROJECT_ROOT / "evaluation/outcome_v2_open_modern/EXPOSURE.md"
SOURCE_SEAL_PATH = PROJECT_ROOT / "evaluation/outcome_v2_open_modern/source_seal.json"
VALIDATION_LOCK_PATH = PROJECT_ROOT / "evaluation/outcome_v2_open_modern/validation_lock.json"
REPOSITORY_URL = "https://github.com/dylantirandaz/forecastfm.git"

EXPECTED_TRAIN_COUNT = OPEN_MODERN_TRAIN_COUNT
EXPECTED_VALIDATION_COUNT = OPEN_MODERN_VALIDATION_COUNT
EXPECTED_TRAIN_IDS_SHA256 = OPEN_MODERN_TRAIN_IDS_SHA256
EXPECTED_VALIDATION_IDS_SHA256 = OPEN_MODERN_VALIDATION_IDS_SHA256

LOCKED_CODE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "evaluation/outcome_v2_open_modern/protocol.json",
    "evaluation/outcome_v2_open_modern/EXPOSURE.md",
    "evaluation/outcome_v2_open_modern/source_seal.json",
    "src/forecastfm/integrity.py",
    "src/forecastfm/elo_residual.py",
    "src/forecastfm/open_modern.py",
    "src/forecastfm/open_modern_features.py",
    "src/forecastfm/open_modern_model.py",
    "src/forecastfm/publication.py",
    "examples/run_open_modern_development.py",
    "tests/test_open_modern.py",
    "tests/test_open_modern_features.py",
    "tests/test_open_modern_model.py",
    "tests/test_run_open_modern_development.py",
)


class OpenModernDevelopmentError(ValueError):
    """Raised when the frozen development run cannot be reproduced exactly."""


@dataclass(frozen=True, slots=True)
class DevelopmentLabel:
    """One actual-winner label loaded only after causal features are built."""

    question_id: str
    outcome: int


def load_development_labels(path: Path) -> tuple[DevelopmentLabel, ...]:
    """Load complementary actual-winner indicators from the verified development file."""
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != OPEN_MODERN_DEVELOPMENT_SHA256:
        raise OpenModernDevelopmentError("development label bytes do not match")
    labels: list[DevelopmentLabel] = []
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OpenModernDevelopmentError("development labels are not UTF-8") from error
    with io.StringIO(text, newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != DEVELOPMENT_COLUMNS:
            raise OpenModernDevelopmentError("development columns do not match")
        for line_number, row in enumerate(reader, start=2):
            question_id = _required(row, "game_id", line_number)
            team1_outcome = _binary(row, "prob1_outcome", line_number)
            team2_outcome = _binary(row, "prob2_outcome", line_number)
            if team1_outcome + team2_outcome != 1:
                raise OpenModernDevelopmentError(
                    f"line {line_number}: outcomes must be complementary"
                )
            labels.append(DevelopmentLabel(question_id, team1_outcome))
    if len({label.question_id for label in labels}) != len(labels):
        raise OpenModernDevelopmentError("development label IDs must be unique")
    return tuple(labels)


def build_resolved_development_rows(
    development_path: Path,
    raptor_path: Path,
    source_seal_path: Path,
    protocol_path: Path,
    exposure_path: Path,
) -> tuple[OpenModernResolvedRow, ...]:
    """Build outcome-free features first, then join the separately loaded labels."""
    inputs = load_open_modern_feature_inputs(
        development_path,
        seal_path=source_seal_path,
        protocol_path=protocol_path,
        exposure_path=exposure_path,
    )
    raptor = load_prior_season_raptor(raptor_path, max_allowed_season=max(TRAIN_SEASONS))
    feature_rows = build_open_modern_features(inputs, raptor)
    labels = load_development_labels(development_path)
    require_open_modern_development(
        development_path,
        source_seal_path,
        protocol_path,
        exposure_path,
    )
    return _join_resolved_rows(inputs, feature_rows, labels)


def build_validation_lock(
    rows: Sequence[OpenModernResolvedRow],
    result: OpenModernValidationResult,
    git_revision: str,
) -> dict[str, object]:
    """Build the complete immutable record without touching holdout inputs or answers."""
    source_order = tuple(rows)
    source_training = tuple(row for row in source_order if row.season in TRAIN_SEASONS)
    source_validation = tuple(row for row in source_order if row.season in VALIDATION_SEASONS)
    if (
        len(source_training) != EXPECTED_TRAIN_COUNT
        or len(source_validation) != EXPECTED_VALIDATION_COUNT
    ):
        raise OpenModernDevelopmentError("development split counts do not match the source seal")
    if canonical_sha256([row.question_id for row in source_training]) != EXPECTED_TRAIN_IDS_SHA256:
        raise OpenModernDevelopmentError("training IDs do not match the source seal")
    if (
        canonical_sha256([row.question_id for row in source_validation])
        != EXPECTED_VALIDATION_IDS_SHA256
    ):
        raise OpenModernDevelopmentError("validation IDs do not match the source seal")

    ordered = tuple(sorted(rows, key=_resolved_sort_key))
    training = tuple(row for row in ordered if row.season in TRAIN_SEASONS)
    validation = tuple(row for row in ordered if row.season in VALIDATION_SEASONS)
    if len(git_revision) != 40 or any(
        character not in "0123456789abcdef" for character in git_revision
    ):
        raise OpenModernDevelopmentError("Git revision must be a lowercase 40-character SHA")

    _require_validation_result(result, validation)
    recalibration = _candidate_dict(result.recalibration, validation)
    forecast = _candidate_dict(result.forecast, validation)
    gate_failures = _gate_failures(result)
    advances_to_holdout = not gate_failures
    return {
        "schema_version": 1,
        "status": (
            "validation_passed_holdout_locked"
            if advances_to_holdout
            else "validation_failed_holdout_closed"
        ),
        "git_revision": git_revision,
        "code_sha256": {path: file_sha256(PROJECT_ROOT / path) for path in LOCKED_CODE_PATHS},
        "inputs": {
            "protocol_sha256": OPEN_MODERN_PROTOCOL_SHA256,
            "exposure_sha256": OPEN_MODERN_EXPOSURE_SHA256,
            "source_seal_sha256": OPEN_MODERN_SOURCE_SEAL_SHA256,
            "development_sha256": OPEN_MODERN_DEVELOPMENT_SHA256,
            "development_ordered_ids_sha256": OPEN_MODERN_DEVELOPMENT_IDS_SHA256,
            "training_source_ordered_ids_sha256": EXPECTED_TRAIN_IDS_SHA256,
            "validation_source_ordered_ids_sha256": EXPECTED_VALIDATION_IDS_SHA256,
            "raptor": {
                "commit": RAPTOR_SOURCE_COMMIT,
                "git_blob_sha": RAPTOR_SOURCE_GIT_BLOB,
                "sha256": RAPTOR_SOURCE_SHA256,
                "maximum_allowed_season": max(TRAIN_SEASONS),
            },
        },
        "contracts": {
            "causal_features_sha256": OPEN_MODERN_CAUSAL_FEATURE_CONTRACT_SHA256,
            "model_contract_sha256": OPEN_MODERN_MODEL_CONTRACT_SHA256,
        },
        "fit": {
            "steps": FIT_STEPS,
            "learning_rate": FIT_LEARNING_RATE,
            "forecast_l2_penalty": FORECAST_L2_PENALTY,
            "feature_names": list(OPEN_MODERN_CAUSAL_FEATURE_NAMES),
            "scaler": {
                "kind": "training_only_uncentered_rms",
                "scales": list(result.scaler.scales),
                "sha256": canonical_sha256(list(result.scaler.scales)),
            },
        },
        "training": _cohort_dict(training),
        "validation": {
            **_cohort_dict(validation),
            "raw_source_metrics": _metrics_dict(result.raw_source_metrics),
            "recalibration": recalibration,
            "forecast": forecast,
            "forecast_vs_raw_source": _comparison_dict(result.forecast_vs_raw_source),
            "forecast_vs_recalibration": _comparison_dict(result.forecast_vs_recalibration),
            "maximum_side_swap_gap": result.maximum_side_swap_gap,
            "advances_to_holdout": advances_to_holdout,
            "gate_failures": gate_failures,
        },
        "evaluation_policy": {
            "primary_metric": "mean log loss",
            "ece_bin_count": ECE_BIN_COUNT,
            "bootstrap_block_days": BOOTSTRAP_BLOCK_DAYS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "one_sided_alpha": ONE_SIDED_ALPHA,
            "maximum_side_swap_gap": MAXIMUM_SIDE_SWAP_GAP,
        },
        "holdout": {
            "seasons": [2021, 2022],
            "row_count": OPEN_MODERN_TEST_COUNT,
            "target_free_inputs_sha256": OPEN_MODERN_TEST_INPUTS_SHA256,
            "ordered_ids_sha256": OPEN_MODERN_TEST_IDS_SHA256,
            "inference_policy": (
                "use the locked forecast weights in one fixed chronological full-file pass; "
                "never fit or adapt row by row"
            ),
            "predictions_written": False,
            "answers_opened_for_scoring": False,
            "scored": False,
        },
    }


def main() -> None:
    """Require a clean committed tree, run validation once, and lock the result."""
    publication = require_published_head(PROJECT_ROOT, REPOSITORY_URL)
    locked_paths = tuple(PROJECT_ROOT / path for path in LOCKED_CODE_PATHS)
    require_paths_at_head(PROJECT_ROOT, publication.commit, locked_paths)
    rows = build_resolved_development_rows(
        DEVELOPMENT_PATH,
        RAPTOR_PATH,
        SOURCE_SEAL_PATH,
        PROTOCOL_PATH,
        EXPOSURE_PATH,
    )
    result = fit_open_modern_validation(rows)
    require_paths_at_head(PROJECT_ROOT, publication.commit, locked_paths)
    lock = build_validation_lock(rows, result, publication.commit)
    require_paths_at_head(PROJECT_ROOT, publication.commit, locked_paths)
    write_validation_lock(VALIDATION_LOCK_PATH, lock)
    print(f"Forecast: {result.forecast.spec.candidate_id}")
    print(f"Validation log loss: {result.forecast.metrics.mean_log_loss:.8f}")
    print(f"Advances to holdout: {result.advances_to_holdout}")
    print(f"Validation lock SHA-256: {file_sha256(VALIDATION_LOCK_PATH)}")


def _join_resolved_rows(
    inputs: Sequence[OpenModernInputGame],
    feature_rows: Sequence[OpenModernFeatureRow],
    labels: Sequence[DevelopmentLabel],
) -> tuple[OpenModernResolvedRow, ...]:
    input_by_id = {game.game_id: game for game in inputs}
    feature_by_id = {row.game_id: row for row in feature_rows}
    label_by_id = {label.question_id: label.outcome for label in labels}
    if set(input_by_id) != set(feature_by_id) or set(input_by_id) != set(label_by_id):
        raise OpenModernDevelopmentError("features, inputs, and labels do not cover exact IDs")
    rows = tuple(
        OpenModernResolvedRow(
            question_id=game.game_id,
            season=game.season,
            game_date=game.game_date,
            source_probability=game.prob1,
            features=feature_by_id[game.game_id].features.vector,
            outcome=label_by_id[game.game_id],
        )
        for game in inputs
    )
    return rows


def _candidate_dict(
    candidate: OpenModernCandidateFit,
    validation: Sequence[OpenModernResolvedRow],
) -> dict[str, object]:
    forecasts = [
        {"question_id": row.question_id, "probability": probability}
        for row, probability in zip(
            validation,
            candidate.validation_probabilities,
            strict=True,
        )
    ]
    return {
        "candidate_id": candidate.spec.candidate_id,
        "name": candidate.spec.name,
        "feature_names": list(candidate.spec.feature_names),
        "l2_penalty": candidate.spec.l2_penalty,
        "weights": list(candidate.model.weights),
        "model_sha256": canonical_sha256(
            {
                "feature_names": list(candidate.model.feature_names),
                "weights": list(candidate.model.weights),
            }
        ),
        "validation_forecasts_sha256": canonical_sha256(forecasts),
        "metrics": _metrics_dict(candidate.metrics),
    }


def _require_validation_result(
    result: OpenModernValidationResult,
    validation: Sequence[OpenModernResolvedRow],
) -> None:
    """Reject any result that differs from the one predeclared experiment."""
    count = len(validation)
    if result.scaler.feature_names != OPEN_MODERN_CAUSAL_FEATURE_NAMES:
        raise OpenModernDevelopmentError("validation scaler feature order differs")
    if result.raw_source_metrics.count != count:
        raise OpenModernDevelopmentError("raw-source metric count differs")

    _require_fixed_fit("recalibration", result.recalibration, RECALIBRATION_SPEC, count)
    _require_fixed_fit("forecast", result.forecast, FORECAST_SPEC, count)

    expected_blocks = len(
        {row.game_date - timedelta(days=row.game_date.weekday()) for row in validation}
    )
    _require_comparison(
        result.forecast_vs_raw_source,
        "raw_source_probability",
        count,
        expected_blocks,
    )
    _require_comparison(
        result.forecast_vs_recalibration,
        "fixed_training_recalibration",
        count,
        expected_blocks,
    )
    _require_finite_result(result)
    if result.advances_to_holdout != (not _gate_failures(result)):
        raise OpenModernDevelopmentError("validation gate decision is internally inconsistent")


def _require_fixed_fit(
    name: str,
    candidate: OpenModernCandidateFit,
    expected_spec: OpenModernCandidateSpec,
    count: int,
) -> None:
    if candidate.spec != expected_spec:
        raise OpenModernDevelopmentError(f"{name} specification differs")
    if candidate.model.feature_names != candidate.spec.feature_names:
        raise OpenModernDevelopmentError(f"{name} model features differ")
    if len(candidate.validation_probabilities) != count:
        raise OpenModernDevelopmentError(f"{name} forecast count differs")
    if candidate.metrics.count != count:
        raise OpenModernDevelopmentError(f"{name} metric count differs")


def _require_comparison(
    comparison: OpenModernBaselineComparison,
    expected_name: str,
    expected_games: int,
    expected_blocks: int,
) -> None:
    if comparison.baseline_name != expected_name:
        raise OpenModernDevelopmentError("validation comparison baseline differs")
    if comparison.game_count != expected_games:
        raise OpenModernDevelopmentError("validation comparison game count differs")
    if comparison.calendar_block_count != expected_blocks:
        raise OpenModernDevelopmentError("validation calendar-block count differs")


def _require_finite_result(result: OpenModernValidationResult) -> None:
    numeric_values = (
        *result.scaler.scales,
        *_metric_values(result.raw_source_metrics),
        *result.recalibration.model.weights,
        *result.recalibration.validation_probabilities,
        *_metric_values(result.recalibration.metrics),
        *result.forecast.model.weights,
        *result.forecast.validation_probabilities,
        *_metric_values(result.forecast.metrics),
        result.forecast_vs_raw_source.mean_baseline_relative_log_score,
        result.forecast_vs_raw_source.lower_one_sided_95,
        result.forecast_vs_recalibration.mean_baseline_relative_log_score,
        result.forecast_vs_recalibration.lower_one_sided_95,
        result.maximum_side_swap_gap,
    )
    if not all(isfinite(value) for value in numeric_values):
        raise OpenModernDevelopmentError("validation result contains a non-finite number")
    probabilities = (
        *result.recalibration.validation_probabilities,
        *result.forecast.validation_probabilities,
    )
    if not all(0.0 < probability < 1.0 for probability in probabilities):
        raise OpenModernDevelopmentError("validation probabilities must be strictly interior")
    if result.maximum_side_swap_gap < 0.0:
        raise OpenModernDevelopmentError("side-swap gap must be non-negative")


def _metric_values(metrics: OpenModernValidationMetrics) -> tuple[float, float, float]:
    return (
        metrics.mean_log_loss,
        metrics.mean_brier,
        metrics.expected_calibration_error,
    )


def _cohort_dict(rows: Sequence[OpenModernResolvedRow]) -> dict[str, object]:
    return {
        "seasons": sorted({row.season for row in rows}),
        "row_count": len(rows),
        "ordered_ids_sha256": canonical_sha256([row.question_id for row in rows]),
        "rows_sha256": canonical_sha256(
            [
                {
                    "question_id": row.question_id,
                    "season": row.season,
                    "date": row.game_date.isoformat(),
                    "source_probability": row.source_probability,
                    "features": list(row.features),
                    "outcome": row.outcome,
                }
                for row in rows
            ]
        ),
    }


def _metrics_dict(metrics: OpenModernValidationMetrics) -> dict[str, object]:
    return {
        "count": metrics.count,
        "mean_log_loss": metrics.mean_log_loss,
        "mean_brier": metrics.mean_brier,
        "expected_calibration_error": metrics.expected_calibration_error,
    }


def _comparison_dict(comparison: OpenModernBaselineComparison) -> dict[str, object]:
    return {
        "baseline_name": comparison.baseline_name,
        "game_count": comparison.game_count,
        "calendar_block_count": comparison.calendar_block_count,
        "mean_baseline_relative_log_score": comparison.mean_baseline_relative_log_score,
        "lower_one_sided_95": comparison.lower_one_sided_95,
    }


def _gate_failures(result: OpenModernValidationResult) -> list[str]:
    failures: list[str] = []
    comparisons = (
        result.forecast_vs_raw_source,
        result.forecast_vs_recalibration,
    )
    for comparison in comparisons:
        if comparison.mean_baseline_relative_log_score <= 0.0:
            failures.append(f"non-positive mean versus {comparison.baseline_name}")
        if comparison.lower_one_sided_95 <= 0.0:
            failures.append(f"non-positive lower bound versus {comparison.baseline_name}")
    if result.maximum_side_swap_gap > MAXIMUM_SIDE_SWAP_GAP:
        failures.append("side-swap gap exceeds threshold")
    return failures


def write_validation_lock(path: Path, value: object) -> None:
    """Publish a complete finite validation lock atomically and exactly once."""
    _require_finite_lock_numbers(value)
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as error:
        raise OpenModernDevelopmentError("validation lock is not serializable") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise OpenModernDevelopmentError("validation lock already exists") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _required(row: Mapping[str, str | None], field: str, line_number: int) -> str:
    value = row.get(field)
    if value is None or not value.strip() or value != value.strip():
        raise OpenModernDevelopmentError(f"line {line_number}: {field} is missing or untrimmed")
    return value


def _binary(row: Mapping[str, str | None], field: str, line_number: int) -> int:
    value = _required(row, field, line_number)
    if value == "0":
        return 0
    if value == "1":
        return 1
    raise OpenModernDevelopmentError(f"line {line_number}: {field} must be zero or one")


def _resolved_sort_key(row: OpenModernResolvedRow) -> tuple[int, date, str]:
    return (row.season, row.game_date, row.question_id)


def _require_finite_lock_numbers(value: object) -> None:
    """Recursively reject non-finite numeric values before JSON serialization."""
    if isinstance(value, float) and not isfinite(value):
        raise OpenModernDevelopmentError("validation lock contains a non-finite number")
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        for item in mapping.values():
            _require_finite_lock_numbers(item)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        sequence = cast(Sequence[object], value)
        for item in sequence:
            _require_finite_lock_numbers(item)


if __name__ == "__main__":
    main()
