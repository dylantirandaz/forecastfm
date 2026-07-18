"""Create untouched NBA forecasts without importing or reading answers."""

from pathlib import Path

from forecastfm.integrity import file_sha256
from forecastfm.nba_evaluation_gate import (
    read_nba_evaluation_forecasts_jsonl,
    write_nba_evaluation_forecasts_jsonl,
)
from forecastfm.nba_feature_rows import read_nba_feature_rows_jsonl
from forecastfm.nba_rich_baseline import (
    NbaRichBaselineForecastLock,
    build_nba_rich_baseline_forecast_lock,
    predict_nba_rich_baseline,
    read_nba_rich_baseline_forecast_lock,
    read_nba_rich_baseline_model,
    write_nba_rich_baseline_forecast_lock,
)
from forecastfm.outcome_v2_config import (
    EVALUATION_FEATURE_ROWS_FILENAME,
    EVALUATION_FORECASTS_FILENAME,
    RICH_BASELINE_FORECAST_LOCK_FILENAME,
    RICH_BASELINE_MODEL_FILENAME,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "outcome_v2"
BASELINE_DIRECTORY = PROJECT_ROOT / "prospective" / "outcome_v2" / "rich_baseline"
MODEL_PATH = DATA_DIRECTORY / RICH_BASELINE_MODEL_FILENAME
EVALUATION_FEATURE_ROWS_PATH = DATA_DIRECTORY / EVALUATION_FEATURE_ROWS_FILENAME
FORECASTS_PATH = DATA_DIRECTORY / EVALUATION_FORECASTS_FILENAME
FORECAST_LOCK_PATH = BASELINE_DIRECTORY / RICH_BASELINE_FORECAST_LOCK_FILENAME


def main() -> None:
    """Predict every sealed row, freeze both outputs, and verify their hashes."""
    _require_unused_outputs()
    model = read_nba_rich_baseline_model(MODEL_PATH)
    feature_rows = read_nba_feature_rows_jsonl(EVALUATION_FEATURE_ROWS_PATH)
    forecasts = predict_nba_rich_baseline(model, feature_rows)
    lock = build_nba_rich_baseline_forecast_lock(model, feature_rows, forecasts)

    write_nba_evaluation_forecasts_jsonl(FORECASTS_PATH, forecasts)
    if read_nba_evaluation_forecasts_jsonl(FORECASTS_PATH) != forecasts:
        raise RuntimeError("written rich baseline forecasts could not be reproduced")
    lock_sha256 = write_nba_rich_baseline_forecast_lock(FORECAST_LOCK_PATH, lock)
    verified = read_nba_rich_baseline_forecast_lock(FORECAST_LOCK_PATH, lock_sha256)
    _require_exact_file_bindings(verified)

    print(f"Frozen {len(forecasts):,} answer-free rich baseline forecasts.")
    print(f"Rich baseline forecast-lock SHA-256: {lock_sha256}")


def _require_unused_outputs() -> None:
    for path in (FORECASTS_PATH, FORECAST_LOCK_PATH):
        if path.exists():
            raise FileExistsError(f"rich baseline output already exists: {path}")


def _require_exact_file_bindings(lock: NbaRichBaselineForecastLock) -> None:
    if lock.model_sha256 != file_sha256(MODEL_PATH):
        raise RuntimeError("forecast lock does not bind the exact model file")
    if lock.evaluation_feature_rows_jsonl_sha256 != file_sha256(EVALUATION_FEATURE_ROWS_PATH):
        raise RuntimeError("forecast lock does not bind the exact evaluation rows")
    if lock.forecast_jsonl_sha256 != file_sha256(FORECASTS_PATH):
        raise RuntimeError("forecast lock does not bind the exact forecast file")


if __name__ == "__main__":
    main()
