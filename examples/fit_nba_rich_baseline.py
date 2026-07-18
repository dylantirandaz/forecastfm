"""Fit the frozen rich NBA baseline from sealed training artifacts."""

from pathlib import Path

from forecastfm.integrity import file_sha256
from forecastfm.nba_feature_rows import read_nba_feature_rows_jsonl
from forecastfm.nba_resolutions import read_nba_resolutions_jsonl
from forecastfm.nba_rich_baseline import (
    NbaRichBaselineModel,
    fit_nba_rich_baseline,
    read_nba_rich_baseline_model,
    write_nba_rich_baseline_model,
)
from forecastfm.nba_snapshot_pack import load_snapshot_pack
from forecastfm.outcome_v2_config import (
    FEATURE_ROWS_FILENAME,
    RESOLUTIONS_FILENAME,
    RICH_BASELINE_MODEL_FILENAME,
    SNAPSHOT_PACK_FILENAME,
    outcome_v2_rich_baseline_fit_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "outcome_v2"
FEATURE_ROWS_PATH = DATA_DIRECTORY / FEATURE_ROWS_FILENAME
SNAPSHOT_PACK_PATH = DATA_DIRECTORY / SNAPSHOT_PACK_FILENAME
RESOLUTIONS_PATH = DATA_DIRECTORY / RESOLUTIONS_FILENAME
MODEL_PATH = DATA_DIRECTORY / RICH_BASELINE_MODEL_FILENAME


def main() -> None:
    """Fit once, write once, and re-read the exact model bytes."""
    if MODEL_PATH.exists():
        raise FileExistsError(f"rich baseline model already exists: {MODEL_PATH}")

    feature_rows = read_nba_feature_rows_jsonl(FEATURE_ROWS_PATH)
    snapshot_index = load_snapshot_pack(SNAPSHOT_PACK_PATH)
    resolutions = read_nba_resolutions_jsonl(
        RESOLUTIONS_PATH,
        snapshot_index=snapshot_index,
    )
    model = fit_nba_rich_baseline(
        feature_rows,
        resolutions,
        outcome_v2_rich_baseline_fit_config(),
    )
    _require_exact_training_files(model)
    digest = write_nba_rich_baseline_model(MODEL_PATH, model)
    if read_nba_rich_baseline_model(MODEL_PATH, digest) != model:
        raise RuntimeError("written rich baseline model could not be reproduced")

    print(f"Fitted {model.training_row_count:,} sealed training games.")
    print(f"Rich baseline model SHA-256: {digest}")
    if model.zero_rms_feature_names:
        names = ", ".join(model.zero_rms_feature_names)
        print(f"Audit warning: zero-RMS training features: {names}")


def _require_exact_training_files(model: NbaRichBaselineModel) -> None:
    if model.training_feature_rows_jsonl_sha256 != file_sha256(FEATURE_ROWS_PATH):
        raise RuntimeError("model does not bind the exact training feature-row file")
    if model.training_resolutions_jsonl_sha256 != file_sha256(RESOLUTIONS_PATH):
        raise RuntimeError("model does not bind the exact training resolution file")


if __name__ == "__main__":
    main()
