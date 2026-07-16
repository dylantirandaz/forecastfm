"""Bind the completed Tinker sampler checkpoint to the frozen training lock."""

from datetime import UTC, datetime
from pathlib import Path

from forecastfm.checkpoints import read_final_checkpoint
from forecastfm.integrity import file_sha256
from forecastfm.json_utils import require_string, required_field
from forecastfm.run_lock import (
    build_experiment_lock,
    verify_training_lock,
    write_new_lock,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_LOCK_PATH = PROJECT_ROOT / "prospective" / "training_lock.json"
CHECKPOINT_LOG_PATH = (
    PROJECT_ROOT / "artifacts" / "tinker" / "first_real_nba_sft" / "checkpoints.jsonl"
)
OUTPUT_PATH = PROJECT_ROOT / "prospective" / "experiment.json"


def main() -> None:
    """Create the forecast-ready experiment lock without changing the training lock."""
    verify_training_lock(PROJECT_ROOT, TRAINING_LOCK_PATH)
    checkpoint = read_final_checkpoint(CHECKPOINT_LOG_PATH)
    sampler_path = require_string(required_field(checkpoint, "sampler_path"), "sampler_path")
    record = build_experiment_lock(
        TRAINING_LOCK_PATH,
        sampler_path,
        checkpoint,
        datetime.now(UTC),
    )
    write_new_lock(OUTPUT_PATH, record)
    print(f"Created {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Experiment SHA-256: {file_sha256(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
