"""Bind the completed outcome sampler to its frozen training lock."""

from datetime import UTC, datetime
from pathlib import Path

from examples.freeze_experiment import read_final_checkpoint

from forecastfm.integrity import file_sha256
from forecastfm.json_utils import require_string, required_field
from forecastfm.outcome_config import MAX_STEPS, RUN_NAME
from forecastfm.outcome_run_lock import verify_outcome_training_lock
from forecastfm.run_lock import build_experiment_lock, write_new_lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = PROJECT_ROOT / "prospective" / "outcome_v1" / f"steps_{MAX_STEPS}"
TRAINING_LOCK_PATH = RUN_PATH / "training_lock.json"
CHECKPOINT_LOG_PATH = PROJECT_ROOT / "artifacts" / "tinker" / RUN_NAME / "checkpoints.jsonl"
OUTPUT_PATH = RUN_PATH / "experiment.json"


def main() -> None:
    """Create the forecast-ready outcome experiment lock."""
    verify_outcome_training_lock(PROJECT_ROOT, TRAINING_LOCK_PATH)
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
    print(f"Outcome experiment SHA-256: {file_sha256(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
