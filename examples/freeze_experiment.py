"""Bind the completed Tinker sampler checkpoint to the frozen training lock."""

from datetime import UTC, datetime
from pathlib import Path

from forecastfm.integrity import file_sha256
from forecastfm.json_utils import parse_json_object, require_string, required_field
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


def read_final_checkpoint(path: Path) -> dict[str, object]:
    """Read one final state/sampler pair from the end of a checkpoint log."""
    try:
        lines = tuple(line for line in path.read_text(encoding="utf-8").splitlines() if line)
    except FileNotFoundError as error:
        raise RuntimeError("Tinker checkpoint log is missing; finish training first") from error
    if not lines:
        raise RuntimeError("Tinker checkpoint log is empty")

    records = tuple(parse_json_object(line) for line in lines)
    names = tuple(require_string(required_field(record, "name"), "name") for record in records)
    if names.count("final") != 1:
        raise RuntimeError("Tinker checkpoint log must contain exactly one final record")
    if names[-1] != "final":
        raise RuntimeError("Tinker final checkpoint must be the last record")

    final = records[-1]
    state_path = require_string(required_field(final, "state_path"), "state_path")
    sampler_path = require_string(required_field(final, "sampler_path"), "sampler_path")
    state_suffix = "/weights/final"
    sampler_suffix = "/sampler_weights/final"
    if not state_path.startswith("tinker://") or not state_path.endswith(state_suffix):
        raise RuntimeError("Tinker final state path has an unexpected format")
    if not sampler_path.startswith("tinker://") or not sampler_path.endswith(sampler_suffix):
        raise RuntimeError("Tinker final sampler path has an unexpected format")
    if state_path.removesuffix(state_suffix) != sampler_path.removesuffix(sampler_suffix):
        raise RuntimeError("Tinker final state and sampler must belong to the same run")
    return final


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
