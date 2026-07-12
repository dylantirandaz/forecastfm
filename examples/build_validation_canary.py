"""Freeze a deterministic 64-game validation canary without opening answers."""

import subprocess
from pathlib import Path

from forecastfm.canary import (
    CanaryModels,
    CanarySource,
    build_canary_artifacts,
)
from forecastfm.integrity import file_sha256
from forecastfm.json_utils import (
    parse_json_object,
    require_object,
    require_string,
    required_field,
)
from forecastfm.run_config import BASE_MODEL, decoding_settings
from forecastfm.run_lock import verify_experiment_lock, verify_training_lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.json"
SOURCE_PROMPTS_PATH = PROJECT_ROOT / "data" / "processed" / "nba_elo_validation_prompts.jsonl"
TRAINING_LOCK_PATH = PROJECT_ROOT / "prospective" / "training_lock.json"
EXPERIMENT_PATH = PROJECT_ROOT / "prospective" / "experiment.json"
OUTPUT_DIRECTORY = PROJECT_ROOT / "evaluation" / "validation_canary"
PROMPTS_PATH = OUTPUT_DIRECTORY / "prompts.jsonl"
MANIFEST_PATH = OUTPUT_DIRECTORY / "manifest.json"

EXPECTED_PROMPTS_SHA256 = "a1c018c09107039101ee9426331f4bbd80bc704ffe23c2d72c628995c245b3cb"
EXPECTED_ANSWERS_SHA256 = "2f91173d30ed835d02663761dcf83c23347a6f9cdb2c0305ac719b852b1c460f"
EXPECTED_QUESTION_IDS_SHA256 = "99e2efda839eca77547f0ee91140463a3ac6cb580fcd24e6a9e2dc90619c1dba"


def main() -> None:
    """Verify existing locks, then exclusively create the answer-free canary."""
    if git_output("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked files have uncommitted changes; commit them before freezing")
    training_lock = verify_training_lock(PROJECT_ROOT, TRAINING_LOCK_PATH)
    experiment = verify_experiment_lock(TRAINING_LOCK_PATH, EXPERIMENT_PATH)
    _verify_dataset_hashes()
    source = CanarySource(
        validation_prompts_path=SOURCE_PROMPTS_PATH,
        validation_prompts_sha256=EXPECTED_PROMPTS_SHA256,
        validation_answers_sha256=EXPECTED_ANSWERS_SHA256,
        dataset_manifest_sha256=file_sha256(DATA_MANIFEST_PATH),
        expected_question_ids_sha256=EXPECTED_QUESTION_IDS_SHA256,
    )
    models = CanaryModels(
        training_lock_sha256=file_sha256(TRAINING_LOCK_PATH),
        experiment_sha256=file_sha256(EXPERIMENT_PATH),
        base_model=BASE_MODEL,
        adapter_sampler_path=require_string(
            required_field(experiment, "adapter_sampler_path"),
            "adapter_sampler_path",
        ),
        decoding=decoding_settings(),
        protocol_code_revision=git_output("rev-parse", "HEAD"),
    )
    manifest = build_canary_artifacts(source, models, PROMPTS_PATH, MANIFEST_PATH)
    if manifest.training_lock_sha256 != file_sha256(TRAINING_LOCK_PATH):
        raise RuntimeError("canary does not bind the verified training lock")
    if required_field(training_lock, "status") != "awaiting_trained_sampler":
        raise RuntimeError("training lock has an unexpected status")
    print(f"Frozen 64 games and 128 prompts at {PROMPTS_PATH}.")
    print(f"Manifest: {MANIFEST_PATH}")


def git_output(*arguments: str) -> str:
    """Run a read-only Git query in the project repository."""
    result = subprocess.run(
        ("git", *arguments),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_dataset_hashes() -> None:
    manifest = parse_json_object(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    outputs = require_object(required_field(manifest, "outputs"), "outputs")
    prompt_hash = require_string(
        required_field(outputs, SOURCE_PROMPTS_PATH.name), SOURCE_PROMPTS_PATH.name
    )
    answer_name = "nba_elo_validation_answers.jsonl"
    answer_hash = require_string(required_field(outputs, answer_name), answer_name)
    if prompt_hash != EXPECTED_PROMPTS_SHA256 or answer_hash != EXPECTED_ANSWERS_SHA256:
        raise RuntimeError("dataset manifest differs from the frozen validation commitments")


if __name__ == "__main__":
    main()
