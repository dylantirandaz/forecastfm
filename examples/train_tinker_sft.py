"""Run one small ForecastFM supervised fine-tuning step on Tinker."""

import asyncio
import os
from pathlib import Path

from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from forecastfm.json_utils import (
    parse_json_object,
    require_object,
    require_string,
    required_field,
)
from forecastfm.nba_data import file_sha256
from forecastfm.prompting import MODEL_INPUT_SCHEMA_VERSION
from forecastfm.run_config import (
    BASE_MODEL,
    BATCH_SIZE,
    LEARNING_RATE,
    LORA_RANK,
    MAX_LENGTH,
    MAX_STEPS,
    RENDERER_NAME,
    require_tokenizer_snapshot,
)
from forecastfm.run_lock import verify_training_lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nba_elo_train_sft.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.json"
TRAINING_LOCK_PATH = PROJECT_ROOT / "prospective" / "training_lock.json"
LOG_PATH = PROJECT_ROOT / "artifacts" / "tinker" / "first_real_nba_sft"


def require_prerequisites() -> None:
    """Fail before any API call when the key or local dataset is missing."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Training data not found at {DATA_PATH}. Run examples/build_real_nba_dataset.py first."
        )
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError("Dataset manifest is missing; rebuild the real NBA dataset.")

    manifest = parse_json_object(MANIFEST_PATH.read_text(encoding="utf-8"))
    version = required_field(manifest, "model_input_schema_version")
    if version != MODEL_INPUT_SCHEMA_VERSION:
        raise RuntimeError("Dataset uses a stale model-input schema; rebuild it before training.")
    outputs = require_object(required_field(manifest, "outputs"), "outputs")
    expected_hash = require_string(required_field(outputs, DATA_PATH.name), DATA_PATH.name)
    if file_sha256(DATA_PATH) != expected_hash:
        raise RuntimeError(
            "Training data hash differs from its manifest; rebuild it before training."
        )
    verify_training_lock(PROJECT_ROOT, TRAINING_LOCK_PATH)
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError('TINKER_API_KEY is not set. Run: export TINKER_API_KEY="your-key"')


def build_config() -> train.Config:
    """Build the intentionally tiny, reproducible smoke-test configuration."""
    tokenizer_path = require_tokenizer_snapshot()
    dataset = FromConversationFileBuilder(
        file_path=str(DATA_PATH),
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=str(tokenizer_path),
            renderer_name=RENDERER_NAME,
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE,
            train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        ),
    )
    return train.Config(
        log_path=str(LOG_PATH),
        model_name=BASE_MODEL,
        recipe_name="forecastfm_first_sft",
        renderer_name=RENDERER_NAME,
        dataset_builder=dataset,
        learning_rate=LEARNING_RATE,
        lora_rank=LORA_RANK,
        num_epochs=1,
        max_steps=MAX_STEPS,
        save_every=0,
        eval_every=0,
        infrequent_eval_every=0,
        submit_ahead=0,
    )


def main() -> None:
    """Validate local inputs, then start the billable remote training run."""
    require_prerequisites()
    print(f"Training {BASE_MODEL} for {MAX_STEPS} step on {DATA_PATH.name}.")
    asyncio.run(train.main(build_config()))


if __name__ == "__main__":
    main()
