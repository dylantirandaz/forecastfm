"""Run eight bounded candidate calls on one non-development training pair."""

import asyncio
import os
from pathlib import Path
from typing import cast

import tinker
from examples.build_outcome_development_evaluation import (
    EXPERIMENT_PATH,
    PROJECT_ROOT,
    TRAINING_LOCK_PATH,
)
from examples.run_tinker_outcome_development import (
    create_clients,
    read_api_key,
    render_model_inputs,
    require_package_versions,
    score_orientations,
)
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from forecastfm.json_utils import require_object, require_string, required_field
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.outcome import TokenCodec, require_label_token_ids
from forecastfm.outcome_run_lock import verify_outcome_training_lock
from forecastfm.run_config import require_tokenizer_snapshot
from forecastfm.run_lock import verify_experiment_lock
from forecastfm.tinker_data import ForecastRecord, read_outcome_training_jsonl

LOCAL_ENV_PATH = PROJECT_ROOT / ".env"
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "outcome_v1" / "nba_train_outcome.jsonl"


def load_smoke_pair(path: Path) -> tuple[ForecastRecord, ForecastRecord]:
    """Load one adjacent training side-swap pair without sending its labels."""
    records = read_outcome_training_jsonl(path)
    if len(records) < 2:
        raise RuntimeError("outcome training data has no complete smoke pair")
    original, swapped = records[:2]
    if swapped["question_id"] != f"{original['question_id']}{SIDE_SWAP_SUFFIX}":
        raise RuntimeError("outcome smoke records are not an adjacent side-swap pair")
    if original["label"] == swapped["label"]:
        raise RuntimeError("outcome smoke side-swap labels must be opposite")
    return (
        ForecastRecord(question_id=original["question_id"], messages=original["messages"]),
        ForecastRecord(question_id=swapped["question_id"], messages=swapped["messages"]),
    )


async def run() -> None:
    """Verify both model clients with four drained candidate calls each."""
    training_lock = verify_outcome_training_lock(PROJECT_ROOT, TRAINING_LOCK_PATH)
    experiment = verify_experiment_lock(TRAINING_LOCK_PATH, EXPERIMENT_PATH)
    require_package_versions()
    model = require_object(required_field(training_lock, "model"), "model")
    base_model = require_string(required_field(model, "base_model"), "base_model")
    renderer_name = require_string(required_field(model, "renderer"), "renderer")
    adapter_path = require_string(
        required_field(experiment, "adapter_sampler_path"),
        "adapter_sampler_path",
    )
    tokenizer = get_tokenizer(str(require_tokenizer_snapshot()))
    label_token_ids = require_label_token_ids(cast(TokenCodec, tokenizer))
    renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=base_model)
    prompt_pair = load_smoke_pair(TRAINING_DATA_PATH)
    model_inputs = render_model_inputs(renderer, prompt_pair)

    if not os.environ.get("TINKER_API_KEY"):
        os.environ["TINKER_API_KEY"] = read_api_key(LOCAL_ENV_PATH)
    clients = await create_clients(tinker.ServiceClient(), base_model, adapter_path)
    await score_orientations(clients["base"], model_inputs, label_token_ids)
    print("Base candidate smoke passed with four logical calls.")
    await score_orientations(clients["adapter"], model_inputs, label_token_ids)
    print("Adapter candidate smoke passed with four logical calls.")


def main() -> None:
    """Run the bounded non-development live smoke."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
