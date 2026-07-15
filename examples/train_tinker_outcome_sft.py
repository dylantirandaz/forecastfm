"""Train ForecastFM on realized NBA winners with label-only cross-entropy."""

import asyncio
import os
from pathlib import Path
from random import Random
from typing import cast, override

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer

from forecastfm.json_utils import (
    parse_json_object,
    require_object,
    require_string,
    required_field,
)
from forecastfm.nba_data import file_sha256
from forecastfm.outcome import (
    OPPONENT_LABEL,
    OUTCOME_INPUT_SCHEMA_VERSION,
    TEAM_LABEL,
    TokenCodec,
    require_label_token_ids,
)
from forecastfm.outcome_config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LORA_RANK,
    MAX_LENGTH,
    MAX_STEPS,
    OUTCOME_RENDERER_NAME,
    RUN_NAME,
    SAVE_EVERY,
)
from forecastfm.outcome_run_lock import verify_outcome_training_lock
from forecastfm.run_config import BASE_MODEL, require_tokenizer_snapshot
from forecastfm.tinker_data import OutcomeTrainingRecord, read_outcome_training_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "outcome_v1" / "nba_train_outcome.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "outcome_v1" / "manifest.json"
TRAINING_LOCK_PATH = (
    PROJECT_ROOT / "prospective" / "outcome_v1" / f"steps_{MAX_STEPS}" / "training_lock.json"
)
LOG_PATH = PROJECT_ROOT / "artifacts" / "tinker" / RUN_NAME


class OutcomeDataset(SupervisedDataset):
    """Deterministic batches whose only loss position is the winner label."""

    def __init__(
        self,
        records: tuple[OutcomeTrainingRecord, ...],
        renderer: renderers.Renderer,
        label_token_ids: tuple[int, int],
        batch_size: int,
        max_length: int,
    ) -> None:
        """Retain immutable records and their deterministic rendering contract."""
        self._records = records
        self._renderer = renderer
        self._label_token_ids = label_token_ids
        self._batch_size = batch_size
        self._max_length = max_length
        self._order = list(range(len(records)))

    @override
    def get_batch(self, index: int) -> list[tinker.Datum]:
        """Render one complete batch at the requested shuffled index."""
        if not 0 <= index < len(self):
            raise IndexError(f"outcome batch index is out of range: {index}")
        start = index * self._batch_size
        positions = self._order[start : start + self._batch_size]
        return [self._datum(self._records[position]) for position in positions]

    @override
    def __len__(self) -> int:
        """Return the number of complete, fixed-size batches."""
        return len(self._records) // self._batch_size

    @override
    def set_epoch(self, seed: int = 0) -> None:
        """Recreate and deterministically shuffle the row order."""
        self._order = list(range(len(self._records)))
        Random(seed).shuffle(self._order)

    def _datum(self, record: OutcomeTrainingRecord) -> tinker.Datum:
        messages = [
            renderers.Message(role=message["role"], content=message["content"])
            for message in record["messages"]
        ]
        prompt = self._renderer.build_generation_prompt(messages)
        if prompt.length + 1 > self._max_length:
            raise RuntimeError(f"outcome prompt exceeds max length: {record['question_id']}")

        team_token, opponent_token = self._label_token_ids
        label_tokens = {TEAM_LABEL: team_token, OPPONENT_LABEL: opponent_token}
        label_token = label_tokens[record["label"]]
        prompt_tokens = prompt.to_ints()
        if not prompt_tokens:
            raise RuntimeError("outcome renderer produced an empty prompt")
        target_tokens = [*prompt_tokens[1:], label_token]
        weights = [0.0] * (len(target_tokens) - 1) + [1.0]
        return tinker.Datum(
            model_input=prompt,
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(
                    data=target_tokens,
                    dtype="int64",
                    shape=[len(target_tokens)],
                ),
                "weights": tinker.TensorData(
                    data=weights,
                    dtype="float32",
                    shape=[len(weights)],
                ),
            },
        )


@chz.chz
class OutcomeDatasetBuilder(SupervisedDatasetBuilder):
    """Build the exact label-only dataset used by the cookbook trainer."""

    file_path: str
    tokenizer_path: str
    renderer_name: str
    model_name: str
    batch_size: int
    max_length: int

    @override
    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        """Load verified records and create a local label-only dataset."""
        records = read_outcome_training_jsonl(Path(self.file_path))
        tokenizer = get_tokenizer(self.tokenizer_path)
        token_ids = require_label_token_ids(cast(TokenCodec, tokenizer))
        renderer = renderers.get_renderer(
            self.renderer_name,
            tokenizer,
            model_name=self.model_name,
        )
        dataset = OutcomeDataset(
            records,
            renderer,
            token_ids,
            self.batch_size,
            self.max_length,
        )
        if len(dataset) == 0:
            raise RuntimeError("outcome dataset does not contain one complete batch")
        return dataset, None


def require_prerequisites() -> None:
    """Fail locally before any paid call when an outcome input is unsafe."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Outcome data not found at {DATA_PATH}. Run examples/build_outcome_dataset.py first."
        )
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError("Outcome manifest is missing; rebuild outcome v1.")

    manifest = parse_json_object(MANIFEST_PATH.read_text(encoding="utf-8"))
    version = required_field(manifest, "outcome_input_schema_version")
    if version != OUTCOME_INPUT_SCHEMA_VERSION:
        raise RuntimeError("Outcome data uses a stale input schema; rebuild it.")
    outputs = require_object(required_field(manifest, "outputs"), "outputs")
    expected_hash = require_string(required_field(outputs, DATA_PATH.name), DATA_PATH.name)
    if file_sha256(DATA_PATH) != expected_hash:
        raise RuntimeError("Outcome training data differs from its manifest; rebuild it.")

    verify_outcome_training_lock(PROJECT_ROOT, TRAINING_LOCK_PATH)
    records = read_outcome_training_jsonl(DATA_PATH)
    if len(records) < BATCH_SIZE:
        raise RuntimeError("Outcome training data does not contain one complete batch")
    tokenizer = get_tokenizer(str(require_tokenizer_snapshot()))
    require_label_token_ids(cast(TokenCodec, tokenizer))
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError('TINKER_API_KEY is not set. Run: export TINKER_API_KEY="your-key"')


def build_config() -> train.Config:
    """Build the first 32-step outcome-classification canary."""
    tokenizer_path = require_tokenizer_snapshot()
    dataset = OutcomeDatasetBuilder(
        file_path=str(DATA_PATH),
        tokenizer_path=str(tokenizer_path),
        renderer_name=OUTCOME_RENDERER_NAME,
        model_name=BASE_MODEL,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    return train.Config(
        log_path=str(LOG_PATH),
        model_name=BASE_MODEL,
        recipe_name=RUN_NAME,
        renderer_name=OUTCOME_RENDERER_NAME,
        dataset_builder=dataset,
        learning_rate=LEARNING_RATE,
        lora_rank=LORA_RANK,
        num_epochs=1,
        max_steps=MAX_STEPS,
        save_every=SAVE_EVERY,
        eval_every=0,
        infrequent_eval_every=0,
        submit_ahead=0,
    )


def main() -> None:
    """Validate every local input, then start the billable outcome run."""
    require_prerequisites()
    config = build_config()
    print(f"Training {BASE_MODEL} for {MAX_STEPS} steps on realized winners from {DATA_PATH.name}.")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
