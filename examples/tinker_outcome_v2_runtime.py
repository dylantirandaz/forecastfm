"""Late-imported Tinker runtime for one committed outcome-v2 SFT run."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from math import log
from typing import Protocol, cast, override

import tinker
import torch
from examples.train_tinker_outcome_sft import OutcomeDataset
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_float,
    require_object,
    required_field,
)
from forecastfm.outcome import (
    OPPONENT_OUTCOME,
    TEAM_LABEL,
    TEAM_OUTCOME,
    TokenCodec,
    require_label_token_ids,
)
from forecastfm.outcome_v2_config import (
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPS,
    BATCH_SIZE,
    FINAL_CHECKPOINT_TTL_SECONDS,
    FINAL_SAMPLER_NAME,
    FINAL_STATE_NAME,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    LORA_RANK,
    LORA_SEED,
    MAX_LENGTH,
    MAX_STEPS,
    OUTCOME_RENDERER_NAME,
    SHUFFLE_SEED,
    TRAIN_ATTN,
    TRAIN_MLP,
    TRAIN_UNEMBED,
    WEIGHT_DECAY,
)
from forecastfm.outcome_v2_preflight import PreparedOutcomeV2Run
from forecastfm.outcome_v2_prompt import OUTCOME_V2_SYSTEM_PROMPT
from forecastfm.run_config import (
    BASE_MODEL,
    require_pinned_tinker_packages,
    require_tokenizer_snapshot,
)
from forecastfm.tinker_data import OutcomeTrainingRecord, read_outcome_training_jsonl_bytes

_RUN_LOCK_METADATA_KEY = "outcome_v2_run_lock_sha256"
_SHA256_CHARACTERS = frozenset("0123456789abcdef")

type _CustomLoss = Callable[
    [list[tinker.Datum], list[torch.Tensor]],
    tuple[torch.Tensor, dict[str, float]],
]
type _TrainingClientFactory = Callable[[str], Awaitable[TrainingClient]]


class _ApiFuture[T](Protocol):
    async def result_async(self) -> T:
        """Return the completed remote operation."""
        ...


class _SavedPath(Protocol):
    path: str


class TrainingClient(Protocol):
    """Small Tinker client surface used by the bounded runtime."""

    async def forward_backward_custom_async(
        self,
        data: list[tinker.Datum],
        loss_fn: _CustomLoss,
    ) -> _ApiFuture[object]:
        """Submit one custom-logprob forward/backward operation."""
        ...

    async def optim_step_async(
        self,
        adam_params: tinker.AdamParams,
    ) -> _ApiFuture[object]:
        """Submit one optimizer step."""
        ...

    async def save_state_async(
        self,
        name: str,
        ttl_seconds: int | None = None,
        overwrite: bool = False,
    ) -> _ApiFuture[_SavedPath]:
        """Save final trainable state."""
        ...

    async def save_weights_for_sampler_async(
        self,
        name: str,
        ttl_seconds: int | None = None,
    ) -> _ApiFuture[_SavedPath]:
        """Save final sampler weights."""
        ...


@dataclass(frozen=True, slots=True)
class OutcomeV2RuntimeResult:
    """Immutable Tinker paths returned after the bounded SFT loop."""

    state_path: str
    sampler_path: str

    def __post_init__(self) -> None:
        _require_tinker_path(self.state_path, "state_path")
        _require_tinker_path(self.sampler_path, "sampler_path")


@dataclass(frozen=True, slots=True)
class _EloOffsetExample:
    """One rendered prompt plus its causal Elo offset and realized winner."""

    datum: tinker.Datum
    elo_team_probability: float
    team_won: bool


type _TrainingBatch = tuple[_EloOffsetExample, ...]


class _EloOffsetOutcomeDataset(OutcomeDataset):
    """Pair-preserving dataset for candidate-token Elo-offset cross-entropy."""

    @override
    def __len__(self) -> int:
        pair_count = len(self._pair_order)
        return (pair_count + self._pairs_per_batch - 1) // self._pairs_per_batch

    def get_elo_offset_batch(self, index: int) -> _TrainingBatch:
        """Render one shuffled batch and retain its sealed prior and outcome."""
        if not 0 <= index < len(self):
            raise IndexError(f"outcome batch index is out of range: {index}")
        start = index * self._pairs_per_batch
        pair_positions = self._pair_order[start : start + self._pairs_per_batch]
        positions = (
            record_position
            for pair_position in pair_positions
            for record_position in (pair_position * 2, pair_position * 2 + 1)
        )
        return tuple(self._elo_offset_example(self._records[position]) for position in positions)

    def _elo_offset_example(self, record: OutcomeTrainingRecord) -> _EloOffsetExample:
        prompt = self._prompt(record)
        return _EloOffsetExample(
            datum=_candidate_logprob_datum(prompt, self._label_token_ids),
            elo_team_probability=_elo_team_probability(record),
            team_won=record["label"] == TEAM_LABEL,
        )


def run_paid(
    prepared: PreparedOutcomeV2Run,
    run_lock_sha256: str,
) -> OutcomeV2RuntimeResult:
    """Run the bounded paid training job from a synchronous orchestrator."""
    return asyncio.run(train_outcome_v2(prepared, run_lock_sha256))


async def train_outcome_v2(
    prepared: PreparedOutcomeV2Run,
    run_lock_sha256: str,
    *,
    create_training_client: _TrainingClientFactory | None = None,
) -> OutcomeV2RuntimeResult:
    """Render all selected batches locally, then execute one bounded paid SFT run."""
    _require_sha256(run_lock_sha256)
    require_pinned_tinker_packages()
    batches = _prepare_training_batches(prepared)

    factory = create_training_client or _create_tinker_training_client
    training_client = await factory(run_lock_sha256)
    await _train_batches(training_client, batches)
    return await _save_final_paths(training_client)


def _prepare_training_batches(prepared: PreparedOutcomeV2Run) -> tuple[_TrainingBatch, ...]:
    records = read_outcome_training_jsonl_bytes(
        prepared.training_jsonl,
        expected_system_prompt=OUTCOME_V2_SYSTEM_PROMPT,
    )
    if len(records) != prepared.proof.row_count:
        raise RuntimeError("prepared row count differs from the parsed training bytes")

    tokenizer = get_tokenizer(str(require_tokenizer_snapshot()))
    label_token_ids = require_label_token_ids(cast(TokenCodec, tokenizer))
    renderer = renderers.get_renderer(
        OUTCOME_RENDERER_NAME,
        tokenizer,
        model_name=BASE_MODEL,
    )
    dataset = _EloOffsetOutcomeDataset(
        records,
        renderer,
        label_token_ids,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    dataset.set_epoch(SHUFFLE_SEED)
    if len(dataset) < MAX_STEPS:
        raise RuntimeError(f"outcome-v2 requires at least {MAX_STEPS} training batches")
    return tuple(dataset.get_elo_offset_batch(index) for index in range(MAX_STEPS))


async def _create_tinker_training_client(run_lock_sha256: str) -> TrainingClient:
    metadata = {_RUN_LOCK_METADATA_KEY: run_lock_sha256}
    service_client = tinker.ServiceClient(user_metadata=metadata)
    client = await service_client.create_lora_training_client_async(
        base_model=BASE_MODEL,
        rank=LORA_RANK,
        seed=LORA_SEED,
        train_mlp=TRAIN_MLP,
        train_attn=TRAIN_ATTN,
        train_unembed=TRAIN_UNEMBED,
        user_metadata=metadata,
    )
    return cast(TrainingClient, client)


async def _train_batches(
    training_client: TrainingClient,
    batches: tuple[_TrainingBatch, ...],
) -> None:
    adam_params = tinker.AdamParams(
        learning_rate=LEARNING_RATE,
        beta1=ADAM_BETA1,
        beta2=ADAM_BETA2,
        eps=ADAM_EPS,
        weight_decay=WEIGHT_DECAY,
        grad_clip_norm=GRAD_CLIP_NORM,
    )
    for batch in batches:
        data = [example.datum for example in batch]
        forward_backward = await training_client.forward_backward_custom_async(
            data,
            _elo_offset_loss(batch),
        )
        optimizer = await training_client.optim_step_async(adam_params)
        await forward_backward.result_async()
        await optimizer.result_async()


def _candidate_logprob_datum(
    prompt: tinker.ModelInput,
    label_token_ids: tuple[int, int],
) -> tinker.Datum:
    """Request both candidate log-probabilities only at the next-token position."""
    target_tokens = [0] * (2 * (prompt.length - 1)) + list(label_token_ids)
    weights = [0.0] * (2 * (prompt.length - 1)) + [1.0, 1.0]
    shape = [prompt.length, 2]
    return tinker.Datum(
        model_input=prompt,
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=target_tokens,
                dtype="int64",
                shape=shape,
            ),
            "weights": tinker.TensorData(
                data=weights,
                dtype="float32",
                shape=shape,
            ),
        },
    )


def _elo_offset_loss(batch: _TrainingBatch) -> _CustomLoss:
    """Bind one batch's sealed Elo priors and realized winners to its loss."""
    expected_data = tuple(example.datum for example in batch)

    def loss_fn(
        data: list[tinker.Datum],
        logprobs: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if len(data) != len(expected_data) or any(
            actual is not expected for actual, expected in zip(data, expected_data, strict=True)
        ):
            raise RuntimeError("custom loss received a different outcome-v2 batch")
        return _elo_offset_binary_cross_entropy(batch, logprobs)

    return loss_fn


def _elo_offset_binary_cross_entropy(
    batch: _TrainingBatch,
    logprobs: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply ordinary winner CE after adding each causal Elo log-odds offset."""
    if not batch or len(logprobs) != len(batch):
        raise RuntimeError("custom loss output count differs from the outcome-v2 batch")

    losses: list[torch.Tensor] = []
    for example, candidate_logprobs in zip(batch, logprobs, strict=True):
        expected_shape = (example.datum.model_input.length, 2)
        if tuple(candidate_logprobs.shape) != expected_shape:
            raise RuntimeError("custom loss received an unexpected candidate-logprob shape")
        residual_logit = candidate_logprobs[-1, 0] - candidate_logprobs[-1, 1]
        final_logit = _probability_logit(example.elo_team_probability) + residual_logit
        target = final_logit.new_tensor(float(example.team_won))
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(final_logit, target))

    loss = torch.stack(losses).mean()
    return loss, {"outcome_v2/elo_offset_binary_cross_entropy": loss.detach().item()}


def _elo_team_probability(record: OutcomeTrainingRecord) -> float:
    """Read the already preflight-bound Elo prior from immutable prompt bytes."""
    try:
        prompt = parse_json_object(record["messages"][1]["content"])
        prior = require_object(required_field(prompt, "prior"), "prior")
        team = require_float(required_field(prior, TEAM_OUTCOME), f"prior.{TEAM_OUTCOME}")
        opponent = require_float(
            required_field(prior, OPPONENT_OUTCOME),
            f"prior.{OPPONENT_OUTCOME}",
        )
    except (IndexError, JsonFormatError) as error:
        raise RuntimeError("outcome-v2 prompt does not contain a valid Elo prior") from error
    if not 0.0 < team < 1.0 or not 0.0 < opponent < 1.0:
        raise RuntimeError("outcome-v2 Elo probabilities must be interior")
    if abs(team + opponent - 1.0) > 1e-9:
        raise RuntimeError("outcome-v2 Elo probabilities must sum to one")
    return team


def _probability_logit(probability: float) -> float:
    """Convert one validated interior probability to stable log-odds."""
    return log(probability) - log(1.0 - probability)


async def _save_final_paths(training_client: TrainingClient) -> OutcomeV2RuntimeResult:
    state_future = await training_client.save_state_async(
        FINAL_STATE_NAME,
        ttl_seconds=FINAL_CHECKPOINT_TTL_SECONDS,
        overwrite=False,
    )
    state = await state_future.result_async()
    sampler_future = await training_client.save_weights_for_sampler_async(
        FINAL_SAMPLER_NAME,
        ttl_seconds=FINAL_CHECKPOINT_TTL_SECONDS,
    )
    sampler = await sampler_future.result_async()
    return OutcomeV2RuntimeResult(state_path=state.path, sampler_path=sampler.path)


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in _SHA256_CHARACTERS for character in value):
        raise RuntimeError("run lock SHA-256 must be a lowercase digest")


def _require_tinker_path(value: str, field_name: str) -> None:
    if not value.startswith("tinker://"):
        raise RuntimeError(f"{field_name} must be an immutable tinker:// path")
