"""Offline tests for the bounded outcome-v2 Tinker runtime."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from math import log
from pathlib import Path
from typing import cast

import pytest
import tinker
import torch
from examples import tinker_outcome_v2_runtime as runtime
from tinker_cookbook import renderers

from forecastfm.integrity import bytes_sha256, canonical_json
from forecastfm.nba_data import side_swap_nba_example
from forecastfm.outcome import OPPONENT_LABEL, TEAM_LABEL
from forecastfm.outcome_v2_preflight import OutcomeV2Preflight, PreparedOutcomeV2Run
from forecastfm.outcome_v2_prompt import OUTCOME_V2_SYSTEM_PROMPT
from forecastfm.tinker_data import (
    OutcomeTrainingRecord,
    build_outcome_training_record,
    read_outcome_training_jsonl_bytes,
)
from tests.helpers import make_nba_training_example

_RUN_LOCK_SHA256 = "a" * 64

type _CustomLoss = Callable[
    [list[tinker.Datum], list[torch.Tensor]],
    tuple[torch.Tensor, dict[str, float]],
]


@dataclass(frozen=True, slots=True)
class _SavedPath:
    path: str


class _Future[T]:
    def __init__(self, result: T) -> None:
        self._result = result

    async def result_async(self) -> T:
        return self._result


class _FakeRenderer:
    def __init__(self, rendered_ids: list[str]) -> None:
        self._rendered_ids = rendered_ids

    def build_generation_prompt(
        self,
        messages: list[renderers.Message],
        role: str = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        assert [message["role"] for message in messages] == ["system", "user"]
        assert role == "assistant"
        assert prefill is None
        content = messages[1]["content"]
        assert isinstance(content, str)
        self._rendered_ids.append(content)
        return tinker.ModelInput.from_ints([1, 2, 3])


class _FakeTokenCodec:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return {TEAM_LABEL: [10], OPPONENT_LABEL: [20]}[text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        return {10: TEAM_LABEL, 20: OPPONENT_LABEL}[token_ids[0]]


class _FakeTrainingClient:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.losses: list[_CustomLoss] = []
        self.data: list[list[tinker.Datum]] = []
        self.optimizer_steps = 0

    async def forward_backward_custom_async(
        self,
        data: list[tinker.Datum],
        loss_fn: _CustomLoss,
    ) -> _Future[object]:
        self.batch_sizes.append(len(data))
        self.losses.append(loss_fn)
        self.data.append(data)
        return _Future(object())

    async def optim_step_async(self, adam_params: tinker.AdamParams) -> _Future[object]:
        assert adam_params.learning_rate == runtime.LEARNING_RATE
        assert adam_params.beta1 == runtime.ADAM_BETA1
        assert adam_params.beta2 == runtime.ADAM_BETA2
        assert adam_params.eps == runtime.ADAM_EPS
        assert adam_params.weight_decay == runtime.WEIGHT_DECAY
        assert adam_params.grad_clip_norm == runtime.GRAD_CLIP_NORM
        self.optimizer_steps += 1
        return _Future(object())

    async def save_state_async(
        self,
        name: str,
        ttl_seconds: int | None = None,
        overwrite: bool = False,
    ) -> _Future[_SavedPath]:
        assert name == runtime.FINAL_STATE_NAME
        assert ttl_seconds == runtime.FINAL_CHECKPOINT_TTL_SECONDS
        assert not overwrite
        return _Future(_SavedPath("tinker://run/state/final"))

    async def save_weights_for_sampler_async(
        self,
        name: str,
        ttl_seconds: int | None = None,
    ) -> _Future[_SavedPath]:
        assert name == runtime.FINAL_SAMPLER_NAME
        assert ttl_seconds == runtime.FINAL_CHECKPOINT_TTL_SECONDS
        return _Future(_SavedPath("tinker://run/sampler/final"))


def _records(pair_count: int) -> tuple[OutcomeTrainingRecord, ...]:
    records: list[OutcomeTrainingRecord] = []
    for index in range(pair_count):
        template = make_nba_training_example()
        original = replace(
            template,
            case=replace(
                template.case,
                question=replace(template.case.question, question_id=f"runtime-game-{index}"),
            ),
        )
        for example in (original, side_swap_nba_example(original)):
            record = build_outcome_training_record(example)
            record["messages"][0]["content"] = OUTCOME_V2_SYSTEM_PROMPT
            records.append(record)
    return tuple(records)


def _training_bytes(pair_count: int) -> bytes:
    text = "".join(f"{canonical_json(record)}\n" for record in _records(pair_count))
    return text.encode("utf-8")


def _prepared(training_jsonl: bytes, pair_count: int) -> PreparedOutcomeV2Run:
    proof = OutcomeV2Preflight(
        manifest_sha256="1" * 64,
        action_at=datetime(2026, 7, 17, tzinfo=UTC),
        action_time_source="internal_paid_preparation",
        untouched_evaluation_seasons=(2025, 2026),
        training_sha256=bytes_sha256(training_jsonl),
        feature_rows_sha256="2" * 64,
        snapshot_pack_sha256="3" * 64,
        evidence_bundles_sha256="4" * 64,
        elo_states_sha256="5" * 64,
        elo_replay_sha256="6" * 64,
        seasons_sha256="7" * 64,
        resolutions_sha256="8" * 64,
        rights_lock_sha256="9" * 64,
        evaluation_feature_rows_sha256="0" * 64,
        evaluation_elo_replay_sha256="a" * 64,
        evaluation_elo_states_sha256="b" * 64,
        evaluation_resolutions_sha256="c" * 64,
        calibration_sha256="0" * 64,
        rich_baseline_model_sha256="e" * 64,
        rich_baseline_forecast_lock_sha256="f" * 64,
        evaluation_report_sha256="d" * 64,
        row_count=pair_count * 2,
        pair_count=pair_count,
        batch_size=runtime.BATCH_SIZE,
    )
    return PreparedOutcomeV2Run(proof, training_jsonl)


def _configure_local_rendering(
    monkeypatch: pytest.MonkeyPatch,
    rendered_ids: list[str],
) -> None:
    def fake_get_tokenizer(_path: str) -> _FakeTokenCodec:
        return _FakeTokenCodec()

    def fake_get_renderer(*_args: object, **_kwargs: object) -> renderers.Renderer:
        return cast(renderers.Renderer, _FakeRenderer(rendered_ids))

    monkeypatch.setattr(runtime, "require_tokenizer_snapshot", lambda: Path("/pinned/tokenizer"))
    monkeypatch.setattr(runtime, "get_tokenizer", fake_get_tokenizer)
    monkeypatch.setattr(runtime.renderers, "get_renderer", fake_get_renderer)


def test_runtime_prepares_every_batch_before_client_and_retains_partial_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_jsonl = _training_bytes(pair_count=8)
    prepared = _prepared(training_jsonl, pair_count=8)
    rendered_ids: list[str] = []
    parsed_bytes: list[bytes] = []
    client = _FakeTrainingClient()
    _configure_local_rendering(monkeypatch, rendered_ids)
    monkeypatch.setattr(runtime, "MAX_STEPS", 2)

    def parse_bytes(
        data: bytes,
        *,
        expected_system_prompt: str,
    ) -> tuple[OutcomeTrainingRecord, ...]:
        parsed_bytes.append(data)
        return read_outcome_training_jsonl_bytes(
            data,
            expected_system_prompt=expected_system_prompt,
        )

    async def create_client(run_lock_sha256: str) -> runtime.TrainingClient:
        assert run_lock_sha256 == _RUN_LOCK_SHA256
        assert len(rendered_ids) == 16
        return cast(runtime.TrainingClient, client)

    monkeypatch.setattr(runtime, "read_outcome_training_jsonl_bytes", parse_bytes)

    result = asyncio.run(
        runtime.train_outcome_v2(
            prepared,
            _RUN_LOCK_SHA256,
            create_training_client=create_client,
        )
    )

    assert parsed_bytes == [training_jsonl]
    assert client.batch_sizes == [14, 2]
    assert len(client.losses) == 2
    assert all(callable(loss) for loss in client.losses)
    assert client.optimizer_steps == 2
    assert all(
        datum.loss_fn_inputs["target_tokens"].shape == [3, 2]
        and datum.loss_fn_inputs["target_tokens"].data[-2:] == [10, 20]
        and datum.loss_fn_inputs["weights"].data[-2:] == [1.0, 1.0]
        for batch in client.data
        for datum in batch
    )
    assert result == runtime.OutcomeV2RuntimeResult(
        state_path="tinker://run/state/final",
        sampler_path="tinker://run/sampler/final",
    )


def test_custom_loss_adds_elo_log_odds_before_binary_cross_entropy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(_training_bytes(pair_count=1), pair_count=1)
    client = _FakeTrainingClient()
    _configure_local_rendering(monkeypatch, [])
    monkeypatch.setattr(runtime, "MAX_STEPS", 1)

    async def create_client(_run_lock_sha256: str) -> runtime.TrainingClient:
        return cast(runtime.TrainingClient, client)

    asyncio.run(
        runtime.train_outcome_v2(
            prepared,
            _RUN_LOCK_SHA256,
            create_training_client=create_client,
        )
    )
    data = client.data[0]
    logprobs = [torch.zeros((datum.model_input.length, 2), requires_grad=True) for datum in data]
    loss, metrics = client.losses[0](data, logprobs)

    assert loss.item() == pytest.approx(-log(0.4))
    assert metrics == {"outcome_v2/elo_offset_binary_cross_entropy": pytest.approx(-log(0.4))}
    assert loss.requires_grad

    symmetric_residuals = [
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [log(4.0), 0.0]], requires_grad=True),
        torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, log(4.0)]], requires_grad=True),
    ]
    corrected_loss, _ = client.losses[0](data, symmetric_residuals)
    assert corrected_loss.item() == pytest.approx(-log(8.0 / 11.0))


def test_runtime_rejects_parsed_row_mismatch_before_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_jsonl = _training_bytes(pair_count=1)
    prepared = _prepared(training_jsonl, pair_count=1)
    wrong_proof = replace(prepared.proof, row_count=4, pair_count=2)
    forged = PreparedOutcomeV2Run(wrong_proof, training_jsonl)
    called = False

    async def create_client(_run_lock_sha256: str) -> runtime.TrainingClient:
        nonlocal called
        called = True
        return cast(runtime.TrainingClient, _FakeTrainingClient())

    with pytest.raises(RuntimeError, match="row count differs"):
        asyncio.run(
            runtime.train_outcome_v2(
                forged,
                _RUN_LOCK_SHA256,
                create_training_client=create_client,
            )
        )
    assert not called


def test_runtime_rejects_too_few_batches_before_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_jsonl = _training_bytes(pair_count=1)
    prepared = _prepared(training_jsonl, pair_count=1)
    called = False
    _configure_local_rendering(monkeypatch, [])
    monkeypatch.setattr(runtime, "MAX_STEPS", 2)

    async def create_client(_run_lock_sha256: str) -> runtime.TrainingClient:
        nonlocal called
        called = True
        return cast(runtime.TrainingClient, _FakeTrainingClient())

    with pytest.raises(RuntimeError, match="requires at least 2 training batches"):
        asyncio.run(
            runtime.train_outcome_v2(
                prepared,
                _RUN_LOCK_SHA256,
                create_training_client=create_client,
            )
        )
    assert not called


def test_default_client_factory_binds_seed_model_rank_and_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    client = _FakeTrainingClient()
    prepared = _prepared(_training_bytes(pair_count=1), pair_count=1)
    _configure_local_rendering(monkeypatch, [])
    monkeypatch.setattr(runtime, "MAX_STEPS", 1)

    class FakeServiceClient:
        def __init__(self, *, user_metadata: dict[str, str]) -> None:
            calls.append(("service_metadata", user_metadata))

        async def create_lora_training_client_async(
            self,
            **kwargs: object,
        ) -> _FakeTrainingClient:
            calls.append(("training_parameters", kwargs))
            return client

    monkeypatch.setattr(runtime.tinker, "ServiceClient", FakeServiceClient)

    result = asyncio.run(runtime.train_outcome_v2(prepared, _RUN_LOCK_SHA256))

    metadata = {"outcome_v2_run_lock_sha256": _RUN_LOCK_SHA256}
    assert result.state_path == "tinker://run/state/final"
    assert calls == [
        ("service_metadata", metadata),
        (
            "training_parameters",
            {
                "base_model": runtime.BASE_MODEL,
                "rank": runtime.LORA_RANK,
                "seed": runtime.LORA_SEED,
                "train_mlp": runtime.TRAIN_MLP,
                "train_attn": runtime.TRAIN_ATTN,
                "train_unembed": runtime.TRAIN_UNEMBED,
                "user_metadata": metadata,
            },
        ),
    ]


def test_run_paid_calls_bounded_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = _prepared(_training_bytes(pair_count=1), pair_count=1)
    expected = runtime.OutcomeV2RuntimeResult(
        state_path="tinker://run/state/final",
        sampler_path="tinker://run/sampler/final",
    )

    async def train(
        actual_prepared: PreparedOutcomeV2Run,
        run_lock_sha256: str,
    ) -> runtime.OutcomeV2RuntimeResult:
        assert actual_prepared is prepared
        assert run_lock_sha256 == _RUN_LOCK_SHA256
        return expected

    monkeypatch.setattr(runtime, "train_outcome_v2", train)

    assert runtime.run_paid(prepared, _RUN_LOCK_SHA256) == expected


@pytest.mark.parametrize("path", ["", "https://example.com/model", "run/state/final"])
def test_runtime_result_requires_immutable_tinker_paths(path: str) -> None:
    with pytest.raises(RuntimeError, match="immutable tinker:// path"):
        runtime.OutcomeV2RuntimeResult(state_path=path, sampler_path="tinker://sampler")
