"""Offline tests for the paid Tinker canary boundary."""

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast, override

import pytest
import tinker
from examples import run_tinker_canary
from tinker_cookbook import renderers

from forecastfm import canary as canary_core
from forecastfm.canary import CanaryManifest, CanaryPrompt, CanaryValidationError
from forecastfm.integrity import canonical_sha256
from forecastfm.prompting import ChatMessage
from forecastfm.run_config import MAX_TOKENS, SEED, TEMPERATURE, TOP_K, TOP_P


@dataclass(frozen=True)
class FakeSequence:
    """The public sequence fields consumed by the runner."""

    tokens: list[int]
    stop_reason: str


@dataclass(frozen=True)
class FakeResponse:
    """The public response fields consumed by the runner."""

    sequences: tuple[FakeSequence, ...]


class FakeTokenizer:
    """Decode generated token IDs without changing them."""

    calls: list[tuple[list[int], bool]]

    def __init__(self) -> None:
        self.calls = []

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        """Return a deterministic full-token rendering."""
        self.calls.append((token_ids, skip_special_tokens))
        return "raw:" + ",".join(str(token) for token in token_ids)


class FakeRenderer:
    """Render and parse a small deterministic chat exchange."""

    prompt_messages: list[list[renderers.Message]]
    parsed_tokens: list[list[int]]

    def __init__(self) -> None:
        self.prompt_messages = []
        self.parsed_tokens = []

    def get_stop_sequences(self) -> list[int]:
        """Return one fake chat terminator."""
        return [99]

    def build_generation_prompt(
        self,
        messages: list[renderers.Message],
        role: str = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        """Return fixed prompt tokens while retaining the exact messages."""
        assert role == "assistant"
        assert prefill is None
        self.prompt_messages.append(messages)
        return tinker.ModelInput.from_ints([1, 2, 3])

    def parse_response(
        self,
        response: list[int],
    ) -> tuple[renderers.Message, renderers.ParseTermination]:
        """Return one clean JSON model message."""
        self.parsed_tokens.append(response)
        message = renderers.Message(
            role="assistant",
            content='{"probabilities":{"opponent_wins":0.4,"team_wins":0.6}}',
        )
        return message, renderers.ParseTermination.STOP_SEQUENCE


class ToolRenderer(FakeRenderer):
    """Return a renderer-level tool-call marker that must invalidate the row."""

    @override
    def parse_response(
        self,
        response: list[int],
    ) -> tuple[renderers.Message, renderers.ParseTermination]:
        """Return valid-looking JSON accompanied by a tool-call field."""
        message, termination = super().parse_response(response)
        message["tool_calls"] = []
        return message, termination


class FakeSamplingClient:
    """Return one response or one configured provider exception."""

    calls: list[tuple[tinker.ModelInput, int, tinker.SamplingParams]]

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = []

    async def sample_async(
        self,
        prompt: tinker.ModelInput,
        num_samples: int,
        sampling_params: tinker.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> tinker.SampleResponse:
        """Record exactly one invocation and return its configured result."""
        assert not include_prompt_logprobs
        assert topk_prompt_logprobs == 0
        self.calls.append((prompt, num_samples, sampling_params))
        if self.fail:
            raise RuntimeError("sensitive-provider-message")
        response = FakeResponse(sequences=(FakeSequence([7, 8, 99], "stop"),))
        return cast(tinker.SampleResponse, response)


class RetrySettings(Protocol):
    """Retry property exposed by the SDK configuration object."""

    enable_retry_logic: bool


class FakeServiceClient:
    """Capture construction of both sampling clients."""

    calls: list[dict[str, object]]

    def __init__(self) -> None:
        self.calls = []

    async def create_sampling_client_async(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: object | None = None,
    ) -> tinker.SamplingClient:
        """Capture one client configuration without contacting Tinker."""
        self.calls.append(
            {
                "model_path": model_path,
                "base_model": base_model,
                "retry_config": retry_config,
            }
        )
        return cast(tinker.SamplingClient, FakeSamplingClient())


def _prompt(sequence: int = 0) -> CanaryPrompt:
    messages = (
        ChatMessage(role="system", content="system"),
        ChatMessage(role="user", content="user"),
    )
    return CanaryPrompt(
        sequence=sequence,
        question_id="nba-question",
        variant="original",
        messages=messages,
        prompt_sha256=canonical_sha256(list(messages)),
    )


def _runtime(
    client: FakeSamplingClient,
) -> tuple[run_tinker_canary.SamplingRuntime, FakeTokenizer, FakeRenderer]:
    tokenizer = FakeTokenizer()
    renderer = FakeRenderer()
    params = run_tinker_canary.build_sampling_params(cast(renderers.Renderer, renderer))
    runtime = run_tinker_canary.SamplingRuntime(
        client=cast(tinker.SamplingClient, client),
        tokenizer=tokenizer,
        renderer=cast(renderers.Renderer, renderer),
        params=params,
    )
    return runtime, tokenizer, renderer


def test_sampling_params_match_frozen_decoding_policy() -> None:
    renderer = FakeRenderer()

    params = run_tinker_canary.build_sampling_params(cast(renderers.Renderer, renderer))

    assert params.max_tokens == MAX_TOKENS
    assert params.temperature == TEMPERATURE
    assert params.top_k == TOP_K
    assert params.top_p == TOP_P
    assert params.seed == SEED
    assert params.stop == [99]


def test_generation_preserves_exact_trace_and_calls_provider_once() -> None:
    client = FakeSamplingClient()
    runtime, tokenizer, renderer = _runtime(client)

    records = asyncio.run(run_tinker_canary.generate_arm((_prompt(),), "base", runtime, 0))

    assert len(client.calls) == 1
    assert client.calls[0][1] == 1
    assert len(records) == 1
    record = records[0]
    assert record.attempt_id == "validation-canary-v1:base:000"
    assert record.prompt_tokens == (1, 2, 3)
    assert record.response_tokens == (7, 8, 99)
    assert record.raw_response == "raw:7,8,99"
    assert record.parsed_response.startswith('{"probabilities"')
    assert record.status == "completed"
    assert record.termination == "stop_sequence"
    assert record.stop_reason == "stop"
    assert record.error is None
    assert tokenizer.calls == [([7, 8, 99], False)]
    assert renderer.parsed_tokens == [[7, 8, 99]]


def test_provider_exception_becomes_one_error_row_without_message_or_retry() -> None:
    client = FakeSamplingClient(fail=True)
    runtime, _tokenizer, renderer = _runtime(client)

    records = asyncio.run(run_tinker_canary.generate_arm((_prompt(),), "adapter", runtime, 0))

    assert len(client.calls) == 1
    assert renderer.parsed_tokens == []
    record = records[0]
    assert record.attempt_id == "validation-canary-v1:adapter:000"
    assert record.prompt_tokens == (1, 2, 3)
    assert record.response_tokens == ()
    assert record.raw_response == ""
    assert record.parsed_response == ""
    assert record.status == "error"
    assert record.termination is None
    assert record.stop_reason is None
    assert record.error == "provider_exception:RuntimeError"
    assert "sensitive" not in record.error


def test_renderer_tool_call_becomes_one_invalid_row() -> None:
    client = FakeSamplingClient()
    tokenizer = FakeTokenizer()
    renderer = ToolRenderer()
    runtime = run_tinker_canary.SamplingRuntime(
        client=cast(tinker.SamplingClient, client),
        tokenizer=tokenizer,
        renderer=cast(renderers.Renderer, renderer),
        params=run_tinker_canary.build_sampling_params(cast(renderers.Renderer, renderer)),
    )

    records = asyncio.run(run_tinker_canary.generate_arm((_prompt(),), "base", runtime, 0))

    assert len(client.calls) == 1
    assert records[0].status == "error"
    assert records[0].error == "renderer_tool_call"


def test_sampling_clients_select_both_arms_and_disable_logical_retries() -> None:
    service = FakeServiceClient()

    asyncio.run(
        run_tinker_canary.create_sampling_clients(
            cast(tinker.ServiceClient, service),
            "tinker://run/sampler_weights/final",
        )
    )

    assert len(service.calls) == 2
    assert service.calls[0]["base_model"] == run_tinker_canary.BASE_MODEL
    assert service.calls[0]["model_path"] is None
    assert service.calls[1]["base_model"] is None
    assert service.calls[1]["model_path"] == "tinker://run/sampler_weights/final"
    for call in service.calls:
        config = cast(RetrySettings, call["retry_config"])
        assert config.enable_retry_logic is False


def test_attempt_marker_is_exclusive_and_contains_no_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    prompts_path = tmp_path / "prompts.jsonl"
    marker_path = tmp_path / "raw" / "attempt.json"
    manifest_path.write_text("manifest", encoding="utf-8")
    prompts_path.write_text("prompts", encoding="utf-8")
    loaded = cast(
        tuple[CanaryManifest, tuple[CanaryPrompt, ...]],
        (object(), ()),
    )

    def accept_canary(
        _manifest: Path,
        _prompts: Path,
    ) -> tuple[CanaryManifest, tuple[CanaryPrompt, ...]]:
        return loaded

    monkeypatch.setattr(canary_core, "load_canary", accept_canary)
    monkeypatch.setenv("TINKER_API_KEY", "never-write-this")

    digest = run_tinker_canary.write_attempt_marker(
        marker_path,
        manifest_path,
        prompts_path,
    )

    text = marker_path.read_text(encoding="utf-8")
    assert len(digest) == 64
    assert "never-write-this" not in text
    with pytest.raises(CanaryValidationError, match="refusing to replace"):
        run_tinker_canary.write_attempt_marker(marker_path, manifest_path, prompts_path)


def test_generation_source_has_no_historical_target_path() -> None:
    source = Path(run_tinker_canary.__file__).read_text(encoding="utf-8")

    assert "nba_elo_validation_" + "answers.jsonl" not in source
    assert "score_historical" not in source
