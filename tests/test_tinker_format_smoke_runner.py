"""Offline tests for the two-call Tinker format smoke."""

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast, override

import pytest
import tinker
from examples import run_tinker_format_smoke
from tinker_cookbook import renderers

from forecastfm.integrity import canonical_sha256, file_sha256, text_sha256
from forecastfm.json_utils import (
    parse_json_object,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.prompting import ChatMessage
from forecastfm.run_config import MAX_TOKENS, SEED, TEMPERATURE, TOP_K, TOP_P
from tests.test_tinker_canary_runner import (
    FakeRenderer,
    FakeSamplingClient,
    FakeServiceClient,
    FakeTokenizer,
    RetrySettings,
    ToolRenderer,
)

VALID_PREDICTION = '{"probabilities":{"team_wins":0.6,"opponent_wins":0.4}}'


class OrderedRenderer(FakeRenderer):
    """Require the base artifact before rendering the adapter prompt."""

    def __init__(self, base_path: Path) -> None:
        super().__init__()
        self.base_path = base_path

    @override
    def build_generation_prompt(
        self,
        messages: list[renderers.Message],
        role: str = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        if self.prompt_messages:
            assert self.base_path.is_file()
        return super().build_generation_prompt(messages, role, prefill)


@dataclass(frozen=True)
class ArtifactPaths:
    """Temporary paths used by the real exclusive artifact writers."""

    attempt: Path
    base: Path
    adapter: Path
    seal: Path
    result: Path


def _inputs() -> run_tinker_format_smoke.SmokeInputs:
    messages = (
        ChatMessage(role="system", content="system"),
        ChatMessage(role="user", content="user"),
    )
    return run_tinker_format_smoke.SmokeInputs(
        messages=messages,
        training_messages=(
            *messages,
            ChatMessage(role="assistant", content=VALID_PREDICTION),
        ),
        prompt_sha256="a" * 64,
        adapter_sampler_path="tinker://run/sampler_weights/final",
        tokenizer_path=Path("/pinned/tokenizer"),
        protocol_revision="b" * 40,
    )


def _runtime(renderer: FakeRenderer) -> run_tinker_format_smoke.SamplingRuntime:
    rendered = cast(renderers.Renderer, renderer)
    return run_tinker_format_smoke.SamplingRuntime(
        tokenizer=FakeTokenizer(),
        renderer=rendered,
        params=run_tinker_format_smoke.build_sampling_params(rendered),
    )


def _record(role: run_tinker_format_smoke.ModelRole) -> run_tinker_format_smoke.SmokeRecord:
    return run_tinker_format_smoke.SmokeRecord(
        model_role=role,
        prompt_sha256="a" * 64,
        prompt_tokens=(1, 2, 3),
        response_tokens=(7, 8, 99),
        raw_response="raw:7,8,99",
        parsed_response=VALID_PREDICTION,
        status="completed",
        termination="stop_sequence",
        stop_reason="stop",
        error=None,
    )


def _accept_training_lock(_root: Path, _path: Path) -> dict[str, object]:
    return {}


def _accept_experiment_lock(_training: Path, _experiment: Path) -> dict[str, object]:
    return {"adapter_sampler_path": "tinker://run/sampler_weights/final"}


def _configure_artifacts(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ArtifactPaths:
    raw = directory / "raw"
    paths = ArtifactPaths(
        attempt=raw / "attempt.json",
        base=raw / "base.json",
        adapter=raw / "adapter.json",
        seal=raw / "manifest.json",
        result=directory / "result.json",
    )
    training = directory / "training.jsonl"
    training_lock = directory / "training_lock.json"
    experiment = directory / "experiment.json"
    training.write_text("training\n", encoding="utf-8")
    training_lock.write_text("lock\n", encoding="utf-8")
    experiment.write_text("experiment\n", encoding="utf-8")
    replacements = {
        "RAW_DIRECTORY": raw,
        "ATTEMPT_PATH": paths.attempt,
        "BASE_OUTPUT_PATH": paths.base,
        "ADAPTER_OUTPUT_PATH": paths.adapter,
        "SEAL_PATH": paths.seal,
        "RESULT_PATH": paths.result,
        "TRAINING_PATH": training,
        "TRAINING_LOCK_PATH": training_lock,
        "EXPERIMENT_PATH": experiment,
    }
    for name, value in replacements.items():
        monkeypatch.setattr(run_tinker_format_smoke, name, value)
    return paths


def test_build_renderers_selects_no_thinking_without_changing_training_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer()
    built = (FakeRenderer(), FakeRenderer())
    calls: list[tuple[str, object, str]] = []

    def get_tokenizer(path: str) -> FakeTokenizer:
        assert path == "/pinned/tokenizer"
        return tokenizer

    def get_renderer(
        name: str,
        received_tokenizer: object,
        *,
        model_name: str,
    ) -> renderers.Renderer:
        calls.append((name, received_tokenizer, model_name))
        return cast(renderers.Renderer, built[len(calls) - 1])

    monkeypatch.setattr(run_tinker_format_smoke, "get_tokenizer", get_tokenizer)
    monkeypatch.setattr(run_tinker_format_smoke.renderers, "get_renderer", get_renderer)

    returned = run_tinker_format_smoke.build_renderers(Path("/pinned/tokenizer"))

    assert returned == (tokenizer, *built)
    assert [call[0] for call in calls] == ["qwen3_5_disable_thinking", "qwen3_5"]
    assert all(call[1] is tokenizer for call in calls)
    assert all(call[2] == run_tinker_format_smoke.BASE_MODEL for call in calls)


def test_run_calls_each_arm_once_and_persists_before_the_next_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _configure_artifacts(tmp_path, monkeypatch)
    inputs = _inputs()
    renderer = OrderedRenderer(paths.base)
    tokenizer = FakeTokenizer()
    base_client = FakeSamplingClient()
    adapter_client = FakeSamplingClient()

    async def clients(
        _service: tinker.ServiceClient,
        _adapter_path: str,
    ) -> tuple[tinker.SamplingClient, tinker.SamplingClient]:
        return cast(tinker.SamplingClient, base_client), cast(tinker.SamplingClient, adapter_client)

    def service() -> tinker.ServiceClient:
        assert paths.attempt.is_file()
        return cast(tinker.ServiceClient, object())

    rendered = cast(renderers.Renderer, renderer)

    def build_renderers(
        _path: Path,
    ) -> tuple[run_tinker_format_smoke.TokenDecoder, renderers.Renderer, renderers.Renderer]:
        return tokenizer, rendered, rendered

    def accept_prefix(
        _inputs_value: run_tinker_format_smoke.SmokeInputs,
        _smoke: renderers.Renderer,
        _training: renderers.Renderer,
    ) -> None:
        return None

    monkeypatch.setattr(run_tinker_format_smoke, "require_run_inputs", lambda: inputs)
    monkeypatch.setattr(run_tinker_format_smoke, "build_renderers", build_renderers)
    monkeypatch.setattr(run_tinker_format_smoke, "_require_training_prefix_match", accept_prefix)
    monkeypatch.setattr(run_tinker_format_smoke, "create_sampling_clients", clients)
    monkeypatch.setattr(run_tinker_format_smoke.tinker, "ServiceClient", service)

    asyncio.run(run_tinker_format_smoke.run())

    assert len(base_client.calls) == len(adapter_client.calls) == 1
    assert len(renderer.prompt_messages) == 2
    for client in (base_client, adapter_client):
        _prompt, num_samples, params = client.calls[0]
        assert num_samples == 1
        assert (params.max_tokens, params.temperature, params.top_k) == (
            MAX_TOKENS,
            TEMPERATURE,
            TOP_K,
        )
        assert (params.top_p, params.seed, params.stop) == (TOP_P, SEED, [99])
    result = parse_json_object(paths.result.read_text(encoding="utf-8"))
    assert required_field(result, "passed") is True


def test_gate_rejects_length_invalid_json_and_tool_output() -> None:
    valid = _record("base")
    length = replace(valid, termination="malformed", stop_reason="length")
    invalid_json = replace(valid, parsed_response="not JSON")
    tool_client = FakeSamplingClient()
    tool = asyncio.run(
        run_tinker_format_smoke.sample_once(
            "adapter",
            cast(tinker.SamplingClient, tool_client),
            _inputs(),
            _runtime(ToolRenderer()),
        )
    )

    assert run_tinker_format_smoke.gate_record(length)["reason"] == ("unclean_renderer_termination")
    assert run_tinker_format_smoke.gate_record(invalid_json)["reason"] == (
        "invalid_prediction_json"
    )
    assert tool.status == "renderer_error"
    assert run_tinker_format_smoke.gate_record(tool)["reason"] == "renderer_tool_call"
    assert len(tool_client.calls) == 1


def test_provider_exception_is_recorded_once_without_sensitive_detail() -> None:
    client = FakeSamplingClient(fail=True)

    record = asyncio.run(
        run_tinker_format_smoke.sample_once(
            "base",
            cast(tinker.SamplingClient, client),
            _inputs(),
            _runtime(FakeRenderer()),
        )
    )

    assert len(client.calls) == 1
    assert record.error == "provider_exception:RuntimeError"
    assert "sensitive" not in record.error
    assert run_tinker_format_smoke.gate_record(record)["passed"] is False


def test_sampling_clients_select_both_arms_and_disable_retries() -> None:
    service = FakeServiceClient()

    asyncio.run(
        run_tinker_format_smoke.create_sampling_clients(
            cast(tinker.ServiceClient, service),
            "tinker://run/sampler_weights/final",
        )
    )

    assert len(service.calls) == 2
    assert service.calls[0]["base_model"] == run_tinker_format_smoke.BASE_MODEL
    assert service.calls[1]["model_path"] == "tinker://run/sampler_weights/final"
    for call in service.calls:
        assert cast(RetrySettings, call["retry_config"]).enable_retry_logic is False


def test_artifacts_are_write_once_and_seal_detects_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _configure_artifacts(tmp_path, monkeypatch)
    inputs = _inputs()
    base, adapter = _record("base"), _record("adapter")

    run_tinker_format_smoke.write_attempt_marker(inputs)
    with pytest.raises(FileExistsError):
        run_tinker_format_smoke.write_attempt_marker(inputs)
    run_tinker_format_smoke.write_record(paths.base, base)
    with pytest.raises(FileExistsError):
        run_tinker_format_smoke.write_record(paths.base, base)
    run_tinker_format_smoke.write_record(paths.adapter, adapter)
    seal_hash, seal = run_tinker_format_smoke.seal_outputs(base, adapter)
    with pytest.raises(FileExistsError):
        run_tinker_format_smoke.seal_outputs(base, adapter)
    run_tinker_format_smoke.write_gate_result(base, adapter, seal, seal_hash)

    sealed = parse_json_object(paths.seal.read_text(encoding="utf-8"))
    expected = require_string(required_field(sealed, "base_sha256"), "base_sha256")
    assert expected == file_sha256(paths.base)
    paths.base.write_text(paths.base.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(run_tinker_format_smoke.SmokeValidationError, match="differ"):
        run_tinker_format_smoke.write_gate_result(base, adapter, seal, seal_hash)


def test_verified_smoke_input_omits_the_training_assistant_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_messages = [
        {"role": "system", "content": "system"},
        {
            "role": "user",
            "content": '{"outcomes":["team_wins","opponent_wins"]}',
        },
        {"role": "assistant", "content": VALID_PREDICTION},
    ]
    source_line = json.dumps({"messages": source_messages}, sort_keys=True)
    training_path = tmp_path / "training.jsonl"
    training_path.write_text(source_line + "\n", encoding="utf-8")
    monkeypatch.setattr(run_tinker_format_smoke, "verify_training_lock", _accept_training_lock)
    monkeypatch.setattr(
        run_tinker_format_smoke,
        "verify_experiment_lock",
        _accept_experiment_lock,
    )
    monkeypatch.setattr(run_tinker_format_smoke, "_require_package_versions", lambda: None)
    monkeypatch.setattr(
        run_tinker_format_smoke,
        "require_tokenizer_snapshot",
        lambda: Path("/pinned"),
    )
    monkeypatch.setattr(run_tinker_format_smoke, "_require_published_revision", lambda: "b" * 40)
    monkeypatch.setattr(run_tinker_format_smoke, "TRAINING_PATH", training_path)
    monkeypatch.setattr(
        run_tinker_format_smoke,
        "EXPECTED_TRAINING_ROW_SHA256",
        text_sha256(source_line),
    )
    monkeypatch.setattr(
        run_tinker_format_smoke,
        "EXPECTED_PROMPT_SHA256",
        canonical_sha256(source_messages[:2]),
    )
    monkeypatch.setattr(run_tinker_format_smoke, "ATTEMPT_PATH", tmp_path / "attempt.json")
    monkeypatch.setattr(run_tinker_format_smoke, "BASE_OUTPUT_PATH", tmp_path / "base.json")
    monkeypatch.setattr(run_tinker_format_smoke, "ADAPTER_OUTPUT_PATH", tmp_path / "adapter.json")
    monkeypatch.setattr(run_tinker_format_smoke, "SEAL_PATH", tmp_path / "seal.json")
    monkeypatch.setattr(run_tinker_format_smoke, "RESULT_PATH", tmp_path / "result.json")

    inputs = run_tinker_format_smoke.require_run_inputs()
    with training_path.open(encoding="utf-8") as file:
        source = parse_json_object(file.readline())
    messages = require_list(required_field(source, "messages"), "messages")
    assistant = require_object(messages[2], "assistant message")
    target = require_string(required_field(assistant, "content"), "assistant content")

    assert tuple(message["role"] for message in inputs.messages) == ("system", "user")
    assert target == inputs.training_messages[2]["content"]
    assert all(message["content"] != target for message in inputs.messages)
