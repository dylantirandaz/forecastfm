"""Run one format-only check on the Tinker base model and adapter."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import tinker
from examples.run_tinker_sft_local import read_api_key
from tinker.lib.retry_handler import RetryConfig  # pyright: ignore[reportMissingTypeStubs]
from tinker_cookbook import renderers
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.tokenizer_utils import get_tokenizer

from forecastfm.integrity import canonical_sha256, file_sha256, text_sha256
from forecastfm.json_utils import (
    parse_json_object,
    require_exact_keys,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.prompting import ChatMessage, parse_prediction
from forecastfm.run_config import (
    BASE_MODEL,
    MAX_TOKENS,
    SEED,
    TEMPERATURE,
    TINKER_COOKBOOK_VERSION,
    TINKER_VERSION,
    TOP_K,
    TOP_P,
    decoding_settings,
    require_tokenizer_snapshot,
)
from forecastfm.run_config import (
    RENDERER_NAME as TRAINING_RENDERER_NAME,
)
from forecastfm.run_lock import verify_experiment_lock, verify_training_lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV_PATH = PROJECT_ROOT / ".env"
TRAINING_PATH = PROJECT_ROOT / "data" / "processed" / "nba_elo_train_sft.jsonl"
TRAINING_LOCK_PATH = PROJECT_ROOT / "prospective" / "training_lock.json"
EXPERIMENT_PATH = PROJECT_ROOT / "prospective" / "experiment.json"
OUTPUT_DIRECTORY = PROJECT_ROOT / "evaluation" / "format_smoke" / "v1"
RAW_DIRECTORY = OUTPUT_DIRECTORY / "raw"
ATTEMPT_PATH = RAW_DIRECTORY / "attempt.json"
BASE_OUTPUT_PATH = RAW_DIRECTORY / "base.json"
ADAPTER_OUTPUT_PATH = RAW_DIRECTORY / "adapter.json"
SEAL_PATH = RAW_DIRECTORY / "manifest.json"
RESULT_PATH = OUTPUT_DIRECTORY / "result.json"

SMOKE_RENDERER_NAME = "qwen3_5_disable_thinking"
TRAINING_ROW_INDEX = 0
EXPECTED_TRAINING_ROW_SHA256 = "b732571ef4607ec872792db5910e5bf58099c05534f1ff1869a95e86c23b9d99"
EXPECTED_PROMPT_SHA256 = "dc7037a096f47b37805c58e54f48d11f715ca22e9056c71cc2c8b76058c90918"
EXPECTED_PROMPT_TOKEN_COUNT = 182
OUTCOMES = ("team_wins", "opponent_wins")
MODEL_ROLES = ("base", "adapter")

type ModelRole = Literal["base", "adapter"]
type GenerationStatus = Literal["completed", "provider_error", "renderer_error"]


class SmokeValidationError(ValueError):
    """Raised when a smoke artifact or response violates the frozen protocol."""


class TokenDecoder(Protocol):
    """The tokenizer operation needed to retain the exact provider response."""

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        """Decode token IDs without removing special tokens."""
        ...


@dataclass(frozen=True, slots=True)
class SmokeInputs:
    """Verified local inputs required before creating a remote client."""

    messages: tuple[ChatMessage, ...]
    training_messages: tuple[ChatMessage, ...]
    prompt_sha256: str
    adapter_sampler_path: str
    tokenizer_path: Path
    protocol_revision: str


@dataclass(frozen=True, slots=True)
class SmokeRecord:
    """One immutable provider attempt for one model arm."""

    model_role: ModelRole
    prompt_sha256: str
    prompt_tokens: tuple[int, ...]
    response_tokens: tuple[int, ...]
    raw_response: str
    parsed_response: str
    status: GenerationStatus
    termination: str | None
    stop_reason: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class SamplingRuntime:
    """Objects shared by the two smoke requests."""

    tokenizer: TokenDecoder
    renderer: renderers.Renderer
    params: tinker.SamplingParams


@dataclass(frozen=True, slots=True)
class SmokeSeal:
    """Hashes binding the attempt marker and both paid responses."""

    attempt_sha256: str
    base_sha256: str
    adapter_sha256: str


def require_run_inputs() -> SmokeInputs:
    """Verify locks, prompt source, clean publication, and unused output paths."""
    verify_training_lock(PROJECT_ROOT, TRAINING_LOCK_PATH)
    experiment = verify_experiment_lock(TRAINING_LOCK_PATH, EXPERIMENT_PATH)
    _require_package_versions()
    for path in (ATTEMPT_PATH, BASE_OUTPUT_PATH, ADAPTER_OUTPUT_PATH, SEAL_PATH, RESULT_PATH):
        if path.exists():
            raise FileExistsError(f"format-smoke output already exists: {path}")

    training_messages = _load_training_conversation(TRAINING_PATH)
    messages = training_messages[:2]
    return SmokeInputs(
        messages=messages,
        training_messages=training_messages,
        prompt_sha256=canonical_sha256(list(messages)),
        adapter_sampler_path=require_string(
            required_field(experiment, "adapter_sampler_path"),
            "adapter_sampler_path",
        ),
        tokenizer_path=require_tokenizer_snapshot(),
        protocol_revision=_require_published_revision(),
    )


def build_renderers(
    tokenizer_path: Path,
) -> tuple[TokenDecoder, renderers.Renderer, renderers.Renderer]:
    """Build the smoke and original training renderers from one pinned tokenizer."""
    tokenizer = get_tokenizer(str(tokenizer_path))
    smoke_renderer = renderers.get_renderer(
        SMOKE_RENDERER_NAME,
        tokenizer,
        model_name=BASE_MODEL,
    )
    training_renderer = renderers.get_renderer(
        TRAINING_RENDERER_NAME,
        tokenizer,
        model_name=BASE_MODEL,
    )
    return cast(TokenDecoder, tokenizer), smoke_renderer, training_renderer


def build_sampling_params(renderer: renderers.Renderer) -> tinker.SamplingParams:
    """Build the same deterministic decoding policy as the failed canary."""
    return tinker.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        seed=SEED,
        stop=renderer.get_stop_sequences(),
    )


async def create_sampling_clients(
    service: tinker.ServiceClient,
    adapter_sampler_path: str,
) -> tuple[tinker.SamplingClient, tinker.SamplingClient]:
    """Create the base and immutable adapter clients with logical retries disabled."""
    retry_config = RetryConfig(enable_retry_logic=False)
    base = await service.create_sampling_client_async(
        base_model=BASE_MODEL,
        retry_config=retry_config,
    )
    adapter = await service.create_sampling_client_async(
        model_path=adapter_sampler_path,
        retry_config=retry_config,
    )
    return base, adapter


async def sample_once(
    model_role: ModelRole,
    client: tinker.SamplingClient,
    inputs: SmokeInputs,
    runtime: SamplingRuntime,
) -> SmokeRecord:
    """Make exactly one provider call and retain its exact response."""
    model_input = runtime.renderer.build_generation_prompt(_renderer_messages(inputs.messages))
    prompt_tokens = tuple(model_input.to_ints())
    try:
        response = await client.sample_async(
            prompt=model_input,
            num_samples=1,
            sampling_params=runtime.params,
        )
    except Exception as error:  # The protocol records once and never retries.
        return _error_record(
            model_role,
            inputs.prompt_sha256,
            prompt_tokens,
            "provider_error",
            f"provider_exception:{type(error).__name__}",
        )

    if len(response.sequences) != 1:
        return _error_record(
            model_role,
            inputs.prompt_sha256,
            prompt_tokens,
            "provider_error",
            f"provider_sequence_count:{len(response.sequences)}",
        )

    sequence = response.sequences[0]
    response_tokens = tuple(sequence.tokens)
    raw_response = ""
    try:
        raw_response = runtime.tokenizer.decode(
            list(response_tokens),
            skip_special_tokens=False,
        )
        message, termination = runtime.renderer.parse_response(list(response_tokens))
        parsed_response = renderers.get_text_content(message)
    except Exception as error:
        return SmokeRecord(
            model_role=model_role,
            prompt_sha256=inputs.prompt_sha256,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            raw_response=raw_response,
            parsed_response="",
            status="renderer_error",
            termination=None,
            stop_reason=sequence.stop_reason,
            error=f"renderer_exception:{type(error).__name__}",
        )
    has_tool_call = "tool_calls" in message or "unparsed_tool_calls" in message
    return SmokeRecord(
        model_role=model_role,
        prompt_sha256=inputs.prompt_sha256,
        prompt_tokens=prompt_tokens,
        response_tokens=response_tokens,
        raw_response=raw_response,
        parsed_response=parsed_response,
        status="renderer_error" if has_tool_call else "completed",
        termination=termination.value,
        stop_reason=sequence.stop_reason,
        error="renderer_tool_call" if has_tool_call else None,
    )


def gate_record(record: SmokeRecord) -> dict[str, object]:
    """Apply the strict target-free format gate without repairing output."""
    reason: str | None = None
    probabilities: dict[str, float] | None = None
    if record.status != "completed" or record.error is not None:
        reason = record.error or record.status
    elif record.termination != "stop_sequence":
        reason = "unclean_renderer_termination"
    elif record.stop_reason != "stop":
        reason = "unclean_provider_stop"
    else:
        try:
            prediction = parse_prediction(record.parsed_response, OUTCOMES)
        except ValueError:
            reason = "invalid_prediction_json"
        else:
            probabilities = prediction.distribution.as_dict()
    return {
        "passed": reason is None,
        "reason": reason,
        "status": record.status,
        "termination": record.termination,
        "stop_reason": record.stop_reason,
        "response_token_count": len(record.response_tokens),
        "probabilities": probabilities,
    }


def write_attempt_marker(inputs: SmokeInputs) -> str:
    """Exclusively mark the single run before remote-client construction."""
    value = {
        "schema_version": 1,
        "kind": "forecastfm_format_smoke_attempt",
        "status": "started_before_remote_client",
        "protocol_revision": inputs.protocol_revision,
        "runner_sha256": file_sha256(Path(__file__)),
        "training_lock_sha256": file_sha256(TRAINING_LOCK_PATH),
        "experiment_sha256": file_sha256(EXPERIMENT_PATH),
        "training_path_sha256": file_sha256(TRAINING_PATH),
        "training_row_index": TRAINING_ROW_INDEX,
        "assistant_target_sent": False,
        "prompt_sha256": inputs.prompt_sha256,
        "renderer": SMOKE_RENDERER_NAME,
        "training_renderer": TRAINING_RENDERER_NAME,
        "prompt_token_count": EXPECTED_PROMPT_TOKEN_COUNT,
        "decoding": decoding_settings(),
        "model_roles": list(MODEL_ROLES),
    }
    return _write_new_json(ATTEMPT_PATH, value)


def write_record(path: Path, record: SmokeRecord) -> str:
    """Durably write one paid response before starting another request."""
    _validate_record(record)
    value = {
        "schema_version": 1,
        "kind": "forecastfm_format_smoke_record",
        "record": _record_to_dict(record),
    }
    return _write_new_json(path, value)


def seal_outputs(base: SmokeRecord, adapter: SmokeRecord) -> tuple[str, SmokeSeal]:
    """Exclusively bind the attempt marker and complete paired outputs."""
    _validate_pair(base, adapter)
    seal = SmokeSeal(
        attempt_sha256=file_sha256(ATTEMPT_PATH),
        base_sha256=file_sha256(BASE_OUTPUT_PATH),
        adapter_sha256=file_sha256(ADAPTER_OUTPUT_PATH),
    )
    value = {
        "schema_version": 1,
        "kind": "forecastfm_format_smoke_seal",
        "status": "sealed_before_gate",
        "attempt_sha256": seal.attempt_sha256,
        "base_sha256": seal.base_sha256,
        "adapter_sha256": seal.adapter_sha256,
    }
    return _write_new_json(SEAL_PATH, value), seal


def write_gate_result(
    base_record: SmokeRecord,
    adapter_record: SmokeRecord,
    seal: SmokeSeal,
    seal_sha256: str,
) -> str:
    """Verify the raw seal, then write the target-free format gate."""
    _verify_seal(seal, seal_sha256)
    base = gate_record(base_record)
    adapter = gate_record(adapter_record)
    value = {
        "schema_version": 1,
        "kind": "forecastfm_format_smoke_result",
        "passed": base["passed"] is True and adapter["passed"] is True,
        "base": base,
        "adapter": adapter,
        "attempt_sha256": file_sha256(ATTEMPT_PATH),
        "seal_sha256": file_sha256(SEAL_PATH),
    }
    return _write_new_json(RESULT_PATH, value)


async def run() -> None:
    """Run exactly two paid requests and seal their raw outputs."""
    inputs = require_run_inputs()
    tokenizer, renderer, training_renderer = build_renderers(inputs.tokenizer_path)
    _require_training_prefix_match(inputs, renderer, training_renderer)
    runtime = SamplingRuntime(tokenizer, renderer, build_sampling_params(renderer))
    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    attempt_hash = write_attempt_marker(inputs)
    service = tinker.ServiceClient()
    base_client, adapter_client = await create_sampling_clients(
        service,
        inputs.adapter_sampler_path,
    )
    base = await sample_once("base", base_client, inputs, runtime)
    base_hash = write_record(BASE_OUTPUT_PATH, base)
    print("Generated 1/2: base format smoke.")
    adapter = await sample_once("adapter", adapter_client, inputs, runtime)
    adapter_hash = write_record(ADAPTER_OUTPUT_PATH, adapter)
    print("Generated 2/2: adapter format smoke.")
    seal_hash, seal = seal_outputs(base, adapter)
    result_hash = write_gate_result(base, adapter, seal, seal_hash)
    print(f"Attempt marker SHA-256: {attempt_hash}")
    print(f"Base output SHA-256: {base_hash}")
    print(f"Adapter output SHA-256: {adapter_hash}")
    print(f"Seal SHA-256: {seal_hash}")
    print(f"Gate result SHA-256: {result_hash}")
    print(RESULT_PATH.read_text(encoding="utf-8"), end="")


def _load_training_conversation(path: Path) -> tuple[ChatMessage, ...]:
    """Load and verify frozen row zero for the local prefix assertion."""
    with path.open(encoding="utf-8") as file:
        first_line = file.readline()
    if text_sha256(first_line.rstrip("\n")) != EXPECTED_TRAINING_ROW_SHA256:
        raise SmokeValidationError("training row zero differs from the frozen smoke source")
    record = parse_json_object(first_line)
    require_exact_keys(record, {"messages"}, "training conversation")
    values = require_list(required_field(record, "messages"), "messages")
    if len(values) != 3:
        raise SmokeValidationError("training row zero must contain three messages")
    messages = tuple(_chat_message(value) for value in values)
    if tuple(message["role"] for message in messages) != ("system", "user", "assistant"):
        raise SmokeValidationError("training row zero has an unexpected role sequence")
    prompt = messages[:2]
    user = parse_json_object(prompt[1]["content"])
    outcomes = tuple(
        require_string(value, "outcomes item")
        for value in require_list(required_field(user, "outcomes"), "outcomes")
    )
    if outcomes != OUTCOMES:
        raise SmokeValidationError("training row zero has unexpected outcomes")
    if canonical_sha256(list(prompt)) != EXPECTED_PROMPT_SHA256:
        raise SmokeValidationError("training prompt differs from the frozen smoke prompt")
    return messages


def _require_training_prefix_match(
    inputs: SmokeInputs,
    smoke_renderer: renderers.Renderer,
    training_renderer: renderers.Renderer,
) -> None:
    """Prove the smoke prompt recreates the exact context before the SFT target."""
    prompt_messages = _renderer_messages(inputs.messages)
    generation_tokens = smoke_renderer.build_generation_prompt(prompt_messages).to_ints()
    training_messages = _renderer_messages(inputs.training_messages)
    supervised, weights = training_renderer.build_supervised_example(
        training_messages,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    if len(generation_tokens) != EXPECTED_PROMPT_TOKEN_COUNT:
        raise SmokeValidationError("no-thinking prompt has an unexpected token count")
    first_weighted = next(
        (index for index in range(len(weights)) if float(weights[index]) != 0.0),
        None,
    )
    if first_weighted != len(generation_tokens):
        raise SmokeValidationError("no-thinking prompt does not end at the first SFT target token")
    if supervised.to_ints()[:first_weighted] != generation_tokens:
        raise SmokeValidationError("no-thinking prompt does not match the SFT target prefix")


def _renderer_messages(messages: Sequence[ChatMessage]) -> list[renderers.Message]:
    return [
        renderers.Message(role=message["role"], content=message["content"]) for message in messages
    ]


def _chat_message(value: object) -> ChatMessage:
    record = require_object(value, "message")
    require_exact_keys(record, {"role", "content"}, "message")
    role = require_string(required_field(record, "role"), "message.role")
    if role not in {"system", "user", "assistant"}:
        raise SmokeValidationError(f"unsupported message role: {role}")
    return ChatMessage(
        role=cast(Literal["system", "user", "assistant"], role),
        content=require_string(required_field(record, "content"), "message.content"),
    )


def _require_published_revision() -> str:
    if _git_output("status", "--porcelain", "--untracked-files=no"):
        raise SmokeValidationError("tracked files must be clean before the format smoke")
    runner_path = str(Path(__file__).resolve().relative_to(PROJECT_ROOT))
    if _git_output("status", "--porcelain", "--", runner_path):
        raise SmokeValidationError("format-smoke runner must be committed before use")
    head = _git_output("rev-parse", "HEAD")
    if head != _git_output("rev-parse", "origin/main"):
        raise SmokeValidationError("format-smoke protocol must be published to origin/main")
    return head


def _require_package_versions() -> None:
    expected = {
        "tinker": TINKER_VERSION,
        "tinker-cookbook": TINKER_COOKBOOK_VERSION,
    }
    for package, version in expected.items():
        if importlib.metadata.version(package) != version:
            raise SmokeValidationError(f"installed {package} version differs from the lock")


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _error_record(
    model_role: ModelRole,
    prompt_sha256: str,
    prompt_tokens: tuple[int, ...],
    status: Literal["provider_error"],
    error: str,
) -> SmokeRecord:
    return SmokeRecord(
        model_role=model_role,
        prompt_sha256=prompt_sha256,
        prompt_tokens=prompt_tokens,
        response_tokens=(),
        raw_response="",
        parsed_response="",
        status=status,
        termination=None,
        stop_reason=None,
        error=error,
    )


def _record_to_dict(record: SmokeRecord) -> dict[str, object]:
    return {
        "model_role": record.model_role,
        "prompt_sha256": record.prompt_sha256,
        "prompt_tokens": list(record.prompt_tokens),
        "response_tokens": list(record.response_tokens),
        "raw_response": record.raw_response,
        "parsed_response": record.parsed_response,
        "status": record.status,
        "termination": record.termination,
        "stop_reason": record.stop_reason,
        "error": record.error,
    }


def _validate_pair(base: SmokeRecord, adapter: SmokeRecord) -> None:
    records = (base, adapter)
    if tuple(record.model_role for record in records) != MODEL_ROLES:
        raise SmokeValidationError("outputs must contain exactly one base and one adapter record")
    if base.prompt_sha256 != adapter.prompt_sha256:
        raise SmokeValidationError("base and adapter prompt hashes differ")
    if base.prompt_tokens != adapter.prompt_tokens:
        raise SmokeValidationError("base and adapter prompt tokens differ")
    for record in records:
        _validate_record(record)


def _verify_seal(seal: SmokeSeal, seal_sha256: str) -> None:
    actual = SmokeSeal(
        attempt_sha256=file_sha256(ATTEMPT_PATH),
        base_sha256=file_sha256(BASE_OUTPUT_PATH),
        adapter_sha256=file_sha256(ADAPTER_OUTPUT_PATH),
    )
    if actual != seal or file_sha256(SEAL_PATH) != seal_sha256:
        raise SmokeValidationError("format-smoke raw artifacts differ from their seal")


def _validate_record(record: SmokeRecord) -> None:
    if not record.prompt_tokens:
        raise SmokeValidationError("prompt tokens must not be empty")
    if record.status == "provider_error":
        has_response = bool(record.response_tokens)
        has_termination = record.termination is not None or record.stop_reason is not None
        if has_response or has_termination or not record.error:
            raise SmokeValidationError("provider-error response fields are inconsistent")
    elif record.status == "renderer_error":
        if not record.response_tokens or record.stop_reason is None or not record.error:
            raise SmokeValidationError("renderer-error response fields are inconsistent")
    elif record.error is not None or record.termination is None or record.stop_reason is None:
        raise SmokeValidationError("completed response fields are inconsistent")


def _write_new_json(path: Path, value: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    return file_sha256(path)


def main() -> None:
    """Load the ignored local key without printing it, then run the smoke test."""
    if not os.environ.get("TINKER_API_KEY"):
        os.environ["TINKER_API_KEY"] = read_api_key(LOCAL_ENV_PATH)
    asyncio.run(run())


if __name__ == "__main__":
    main()
