"""Generate the frozen validation canary with the base and adapter models."""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import tinker
from examples.run_tinker_sft_local import read_api_key
from tinker.lib.retry_handler import RetryConfig  # pyright: ignore[reportMissingTypeStubs]
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from forecastfm.canary import (
    CanaryPrompt,
    CompletedGeneration,
    GenerationRecord,
    failed_generation,
    load_canary,
    seal_generation_outputs,
    successful_generation,
    write_attempt_marker,
    write_generation_records,
)
from forecastfm.json_utils import require_string, required_field
from forecastfm.run_config import (
    BASE_MODEL,
    MAX_TOKENS,
    RENDERER_NAME,
    SEED,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    require_tokenizer_snapshot,
)
from forecastfm.run_lock import verify_experiment_lock, verify_training_lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV_PATH = PROJECT_ROOT / ".env"
TRAINING_LOCK_PATH = PROJECT_ROOT / "prospective" / "training_lock.json"
EXPERIMENT_LOCK_PATH = PROJECT_ROOT / "prospective" / "experiment.json"
CANARY_DIRECTORY = PROJECT_ROOT / "evaluation" / "validation_canary"
CANARY_MANIFEST_PATH = CANARY_DIRECTORY / "manifest.json"
CANARY_PROMPTS_PATH = CANARY_DIRECTORY / "prompts.jsonl"
RAW_DIRECTORY = CANARY_DIRECTORY / "raw"
ATTEMPT_MARKER_PATH = RAW_DIRECTORY / "attempt.json"
BASE_OUTPUT_PATH = RAW_DIRECTORY / "base.jsonl"
ADAPTER_OUTPUT_PATH = RAW_DIRECTORY / "adapter.jsonl"
OUTPUT_MANIFEST_PATH = RAW_DIRECTORY / "manifest.json"

type ModelRole = Literal["base", "adapter"]


class TokenDecoder(Protocol):
    """The one tokenizer operation generation needs after rendering."""

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        """Decode generated tokens without dropping special tokens."""
        ...


@dataclass(frozen=True)
class RunInputs:
    """Verified inputs required before constructing a remote client."""

    prompts: tuple[CanaryPrompt, ...]
    adapter_sampler_path: str
    tokenizer_path: Path


@dataclass(frozen=True)
class SamplingRuntime:
    """Objects shared by every request for one model arm."""

    client: tinker.SamplingClient
    tokenizer: TokenDecoder
    renderer: renderers.Renderer
    params: tinker.SamplingParams


def require_run_inputs() -> RunInputs:
    """Verify every frozen input and require unused output paths."""
    verify_training_lock(PROJECT_ROOT, TRAINING_LOCK_PATH)
    experiment = verify_experiment_lock(TRAINING_LOCK_PATH, EXPERIMENT_LOCK_PATH)
    _manifest, prompts = load_canary(CANARY_MANIFEST_PATH, CANARY_PROMPTS_PATH)
    adapter_path = require_string(
        required_field(experiment, "adapter_sampler_path"),
        "adapter_sampler_path",
    )
    for path in (
        ATTEMPT_MARKER_PATH,
        BASE_OUTPUT_PATH,
        ADAPTER_OUTPUT_PATH,
        OUTPUT_MANIFEST_PATH,
    ):
        if path.exists():
            raise FileExistsError(f"canary output already exists: {path}")
    tokenizer_path = require_tokenizer_snapshot()
    return RunInputs(
        prompts=prompts,
        adapter_sampler_path=adapter_path,
        tokenizer_path=tokenizer_path,
    )


def build_renderer(tokenizer_path: Path) -> tuple[TokenDecoder, renderers.Renderer]:
    """Build the locked local tokenizer and Qwen renderer."""
    tokenizer = get_tokenizer(str(tokenizer_path))
    renderer = renderers.get_renderer(
        RENDERER_NAME,
        tokenizer,
        model_name=BASE_MODEL,
    )
    return cast(TokenDecoder, tokenizer), renderer


def build_sampling_params(renderer: renderers.Renderer) -> tinker.SamplingParams:
    """Build the exact decoding policy from the training lock."""
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
    """Create both model arms with SDK-level logical retries disabled."""
    retry_config = RetryConfig(enable_retry_logic=False)
    base_client = await service.create_sampling_client_async(
        base_model=BASE_MODEL,
        retry_config=retry_config,
    )
    adapter_client = await service.create_sampling_client_async(
        model_path=adapter_sampler_path,
        retry_config=retry_config,
    )
    return base_client, adapter_client


def _render_messages(prompt: CanaryPrompt) -> list[renderers.Message]:
    return [
        renderers.Message(role=message["role"], content=message["content"])
        for message in prompt.messages
    ]


async def generate_arm(
    prompts: tuple[CanaryPrompt, ...],
    model_role: ModelRole,
    runtime: SamplingRuntime,
    completed_before: int,
) -> tuple[GenerationRecord, ...]:
    """Make exactly one remote sampling call for each frozen prompt."""
    records: list[GenerationRecord] = []
    total = len(prompts) * 2
    for index, prompt in enumerate(prompts, start=1):
        model_input = runtime.renderer.build_generation_prompt(_render_messages(prompt))
        prompt_tokens = tuple(model_input.to_ints())
        try:
            response = await runtime.client.sample_async(
                prompt=model_input,
                num_samples=1,
                sampling_params=runtime.params,
            )
        except Exception as error:  # The frozen policy records failures and never resamples.
            record = failed_generation(
                prompt,
                model_role,
                prompt_tokens,
                f"provider_exception:{type(error).__name__}",
            )
        else:
            if len(response.sequences) != 1:
                record = failed_generation(
                    prompt,
                    model_role,
                    prompt_tokens,
                    f"provider_sequence_count:{len(response.sequences)}",
                )
            else:
                sequence = response.sequences[0]
                response_tokens = tuple(sequence.tokens)
                raw_response = runtime.tokenizer.decode(
                    list(response_tokens),
                    skip_special_tokens=False,
                )
                message, termination = runtime.renderer.parse_response(list(response_tokens))
                if "tool_calls" in message or "unparsed_tool_calls" in message:
                    record = failed_generation(
                        prompt,
                        model_role,
                        prompt_tokens,
                        "renderer_tool_call",
                    )
                else:
                    parsed_response = renderers.get_text_content(message)
                    record = successful_generation(
                        prompt,
                        model_role,
                        CompletedGeneration(
                            prompt_tokens=prompt_tokens,
                            response_tokens=response_tokens,
                            raw_response=raw_response,
                            parsed_response=parsed_response,
                            termination=termination.value,
                            stop_reason=sequence.stop_reason,
                        ),
                    )
        records.append(record)
        completed = completed_before + index
        print(f"Generated {completed}/{total}: {model_role} prompt {prompt.sequence}.")
    return tuple(records)


async def run() -> None:
    """Run both frozen model arms and seal complete raw outputs."""
    inputs = require_run_inputs()
    RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)
    attempt_hash = write_attempt_marker(
        ATTEMPT_MARKER_PATH,
        CANARY_MANIFEST_PATH,
        CANARY_PROMPTS_PATH,
    )
    tokenizer, renderer = build_renderer(inputs.tokenizer_path)
    sampling_params = build_sampling_params(renderer)
    service = tinker.ServiceClient()
    base_client, adapter_client = await create_sampling_clients(
        service,
        inputs.adapter_sampler_path,
    )

    base_records = await generate_arm(
        inputs.prompts,
        "base",
        SamplingRuntime(base_client, tokenizer, renderer, sampling_params),
        0,
    )
    adapter_records = await generate_arm(
        inputs.prompts,
        "adapter",
        SamplingRuntime(adapter_client, tokenizer, renderer, sampling_params),
        len(inputs.prompts),
    )

    base_hash = write_generation_records(
        BASE_OUTPUT_PATH,
        base_records,
        inputs.prompts,
        "base",
    )
    adapter_hash = write_generation_records(
        ADAPTER_OUTPUT_PATH,
        adapter_records,
        inputs.prompts,
        "adapter",
    )
    seal_generation_outputs(
        OUTPUT_MANIFEST_PATH,
        CANARY_MANIFEST_PATH,
        CANARY_PROMPTS_PATH,
        BASE_OUTPUT_PATH,
        ADAPTER_OUTPUT_PATH,
    )
    print(f"Attempt marker SHA-256: {attempt_hash}")
    print(f"Base output SHA-256: {base_hash}")
    print(f"Adapter output SHA-256: {adapter_hash}")
    print(f"Sealed paired outputs at {OUTPUT_MANIFEST_PATH}.")


def main() -> None:
    """Load the ignored local key without printing it, then run generation."""
    if not os.environ.get("TINKER_API_KEY"):
        os.environ["TINKER_API_KEY"] = read_api_key(LOCAL_ENV_PATH)
    asyncio.run(run())


if __name__ == "__main__":
    main()
