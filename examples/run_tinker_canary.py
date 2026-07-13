"""Generate a frozen validation canary with the base and adapter models."""

from __future__ import annotations

import asyncio
import importlib.metadata
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import tinker
from examples.build_validation_canary_v2 import (
    EXPECTED_REMOTE_URL,
    protocol_commitments,
    v2_decoding_settings,
    verify_format_smoke,
    verify_retired_v1,
    verify_v2_manifest,
    verify_v2_selection,
)
from examples.run_tinker_sft_local import read_api_key
from tinker.lib.retry_handler import RetryConfig  # pyright: ignore[reportMissingTypeStubs]
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from forecastfm.canary import (
    PROMPT_COUNT,
    V1_PROTOCOL,
    CanaryManifest,
    CanaryPrompt,
    CanaryProtocol,
    CanaryValidationError,
    CompletedGeneration,
    GenerationRecord,
    RendererFailedGeneration,
    failed_generation,
    load_canary,
    renderer_failed_generation,
    seal_generation_outputs,
    successful_generation,
    write_attempt_marker,
    write_generation_records,
)
from forecastfm.integrity import file_sha256
from forecastfm.json_utils import require_string, required_field
from forecastfm.run_config import (
    BASE_MODEL,
    MAX_TOKENS,
    RENDERER_NAME,
    SEED,
    TEMPERATURE,
    TINKER_COOKBOOK_VERSION,
    TINKER_VERSION,
    TOP_K,
    TOP_P,
    decoding_settings,
    require_tokenizer_snapshot,
)
from forecastfm.run_lock import verify_experiment_lock, verify_training_lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV_PATH = PROJECT_ROOT / ".env"
TRAINING_LOCK_PATH = PROJECT_ROOT / "prospective" / "training_lock.json"
EXPERIMENT_LOCK_PATH = PROJECT_ROOT / "prospective" / "experiment.json"
CANARY_DIRECTORY = PROJECT_ROOT / "evaluation" / "validation_canary"
REMOTE_NAME = "origin"
REMOTE_REF = "refs/heads/main"
PROTOCOL_PATHS = (
    "src/forecastfm",
    "examples/build_validation_canary_v2.py",
    "examples/run_tinker_canary.py",
    "examples/run_tinker_canary_v2.py",
    "examples/score_validation_canary_v2.py",
)

type ModelRole = Literal["base", "adapter"]


@dataclass(frozen=True, slots=True)
class CanaryRunConfig:
    """The few protocol choices that differ between canary runs."""

    directory: Path
    protocol: CanaryProtocol
    entrypoint_path: Path
    require_published_commitments: bool = False

    @property
    def manifest_path(self) -> Path:
        """Return the frozen prompt manifest path."""
        return self.directory / "manifest.json"

    @property
    def prompts_path(self) -> Path:
        """Return the frozen target-free prompt path."""
        return self.directory / "prompts.jsonl"

    @property
    def raw_directory(self) -> Path:
        """Return the immutable raw-output directory."""
        return self.directory / "raw"

    @property
    def attempt_path(self) -> Path:
        """Return the exclusive attempt-marker path."""
        return self.raw_directory / "attempt.json"

    @property
    def base_output_path(self) -> Path:
        """Return the base-model generation path."""
        return self.raw_directory / "base.jsonl"

    @property
    def adapter_output_path(self) -> Path:
        """Return the adapter generation path."""
        return self.raw_directory / "adapter.jsonl"

    @property
    def output_manifest_path(self) -> Path:
        """Return the paired raw-output seal path."""
        return self.raw_directory / "manifest.json"


V1_CONFIG = CanaryRunConfig(
    directory=CANARY_DIRECTORY,
    protocol=V1_PROTOCOL,
    entrypoint_path=Path(__file__).resolve(),
)
CANARY_MANIFEST_PATH = V1_CONFIG.manifest_path
CANARY_PROMPTS_PATH = V1_CONFIG.prompts_path
RAW_DIRECTORY = V1_CONFIG.raw_directory
ATTEMPT_MARKER_PATH = V1_CONFIG.attempt_path
BASE_OUTPUT_PATH = V1_CONFIG.base_output_path
ADAPTER_OUTPUT_PATH = V1_CONFIG.adapter_output_path
OUTPUT_MANIFEST_PATH = V1_CONFIG.output_manifest_path


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


@dataclass(frozen=True, slots=True)
class RunInputs:
    """Verified inputs required before constructing a remote client."""

    prompts: tuple[CanaryPrompt, ...]
    adapter_sampler_path: str
    tokenizer_path: Path


@dataclass(frozen=True, slots=True)
class SamplingRuntime:
    """Objects shared by every request for one model arm."""

    client: tinker.SamplingClient
    tokenizer: TokenDecoder
    renderer: renderers.Renderer
    params: tinker.SamplingParams
    attempt_namespace: str = V1_PROTOCOL.attempt_namespace


def require_run_inputs(config: CanaryRunConfig = V1_CONFIG) -> RunInputs:
    """Verify every frozen input and require unused output paths."""
    verify_training_lock(PROJECT_ROOT, TRAINING_LOCK_PATH)
    experiment = verify_experiment_lock(TRAINING_LOCK_PATH, EXPERIMENT_LOCK_PATH)
    manifest, prompts = load_canary(config.manifest_path, config.prompts_path)
    adapter_path = require_string(
        required_field(experiment, "adapter_sampler_path"),
        "adapter_sampler_path",
    )
    _require_manifest_bindings(manifest, adapter_path, config)
    if len(prompts) != PROMPT_COUNT:
        raise CanaryValidationError(f"canary runner requires exactly {PROMPT_COUNT} prompts")
    if config.require_published_commitments:
        _require_package_versions()
        _require_protocol_commitments(manifest)
        require_published_revision(config, manifest.protocol_code_revision)
    for path in _output_paths(config):
        if path.exists():
            raise FileExistsError(f"canary output already exists: {path}")
    return RunInputs(
        prompts=prompts,
        adapter_sampler_path=adapter_path,
        tokenizer_path=require_tokenizer_snapshot(),
    )


def build_renderer(
    tokenizer_path: Path,
    renderer_name: str = RENDERER_NAME,
) -> tuple[TokenDecoder, renderers.Renderer]:
    """Build the pinned local tokenizer and selected Qwen renderer."""
    tokenizer = get_tokenizer(str(tokenizer_path))
    renderer = renderers.get_renderer(
        renderer_name,
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
        except Exception as error:  # The frozen policy records once and never retries.
            record = failed_generation(
                prompt,
                model_role,
                prompt_tokens,
                f"provider_exception:{type(error).__name__}",
                runtime.attempt_namespace,
            )
        else:
            record = _record_response(
                prompt,
                model_role,
                prompt_tokens,
                response,
                runtime,
            )
        records.append(record)
        completed = completed_before + index
        print(f"Generated {completed}/{total}: {model_role} prompt {prompt.sequence}.")
    return tuple(records)


def _record_response(
    prompt: CanaryPrompt,
    model_role: ModelRole,
    prompt_tokens: tuple[int, ...],
    response: tinker.SampleResponse,
    runtime: SamplingRuntime,
) -> GenerationRecord:
    if len(response.sequences) != 1:
        return failed_generation(
            prompt,
            model_role,
            prompt_tokens,
            f"provider_sequence_count:{len(response.sequences)}",
            runtime.attempt_namespace,
        )
    sequence = response.sequences[0]
    response_tokens = tuple(sequence.tokens)
    raw_response = ""
    parsed_response = ""
    termination_text: str | None = None
    try:
        raw_response = runtime.tokenizer.decode(
            list(response_tokens),
            skip_special_tokens=False,
        )
        message, termination = runtime.renderer.parse_response(list(response_tokens))
        termination_text = termination.value
        parsed_response = renderers.get_text_content(message)
    except Exception as error:
        return renderer_failed_generation(
            prompt,
            model_role,
            RendererFailedGeneration(
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                raw_response=raw_response,
                parsed_response=parsed_response,
                termination=termination_text,
                stop_reason=sequence.stop_reason,
                error=f"renderer_exception:{type(error).__name__}",
            ),
            runtime.attempt_namespace,
        )
    if "tool_calls" in message or "unparsed_tool_calls" in message:
        return renderer_failed_generation(
            prompt,
            model_role,
            RendererFailedGeneration(
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                raw_response=raw_response,
                parsed_response=parsed_response,
                termination=termination_text,
                stop_reason=sequence.stop_reason,
                error="renderer_tool_call",
            ),
            runtime.attempt_namespace,
        )
    return successful_generation(
        prompt,
        model_role,
        CompletedGeneration(
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            raw_response=raw_response,
            parsed_response=parsed_response,
            termination=termination_text,
            stop_reason=sequence.stop_reason,
        ),
        runtime.attempt_namespace,
    )


async def run(config: CanaryRunConfig = V1_CONFIG) -> None:
    """Run both frozen model arms and seal complete raw outputs."""
    inputs = require_run_inputs(config)
    tokenizer, renderer = build_renderer(inputs.tokenizer_path, config.protocol.renderer_name)
    sampling_params = build_sampling_params(renderer)
    config.raw_directory.mkdir(parents=True, exist_ok=True)
    attempt_hash = write_attempt_marker(
        config.attempt_path,
        config.manifest_path,
        config.prompts_path,
    )
    service = tinker.ServiceClient()
    base_client, adapter_client = await create_sampling_clients(
        service,
        inputs.adapter_sampler_path,
    )

    base_records = await generate_arm(
        inputs.prompts,
        "base",
        SamplingRuntime(
            base_client,
            tokenizer,
            renderer,
            sampling_params,
            config.protocol.attempt_namespace,
        ),
        0,
    )
    base_hash = write_generation_records(
        config.base_output_path,
        base_records,
        inputs.prompts,
        "base",
        config.protocol.attempt_namespace,
    )
    adapter_records = await generate_arm(
        inputs.prompts,
        "adapter",
        SamplingRuntime(
            adapter_client,
            tokenizer,
            renderer,
            sampling_params,
            config.protocol.attempt_namespace,
        ),
        len(inputs.prompts),
    )
    adapter_hash = write_generation_records(
        config.adapter_output_path,
        adapter_records,
        inputs.prompts,
        "adapter",
        config.protocol.attempt_namespace,
    )
    seal_generation_outputs(
        config.output_manifest_path,
        config.manifest_path,
        config.prompts_path,
        config.base_output_path,
        config.adapter_output_path,
    )
    print(f"Attempt marker SHA-256: {attempt_hash}")
    print(f"Base output SHA-256: {base_hash}")
    print(f"Adapter output SHA-256: {adapter_hash}")
    print(f"Sealed paired outputs at {config.output_manifest_path}.")


def expected_decoding(config: CanaryRunConfig) -> dict[str, object]:
    """Return the decoding commitment required by one protocol config."""
    if config.require_published_commitments:
        return v2_decoding_settings()
    return decoding_settings()


def _require_manifest_bindings(
    manifest: CanaryManifest,
    adapter_path: str,
    config: CanaryRunConfig,
) -> None:
    expected = (
        file_sha256(TRAINING_LOCK_PATH),
        file_sha256(EXPERIMENT_LOCK_PATH),
        BASE_MODEL,
        adapter_path,
    )
    actual = (
        manifest.training_lock_sha256,
        manifest.experiment_sha256,
        manifest.base_model,
        manifest.adapter_sampler_path,
    )
    if actual != expected:
        raise CanaryValidationError("canary manifest differs from the active model locks")
    if manifest.decoding != expected_decoding(config):
        raise CanaryValidationError("canary manifest has an unexpected decoding commitment")


def _require_protocol_commitments(
    manifest: CanaryManifest,
) -> None:
    retired_ids = verify_retired_v1()
    verify_format_smoke()
    selected_ids = verify_v2_selection(retired_ids)
    verify_v2_manifest(manifest, selected_ids)
    if manifest.protocol_commitments != protocol_commitments():
        raise CanaryValidationError("v2 protocol commitments differ from committed evidence")


def _require_package_versions() -> None:
    expected = {
        "tinker": TINKER_VERSION,
        "tinker-cookbook": TINKER_COOKBOOK_VERSION,
    }
    for package, version in expected.items():
        if importlib.metadata.version(package) != version:
            raise CanaryValidationError(f"installed {package} version differs from the lock")


def require_published_revision(config: CanaryRunConfig, protocol_revision: str) -> str:
    """Require clean, unchanged protocol code at the authoritative remote head."""
    if not _is_revision(protocol_revision):
        raise CanaryValidationError("canary protocol revision is invalid")
    if _git_output("status", "--porcelain", "--untracked-files=all"):
        raise CanaryValidationError("working tree must be clean before the v2 canary")
    head = _git_output("rev-parse", "HEAD")
    if not _is_revision(head):
        raise CanaryValidationError("Git HEAD is not a valid revision")
    _require_paths_at_head(config, head)
    _git_output("merge-base", "--is-ancestor", protocol_revision, head)
    changed_protocol = _git_output(
        "diff",
        "--name-only",
        f"{protocol_revision}..{head}",
        "--",
        *PROTOCOL_PATHS,
    )
    if changed_protocol:
        raise CanaryValidationError("canary protocol code changed after cohort freeze")
    if _authoritative_remote_revision() != head:
        raise CanaryValidationError("v2 canary protocol must be published to origin/main")
    return head


def _require_paths_at_head(config: CanaryRunConfig, head: str) -> None:
    """Require exact committed bytes for the shared runner and frozen cohort."""
    required_paths = {
        Path(__file__).resolve(),
        config.entrypoint_path.resolve(),
        config.manifest_path.resolve(),
        config.prompts_path.resolve(),
    }
    for path in required_paths:
        relative = str(path.relative_to(PROJECT_ROOT))
        if _git_output("ls-files", "--error-unmatch", "--", relative) != relative:
            raise CanaryValidationError(f"v2 protocol path must be committed: {relative}")
        head_object = _git_output("rev-parse", f"{head}:{relative}")
        working_object = _git_output("hash-object", "--", relative)
        if head_object != working_object:
            raise CanaryValidationError(f"local v2 protocol path differs from HEAD: {relative}")


def _authoritative_remote_revision() -> str:
    if _git_output("remote", "get-url", REMOTE_NAME) != EXPECTED_REMOTE_URL:
        raise CanaryValidationError("origin does not point to the frozen forecastfm repository")
    output = _git_output("ls-remote", "--exit-code", REMOTE_NAME, REMOTE_REF)
    lines = output.splitlines()
    if len(lines) != 1:
        raise CanaryValidationError("origin/main returned an unexpected publication record")
    fields = lines[0].split()
    if len(fields) != 2 or fields[1] != REMOTE_REF or not _is_revision(fields[0]):
        raise CanaryValidationError("origin/main returned an invalid publication record")
    return fields[0]


def _git_output(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise CanaryValidationError("Git publication verification failed") from error
    return result.stdout.strip()


def _is_revision(value: str) -> bool:
    return len(value) in {40, 64} and all(character in "0123456789abcdef" for character in value)


def _output_paths(config: CanaryRunConfig) -> tuple[Path, ...]:
    return (
        config.attempt_path,
        config.base_output_path,
        config.adapter_output_path,
        config.output_manifest_path,
    )


def main(config: CanaryRunConfig = V1_CONFIG) -> None:
    """Load the ignored local key without printing it, then run generation."""
    if not os.environ.get("TINKER_API_KEY"):
        os.environ["TINKER_API_KEY"] = read_api_key(LOCAL_ENV_PATH)
    asyncio.run(run(config))


if __name__ == "__main__":
    main()
