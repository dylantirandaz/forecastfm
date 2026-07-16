"""Run the published answer-blind outcome development evaluation."""

import asyncio
import fcntl
import hashlib
import importlib.metadata
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from math import exp, isfinite
from pathlib import Path
from typing import cast

import tinker
from examples.build_outcome_development_evaluation import (
    ATTEMPT_PATH,
    EXPECTED_GAME_COUNT,
    EXPECTED_REMOTE_URL,
    EXPERIMENT_PATH,
    MANIFEST_PATH,
    PROMPTS_PATH,
    PROTOCOL_PATHS,
    RAW_DIRECTORY,
    SCORING_POLICY,
    SOURCE_MANIFEST_PATH,
    TRAINING_LOCK_PATH,
    TRANSPORT_RETRY_NOTE,
    source_hashes,
)
from examples.tinker_outcome_inference import CandidateLogprobClient
from tinker.lib.retry_handler import RetryConfig  # pyright: ignore[reportMissingTypeStubs]
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from forecastfm.integrity import file_sha256
from forecastfm.json_utils import (
    parse_json_object,
    require_exact_keys,
    require_object,
    require_string,
    required_field,
)
from forecastfm.outcome import (
    TEAM_OUTCOME,
    TokenCodec,
    prediction_from_logprobs,
    require_label_token_ids,
)
from forecastfm.outcome_evaluation import (
    EvaluationPaths,
    ModelRole,
    OrientationResult,
    OutcomeEvaluationError,
    OutcomeEvaluationManifest,
    OutcomeEvaluationRecord,
    completed_record,
    failed_record,
    load_prompt_pairs,
    read_manifest,
    read_records,
    record_from_dict,
    record_to_dict,
    seal_outputs,
    verify_attempt_marker,
    write_records,
)
from forecastfm.outcome_run_lock import verify_outcome_training_lock
from forecastfm.publication import (
    require_paths_at_head,
    require_protocol_unchanged,
    require_published_head,
)
from forecastfm.run_config import (
    TINKER_COOKBOOK_VERSION,
    TINKER_VERSION,
    require_tokenizer_snapshot,
)
from forecastfm.run_lock import verify_experiment_lock
from forecastfm.tinker_data import ForecastRecord

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV_PATH = PROJECT_ROOT / ".env"
JOURNAL_PATH = RAW_DIRECTORY / "journal.jsonl"
BASE_PATH = RAW_DIRECTORY / "base.jsonl"
ADAPTER_PATH = RAW_DIRECTORY / "adapter.jsonl"
SEAL_PATH = RAW_DIRECTORY / "manifest.json"
PATHS = EvaluationPaths(
    MANIFEST_PATH,
    PROMPTS_PATH,
    ATTEMPT_PATH,
    JOURNAL_PATH,
    BASE_PATH,
    ADAPTER_PATH,
    SEAL_PATH,
)
_STARTED_KEYS = {
    "schema_version",
    "kind",
    "sequence",
    "model_role",
    "question_id",
    "attempt_sha256",
}
_COMPLETED_KEYS = {"schema_version", "kind", "attempt_sha256", "record"}
_RECOVERY_KEYS = {
    "schema_version",
    "kind",
    "attempt_sha256",
    "discarded_byte_count",
    "discarded_sha256",
}
type UnitKey = tuple[int, ModelRole]


class CandidateCallError(RuntimeError):
    """Sanitized terminal types from fully drained candidate calls."""

    def __init__(self, error_types: tuple[str, ...]) -> None:
        """Retain only stable exception type names."""
        self.error_types = error_types
        super().__init__(",".join(error_types))


class CandidateResponseError(RuntimeError):
    """Stable local validation code for one malformed candidate response."""

    def __init__(self, code: str) -> None:
        """Retain a non-sensitive response error code."""
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class RunInputs:
    """All locally verified values needed by remote inference."""

    manifest: OutcomeEvaluationManifest
    prompt_pairs: tuple[tuple[ForecastRecord, ForecastRecord], ...]
    renderer: renderers.Renderer
    label_token_ids: tuple[int, int]


def require_inputs() -> RunInputs:
    """Verify published target-free inputs before any attempt or remote client."""
    proof = require_published_head(PROJECT_ROOT, EXPECTED_REMOTE_URL, require_clean=False)
    manifest = read_manifest(MANIFEST_PATH)
    required_paths = (
        *PROTOCOL_PATHS,
        SOURCE_MANIFEST_PATH,
        TRAINING_LOCK_PATH,
        EXPERIMENT_PATH,
        MANIFEST_PATH,
        PROMPTS_PATH,
        ATTEMPT_PATH,
    )
    require_paths_at_head(PROJECT_ROOT, proof.commit, required_paths)
    require_protocol_unchanged(
        PROJECT_ROOT,
        manifest.protocol_revision,
        proof.commit,
        PROTOCOL_PATHS,
    )
    verify_manifest_bindings(manifest)
    verify_attempt_marker(ATTEMPT_PATH, MANIFEST_PATH, PROMPTS_PATH)
    require_package_versions()
    prompt_pairs = load_prompt_pairs(manifest, PROMPTS_PATH)
    tokenizer = get_tokenizer(str(require_tokenizer_snapshot()))
    label_token_ids = require_label_token_ids(cast(TokenCodec, tokenizer))
    if label_token_ids != (manifest.team_token_id, manifest.opponent_token_id):
        raise OutcomeEvaluationError("local label token IDs differ from the manifest")
    renderer = renderers.get_renderer(
        manifest.renderer_name,
        tokenizer,
        model_name=manifest.base_model,
    )
    for prompt_pair in prompt_pairs:
        _render_prompt_pair(renderer, prompt_pair)
    if SEAL_PATH.exists():
        raise FileExistsError("sealed outcome outputs already exist")
    return RunInputs(manifest, prompt_pairs, renderer, label_token_ids)


async def create_clients(
    service: tinker.ServiceClient,
    base_model: str,
    adapter_sampler_path: str,
) -> dict[ModelRole, tinker.SamplingClient]:
    """Create base and adapter clients with high-level retries disabled."""
    retry_config = RetryConfig(enable_retry_logic=False)
    base = await service.create_sampling_client_async(
        base_model=base_model,
        retry_config=retry_config,
    )
    adapter = await service.create_sampling_client_async(
        model_path=adapter_sampler_path,
        retry_config=retry_config,
    )
    return {"base": base, "adapter": adapter}


async def run() -> None:
    """Complete every arm once, compile ordered outputs, and seal them."""
    with exclusive_runner_lock():
        inputs = require_inputs()
        RAW_DIRECTORY.mkdir(parents=True, exist_ok=True)
        completed, started = read_journal(inputs.manifest, recover_partial=True)
        terminalize_interrupted(inputs, completed, started)
        _validate_completed_prompt_tokens(inputs, completed)
        _validate_existing_compiled(inputs.manifest, completed)
        pending = _pending_units(inputs.manifest, completed)
        if pending:
            if not os.environ.get("TINKER_API_KEY"):
                os.environ["TINKER_API_KEY"] = read_api_key(LOCAL_ENV_PATH)
            clients = await create_clients(
                tinker.ServiceClient(),
                inputs.manifest.base_model,
                inputs.manifest.adapter_sampler_path,
            )
            await run_pending(inputs, clients, completed, pending)
        _compile_and_seal(inputs, completed)


@contextmanager
def exclusive_runner_lock() -> Generator[None]:
    """Hold the attempt file lock for one complete runner lifecycle."""
    try:
        attempt_file = ATTEMPT_PATH.open("rb")
    except FileNotFoundError as error:
        raise OutcomeEvaluationError("published attempt commitment is missing") from error
    with attempt_file:
        try:
            fcntl.flock(attempt_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise OutcomeEvaluationError("another outcome runner is already active") from error
        try:
            yield
        finally:
            fcntl.flock(attempt_file.fileno(), fcntl.LOCK_UN)


def read_journal(
    manifest: OutcomeEvaluationManifest,
    *,
    recover_partial: bool = False,
) -> tuple[dict[UnitKey, OutcomeEvaluationRecord], set[UnitKey]]:
    """Reconstruct terminal and in-flight units from the durable journal."""
    completed: dict[UnitKey, OutcomeEvaluationRecord] = {}
    started: set[UnitKey] = set()
    if not JOURNAL_PATH.exists():
        return completed, started
    text = JOURNAL_PATH.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        if not recover_partial:
            raise OutcomeEvaluationError("journal must end with a newline")
        _recover_partial_journal()
        text = JOURNAL_PATH.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), 1):
        event = parse_json_object(line)
        kind = require_string(required_field(event, "kind"), "kind")
        if kind == "started":
            _read_started_event(event, manifest, started, completed)
        elif kind == "completed":
            _read_completed_event(event, manifest, started, completed)
        elif kind == "recovered_partial_tail":
            _read_recovery_event(event)
        else:
            raise OutcomeEvaluationError(f"unknown journal event on line {line_number}")
    return completed, started


def _read_started_event(
    event: dict[str, object],
    manifest: OutcomeEvaluationManifest,
    started: set[UnitKey],
    completed: dict[UnitKey, OutcomeEvaluationRecord],
) -> None:
    require_exact_keys(event, _STARTED_KEYS, "started journal event")
    if required_field(event, "schema_version") != 1:
        raise OutcomeEvaluationError("started journal event has the wrong schema")
    _require_attempt_binding(event)
    sequence = _journal_sequence(event, manifest)
    model_role = _journal_role(event)
    key = (sequence, model_role)
    if started - set(completed):
        raise OutcomeEvaluationError("journal contains overlapping active arms")
    if key in started or key in completed:
        raise OutcomeEvaluationError("journal contains a duplicate started event")
    if required_field(event, "question_id") != manifest.question_ids[sequence]:
        raise OutcomeEvaluationError("journal started event has the wrong question ID")
    started.add(key)


def _read_completed_event(
    event: dict[str, object],
    manifest: OutcomeEvaluationManifest,
    started: set[UnitKey],
    completed: dict[UnitKey, OutcomeEvaluationRecord],
) -> None:
    require_exact_keys(event, _COMPLETED_KEYS, "completed journal event")
    if required_field(event, "schema_version") != 1:
        raise OutcomeEvaluationError("completed journal event has the wrong schema")
    _require_attempt_binding(event)
    record = record_from_dict(required_field(event, "record"))
    if not 0 <= record.sequence < manifest.game_count:
        raise OutcomeEvaluationError("journal completed sequence is out of range")
    key = (record.sequence, record.model_role)
    if key not in started or key in completed:
        raise OutcomeEvaluationError("journal completed event has no unique start")
    if manifest.question_ids[record.sequence] != record.question_id:
        raise OutcomeEvaluationError("journal completed event has the wrong question ID")
    completed[key] = record


def _read_recovery_event(event: dict[str, object]) -> None:
    require_exact_keys(event, _RECOVERY_KEYS, "journal recovery event")
    if required_field(event, "schema_version") != 1:
        raise OutcomeEvaluationError("journal recovery event has the wrong schema")
    _require_attempt_binding(event)
    byte_count = required_field(event, "discarded_byte_count")
    digest = require_string(required_field(event, "discarded_sha256"), "discarded_sha256")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise OutcomeEvaluationError("journal recovery byte count is invalid")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise OutcomeEvaluationError("journal recovery digest is invalid")


def _recover_partial_journal() -> None:
    data = JOURNAL_PATH.read_bytes()
    if data.endswith(b"\n"):
        return
    last_newline = data.rfind(b"\n")
    complete_length = last_newline + 1
    discarded = data[complete_length:]
    with JOURNAL_PATH.open("r+b") as file:
        file.truncate(complete_length)
        file.flush()
        os.fsync(file.fileno())
    _append_journal(
        {
            "schema_version": 1,
            "kind": "recovered_partial_tail",
            "attempt_sha256": file_sha256(ATTEMPT_PATH),
            "discarded_byte_count": len(discarded),
            "discarded_sha256": hashlib.sha256(discarded).hexdigest(),
        }
    )


def terminalize_interrupted(
    inputs: RunInputs,
    completed: dict[UnitKey, OutcomeEvaluationRecord],
    started: set[UnitKey],
) -> None:
    """Mark ambiguous started-only units failed without calling them again."""
    for sequence, model_role in sorted(started - set(completed), key=_unit_sort_key):
        original, swapped = inputs.prompt_pairs[sequence]
        prompt_tokens = _render_prompt_pair(inputs.renderer, (original, swapped))
        record = failed_record(
            sequence,
            model_role,
            (original["question_id"], swapped["question_id"]),
            prompt_tokens,
            "indeterminate_interrupted",
        )
        _append_completed(record)
        completed[(sequence, model_role)] = record


def _pending_units(
    manifest: OutcomeEvaluationManifest,
    completed: dict[UnitKey, OutcomeEvaluationRecord],
) -> tuple[UnitKey, ...]:
    return tuple(
        (sequence, model_role)
        for model_role in cast(tuple[ModelRole, ModelRole], ("base", "adapter"))
        for sequence in range(manifest.game_count)
        if (sequence, model_role) not in completed
    )


async def run_pending(
    inputs: RunInputs,
    clients: dict[ModelRole, tinker.SamplingClient],
    completed: dict[UnitKey, OutcomeEvaluationRecord],
    pending: tuple[UnitKey, ...],
) -> None:
    """Run ordered arms and stop immediately after one terminal failure."""
    total = inputs.manifest.game_count * 2
    for sequence, model_role in pending:
        original, swapped = inputs.prompt_pairs[sequence]
        append_started(sequence, model_role, original["question_id"])
        record = await score_arm(
            sequence,
            model_role,
            clients[model_role],
            inputs,
            (original, swapped),
        )
        _append_completed(record)
        completed[(sequence, model_role)] = record
        count = len(completed)
        if count == 1 or count % 25 == 0 or count == total:
            print(f"Completed {count}/{total} model-game arms.")
        if record.status == "failed":
            raise OutcomeEvaluationError(
                f"stopped after terminal {model_role} failure for {record.question_id}: "
                f"{record.error}"
            )


async def score_arm(
    sequence: int,
    model_role: ModelRole,
    client: CandidateLogprobClient,
    inputs: RunInputs,
    prompt_pair: tuple[ForecastRecord, ForecastRecord],
) -> OutcomeEvaluationRecord:
    """Make one terminal four-candidate attempt for a game and model arm."""
    original, swapped = prompt_pair
    model_inputs = render_model_inputs(inputs.renderer, prompt_pair)
    prompt_tokens = (
        tuple(model_inputs[0].to_ints()),
        tuple(model_inputs[1].to_ints()),
    )
    try:
        orientations = await score_orientations(
            client,
            model_inputs,
            inputs.label_token_ids,
        )
        return completed_record(
            sequence,
            model_role,
            (original["question_id"], swapped["question_id"]),
            prompt_tokens,
            orientations,
        )
    except CandidateCallError as error:
        failure = f"candidate_call_exception:{','.join(error.error_types)}"
        return failed_record(
            sequence,
            model_role,
            (original["question_id"], swapped["question_id"]),
            prompt_tokens,
            failure,
        )
    except Exception as error:  # One terminal application attempt; never retried here.
        return failed_record(
            sequence,
            model_role,
            (original["question_id"], swapped["question_id"]),
            prompt_tokens,
            f"provider_or_renderer_exception:{type(error).__name__}",
        )


async def score_orientations(
    client: CandidateLogprobClient,
    prompts: tuple[tinker.ModelInput, tinker.ModelInput],
    label_token_ids: tuple[int, int],
) -> tuple[OrientationResult, OrientationResult]:
    """Score both prompt orientations and drain all four candidate calls."""
    results = await asyncio.gather(
        _score_orientation(client, prompts[0], label_token_ids),
        _score_orientation(client, prompts[1], label_token_ids),
        return_exceptions=True,
    )
    _raise_candidate_errors(results)
    return cast(tuple[OrientationResult, OrientationResult], results)


async def _score_orientation(
    client: CandidateLogprobClient,
    prompt: tinker.ModelInput,
    label_token_ids: tuple[int, int],
) -> OrientationResult:
    results = await asyncio.gather(
        _label_logprob(client, prompt, label_token_ids[0]),
        _label_logprob(client, prompt, label_token_ids[1]),
        return_exceptions=True,
    )
    _raise_candidate_errors(results)
    team_logprob, opponent_logprob = cast(tuple[float, float], results)
    probability = prediction_from_logprobs(
        team_logprob,
        opponent_logprob,
    ).distribution.probability_for(TEAM_OUTCOME)
    return OrientationResult(
        team_logprob=team_logprob,
        opponent_logprob=opponent_logprob,
        team_probability=probability,
        valid_label_mass=exp(team_logprob) + exp(opponent_logprob),
    )


async def _label_logprob(
    client: CandidateLogprobClient,
    prompt: tinker.ModelInput,
    token_id: int,
) -> float:
    full_prompt = prompt.append_int(token_id)
    values = await client.compute_logprobs_async(full_prompt)
    if len(values) != full_prompt.length:
        raise CandidateResponseError("unexpected_logprob_count")
    value = values[-1]
    if value is None or not isfinite(value):
        raise CandidateResponseError("missing_or_nonfinite_label_logprob")
    if value > 0.000001:
        raise CandidateResponseError("positive_label_logprob")
    return value


def _raise_candidate_errors(values: tuple[object, ...]) -> None:
    errors = tuple(value for value in values if isinstance(value, BaseException))
    for error in errors:
        if isinstance(error, asyncio.CancelledError):
            raise error
    if errors:
        names = tuple(name for error in errors for name in _candidate_error_types(error))
        raise CandidateCallError(tuple(sorted(set(names))))


def _candidate_error_types(error: BaseException) -> tuple[str, ...]:
    if isinstance(error, CandidateCallError):
        return error.error_types
    if isinstance(error, CandidateResponseError):
        return (error.code,)
    return (type(error).__name__,)


def render_model_inputs(
    renderer: renderers.Renderer,
    prompt_pair: tuple[ForecastRecord, ForecastRecord],
) -> tuple[tinker.ModelInput, tinker.ModelInput]:
    """Render one frozen original/swap prompt pair to exact model inputs."""
    original, swapped = prompt_pair
    rendered: list[tinker.ModelInput] = []
    for record in (original, swapped):
        messages = [
            renderers.Message(role=message["role"], content=message["content"])
            for message in record["messages"]
        ]
        rendered.append(renderer.build_generation_prompt(messages))
    return (rendered[0], rendered[1])


def _render_prompt_pair(
    renderer: renderers.Renderer,
    prompt_pair: tuple[ForecastRecord, ForecastRecord],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    original, swapped = render_model_inputs(renderer, prompt_pair)
    return (tuple(original.to_ints()), tuple(swapped.to_ints()))


def append_started(sequence: int, model_role: ModelRole, question_id: str) -> None:
    """Durably record one arm start before its provider calls."""
    _append_journal(
        {
            "schema_version": 1,
            "kind": "started",
            "sequence": sequence,
            "model_role": model_role,
            "question_id": question_id,
            "attempt_sha256": file_sha256(ATTEMPT_PATH),
        }
    )


def _append_completed(record: OutcomeEvaluationRecord) -> None:
    _append_journal(
        {
            "schema_version": 1,
            "kind": "completed",
            "attempt_sha256": file_sha256(ATTEMPT_PATH),
            "record": record_to_dict(record),
        }
    )


def _append_journal(event: dict[str, object]) -> None:
    with JOURNAL_PATH.open("a", encoding="utf-8") as file:
        file.write(f"{json.dumps(event, sort_keys=True, allow_nan=False)}\n")
        file.flush()
        os.fsync(file.fileno())


def _require_attempt_binding(event: dict[str, object]) -> None:
    if required_field(event, "attempt_sha256") != file_sha256(ATTEMPT_PATH):
        raise OutcomeEvaluationError("journal event differs from the frozen attempt")


def _compile_and_seal(
    inputs: RunInputs,
    completed: dict[UnitKey, OutcomeEvaluationRecord],
) -> None:
    manifest = inputs.manifest
    _validate_completed_prompt_tokens(inputs, completed)
    base = tuple(completed[(sequence, "base")] for sequence in range(manifest.game_count))
    adapter = tuple(completed[(sequence, "adapter")] for sequence in range(manifest.game_count))
    _require_or_write_records(BASE_PATH, base, manifest, "base")
    _require_or_write_records(ADAPTER_PATH, adapter, manifest, "adapter")
    require_journal_matches(manifest, base, adapter)
    seal_outputs(PATHS, datetime.now(UTC))
    print(f"Sealed answer-blind outcome outputs at {SEAL_PATH}.")


def require_journal_matches(
    manifest: OutcomeEvaluationManifest,
    base: tuple[OutcomeEvaluationRecord, ...],
    adapter: tuple[OutcomeEvaluationRecord, ...],
) -> None:
    """Require one durable start and the exact terminal record for every arm."""
    expected = {(record.sequence, record.model_role): record for record in (*base, *adapter)}
    completed, started = read_journal(manifest)
    if completed != expected or started != set(expected):
        raise OutcomeEvaluationError("journal differs from the complete compiled outputs")


def _validate_completed_prompt_tokens(
    inputs: RunInputs,
    completed: dict[UnitKey, OutcomeEvaluationRecord],
) -> None:
    rendered: dict[int, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for (sequence, _model_role), record in completed.items():
        if sequence not in rendered:
            rendered[sequence] = _render_prompt_pair(
                inputs.renderer,
                inputs.prompt_pairs[sequence],
            )
        actual = (record.original_prompt_tokens, record.swapped_prompt_tokens)
        if actual != rendered[sequence]:
            raise OutcomeEvaluationError("stored tokens differ from the frozen rendered prompt")


def _validate_existing_compiled(
    manifest: OutcomeEvaluationManifest,
    completed: dict[UnitKey, OutcomeEvaluationRecord],
) -> None:
    compiled_arms: tuple[tuple[Path, ModelRole], ...] = (
        (BASE_PATH, "base"),
        (ADAPTER_PATH, "adapter"),
    )
    for path, model_role in compiled_arms:
        if not path.exists():
            continue
        records = read_records(path, manifest, model_role)
        try:
            expected = tuple(
                completed[(sequence, model_role)] for sequence in range(manifest.game_count)
            )
        except KeyError as error:
            raise OutcomeEvaluationError(
                f"compiled {model_role} output has no complete journal evidence"
            ) from error
        if records != expected:
            raise OutcomeEvaluationError(f"compiled {model_role} output differs from the journal")


def verify_manifest_bindings(manifest: OutcomeEvaluationManifest) -> None:
    """Rebind the manifest to the published source, locks, and exact policy."""
    training_lock = verify_outcome_training_lock(PROJECT_ROOT, TRAINING_LOCK_PATH)
    experiment = verify_experiment_lock(TRAINING_LOCK_PATH, EXPERIMENT_PATH)
    source_manifest = parse_json_object(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    source_prompt_hash, source_answer_hash = source_hashes(source_manifest)
    locked_data = require_object(required_field(training_lock, "data"), "data")
    model = require_object(required_field(training_lock, "model"), "model")
    expected = (
        manifest.source_manifest_sha256,
        manifest.source_prompts_sha256,
        manifest.source_answers_sha256,
        manifest.frozen_prompts_sha256,
        manifest.training_lock_sha256,
        manifest.experiment_sha256,
        manifest.base_model,
        manifest.renderer_name,
        manifest.adapter_sampler_path,
        manifest.transport_retry_note,
    )
    actual = (
        file_sha256(SOURCE_MANIFEST_PATH),
        source_prompt_hash,
        source_answer_hash,
        source_prompt_hash,
        file_sha256(TRAINING_LOCK_PATH),
        file_sha256(EXPERIMENT_PATH),
        require_string(required_field(model, "base_model"), "base_model"),
        require_string(required_field(model, "renderer"), "renderer"),
        require_string(required_field(experiment, "adapter_sampler_path"), "adapter_sampler_path"),
        TRANSPORT_RETRY_NOTE,
    )
    if actual != expected:
        raise OutcomeEvaluationError("evaluation manifest differs from its source or model locks")
    if required_field(locked_data, "manifest_sha256") != manifest.source_manifest_sha256:
        raise OutcomeEvaluationError("evaluation source differs from the training-locked data")
    if manifest.game_count != EXPECTED_GAME_COUNT:
        raise OutcomeEvaluationError("evaluation does not contain the full development cohort")
    if manifest.scoring_policy != SCORING_POLICY:
        raise OutcomeEvaluationError("evaluation scoring policy differs from the protocol")


def _require_or_write_records(
    path: Path,
    records: tuple[OutcomeEvaluationRecord, ...],
    manifest: OutcomeEvaluationManifest,
    model_role: ModelRole,
) -> None:
    if not path.exists():
        write_records(path, records, manifest, model_role)
        return
    if read_records(path, manifest, model_role) != records:
        raise OutcomeEvaluationError(f"compiled {model_role} output differs from the journal")


def require_package_versions() -> None:
    """Require the exact installed Tinker packages from the training lock."""
    expected = {"tinker": TINKER_VERSION, "tinker-cookbook": TINKER_COOKBOOK_VERSION}
    for package, version in expected.items():
        if importlib.metadata.version(package) != version:
            raise OutcomeEvaluationError(f"installed {package} version differs from the lock")


def read_api_key(path: Path) -> str:
    """Read one ignored local Tinker key assignment without printing it."""
    try:
        name, separator, raw_value = path.read_text(encoding="utf-8").strip().partition("=")
    except FileNotFoundError as error:
        raise RuntimeError(f"local config is missing: {path}") from error
    if name != "TINKER_API_KEY" or not separator:
        raise RuntimeError(".env must contain exactly one TINKER_API_KEY assignment")
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    if not value:
        raise RuntimeError("TINKER_API_KEY is empty in .env")
    return value


def _journal_sequence(
    event: dict[str, object],
    manifest: OutcomeEvaluationManifest,
) -> int:
    value = required_field(event, "sequence")
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < manifest.game_count
    ):
        raise OutcomeEvaluationError("journal sequence is out of range")
    return value


def _journal_role(event: dict[str, object]) -> ModelRole:
    value = require_string(required_field(event, "model_role"), "model_role")
    if value not in {"base", "adapter"}:
        raise OutcomeEvaluationError("journal model role is invalid")
    return cast(ModelRole, value)


def _unit_sort_key(value: UnitKey) -> tuple[int, int]:
    sequence, role = value
    return (0 if role == "base" else 1, sequence)


def main() -> None:
    """Run or safely resume the answer-blind paid evaluation."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
