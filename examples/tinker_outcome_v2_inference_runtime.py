"""Answer-blind, single-application-attempt Tinker inference for outcome-v2."""

from __future__ import annotations

import asyncio
import fcntl
import os
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import BinaryIO, Protocol, cast

import tinker
from tinker.lib.retry_handler import RetryConfig  # pyright: ignore[reportMissingTypeStubs]
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from forecastfm.integrity import bytes_sha256, canonical_json
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_object,
    require_string,
    required_field,
)
from forecastfm.nba_evaluation_gate import (
    read_nba_evaluation_forecasts_jsonl,
    write_nba_evaluation_forecasts_jsonl,
)
from forecastfm.nba_feature_rows import (
    NbaRichFeatureRow,
    read_nba_feature_rows_jsonl_bytes,
)
from forecastfm.outcome import (
    OPPONENT_LABEL,
    TEAM_LABEL,
    TokenCodec,
    require_label_token_ids,
)
from forecastfm.outcome_v2_config import MAX_LENGTH, OUTCOME_RENDERER_NAME
from forecastfm.outcome_v2_inference import (
    InferenceRecord,
    OrientationScore,
    OutcomeV2GenerationArtifacts,
    OutcomeV2GenerationLock,
    OutcomeV2InferenceError,
    binary_forecasts_from_inference_records,
    build_orientation_score,
    build_outcome_v2_prompt_records,
    completed_inference_record,
    failed_inference_record,
    outcome_v2_inference_record_from_payload,
    outcome_v2_inference_record_payload,
    outcome_v2_prompt_pairs_jsonl_bytes,
    read_outcome_v2_inference_records,
    rendered_prompt_token_ids_sha256,
    sanitize_inference_failure,
    verify_outcome_v2_generation_lock,
    write_outcome_v2_inference_records,
)
from forecastfm.run_config import (
    BASE_MODEL,
    require_pinned_tinker_packages,
    require_tokenizer_snapshot,
)
from forecastfm.tinker_data import ForecastRecord

JOURNAL_SCHEMA_VERSION = 1

_STARTED_KIND = "started"
_TERMINAL_KIND = "terminal"
_GENERATION_LOCK_METADATA_KEY = "outcome_v2_generation_lock_sha256"
_EVENT_KEYS = {
    "schema_version",
    "kind",
    "generation_lock_sha256",
    "sequence",
    "question_id",
    "feature_row_sha256",
    "original_prompt_token_ids_sha256",
    "swapped_prompt_token_ids_sha256",
    "record",
}
type PrecommitVerifier = Callable[[OutcomeV2GenerationLock, Path], None]
type SamplingClientFactory = Callable[
    [str, str, RetryConfig],
    Awaitable[CandidateLogprobClient],
]
type JsonObject = dict[str, object]


class CandidateLogprobClient(Protocol):
    """Minimal sampling-client surface used by fixed-candidate inference."""

    async def compute_logprobs_async(
        self,
        prompt: tinker.ModelInput,
    ) -> list[float | None]:
        """Return one value per input token."""
        ...


@dataclass(frozen=True, slots=True)
class OutcomeV2InferenceRuntimePaths:
    """Immutable inputs and append-only outputs for one generation attempt."""

    artifacts: OutcomeV2GenerationArtifacts
    generation_lock_path: Path
    prompts_path: Path
    journal_path: Path
    inference_records_path: Path
    forecasts_path: Path


@dataclass(frozen=True, slots=True)
class RenderedOutcomeV2Game:
    """One row and its four fully rendered candidate inputs."""

    row: NbaRichFeatureRow
    original_prompt_token_ids_sha256: str
    swapped_prompt_token_ids_sha256: str
    candidate_inputs: tuple[
        tinker.ModelInput,
        tinker.ModelInput,
        tinker.ModelInput,
        tinker.ModelInput,
    ]


@dataclass(frozen=True, slots=True)
class PreparedOutcomeV2Inference:
    """All local checks and rendering completed before client construction."""

    generation_lock: OutcomeV2GenerationLock
    sampler_path: str
    games: tuple[RenderedOutcomeV2Game, ...]


@dataclass(slots=True)
class _JournalState:
    started: set[int]
    terminal: dict[int, InferenceRecord]


async def run_outcome_v2_inference(
    paths: OutcomeV2InferenceRuntimePaths,
    verify_precommit: PrecommitVerifier,
    *,
    create_sampling_client: SamplingClientFactory | None = None,
) -> tuple[InferenceRecord, ...]:
    """Run every unstarted game once and compile complete canonical records."""
    prepared = prepare_outcome_v2_inference(paths)
    rows = tuple(game.row for game in prepared.games)
    _write_or_verify_bytes(
        paths.prompts_path,
        outcome_v2_prompt_pairs_jsonl_bytes(rows),
        "outcome-v2 prompt pairs",
    )
    verify_precommit(prepared.generation_lock, paths.generation_lock_path)
    factory = create_sampling_client or _create_sampling_client

    with _exclusive_journal(paths.journal_path) as journal:
        state = _read_journal(journal, prepared)
        _terminalize_interrupted(journal, prepared, state)
        pending = tuple(
            sequence for sequence in range(len(prepared.games)) if sequence not in state.terminal
        )
        if pending:
            retry_config = RetryConfig(enable_retry_logic=False)
            client = await factory(
                prepared.sampler_path,
                prepared.generation_lock.sha256,
                retry_config,
            )
            await _run_pending(journal, prepared, state, client, pending)
        records = tuple(state.terminal[index] for index in range(len(prepared.games)))
        compiled = _compile_records(paths.inference_records_path, prepared, records)
        _compile_forecasts(paths.forecasts_path, prepared, compiled)
        return compiled


def prepare_outcome_v2_inference(
    paths: OutcomeV2InferenceRuntimePaths,
) -> PreparedOutcomeV2Inference:
    """Verify locks and render all candidate inputs locally before any client exists."""
    require_pinned_tinker_packages()
    generation_lock = verify_outcome_v2_generation_lock(
        paths.artifacts,
        paths.generation_lock_path,
    )
    feature_bytes = _read_feature_bytes(paths.artifacts.feature_rows_path)
    rows = read_nba_feature_rows_jsonl_bytes(feature_bytes)
    lock_record = generation_lock.to_record()
    if bytes_sha256(feature_bytes) != _string(lock_record, "feature_rows_sha256"):
        raise RuntimeError("feature rows differ from the verified generation lock")

    tokenizer = get_tokenizer(str(require_tokenizer_snapshot()))
    label_token_ids = require_label_token_ids(cast(TokenCodec, tokenizer))
    _require_locked_runtime(lock_record, label_token_ids)
    renderer = renderers.get_renderer(
        OUTCOME_RENDERER_NAME,
        tokenizer,
        model_name=BASE_MODEL,
    )
    games = _render_games(renderer, rows, label_token_ids)
    if paths.artifacts.feature_rows_path.read_bytes() != feature_bytes:
        raise RuntimeError("feature rows changed during local inference preparation")
    if (
        verify_outcome_v2_generation_lock(
            paths.artifacts,
            paths.generation_lock_path,
        )
        != generation_lock
    ):
        raise RuntimeError("generation inputs changed during local inference preparation")
    return PreparedOutcomeV2Inference(
        generation_lock=generation_lock,
        sampler_path=_string(lock_record, "sampler_path"),
        games=games,
    )


async def _create_sampling_client(
    sampler_path: str,
    generation_lock_sha256: str,
    retry_config: RetryConfig,
) -> CandidateLogprobClient:
    service = tinker.ServiceClient(
        user_metadata={_GENERATION_LOCK_METADATA_KEY: generation_lock_sha256}
    )
    client = await service.create_sampling_client_async(
        model_path=sampler_path,
        retry_config=retry_config,
    )
    return cast(CandidateLogprobClient, client)


async def _run_pending(
    journal: BinaryIO,
    prepared: PreparedOutcomeV2Inference,
    state: _JournalState,
    client: CandidateLogprobClient,
    pending: tuple[int, ...],
) -> None:
    for sequence in pending:
        game = prepared.games[sequence]
        _append_event(journal, _started_event(prepared, sequence, game))
        state.started.add(sequence)
        record = await _score_game(client, prepared.generation_lock, sequence, game)
        _append_event(journal, _terminal_event(prepared, sequence, game, record))
        state.terminal[sequence] = record


async def _score_game(
    client: CandidateLogprobClient,
    generation_lock: OutcomeV2GenerationLock,
    sequence: int,
    game: RenderedOutcomeV2Game,
) -> InferenceRecord:
    results = await asyncio.gather(
        client.compute_logprobs_async(game.candidate_inputs[0]),
        client.compute_logprobs_async(game.candidate_inputs[1]),
        client.compute_logprobs_async(game.candidate_inputs[2]),
        client.compute_logprobs_async(game.candidate_inputs[3]),
        return_exceptions=True,
    )
    remote_error = next(
        (value for value in results if isinstance(value, BaseException)),
        None,
    )
    if remote_error is not None:
        return _failed_record(
            generation_lock,
            sequence,
            game,
            sanitize_inference_failure(remote_error),
        )
    try:
        scores = _orientation_scores(game, cast(Sequence[object], results))
        return completed_inference_record(
            generation_lock,
            sequence,
            game.row,
            original_prompt_token_ids_sha256=game.original_prompt_token_ids_sha256,
            swapped_prompt_token_ids_sha256=game.swapped_prompt_token_ids_sha256,
            original=scores[0],
            swapped=scores[1],
        )
    except (OutcomeV2InferenceError, RuntimeError, TypeError, ValueError):
        return _failed_record(
            generation_lock,
            sequence,
            game,
            "candidate_output_invalid",
        )


def _orientation_scores(
    game: RenderedOutcomeV2Game,
    results: Sequence[object],
) -> tuple[OrientationScore, OrientationScore]:
    if len(results) != 4:
        raise RuntimeError("candidate result count differs from the locked call policy")
    values = tuple(
        _final_label_logprob(model_input, result)
        for model_input, result in zip(game.candidate_inputs, results, strict=True)
    )
    return (
        build_orientation_score(
            game.row.elo_team_win_probability,
            values[0],
            values[1],
        ),
        build_orientation_score(
            game.row.elo_opponent_win_probability,
            values[2],
            values[3],
        ),
    )


def _final_label_logprob(model_input: tinker.ModelInput, result: object) -> float:
    if not isinstance(result, list):
        raise RuntimeError("candidate response must be a list")
    values = cast(list[object], result)
    if len(values) != model_input.length:
        raise RuntimeError("candidate response has the wrong length")
    value = values[-1]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError("candidate response is missing its final label score")
    score = float(value)
    if not isfinite(score) or score > 0.0:
        raise RuntimeError("candidate label score must be finite and non-positive")
    return score


def _failed_record(
    generation_lock: OutcomeV2GenerationLock,
    sequence: int,
    game: RenderedOutcomeV2Game,
    reason: str,
) -> InferenceRecord:
    return failed_inference_record(
        generation_lock,
        sequence,
        game.row,
        original_prompt_token_ids_sha256=game.original_prompt_token_ids_sha256,
        swapped_prompt_token_ids_sha256=game.swapped_prompt_token_ids_sha256,
        failure_reason=reason,
    )


def _render_games(
    renderer: renderers.Renderer,
    rows: tuple[NbaRichFeatureRow, ...],
    label_token_ids: tuple[int, int],
) -> tuple[RenderedOutcomeV2Game, ...]:
    records = build_outcome_v2_prompt_records(rows)
    games: list[RenderedOutcomeV2Game] = []
    for sequence, row in enumerate(rows):
        original = records[sequence * 2]
        swapped = records[sequence * 2 + 1]
        games.append(_render_game(renderer, row, original, swapped, label_token_ids))
    return tuple(games)


def _render_game(
    renderer: renderers.Renderer,
    row: NbaRichFeatureRow,
    original: ForecastRecord,
    swapped: ForecastRecord,
    label_token_ids: tuple[int, int],
) -> RenderedOutcomeV2Game:
    prompts = (
        _render_prompt(renderer, original),
        _render_prompt(renderer, swapped),
    )
    prompt_tokens = (tuple(prompts[0].to_ints()), tuple(prompts[1].to_ints()))
    for tokens in prompt_tokens:
        if not tokens:
            raise RuntimeError("outcome-v2 renderer produced an empty prompt")
        if len(tokens) + 1 > MAX_LENGTH:
            raise RuntimeError("outcome-v2 candidate input exceeds the locked maximum length")
    candidates = (
        _append_and_check(prompts[0], prompt_tokens[0], label_token_ids[0]),
        _append_and_check(prompts[0], prompt_tokens[0], label_token_ids[1]),
        _append_and_check(prompts[1], prompt_tokens[1], label_token_ids[0]),
        _append_and_check(prompts[1], prompt_tokens[1], label_token_ids[1]),
    )
    return RenderedOutcomeV2Game(
        row=row,
        original_prompt_token_ids_sha256=rendered_prompt_token_ids_sha256(prompt_tokens[0]),
        swapped_prompt_token_ids_sha256=rendered_prompt_token_ids_sha256(prompt_tokens[1]),
        candidate_inputs=candidates,
    )


def _render_prompt(
    renderer: renderers.Renderer,
    record: ForecastRecord,
) -> tinker.ModelInput:
    messages = [
        renderers.Message(role=message["role"], content=message["content"])
        for message in record["messages"]
    ]
    return renderer.build_generation_prompt(messages)


def _append_and_check(
    prompt: tinker.ModelInput,
    prompt_tokens: tuple[int, ...],
    label_token_id: int,
) -> tinker.ModelInput:
    candidate = prompt.append_int(label_token_id)
    if tuple(candidate.to_ints()) != (*prompt_tokens, label_token_id):
        raise RuntimeError("candidate input does not append exactly one locked label token")
    return candidate


def _require_locked_runtime(
    lock_record: Mapping[str, object],
    label_token_ids: tuple[int, int],
) -> None:
    if _string(lock_record, "renderer_name") != OUTCOME_RENDERER_NAME:
        raise RuntimeError("generation lock uses a different renderer")
    labels = require_object(required_field(lock_record, "label_token_ids"), "label_token_ids")
    locked_ids = (_integer(labels, TEAM_LABEL), _integer(labels, OPPONENT_LABEL))
    if locked_ids != label_token_ids:
        raise RuntimeError("local label token IDs differ from the generation lock")


def _read_feature_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise RuntimeError("cannot read outcome-v2 inference feature rows") from error


@contextmanager
def _exclusive_journal(path: Path) -> Generator[BinaryIO, None, None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        journal = path.open("a+b")
    except OSError as error:
        raise RuntimeError("cannot open the outcome-v2 inference journal") from error
    with journal:
        try:
            fcntl.flock(journal.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another outcome-v2 inference runner is active") from error
        try:
            yield journal
        finally:
            fcntl.flock(journal.fileno(), fcntl.LOCK_UN)


def _read_journal(
    journal: BinaryIO,
    prepared: PreparedOutcomeV2Inference,
) -> _JournalState:
    journal.seek(0)
    value = journal.read()
    value = _recover_partial_tail(journal, value)
    state = _JournalState(set(), {})
    for line_number, line in enumerate(value.splitlines(), start=1):
        event = _parse_event(line, line_number)
        sequence = _validate_event_binding(event, prepared)
        kind = _string(event, "kind")
        if kind == _STARTED_KIND:
            _accept_started(state, sequence, event)
        elif kind == _TERMINAL_KIND:
            _accept_terminal(state, sequence, event, prepared)
        else:
            raise RuntimeError(f"unsupported inference journal event at line {line_number}")
    return state


def _parse_event(line: bytes, line_number: int) -> JsonObject:
    try:
        text = line.decode("utf-8")
        event = parse_json_object(text)
        require_exact_keys(event, _EVENT_KEYS, "inference journal event")
    except (JsonFormatError, UnicodeError) as error:
        raise RuntimeError(f"invalid inference journal event at line {line_number}") from error
    if text != canonical_json(event):
        raise RuntimeError(f"noncanonical inference journal event at line {line_number}")
    if _integer(event, "schema_version") != JOURNAL_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported inference journal schema at line {line_number}")
    return event


def _validate_event_binding(
    event: Mapping[str, object],
    prepared: PreparedOutcomeV2Inference,
) -> int:
    sequence = _integer(event, "sequence")
    if not 0 <= sequence < len(prepared.games):
        raise RuntimeError("inference journal sequence is outside the generation lock")
    game = prepared.games[sequence]
    expected = _binding_payload(prepared, sequence, game)
    actual = {name: required_field(event, name) for name in expected}
    if actual != expected:
        raise RuntimeError("inference journal event differs from the locked game")
    return sequence


def _accept_started(
    state: _JournalState,
    sequence: int,
    event: Mapping[str, object],
) -> None:
    if required_field(event, "record") is not None:
        raise RuntimeError("started inference event cannot contain a terminal record")
    if sequence in state.started or sequence in state.terminal:
        raise RuntimeError("duplicate started inference event")
    state.started.add(sequence)


def _accept_terminal(
    state: _JournalState,
    sequence: int,
    event: Mapping[str, object],
    prepared: PreparedOutcomeV2Inference,
) -> None:
    if sequence not in state.started or sequence in state.terminal:
        raise RuntimeError("terminal inference event lacks one unique start")
    payload = require_object(required_field(event, "record"), "record")
    record = outcome_v2_inference_record_from_payload(payload)
    _require_terminal_binding(
        record,
        sequence,
        prepared.games[sequence],
        prepared.generation_lock.sha256,
    )
    state.terminal[sequence] = record


def _recover_partial_tail(journal: BinaryIO, value: bytes) -> bytes:
    """Discard only a torn final event; all complete append-only events remain."""
    if not value or value.endswith(b"\n"):
        return value
    boundary = value.rfind(b"\n") + 1
    journal.seek(boundary)
    journal.truncate()
    journal.flush()
    os.fsync(journal.fileno())
    return value[:boundary]


def _terminalize_interrupted(
    journal: BinaryIO,
    prepared: PreparedOutcomeV2Inference,
    state: _JournalState,
) -> None:
    for sequence in sorted(state.started - state.terminal.keys()):
        game = prepared.games[sequence]
        record = _failed_record(
            prepared.generation_lock,
            sequence,
            game,
            "interrupted_after_start",
        )
        _append_event(journal, _terminal_event(prepared, sequence, game, record))
        state.terminal[sequence] = record


def _compile_records(
    path: Path,
    prepared: PreparedOutcomeV2Inference,
    records: tuple[InferenceRecord, ...],
) -> tuple[InferenceRecord, ...]:
    if path.exists():
        existing = read_outcome_v2_inference_records(path, prepared.generation_lock)
        if existing != records:
            raise RuntimeError("compiled inference records differ from the durable journal")
        return existing
    write_outcome_v2_inference_records(path, records, prepared.generation_lock)
    return records


def _compile_forecasts(
    path: Path,
    prepared: PreparedOutcomeV2Inference,
    records: tuple[InferenceRecord, ...],
) -> None:
    forecasts = binary_forecasts_from_inference_records(records, prepared.generation_lock)
    if path.exists():
        if read_nba_evaluation_forecasts_jsonl(path) != forecasts:
            raise RuntimeError("compiled forecasts differ from the durable inference records")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_nba_evaluation_forecasts_jsonl(path, forecasts)


def _write_or_verify_bytes(path: Path, expected: bytes, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as file:
            if file.write(expected) != len(expected):
                raise OSError(f"short write while creating {description}")
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError:
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise RuntimeError(f"cannot read existing {description}") from error
        if actual != expected:
            raise RuntimeError(f"existing {description} differs from locked inputs") from None
    except OSError as error:
        raise RuntimeError(f"cannot create {description}") from error


def _started_event(
    prepared: PreparedOutcomeV2Inference,
    sequence: int,
    game: RenderedOutcomeV2Game,
) -> JsonObject:
    return {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "kind": _STARTED_KIND,
        **_binding_payload(prepared, sequence, game),
        "record": None,
    }


def _terminal_event(
    prepared: PreparedOutcomeV2Inference,
    sequence: int,
    game: RenderedOutcomeV2Game,
    record: InferenceRecord,
) -> JsonObject:
    return {
        "schema_version": JOURNAL_SCHEMA_VERSION,
        "kind": _TERMINAL_KIND,
        **_binding_payload(prepared, sequence, game),
        "record": outcome_v2_inference_record_payload(record),
    }


def _binding_payload(
    prepared: PreparedOutcomeV2Inference,
    sequence: int,
    game: RenderedOutcomeV2Game,
) -> JsonObject:
    return {
        "generation_lock_sha256": prepared.generation_lock.sha256,
        "sequence": sequence,
        "question_id": game.row.question_id,
        "feature_row_sha256": game.row.row_sha256,
        "original_prompt_token_ids_sha256": game.original_prompt_token_ids_sha256,
        "swapped_prompt_token_ids_sha256": game.swapped_prompt_token_ids_sha256,
    }


def _append_event(journal: BinaryIO, event: Mapping[str, object]) -> None:
    value = f"{canonical_json(event)}\n".encode()
    journal.seek(0, os.SEEK_END)
    if journal.write(value) != len(value):
        raise OSError("short write while appending the inference journal")
    journal.flush()
    os.fsync(journal.fileno())


def _require_terminal_binding(
    record: InferenceRecord,
    sequence: int,
    game: RenderedOutcomeV2Game,
    generation_lock_sha256: str,
) -> None:
    expected = (
        sequence,
        generation_lock_sha256,
        game.row.question_id,
        game.row.row_sha256,
        game.original_prompt_token_ids_sha256,
        game.swapped_prompt_token_ids_sha256,
    )
    actual = (
        record.sequence,
        record.generation_lock_sha256,
        record.question_id,
        record.feature_row_sha256,
        record.original_prompt_token_ids_sha256,
        record.swapped_prompt_token_ids_sha256,
    )
    if actual != expected:
        raise RuntimeError("journal terminal record differs from its locked game")
    if record.status == "completed":
        if record.original is None or record.swapped is None:
            raise RuntimeError("completed journal record is missing orientation scores")
        priors = (
            record.original.elo_team_probability,
            record.swapped.elo_team_probability,
        )
        expected_priors = (
            game.row.elo_team_win_probability,
            game.row.elo_opponent_win_probability,
        )
        if priors != expected_priors:
            raise RuntimeError("journal terminal record differs from its locked Elo priors")


def _string(record: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = required_field(record, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{field_name} must be an integer")
    return value
