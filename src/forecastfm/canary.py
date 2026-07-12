"""Frozen, answer-free validation canaries and their sealed generation records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Literal

from forecastfm.integrity import canonical_json, canonical_sha256, file_sha256, text_sha256
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_float,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.nba_data import elo_venue_probability
from forecastfm.prompting import SYSTEM_PROMPT, ChatMessage, parse_prediction

CANARY_SCHEMA_VERSION = 1
GENERATION_SCHEMA_VERSION = 1
SEAL_SCHEMA_VERSION = 1
CANARY_SIZE = 64
PROMPT_COUNT = CANARY_SIZE * 2
INVALID_MAE = 1.0
INVALID_BRIER = 1.0
MINIMUM_LOG_PROBABILITY = 1e-15
INVALID_LOG_LOSS = -log(MINIMUM_LOG_PROBABILITY)

VALIDATION_PROMPTS_LABEL = "data/processed/nba_elo_validation_prompts.jsonl"
FROZEN_PROMPTS_LABEL = "evaluation/validation_canary/prompts.jsonl"
SELECTION_ALGORITHM = "lexicographic_question_id_first_64_v1"
SIDE_SWAP_TRANSFORM = "swap_prior_probabilities_and_home_away_v1"
ATTEMPT_FILENAME = "attempt.json"
OUTCOMES = ("team_wins", "opponent_wins")

type Variant = Literal["original", "side_swap"]
type ModelRole = Literal["base", "adapter"]
type GenerationStatus = Literal["completed", "error"]

VARIANTS: tuple[Variant, ...] = ("original", "side_swap")
MODEL_ROLES: tuple[ModelRole, ...] = ("base", "adapter")

_QUESTION_KEYS = {"question", "resolution_rule", "outcomes", "prior", "evidence"}
_PROMPT_RECORD_KEYS = {
    "schema_version",
    "sequence",
    "question_id",
    "variant",
    "messages",
    "prompt_sha256",
}
_GENERATION_KEYS = {
    "schema_version",
    "sequence",
    "model_role",
    "question_id",
    "variant",
    "prompt_sha256",
    "attempt_id",
    "prompt_tokens",
    "response_tokens",
    "raw_response",
    "parsed_response",
    "status",
    "termination",
    "stop_reason",
    "error",
}


class CanaryValidationError(ValueError):
    """Raised when a canary artifact is incomplete, stale, or target-contaminated."""


@dataclass(frozen=True, slots=True)
class CanarySource:
    """Verified source commitments used without opening the answer file."""

    validation_prompts_path: Path
    validation_prompts_sha256: str
    validation_answers_sha256: str
    dataset_manifest_sha256: str
    expected_question_ids_sha256: str


@dataclass(frozen=True, slots=True)
class CanaryModels:
    """Frozen model identities and decoding settings for a paired canary."""

    training_lock_sha256: str
    experiment_sha256: str
    base_model: str
    adapter_sampler_path: str
    decoding: Mapping[str, object]
    protocol_code_revision: str


@dataclass(frozen=True, slots=True)
class CanaryPrompt:
    """One exact target-free generation input."""

    sequence: int
    question_id: str
    variant: Variant
    messages: tuple[ChatMessage, ...]
    prompt_sha256: str


@dataclass(frozen=True, slots=True)
class CanaryManifest:
    """Commitments for the frozen canary, models, and prompt artifact."""

    source_prompt_sha256: str
    source_answer_sha256: str
    dataset_manifest_sha256: str
    training_lock_sha256: str
    experiment_sha256: str
    base_model: str
    adapter_sampler_path: str
    decoding: dict[str, object]
    protocol_code_revision: str
    question_ids: tuple[str, ...]
    question_ids_sha256: str
    prompts_sha256: str


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """Exactly one immutable provider attempt for one frozen prompt."""

    sequence: int
    model_role: ModelRole
    question_id: str
    variant: Variant
    prompt_sha256: str
    attempt_id: str
    prompt_tokens: tuple[int, ...]
    response_tokens: tuple[int, ...]
    raw_response: str
    parsed_response: str
    status: GenerationStatus
    termination: str | None
    stop_reason: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class CompletedGeneration:
    """Exact token and text trace returned by a completed renderer attempt."""

    prompt_tokens: tuple[int, ...]
    response_tokens: tuple[int, ...]
    raw_response: str
    parsed_response: str
    termination: str
    stop_reason: str


@dataclass(frozen=True, slots=True)
class GenerationSeal:
    """Hashes binding both complete model outputs to one canary."""

    manifest_sha256: str
    prompts_sha256: str
    attempt_sha256: str
    base_sha256: str
    adapter_sha256: str


@dataclass(frozen=True, slots=True)
class SealedGenerations:
    """Verified paired outputs that are safe to pass to scoring."""

    manifest: CanaryManifest
    prompts: tuple[CanaryPrompt, ...]
    base: tuple[GenerationRecord, ...]
    adapter: tuple[GenerationRecord, ...]
    seal: GenerationSeal


@dataclass(frozen=True, slots=True)
class PrimaryModelMetrics:
    """Answer-free metrics for one model on the complete canary."""

    game_count: int
    valid_original_count: int
    valid_side_swap_count: int
    valid_pair_count: int
    schema_valid_rate: float
    side_swap_valid_rate: float
    valid_pair_rate: float
    oracle_mae: float
    valid_only_oracle_mae: float | None
    side_swap_mae: float
    valid_only_side_swap_mae: float | None


@dataclass(frozen=True, slots=True)
class PrimaryComparison:
    """Paired answer-free metrics and adapter-minus-base deltas."""

    base: PrimaryModelMetrics
    adapter: PrimaryModelMetrics
    adapter_minus_base_oracle_mae: float
    adapter_minus_base_side_swap_mae: float


def build_canary_artifacts(
    source: CanarySource,
    models: CanaryModels,
    prompts_path: Path,
    manifest_path: Path,
) -> CanaryManifest:
    """Select prompts without answers and exclusively create both frozen artifacts."""
    _require_new_paths((prompts_path, manifest_path))
    source_records = _load_source_prompts(source)
    selected = tuple(sorted(source_records, key=lambda item: item.question_id)[:CANARY_SIZE])
    if len(selected) != CANARY_SIZE:
        raise CanaryValidationError(
            f"validation source must contain at least {CANARY_SIZE} prompts"
        )
    prompts = _paired_prompts(selected)
    prompts_text = _jsonl_text(_prompt_to_dict(prompt) for prompt in prompts)
    manifest = _manifest(source, models, selected, text_sha256(prompts_text))
    _write_new_text(prompts_path, prompts_text)
    _write_new_text(manifest_path, _manifest_text(manifest))
    return manifest


def load_canary(
    manifest_path: Path,
    prompts_path: Path,
) -> tuple[CanaryManifest, tuple[CanaryPrompt, ...]]:
    """Load a manifest and its exact, complete target-free prompt artifact."""
    manifest = _manifest_from_dict(parse_json_object(manifest_path.read_text(encoding="utf-8")))
    if file_sha256(prompts_path) != manifest.prompts_sha256:
        raise CanaryValidationError("frozen prompts differ from the canary manifest")
    prompts = tuple(_prompt_from_dict(item) for item in _load_jsonl_objects(prompts_path))
    _validate_frozen_prompts(prompts, manifest.question_ids)
    return manifest, prompts


def successful_generation(
    prompt: CanaryPrompt,
    model_role: ModelRole,
    completed: CompletedGeneration,
) -> GenerationRecord:
    """Create one completed generation row without repairing its response."""
    if not completed.termination.strip() or not completed.stop_reason.strip():
        raise CanaryValidationError("termination and stop_reason must not be empty")
    return GenerationRecord(
        sequence=prompt.sequence,
        model_role=_model_role(model_role),
        question_id=prompt.question_id,
        variant=prompt.variant,
        prompt_sha256=prompt.prompt_sha256,
        attempt_id=_attempt_id(model_role, prompt.sequence),
        prompt_tokens=_token_tuple(completed.prompt_tokens, "prompt_tokens", require_nonempty=True),
        response_tokens=_token_tuple(completed.response_tokens, "response_tokens"),
        raw_response=completed.raw_response,
        parsed_response=completed.parsed_response,
        status="completed",
        termination=completed.termination,
        stop_reason=completed.stop_reason,
        error=None,
    )


def failed_generation(
    prompt: CanaryPrompt,
    model_role: ModelRole,
    prompt_tokens: Sequence[int],
    error: str,
) -> GenerationRecord:
    """Create the sole immutable error row for a failed provider attempt."""
    if not error.strip():
        raise CanaryValidationError("generation error must not be empty")
    return GenerationRecord(
        sequence=prompt.sequence,
        model_role=_model_role(model_role),
        question_id=prompt.question_id,
        variant=prompt.variant,
        prompt_sha256=prompt.prompt_sha256,
        attempt_id=_attempt_id(model_role, prompt.sequence),
        prompt_tokens=_token_tuple(prompt_tokens, "prompt_tokens", require_nonempty=True),
        response_tokens=(),
        raw_response="",
        parsed_response="",
        status="error",
        termination=None,
        stop_reason=None,
        error=error,
    )


def write_generation_records(
    path: Path,
    records: Sequence[GenerationRecord],
    prompts: Sequence[CanaryPrompt],
    model_role: ModelRole,
) -> str:
    """Exclusively write one exact ordered generation row per prompt."""
    _validate_generation_coverage(records, prompts, model_role)
    text = _jsonl_text(_generation_to_dict(record) for record in records)
    _write_new_text(path, text)
    return text_sha256(text)


def write_attempt_marker(path: Path, manifest_path: Path, prompts_path: Path) -> str:
    """Exclusively mark the single allowed run before constructing a remote client."""
    load_canary(manifest_path, prompts_path)
    value = {
        "schema_version": 1,
        "kind": "forecastfm_canary_attempt",
        "status": "started_before_remote_client",
        "manifest_sha256": file_sha256(manifest_path),
        "prompts_sha256": file_sha256(prompts_path),
    }
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    _write_new_text(path, text)
    return text_sha256(text)


def load_generation_records(
    path: Path,
    prompts: Sequence[CanaryPrompt],
    model_role: ModelRole,
) -> tuple[GenerationRecord, ...]:
    """Load a generation file only when it exactly covers the frozen prompt order."""
    records = tuple(_generation_from_dict(item) for item in _load_jsonl_objects(path))
    _validate_generation_coverage(records, prompts, model_role)
    return records


def seal_generation_outputs(
    seal_path: Path,
    manifest_path: Path,
    prompts_path: Path,
    base_path: Path,
    adapter_path: Path,
) -> GenerationSeal:
    """Validate complete paired outputs and exclusively create their hash seal."""
    _, prompts = load_canary(manifest_path, prompts_path)
    base = load_generation_records(base_path, prompts, "base")
    adapter = load_generation_records(adapter_path, prompts, "adapter")
    _validate_paired_prompt_tokens(base, adapter)
    attempt_path = base_path.parent / ATTEMPT_FILENAME
    _validate_attempt_marker(attempt_path, manifest_path, prompts_path)
    seal = GenerationSeal(
        manifest_sha256=file_sha256(manifest_path),
        prompts_sha256=file_sha256(prompts_path),
        attempt_sha256=file_sha256(attempt_path),
        base_sha256=file_sha256(base_path),
        adapter_sha256=file_sha256(adapter_path),
    )
    _write_new_text(seal_path, _seal_text(seal))
    return seal


def load_sealed_generations(
    seal_path: Path,
    manifest_path: Path,
    prompts_path: Path,
    base_path: Path,
    adapter_path: Path,
) -> SealedGenerations:
    """Verify a seal and return the two complete immutable generation files."""
    seal = _seal_from_dict(parse_json_object(seal_path.read_text(encoding="utf-8")))
    _verify_seal(seal, manifest_path, prompts_path, base_path, adapter_path)
    manifest, prompts = load_canary(manifest_path, prompts_path)
    base = load_generation_records(base_path, prompts, "base")
    adapter = load_generation_records(adapter_path, prompts, "adapter")
    _validate_paired_prompt_tokens(base, adapter)
    return SealedGenerations(
        manifest=manifest,
        prompts=prompts,
        base=base,
        adapter=adapter,
        seal=seal,
    )


def score_primary(generations: SealedGenerations) -> PrimaryComparison:
    """Score the sealed pair using only prompt-visible priors and venues."""
    base = _primary_model_metrics(generations.prompts, generations.base)
    adapter = _primary_model_metrics(generations.prompts, generations.adapter)
    return PrimaryComparison(
        base=base,
        adapter=adapter,
        adapter_minus_base_oracle_mae=adapter.oracle_mae - base.oracle_mae,
        adapter_minus_base_side_swap_mae=adapter.side_swap_mae - base.side_swap_mae,
    )


def _load_source_prompts(source: CanarySource) -> tuple[CanaryPrompt, ...]:
    if source.validation_prompts_path.name != Path(VALIDATION_PROMPTS_LABEL).name:
        raise CanaryValidationError("canary source must be the validation prompt file")
    if file_sha256(source.validation_prompts_path) != source.validation_prompts_sha256:
        raise CanaryValidationError("validation prompt digest differs from its commitment")
    prompts: list[CanaryPrompt] = []
    for sequence, item in enumerate(_load_jsonl_objects(source.validation_prompts_path)):
        try:
            require_exact_keys(item, {"question_id", "messages"}, "source prompt")
            messages = _messages(required_field(item, "messages"))
            question_id = require_string(required_field(item, "question_id"), "question_id")
        except JsonFormatError as error:
            raise CanaryValidationError(f"invalid source prompt row {sequence}") from error
        prompts.append(_new_prompt(sequence, question_id, "original", messages))
    if len({item.question_id for item in prompts}) != len(prompts):
        raise CanaryValidationError("validation question_id values must be unique")
    return tuple(prompts)


def _paired_prompts(selected: Sequence[CanaryPrompt]) -> tuple[CanaryPrompt, ...]:
    result: list[CanaryPrompt] = []
    for source in selected:
        original = _new_prompt(len(result), source.question_id, "original", source.messages)
        result.append(original)
        swapped_messages = _side_swap_messages(source.messages)
        result.append(_new_prompt(len(result), source.question_id, "side_swap", swapped_messages))
    return tuple(result)


def _new_prompt(
    sequence: int,
    question_id: str,
    variant: Variant,
    messages: tuple[ChatMessage, ...],
) -> CanaryPrompt:
    _validate_messages(messages)
    return CanaryPrompt(
        sequence=sequence,
        question_id=question_id,
        variant=variant,
        messages=messages,
        prompt_sha256=canonical_sha256(list(messages)),
    )


def _side_swap_messages(messages: tuple[ChatMessage, ...]) -> tuple[ChatMessage, ...]:
    user = parse_json_object(messages[1]["content"])
    _validate_question(user)
    prior = require_object(required_field(user, "prior"), "prior")
    team = require_float(required_field(prior, "team_wins"), "prior.team_wins")
    opponent = require_float(required_field(prior, "opponent_wins"), "prior.opponent_wins")
    prior["team_wins"], prior["opponent_wins"] = opponent, team
    evidence = require_list(required_field(user, "evidence"), "evidence")
    venue = _venue(require_string(evidence[0], "evidence[0]"))
    swapped_venue = {"home": "away", "away": "home", "neutral": "neutral"}[venue]
    evidence[0] = f"Venue for the listed team: {swapped_venue}."
    user["prior"], user["evidence"] = prior, evidence
    return (
        ChatMessage(role="system", content=messages[0]["content"]),
        ChatMessage(role="user", content=json.dumps(user, indent=2, sort_keys=True)),
    )


def _messages(value: object) -> tuple[ChatMessage, ...]:
    items = require_list(value, "messages")
    if len(items) != 2:
        raise CanaryValidationError("target-free prompts must contain system and user only")
    result: list[ChatMessage] = []
    for index, item in enumerate(items):
        record = require_object(item, f"messages[{index}]")
        require_exact_keys(record, {"role", "content"}, f"messages[{index}]")
        role = require_string(required_field(record, "role"), f"messages[{index}].role")
        expected_role = ("system", "user")[index]
        if role != expected_role:
            raise CanaryValidationError("target-free prompt roles must be system then user")
        result.append(
            ChatMessage(
                role=expected_role,
                content=require_string(
                    required_field(record, "content"), f"messages[{index}].content"
                ),
            )
        )
    messages = tuple(result)
    _validate_messages(messages)
    return messages


def _validate_messages(messages: Sequence[ChatMessage]) -> None:
    if len(messages) != 2 or tuple(item["role"] for item in messages) != ("system", "user"):
        raise CanaryValidationError("target-free prompt roles must be system then user")
    if messages[0]["content"] != SYSTEM_PROMPT:
        raise CanaryValidationError("canary system prompt differs from the frozen prompt")
    user = parse_json_object(messages[1]["content"])
    _validate_question(user)
    if messages[1]["content"] != json.dumps(user, indent=2, sort_keys=True):
        raise CanaryValidationError("user prompt is not in its exact canonical rendered form")


def _validate_question(record: Mapping[str, object]) -> None:
    require_exact_keys(record, _QUESTION_KEYS, "question")
    outcomes = require_list(required_field(record, "outcomes"), "outcomes")
    if tuple(outcomes) != OUTCOMES:
        raise CanaryValidationError("canary outcomes differ from the frozen binary outcomes")
    prior = require_object(required_field(record, "prior"), "prior")
    require_exact_keys(prior, set(OUTCOMES), "prior")
    team = require_float(required_field(prior, "team_wins"), "prior.team_wins")
    opponent = require_float(required_field(prior, "opponent_wins"), "prior.opponent_wins")
    if not 0.0 < team < 1.0 or abs(team + opponent - 1.0) > 1e-6:
        raise CanaryValidationError("canary prior must be a strict binary distribution")
    evidence = require_list(required_field(record, "evidence"), "evidence")
    if len(evidence) != 1:
        raise CanaryValidationError("canary must contain exactly one venue evidence card")
    _venue(require_string(evidence[0], "evidence[0]"))


def _venue(text: str) -> str:
    prefix = "Venue for the listed team: "
    if not text.startswith(prefix) or not text.endswith("."):
        raise CanaryValidationError("venue evidence has an unexpected format")
    venue = text[len(prefix) : -1]
    if venue not in {"home", "away", "neutral"}:
        raise CanaryValidationError("venue evidence has an unknown value")
    return venue


def _manifest(
    source: CanarySource,
    models: CanaryModels,
    selected: Sequence[CanaryPrompt],
    prompts_sha256: str,
) -> CanaryManifest:
    question_ids = tuple(item.question_id for item in selected)
    question_ids_sha256 = canonical_sha256(list(question_ids))
    if question_ids_sha256 != source.expected_question_ids_sha256:
        raise CanaryValidationError("selected question IDs differ from the frozen commitment")
    return CanaryManifest(
        source_prompt_sha256=source.validation_prompts_sha256,
        source_answer_sha256=source.validation_answers_sha256,
        dataset_manifest_sha256=source.dataset_manifest_sha256,
        training_lock_sha256=models.training_lock_sha256,
        experiment_sha256=models.experiment_sha256,
        base_model=models.base_model,
        adapter_sampler_path=models.adapter_sampler_path,
        decoding=dict(models.decoding),
        protocol_code_revision=models.protocol_code_revision,
        question_ids=question_ids,
        question_ids_sha256=question_ids_sha256,
        prompts_sha256=prompts_sha256,
    )


def _manifest_to_dict(manifest: CanaryManifest) -> dict[str, object]:
    return {
        "schema_version": CANARY_SCHEMA_VERSION,
        "kind": "forecastfm_validation_canary",
        "status": "frozen_before_generation",
        "source": {
            "dataset_manifest_sha256": manifest.dataset_manifest_sha256,
            "validation_prompts_path": VALIDATION_PROMPTS_LABEL,
            "validation_prompts_sha256": manifest.source_prompt_sha256,
            "validation_answers_sha256": manifest.source_answer_sha256,
            "answers_opened_during_selection": False,
        },
        "selection": {
            "algorithm": SELECTION_ALGORITHM,
            "game_count": CANARY_SIZE,
            "ordered_question_ids": list(manifest.question_ids),
            "ordered_question_ids_sha256": manifest.question_ids_sha256,
        },
        "prompts": {
            "path": FROZEN_PROMPTS_LABEL,
            "count": PROMPT_COUNT,
            "sha256": manifest.prompts_sha256,
            "ordering": "original then side_swap for each ordered question_id",
            "side_swap_transform": SIDE_SWAP_TRANSFORM,
        },
        "models": {
            "protocol_code_revision": manifest.protocol_code_revision,
            "training_lock_sha256": manifest.training_lock_sha256,
            "experiment_sha256": manifest.experiment_sha256,
            "base_model": manifest.base_model,
            "adapter_sampler_path": manifest.adapter_sampler_path,
        },
        "decoding": manifest.decoding,
        "scoring": {
            "primary": "prompt-derived Elo oracle and side-swap consistency",
            "invalid_mae": INVALID_MAE,
            "invalid_brier": INVALID_BRIER,
            "invalid_log_loss": INVALID_LOG_LOSS,
            "historical_answers_require_sealed_outputs": True,
        },
    }


def _manifest_from_dict(record: Mapping[str, object]) -> CanaryManifest:
    keys = {
        "schema_version",
        "kind",
        "status",
        "source",
        "selection",
        "prompts",
        "models",
        "decoding",
        "scoring",
    }
    require_exact_keys(record, keys, "canary manifest")
    _require_manifest_header(record)
    source = require_object(required_field(record, "source"), "source")
    selection = require_object(required_field(record, "selection"), "selection")
    prompts = require_object(required_field(record, "prompts"), "prompts")
    models = require_object(required_field(record, "models"), "models")
    scoring = require_object(required_field(record, "scoring"), "scoring")
    _validate_manifest_sections(source, selection, prompts)
    _validate_model_and_scoring_sections(models, scoring)
    question_ids = _strings(required_field(selection, "ordered_question_ids"), "question_ids")
    manifest = CanaryManifest(
        source_prompt_sha256=_string(source, "validation_prompts_sha256"),
        source_answer_sha256=_string(source, "validation_answers_sha256"),
        dataset_manifest_sha256=_string(source, "dataset_manifest_sha256"),
        training_lock_sha256=_string(models, "training_lock_sha256"),
        experiment_sha256=_string(models, "experiment_sha256"),
        base_model=_string(models, "base_model"),
        adapter_sampler_path=_string(models, "adapter_sampler_path"),
        decoding=require_object(required_field(record, "decoding"), "decoding"),
        protocol_code_revision=_string(models, "protocol_code_revision"),
        question_ids=question_ids,
        question_ids_sha256=_string(selection, "ordered_question_ids_sha256"),
        prompts_sha256=_string(prompts, "sha256"),
    )
    _validate_manifest_values(manifest)
    return manifest


def _require_manifest_header(record: Mapping[str, object]) -> None:
    if required_field(record, "schema_version") != CANARY_SCHEMA_VERSION:
        raise CanaryValidationError("unsupported canary schema version")
    if required_field(record, "kind") != "forecastfm_validation_canary":
        raise CanaryValidationError("unexpected canary manifest kind")
    if required_field(record, "status") != "frozen_before_generation":
        raise CanaryValidationError("canary was not frozen before generation")


def _validate_manifest_sections(
    source: Mapping[str, object],
    selection: Mapping[str, object],
    prompts: Mapping[str, object],
) -> None:
    require_exact_keys(
        source,
        {
            "dataset_manifest_sha256",
            "validation_prompts_path",
            "validation_prompts_sha256",
            "validation_answers_sha256",
            "answers_opened_during_selection",
        },
        "source",
    )
    require_exact_keys(
        selection,
        {"algorithm", "game_count", "ordered_question_ids", "ordered_question_ids_sha256"},
        "selection",
    )
    require_exact_keys(
        prompts,
        {"path", "count", "sha256", "ordering", "side_swap_transform"},
        "prompts",
    )
    _validate_source_section(source)
    _validate_selection_section(selection)
    _validate_prompt_section(prompts)


def _validate_source_section(source: Mapping[str, object]) -> None:
    if _string(source, "validation_prompts_path") != VALIDATION_PROMPTS_LABEL:
        raise CanaryValidationError("manifest does not reference validation prompts")
    if required_field(source, "answers_opened_during_selection") is not False:
        raise CanaryValidationError("canary selection must be answer-free")


def _validate_selection_section(selection: Mapping[str, object]) -> None:
    if _string(selection, "algorithm") != SELECTION_ALGORITHM:
        raise CanaryValidationError("unknown canary selection algorithm")
    if required_field(selection, "game_count") != CANARY_SIZE:
        raise CanaryValidationError("canary game count differs from the frozen count")


def _validate_prompt_section(prompts: Mapping[str, object]) -> None:
    if _string(prompts, "path") != FROZEN_PROMPTS_LABEL:
        raise CanaryValidationError("manifest has an unexpected frozen prompt path")
    if required_field(prompts, "count") != PROMPT_COUNT:
        raise CanaryValidationError("canary prompt count differs from the frozen count")
    if _string(prompts, "side_swap_transform") != SIDE_SWAP_TRANSFORM:
        raise CanaryValidationError("unknown side-swap transformation")
    if _string(prompts, "ordering") != "original then side_swap for each ordered question_id":
        raise CanaryValidationError("unknown canary prompt ordering")


def _validate_model_and_scoring_sections(
    models: Mapping[str, object],
    scoring: Mapping[str, object],
) -> None:
    require_exact_keys(
        models,
        {
            "protocol_code_revision",
            "training_lock_sha256",
            "experiment_sha256",
            "base_model",
            "adapter_sampler_path",
        },
        "models",
    )
    require_exact_keys(
        scoring,
        {
            "primary",
            "invalid_mae",
            "invalid_brier",
            "invalid_log_loss",
            "historical_answers_require_sealed_outputs",
        },
        "scoring",
    )
    expected = (
        "prompt-derived Elo oracle and side-swap consistency",
        INVALID_MAE,
        INVALID_BRIER,
        INVALID_LOG_LOSS,
        True,
    )
    actual = tuple(
        required_field(scoring, field)
        for field in (
            "primary",
            "invalid_mae",
            "invalid_brier",
            "invalid_log_loss",
            "historical_answers_require_sealed_outputs",
        )
    )
    if actual != expected:
        raise CanaryValidationError("manifest scoring policy differs from the frozen policy")


def _validate_manifest_values(manifest: CanaryManifest) -> None:
    hashes = (
        manifest.source_prompt_sha256,
        manifest.source_answer_sha256,
        manifest.dataset_manifest_sha256,
        manifest.training_lock_sha256,
        manifest.experiment_sha256,
        manifest.question_ids_sha256,
        manifest.prompts_sha256,
    )
    if any(not _is_hash(value) for value in hashes):
        raise CanaryValidationError("manifest contains an invalid SHA-256 digest")
    if len(manifest.question_ids) != CANARY_SIZE:
        raise CanaryValidationError("manifest must contain exactly 64 question IDs")
    if tuple(sorted(manifest.question_ids)) != manifest.question_ids:
        raise CanaryValidationError("manifest question IDs are not lexicographically ordered")
    if canonical_sha256(list(manifest.question_ids)) != manifest.question_ids_sha256:
        raise CanaryValidationError("question ID hash differs from the ordered IDs")
    if not manifest.base_model.strip() or not manifest.adapter_sampler_path.startswith("tinker://"):
        raise CanaryValidationError("manifest model identities are incomplete")
    if not _is_git_revision(manifest.protocol_code_revision):
        raise CanaryValidationError("manifest protocol code revision is invalid")


def _validate_frozen_prompts(
    prompts: Sequence[CanaryPrompt],
    question_ids: Sequence[str],
) -> None:
    if len(prompts) != PROMPT_COUNT:
        raise CanaryValidationError("frozen prompt file must contain exactly 128 rows")
    for index, question_id in enumerate(question_ids):
        original, swapped = prompts[index * 2 : index * 2 + 2]
        if original != _new_prompt(index * 2, question_id, "original", original.messages):
            raise CanaryValidationError("original prompt metadata is inconsistent")
        expected_swap = _new_prompt(
            index * 2 + 1,
            question_id,
            "side_swap",
            _side_swap_messages(original.messages),
        )
        if swapped != expected_swap:
            raise CanaryValidationError("side-swap prompt differs from its deterministic source")


def _prompt_to_dict(prompt: CanaryPrompt) -> dict[str, object]:
    return {
        "schema_version": CANARY_SCHEMA_VERSION,
        "sequence": prompt.sequence,
        "question_id": prompt.question_id,
        "variant": prompt.variant,
        "messages": list(prompt.messages),
        "prompt_sha256": prompt.prompt_sha256,
    }


def _prompt_from_dict(record: Mapping[str, object]) -> CanaryPrompt:
    require_exact_keys(record, _PROMPT_RECORD_KEYS, "canary prompt")
    if required_field(record, "schema_version") != CANARY_SCHEMA_VERSION:
        raise CanaryValidationError("unsupported canary prompt schema")
    sequence = _integer(record, "sequence")
    variant = _variant(_string(record, "variant"))
    prompt = _new_prompt(
        sequence,
        _string(record, "question_id"),
        variant,
        _messages(required_field(record, "messages")),
    )
    if prompt.prompt_sha256 != _string(record, "prompt_sha256"):
        raise CanaryValidationError("prompt_sha256 differs from exact messages")
    return prompt


def _generation_to_dict(record: GenerationRecord) -> dict[str, object]:
    return {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "sequence": record.sequence,
        "model_role": record.model_role,
        "question_id": record.question_id,
        "variant": record.variant,
        "prompt_sha256": record.prompt_sha256,
        "attempt_id": record.attempt_id,
        "prompt_tokens": list(record.prompt_tokens),
        "response_tokens": list(record.response_tokens),
        "raw_response": record.raw_response,
        "parsed_response": record.parsed_response,
        "status": record.status,
        "termination": record.termination,
        "stop_reason": record.stop_reason,
        "error": record.error,
    }


def _generation_from_dict(record: Mapping[str, object]) -> GenerationRecord:
    require_exact_keys(record, _GENERATION_KEYS, "generation")
    if required_field(record, "schema_version") != GENERATION_SCHEMA_VERSION:
        raise CanaryValidationError("unsupported generation schema")
    generation = GenerationRecord(
        sequence=_integer(record, "sequence"),
        model_role=_model_role(_string(record, "model_role")),
        question_id=_string(record, "question_id"),
        variant=_variant(_string(record, "variant")),
        prompt_sha256=_string(record, "prompt_sha256"),
        attempt_id=_string(record, "attempt_id"),
        prompt_tokens=_tokens(required_field(record, "prompt_tokens"), "prompt_tokens", True),
        response_tokens=_tokens(required_field(record, "response_tokens"), "response_tokens"),
        raw_response=_string(record, "raw_response"),
        parsed_response=_string(record, "parsed_response"),
        status=_generation_status(_string(record, "status")),
        termination=_optional_string(required_field(record, "termination"), "termination"),
        stop_reason=_optional_string(required_field(record, "stop_reason"), "stop_reason"),
        error=_optional_string(required_field(record, "error"), "error"),
    )
    _validate_generation_shape(generation)
    return generation


def _validate_generation_coverage(
    records: Sequence[GenerationRecord],
    prompts: Sequence[CanaryPrompt],
    model_role: ModelRole,
) -> None:
    if len(records) != len(prompts):
        raise CanaryValidationError("generation file must contain one row per frozen prompt")
    for prompt, record in zip(prompts, records, strict=True):
        _validate_generation_shape(record)
        expected = (
            prompt.sequence,
            model_role,
            prompt.question_id,
            prompt.variant,
            prompt.prompt_sha256,
            _attempt_id(model_role, prompt.sequence),
        )
        actual = (
            record.sequence,
            record.model_role,
            record.question_id,
            record.variant,
            record.prompt_sha256,
            record.attempt_id,
        )
        if actual != expected:
            raise CanaryValidationError("generation rows must exactly match frozen prompt order")


def _validate_generation_shape(record: GenerationRecord) -> None:
    _token_tuple(record.prompt_tokens, "prompt_tokens", require_nonempty=True)
    _token_tuple(record.response_tokens, "response_tokens")
    if not record.prompt_tokens:
        raise CanaryValidationError("prompt_tokens must not be empty")
    if record.status == "completed":
        if (
            record.error is not None
            or record.termination is None
            or not record.termination.strip()
            or record.stop_reason is None
            or not record.stop_reason.strip()
        ):
            raise CanaryValidationError("completed generation fields are inconsistent")
    elif any(
        (
            record.response_tokens,
            record.raw_response,
            record.parsed_response,
            record.termination is not None,
            record.stop_reason is not None,
            record.error is None or not record.error.strip(),
        )
    ):
        raise CanaryValidationError("error generation fields are inconsistent")


def _validate_paired_prompt_tokens(
    base: Sequence[GenerationRecord],
    adapter: Sequence[GenerationRecord],
) -> None:
    base_tokens = tuple(item.prompt_tokens for item in base)
    adapter_tokens = tuple(item.prompt_tokens for item in adapter)
    if base_tokens != adapter_tokens:
        raise CanaryValidationError("base and adapter prompt tokens differ")


def _primary_model_metrics(
    prompts: Sequence[CanaryPrompt],
    records: Sequence[GenerationRecord],
) -> PrimaryModelMetrics:
    oracle_errors: list[float] = []
    valid_oracle_errors: list[float] = []
    swap_errors: list[float] = []
    valid_swap_errors: list[float] = []
    valid_original = valid_swapped = valid_pairs = 0
    for index in range(CANARY_SIZE):
        original_prompt, swapped_prompt = prompts[index * 2 : index * 2 + 2]
        original = parsed_team_probability(records[index * 2])
        swapped = parsed_team_probability(records[index * 2 + 1])
        oracle = _prompt_oracle(original_prompt)
        original_error = INVALID_MAE if original is None else abs(original - oracle)
        oracle_errors.append(original_error)
        if original is not None:
            valid_original += 1
            valid_oracle_errors.append(original_error)
        if swapped is not None:
            valid_swapped += 1
        swap_error = (
            INVALID_MAE if original is None or swapped is None else abs(original + swapped - 1.0)
        )
        swap_errors.append(swap_error)
        if original is not None and swapped is not None:
            valid_pairs += 1
            valid_swap_errors.append(swap_error)
        _prompt_oracle(swapped_prompt)
    return PrimaryModelMetrics(
        game_count=CANARY_SIZE,
        valid_original_count=valid_original,
        valid_side_swap_count=valid_swapped,
        valid_pair_count=valid_pairs,
        schema_valid_rate=valid_original / CANARY_SIZE,
        side_swap_valid_rate=valid_swapped / CANARY_SIZE,
        valid_pair_rate=valid_pairs / CANARY_SIZE,
        oracle_mae=_mean(oracle_errors),
        valid_only_oracle_mae=_optional_mean(valid_oracle_errors),
        side_swap_mae=_mean(swap_errors),
        valid_only_side_swap_mae=_optional_mean(valid_swap_errors),
    )


def parsed_team_probability(record: GenerationRecord) -> float | None:
    """Return the strict parsed team probability, or ``None`` without repair."""
    if (
        record.status == "error"
        or record.termination != "stop_sequence"
        or record.stop_reason != "stop"
    ):
        return None
    try:
        prediction = parse_prediction(record.parsed_response, OUTCOMES)
    except ValueError:
        return None
    return prediction.distribution.probability_for("team_wins")


def _prompt_oracle(prompt: CanaryPrompt) -> float:
    user = parse_json_object(prompt.messages[1]["content"])
    prior = require_object(required_field(user, "prior"), "prior")
    probability = require_float(required_field(prior, "team_wins"), "prior.team_wins")
    evidence = require_list(required_field(user, "evidence"), "evidence")
    return elo_venue_probability(probability, _venue(require_string(evidence[0], "evidence[0]")))


def _seal_text(seal: GenerationSeal) -> str:
    value = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "kind": "forecastfm_canary_generation_seal",
        "status": "sealed_before_scoring",
        "manifest_sha256": seal.manifest_sha256,
        "prompts_sha256": seal.prompts_sha256,
        "attempt_sha256": seal.attempt_sha256,
        "base_sha256": seal.base_sha256,
        "adapter_sha256": seal.adapter_sha256,
    }
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _seal_from_dict(record: Mapping[str, object]) -> GenerationSeal:
    keys = {
        "schema_version",
        "kind",
        "status",
        "manifest_sha256",
        "prompts_sha256",
        "attempt_sha256",
        "base_sha256",
        "adapter_sha256",
    }
    require_exact_keys(record, keys, "generation seal")
    if required_field(record, "schema_version") != SEAL_SCHEMA_VERSION:
        raise CanaryValidationError("unsupported generation seal schema")
    if required_field(record, "kind") != "forecastfm_canary_generation_seal":
        raise CanaryValidationError("unexpected generation seal kind")
    if required_field(record, "status") != "sealed_before_scoring":
        raise CanaryValidationError("generation outputs were not sealed before scoring")
    seal = GenerationSeal(
        manifest_sha256=_string(record, "manifest_sha256"),
        prompts_sha256=_string(record, "prompts_sha256"),
        attempt_sha256=_string(record, "attempt_sha256"),
        base_sha256=_string(record, "base_sha256"),
        adapter_sha256=_string(record, "adapter_sha256"),
    )
    if any(
        not _is_hash(value)
        for value in (
            seal.manifest_sha256,
            seal.prompts_sha256,
            seal.attempt_sha256,
            seal.base_sha256,
            seal.adapter_sha256,
        )
    ):
        raise CanaryValidationError("generation seal contains an invalid digest")
    return seal


def _verify_seal(
    seal: GenerationSeal,
    manifest_path: Path,
    prompts_path: Path,
    base_path: Path,
    adapter_path: Path,
) -> None:
    actual = (
        file_sha256(manifest_path),
        file_sha256(prompts_path),
        file_sha256(base_path.parent / ATTEMPT_FILENAME),
        file_sha256(base_path),
        file_sha256(adapter_path),
    )
    expected = (
        seal.manifest_sha256,
        seal.prompts_sha256,
        seal.attempt_sha256,
        seal.base_sha256,
        seal.adapter_sha256,
    )
    if actual != expected:
        raise CanaryValidationError("generation files differ from their seal")


def _validate_attempt_marker(
    path: Path,
    manifest_path: Path,
    prompts_path: Path,
) -> None:
    record = parse_json_object(path.read_text(encoding="utf-8"))
    keys = {"schema_version", "kind", "status", "manifest_sha256", "prompts_sha256"}
    require_exact_keys(record, keys, "attempt marker")
    expected = (
        1,
        "forecastfm_canary_attempt",
        "started_before_remote_client",
        file_sha256(manifest_path),
        file_sha256(prompts_path),
    )
    actual = (
        required_field(record, "schema_version"),
        required_field(record, "kind"),
        required_field(record, "status"),
        _string(record, "manifest_sha256"),
        _string(record, "prompts_sha256"),
    )
    if actual != expected:
        raise CanaryValidationError("attempt marker differs from the frozen canary")


def _manifest_text(manifest: CanaryManifest) -> str:
    return json.dumps(_manifest_to_dict(manifest), indent=2, sort_keys=True) + "\n"


def _load_jsonl_objects(path: Path) -> tuple[dict[str, object], ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise CanaryValidationError(f"missing JSONL artifact: {path}") from error
    if not text.endswith("\n"):
        raise CanaryValidationError("JSONL artifact must end with a newline")
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise CanaryValidationError(f"blank JSONL record on line {line_number}")
        try:
            records.append(parse_json_object(line))
        except JsonFormatError as error:
            raise CanaryValidationError(f"invalid JSONL record on line {line_number}") from error
    return tuple(records)


def _jsonl_text(records: Iterable[Mapping[str, object]]) -> str:
    return "".join(f"{canonical_json(record)}\n" for record in records)


def _write_new_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as file:
            file.write(text)
    except FileExistsError as error:
        raise CanaryValidationError(f"refusing to replace frozen artifact: {path}") from error


def _require_new_paths(paths: Sequence[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise CanaryValidationError(f"refusing to replace frozen artifacts: {', '.join(existing)}")


def _attempt_id(model_role: ModelRole, sequence: int) -> str:
    return f"validation-canary-v1:{model_role}:{sequence:03d}"


def _token_tuple(
    values: Sequence[int],
    field_name: str,
    require_nonempty: bool = False,
) -> tuple[int, ...]:
    result = tuple(values)
    if require_nonempty and not result:
        raise CanaryValidationError(f"{field_name} must not be empty")
    if any(value < 0 for value in result):
        raise CanaryValidationError(f"{field_name} must contain nonnegative integers")
    return result


def _tokens(value: object, field_name: str, require_nonempty: bool = False) -> tuple[int, ...]:
    items = require_list(value, field_name)
    values: list[int] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int):
            raise CanaryValidationError(f"{field_name} must contain integers")
        values.append(item)
    return _token_tuple(values, field_name, require_nonempty)


def _variant(value: str) -> Variant:
    if value == "original":
        return "original"
    if value == "side_swap":
        return "side_swap"
    raise CanaryValidationError(f"unknown canary variant: {value}")


def _model_role(value: str) -> ModelRole:
    if value == "base":
        return "base"
    if value == "adapter":
        return "adapter"
    raise CanaryValidationError(f"unknown model role: {value}")


def _generation_status(value: str) -> GenerationStatus:
    if value == "completed":
        return "completed"
    if value == "error":
        return "error"
    raise CanaryValidationError(f"unknown generation status: {value}")


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return require_string(value, field_name)


def _strings(value: object, field_name: str) -> tuple[str, ...]:
    return tuple(
        require_string(item, f"{field_name}[{index}]")
        for index, item in enumerate(require_list(value, field_name))
    )


def _string(record: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = required_field(record, field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CanaryValidationError(f"{field_name} must be a nonnegative integer")
    return value


def _is_hash(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_git_revision(value: str) -> bool:
    return len(value) in {40, 64} and all(character in "0123456789abcdef" for character in value)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise CanaryValidationError("cannot average an empty metric")
    return sum(values) / len(values)


def _optional_mean(values: Sequence[float]) -> float | None:
    return None if not values else _mean(values)
