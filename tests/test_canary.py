"""Tests for the frozen validation canary and answer-gated diagnostics."""

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from forecastfm.canary import (
    CANARY_SIZE,
    INVALID_BRIER,
    INVALID_LOG_LOSS,
    INVALID_MAE,
    PROMPT_COUNT,
    CanaryManifest,
    CanaryModels,
    CanaryPrompt,
    CanarySource,
    CanaryValidationError,
    CompletedGeneration,
    GenerationRecord,
    ModelRole,
    SealedGenerations,
    build_canary_artifacts,
    failed_generation,
    load_canary,
    load_generation_records,
    load_sealed_generations,
    parsed_team_probability,
    score_primary,
    seal_generation_outputs,
    successful_generation,
    write_attempt_marker,
    write_generation_records,
)
from forecastfm.canary_history import score_historical
from forecastfm.integrity import canonical_json, canonical_sha256, file_sha256
from forecastfm.models import (
    Distribution,
    EvidenceCard,
    ForecastCase,
    ForecastPrediction,
    ForecastQuestion,
    TrainingExample,
)
from forecastfm.nba_data import elo_venue_probability
from forecastfm.prompting import build_forecast_messages
from forecastfm.serialization import write_jsonl


@dataclass(frozen=True, slots=True)
class BuiltCanary:
    manifest: CanaryManifest
    prompts: tuple[CanaryPrompt, ...]
    manifest_path: Path
    prompts_path: Path
    answers_path: Path


def _example(index: int) -> TrainingExample:
    forecast_at = datetime(2010, 1, 1, tzinfo=UTC) + timedelta(days=index)
    question_id = f"nba-{index:016x}"
    team_probability = round(0.2 + index / 200.0, 7)
    venue = "home" if index % 2 == 0 else "away"
    outcomes = ("team_wins", "opponent_wins")
    case = ForecastCase(
        question=ForecastQuestion(
            question_id=question_id,
            text="Will the listed team defeat its opponent in this NBA game?",
            resolution_rule="Resolve to the team with the higher final score.",
            resolution_source="source",
            outcomes=outcomes,
            forecast_at=forecast_at,
            resolves_at=forecast_at + timedelta(days=2),
        ),
        prior=Distribution(
            outcomes=outcomes,
            probabilities=(team_probability, round(1.0 - team_probability, 7)),
        ),
        prior_source="neutral Elo",
        prior_as_of=forecast_at,
        evidence=(
            EvidenceCard(
                text=f"Venue for the listed team: {venue}.",
                source="source",
                available_at=forecast_at,
            ),
        ),
    )
    target = elo_venue_probability(team_probability, venue)
    return TrainingExample(
        case=case,
        target=ForecastPrediction(
            Distribution(outcomes=outcomes, probabilities=(target, 1.0 - target))
        ),
        target_information_cutoff=forecast_at,
        target_method="Elo fixture",
        realized_outcome="team_wins" if index % 3 else "opponent_wins",
    )


def _write_source(path: Path, examples: tuple[TrainingExample, ...]) -> None:
    records = [
        {
            "question_id": example.case.question.question_id,
            "messages": list(build_forecast_messages(example.case)),
        }
        for example in reversed(examples)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{canonical_json(record)}\n" for record in records),
        encoding="utf-8",
    )


def _build(tmp_path: Path, *, include_answers: bool = True) -> BuiltCanary:
    examples = tuple(_example(index) for index in range(70))
    source_path = tmp_path / "data" / "nba_elo_validation_prompts.jsonl"
    answers_path = tmp_path / "data" / "nba_elo_validation_answers.jsonl"
    _write_source(source_path, examples)
    if include_answers:
        write_jsonl(examples, answers_path)
        answers_hash = file_sha256(answers_path)
    else:
        answers_hash = "b" * 64
    selected_ids = tuple(sorted(item.case.question.question_id for item in examples)[:CANARY_SIZE])
    source = CanarySource(
        validation_prompts_path=source_path,
        validation_prompts_sha256=file_sha256(source_path),
        validation_answers_sha256=answers_hash,
        dataset_manifest_sha256="c" * 64,
        expected_question_ids_sha256=canonical_sha256(list(selected_ids)),
    )
    models = CanaryModels(
        training_lock_sha256="d" * 64,
        experiment_sha256="e" * 64,
        base_model="Qwen/Qwen3.5-4B",
        adapter_sampler_path="tinker://adapter/final",
        decoding={"temperature": 0.0, "seed": 0, "max_attempts": 1},
        protocol_code_revision="f" * 40,
    )
    output = tmp_path / "evaluation" / "validation_canary"
    prompts_path = output / "prompts.jsonl"
    manifest_path = output / "manifest.json"
    manifest = build_canary_artifacts(source, models, prompts_path, manifest_path)
    loaded_manifest, prompts = load_canary(manifest_path, prompts_path)
    assert loaded_manifest == manifest
    return BuiltCanary(manifest, prompts, manifest_path, prompts_path, answers_path)


def _oracle(prompt: CanaryPrompt) -> float:
    user = json.loads(prompt.messages[1]["content"])
    evidence = user["evidence"][0]
    venue = evidence.removeprefix("Venue for the listed team: ").removesuffix(".")
    return elo_venue_probability(user["prior"]["team_wins"], venue)


def _completed(prompt: CanaryPrompt, role: ModelRole) -> GenerationRecord:
    probability = _oracle(prompt)
    response = canonical_json(
        {"probabilities": {"team_wins": probability, "opponent_wins": 1.0 - probability}}
    )
    return successful_generation(
        prompt,
        role,
        CompletedGeneration(
            prompt_tokens=(10, prompt.sequence + 20),
            response_tokens=(30, prompt.sequence + 40),
            raw_response=response,
            parsed_response=response,
            termination="stop_sequence",
            stop_reason="stop",
        ),
    )


def _records(
    prompts: tuple[CanaryPrompt, ...],
    role: ModelRole,
    *,
    fail_first: bool = False,
) -> tuple[GenerationRecord, ...]:
    records = [_completed(prompt, role) for prompt in prompts]
    if fail_first:
        records[0] = failed_generation(
            prompts[0],
            role,
            (10, 20),
            "provider_exception:TimeoutError",
        )
    return tuple(records)


def test_strict_probability_requires_clean_renderer_and_provider_stop(tmp_path: Path) -> None:
    built = _build(tmp_path)
    clean = _completed(built.prompts[0], "base")

    assert parsed_team_probability(clean) is not None
    assert parsed_team_probability(replace(clean, termination="eos")) is None
    assert parsed_team_probability(replace(clean, stop_reason="length")) is None


def _seal(
    tmp_path: Path,
    *,
    fail_first_adapter: bool = False,
) -> tuple[BuiltCanary, SealedGenerations, Path]:
    built = _build(tmp_path)
    raw = built.prompts_path.parent / "raw"
    marker_path = raw / "attempt.json"
    base_path = raw / "base.jsonl"
    adapter_path = raw / "adapter.jsonl"
    seal_path = raw / "manifest.json"
    write_attempt_marker(marker_path, built.manifest_path, built.prompts_path)
    write_generation_records(base_path, _records(built.prompts, "base"), built.prompts, "base")
    write_generation_records(
        adapter_path,
        _records(built.prompts, "adapter", fail_first=fail_first_adapter),
        built.prompts,
        "adapter",
    )
    seal_generation_outputs(
        seal_path,
        built.manifest_path,
        built.prompts_path,
        base_path,
        adapter_path,
    )
    generations = load_sealed_generations(
        seal_path,
        built.manifest_path,
        built.prompts_path,
        base_path,
        adapter_path,
    )
    return built, generations, base_path


def test_builder_selects_lexical_ids_and_creates_deterministic_swaps(tmp_path: Path) -> None:
    built = _build(tmp_path, include_answers=False)

    assert len(built.manifest.question_ids) == CANARY_SIZE
    assert built.manifest.question_ids == tuple(sorted(built.manifest.question_ids))
    assert len(built.prompts) == PROMPT_COUNT
    for original, swapped in zip(built.prompts[::2], built.prompts[1::2], strict=True):
        assert original.variant == "original"
        assert swapped.variant == "side_swap"
        original_user = json.loads(original.messages[1]["content"])
        swapped_user = json.loads(swapped.messages[1]["content"])
        assert swapped_user["prior"]["team_wins"] == original_user["prior"]["opponent_wins"]
        assert _oracle(swapped) == pytest.approx(1.0 - _oracle(original))

    with pytest.raises(CanaryValidationError, match="replace"):
        _build(tmp_path, include_answers=False)


def test_builder_rejects_wrong_digest_target_fields_and_test_path(tmp_path: Path) -> None:
    examples = tuple(_example(index) for index in range(70))
    source_path = tmp_path / "nba_elo_validation_prompts.jsonl"
    _write_source(source_path, examples)
    selected_ids = tuple(sorted(item.case.question.question_id for item in examples)[:CANARY_SIZE])

    def source(path: Path, digest: str) -> CanarySource:
        return CanarySource(
            validation_prompts_path=path,
            validation_prompts_sha256=digest,
            validation_answers_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            expected_question_ids_sha256=canonical_sha256(list(selected_ids)),
        )

    models = CanaryModels(
        "c" * 64,
        "d" * 64,
        "base",
        "tinker://adapter",
        {},
        "e" * 40,
    )
    with pytest.raises(CanaryValidationError, match="digest"):
        build_canary_artifacts(
            source(source_path, "0" * 64), models, tmp_path / "wrong.jsonl", tmp_path / "wrong.json"
        )

    test_path = tmp_path / "nba_elo_test_prompts.jsonl"
    test_path.write_bytes(source_path.read_bytes())
    with pytest.raises(CanaryValidationError, match="validation prompt"):
        build_canary_artifacts(
            source(test_path, file_sha256(test_path)),
            models,
            tmp_path / "test.jsonl",
            tmp_path / "test.json",
        )

    records = source_path.read_text(encoding="utf-8").splitlines()
    contaminated = json.loads(records[0])
    contaminated["target"] = [0.9, 0.1]
    records[0] = canonical_json(contaminated)
    source_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    with pytest.raises(CanaryValidationError, match="invalid source"):
        build_canary_artifacts(
            source(source_path, file_sha256(source_path)),
            models,
            tmp_path / "target.jsonl",
            tmp_path / "target.json",
        )


def test_generation_files_require_exact_order_and_are_create_only(tmp_path: Path) -> None:
    built = _build(tmp_path)
    records = _records(built.prompts, "base")
    output = tmp_path / "base.jsonl"

    with pytest.raises(CanaryValidationError, match="one row"):
        write_generation_records(output, records[:-1], built.prompts, "base")
    digest = write_generation_records(output, records, built.prompts, "base")
    assert digest == file_sha256(output)
    assert load_generation_records(output, built.prompts, "base") == records
    with pytest.raises(CanaryValidationError, match="replace"):
        write_generation_records(output, records, built.prompts, "base")

    reordered = (records[1], records[0], *records[2:])
    with pytest.raises(CanaryValidationError, match="exactly match"):
        write_generation_records(tmp_path / "reordered.jsonl", reordered, built.prompts, "base")


def test_attempt_marker_prevents_rerun_and_seal_detects_tampering(tmp_path: Path) -> None:
    built, generations, base_path = _seal(tmp_path)
    marker = base_path.parent / "attempt.json"
    with pytest.raises(CanaryValidationError, match="replace"):
        write_attempt_marker(marker, built.manifest_path, built.prompts_path)
    assert len(generations.base) == PROMPT_COUNT

    base_path.write_text(base_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CanaryValidationError, match="seal"):
        load_sealed_generations(
            base_path.parent / "manifest.json",
            built.manifest_path,
            built.prompts_path,
            base_path,
            base_path.parent / "adapter.jsonl",
        )


def test_primary_scores_complete_cohort_and_penalizes_invalid_rows(tmp_path: Path) -> None:
    _built, generations, _base_path = _seal(tmp_path, fail_first_adapter=True)
    scores = score_primary(generations)

    assert scores.base.schema_valid_rate == 1.0
    assert scores.base.oracle_mae == pytest.approx(0.0)
    assert scores.base.side_swap_mae == pytest.approx(0.0)
    assert scores.adapter.schema_valid_rate == (CANARY_SIZE - 1) / CANARY_SIZE
    assert scores.adapter.valid_pair_rate == (CANARY_SIZE - 1) / CANARY_SIZE
    assert scores.adapter.oracle_mae == pytest.approx(INVALID_MAE / CANARY_SIZE)
    assert scores.adapter.side_swap_mae == pytest.approx(INVALID_MAE / CANARY_SIZE)


def test_historical_scores_open_only_committed_answers_after_seal(tmp_path: Path) -> None:
    built, generations, _base_path = _seal(tmp_path, fail_first_adapter=True)
    scores = score_historical(generations, built.answers_path)

    assert scores.base.valid_count == CANARY_SIZE
    assert scores.base.teacher_mae == pytest.approx(0.0)
    assert scores.adapter.valid_count == CANARY_SIZE - 1
    assert scores.adapter.teacher_mae == pytest.approx(INVALID_MAE / CANARY_SIZE)
    assert scores.adapter.mean_brier >= INVALID_BRIER / CANARY_SIZE
    assert scores.adapter.mean_log_loss >= INVALID_LOG_LOSS / CANARY_SIZE

    built.answers_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(CanaryValidationError, match="commitment"):
        score_historical(generations, built.answers_path)
