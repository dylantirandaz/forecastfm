"""Tests for frozen answer-blind outcome evaluation artifacts."""

from dataclasses import replace
from datetime import UTC, datetime
from math import log
from pathlib import Path
from typing import cast

import pytest

from forecastfm.integrity import canonical_sha256, file_sha256
from forecastfm.models import TrainingExample
from forecastfm.nba_data import side_swap_nba_example
from forecastfm.outcome_evaluation import (
    EvaluationPaths,
    ModelRole,
    OrientationResult,
    OutcomeEvaluationError,
    OutcomeEvaluationManifest,
    OutcomeEvaluationRecord,
    RecordStatus,
    completed_record,
    failed_record,
    load_sealed_evaluation,
    read_records,
    seal_outputs,
    write_attempt_marker,
    write_manifest,
    write_records,
)
from forecastfm.tinker_data import write_outcome_forecast_jsonl
from tests.helpers import make_nba_training_example

CREATED_AT = datetime(2026, 7, 16, 3, tzinfo=UTC)


def _paths(root: Path) -> EvaluationPaths:
    return EvaluationPaths(
        manifest=root / "manifest.json",
        prompts=root / "prompts.jsonl",
        attempt=root / "raw" / "attempt.json",
        journal=root / "raw" / "journal.jsonl",
        base=root / "raw" / "base.jsonl",
        adapter=root / "raw" / "adapter.jsonl",
        seal=root / "raw" / "manifest.json",
    )


def _examples() -> tuple[TrainingExample, ...]:
    examples: list[TrainingExample] = []
    for index in range(2):
        template = make_nba_training_example()
        original = replace(
            template,
            case=replace(
                template.case,
                question=replace(template.case.question, question_id=f"nba-eval-{index}"),
            ),
        )
        examples.extend((original, side_swap_nba_example(original)))
    return tuple(examples)


def _manifest(paths: EvaluationPaths) -> OutcomeEvaluationManifest:
    question_ids = ("nba-eval-0", "nba-eval-1")
    return OutcomeEvaluationManifest(
        created_at=CREATED_AT.isoformat(),
        protocol_revision="a" * 40,
        source_manifest_sha256="b" * 64,
        source_prompts_sha256=file_sha256(paths.prompts),
        source_answers_sha256="c" * 64,
        frozen_prompts_sha256=file_sha256(paths.prompts),
        training_lock_sha256="d" * 64,
        experiment_sha256="e" * 64,
        base_model="Qwen/Qwen3.5-4B",
        adapter_sampler_path="tinker://run/sampler_weights/final",
        renderer_name="qwen3_5_disable_thinking",
        team_token_id=10,
        opponent_token_id=20,
        game_count=2,
        orientation_count=4,
        logical_calls_per_game_per_arm=4,
        expected_total_logical_calls=16,
        max_active_arms=1,
        application_retries=0,
        transport_retry_note="same logical request may be retransmitted",
        question_ids=question_ids,
        question_ids_sha256=canonical_sha256(list(question_ids)),
        scoring_policy={"primary": "mean_log_loss"},
    )


def _orientation(team_probability: float) -> OrientationResult:
    valid_label_mass = 0.9
    return OrientationResult(
        team_logprob=log(team_probability * valid_label_mass),
        opponent_logprob=log((1.0 - team_probability) * valid_label_mass),
        team_probability=team_probability,
        valid_label_mass=valid_label_mass,
    )


def _records(model_role: ModelRole) -> tuple[OutcomeEvaluationRecord, ...]:
    return tuple(
        completed_record(
            sequence,
            model_role,
            (f"nba-eval-{sequence}", f"nba-eval-{sequence}-side-swap"),
            ((1, sequence + 2), (1, sequence + 3)),
            (_orientation(0.6), _orientation(0.3)),
        )
        for sequence in range(2)
    )


def _build_sealed(root: Path) -> EvaluationPaths:
    paths = _paths(root)
    write_outcome_forecast_jsonl((example.case for example in _examples()), paths.prompts)
    manifest = _manifest(paths)
    write_manifest(paths.manifest, manifest)
    write_attempt_marker(paths.attempt, paths.manifest, paths.prompts, CREATED_AT)
    paths.journal.write_text("sealed test journal\n", encoding="utf-8")
    write_records(paths.base, _records("base"), manifest, "base")
    write_records(paths.adapter, _records("adapter"), manifest, "adapter")
    seal_outputs(paths, CREATED_AT)
    return paths


def test_sealed_outcome_evaluation_round_trips(tmp_path: Path) -> None:
    paths = _build_sealed(tmp_path)

    sealed = load_sealed_evaluation(paths)

    assert sealed.manifest.game_count == 2
    assert len(sealed.prompt_pairs) == 2
    assert sealed.base[0].symmetric_team_probability == pytest.approx(0.65)


def test_seal_rejects_changed_raw_output(tmp_path: Path) -> None:
    paths = _build_sealed(tmp_path)
    with paths.base.open("a", encoding="utf-8") as file:
        file.write("\n")

    with pytest.raises(OutcomeEvaluationError):
        load_sealed_evaluation(paths)


def test_failed_record_cannot_hide_partial_scores() -> None:
    record = failed_record(
        0,
        "base",
        ("nba-eval-0", "nba-eval-0-side-swap"),
        ((1, 2), (1, 3)),
        "provider_error",
    )

    assert record.status == "failed"
    assert record.symmetric_team_probability is None


def test_manifest_rejects_wrong_call_count(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    write_outcome_forecast_jsonl((example.case for example in _examples()), paths.prompts)

    with pytest.raises(OutcomeEvaluationError, match="logical-call"):
        replace(_manifest(paths), expected_total_logical_calls=15)


def test_manifest_rejects_non_tinker_adapter_path(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    write_outcome_forecast_jsonl((example.case for example in _examples()), paths.prompts)

    with pytest.raises(OutcomeEvaluationError, match="sampler path"):
        replace(_manifest(paths), adapter_sampler_path="https://example.com/adapter")


def test_orientation_rejects_derived_values_that_disagree() -> None:
    with pytest.raises(OutcomeEvaluationError, match="mass differs"):
        OrientationResult(log(0.6), log(0.4), 0.6, 0.8)

    with pytest.raises(OutcomeEvaluationError, match="probability differs"):
        OrientationResult(log(0.6), log(0.4), 0.7, 1.0)


def test_record_runtime_rejects_unknown_status() -> None:
    record = _records("base")[0]

    with pytest.raises(OutcomeEvaluationError, match="status is invalid"):
        replace(record, status=cast(RecordStatus, "unknown"))


def test_write_records_rejects_reordered_coverage(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    write_outcome_forecast_jsonl((example.case for example in _examples()), paths.prompts)
    manifest = _manifest(paths)

    with pytest.raises(OutcomeEvaluationError, match="order differs"):
        write_records(paths.base, tuple(reversed(_records("base"))), manifest, "base")


def test_read_records_requires_terminal_newline(tmp_path: Path) -> None:
    paths = _build_sealed(tmp_path)
    paths.base.write_text(paths.base.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")

    with pytest.raises(OutcomeEvaluationError, match="end with a newline"):
        read_records(paths.base, _manifest(paths), "base")


def test_attempt_rejects_changed_prompts_and_duplicate_creation(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    write_outcome_forecast_jsonl((example.case for example in _examples()), paths.prompts)
    write_manifest(paths.manifest, _manifest(paths))
    write_attempt_marker(paths.attempt, paths.manifest, paths.prompts, CREATED_AT)

    with pytest.raises(OutcomeEvaluationError, match="refusing to replace"):
        write_attempt_marker(paths.attempt, paths.manifest, paths.prompts, CREATED_AT)

    changed_path = tmp_path / "changed-prompts.jsonl"
    changed_path.write_text(
        paths.prompts.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(OutcomeEvaluationError, match="prompts differ"):
        write_attempt_marker(
            tmp_path / "wrong-attempt.json", paths.manifest, changed_path, CREATED_AT
        )


def test_seal_rejects_cross_arm_token_mismatch_even_on_failed_row(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    write_outcome_forecast_jsonl((example.case for example in _examples()), paths.prompts)
    manifest = _manifest(paths)
    write_manifest(paths.manifest, manifest)
    write_attempt_marker(paths.attempt, paths.manifest, paths.prompts, CREATED_AT)
    paths.journal.write_text("sealed test journal\n", encoding="utf-8")
    base = (
        failed_record(
            0,
            "base",
            ("nba-eval-0", "nba-eval-0-side-swap"),
            ((1, 2), (1, 3)),
            "provider_error",
        ),
        _records("base")[1],
    )
    adapter = (
        failed_record(
            0,
            "adapter",
            ("nba-eval-0", "nba-eval-0-side-swap"),
            ((9, 2), (1, 3)),
            "provider_error",
        ),
        _records("adapter")[1],
    )
    write_records(paths.base, base, manifest, "base")
    write_records(paths.adapter, adapter, manifest, "adapter")

    with pytest.raises(OutcomeEvaluationError, match="rendered different prompt tokens"):
        seal_outputs(paths, CREATED_AT)


def test_sealed_evaluation_rejects_changed_journal(tmp_path: Path) -> None:
    paths = _build_sealed(tmp_path)
    with paths.journal.open("a", encoding="utf-8") as file:
        file.write("changed\n")

    with pytest.raises(OutcomeEvaluationError, match="journal_sha256"):
        load_sealed_evaluation(paths)
