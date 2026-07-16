"""Tests for publication-gated outcome development scoring."""

from dataclasses import replace
from datetime import UTC, datetime
from math import log
from pathlib import Path
from typing import cast

import pytest
from examples import build_outcome_development_evaluation as builder
from examples import score_outcome_development as scorer

from forecastfm.integrity import canonical_sha256, file_sha256
from forecastfm.models import Distribution, ForecastPrediction, TrainingExample
from forecastfm.nba_data import side_swap_nba_example
from forecastfm.outcome_evaluation import (
    EvaluationPaths,
    ModelRole,
    OrientationResult,
    OutcomeEvaluationError,
    OutcomeEvaluationManifest,
    OutcomeEvaluationRecord,
    SealedOutcomeEvaluation,
    completed_record,
)
from forecastfm.publication import PublicationError, PublicationProof
from forecastfm.serialization import write_jsonl
from forecastfm.tinker_data import build_outcome_forecast_record
from tests.helpers import make_nba_training_example


def _manifest() -> OutcomeEvaluationManifest:
    question_ids = ("nba-score-0", "nba-score-1")
    return OutcomeEvaluationManifest(
        created_at=datetime(2026, 7, 16, 3, tzinfo=UTC).isoformat(),
        protocol_revision="a" * 40,
        source_manifest_sha256="b" * 64,
        source_prompts_sha256="c" * 64,
        source_answers_sha256="d" * 64,
        frozen_prompts_sha256="c" * 64,
        training_lock_sha256="e" * 64,
        experiment_sha256="f" * 64,
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


def _answer(
    index: int,
    realized_outcome: str,
    elo_team_probability: float,
) -> TrainingExample:
    example = make_nba_training_example(realized_outcome)
    question = replace(example.case.question, question_id=f"nba-score-{index}")
    target = ForecastPrediction(
        Distribution(
            outcomes=example.case.question.outcomes,
            probabilities=(elo_team_probability, 1.0 - elo_team_probability),
        )
    )
    return replace(example, case=replace(example.case, question=question), target=target)


def _orientation(probability: float) -> OrientationResult:
    valid_label_mass = 0.9
    return OrientationResult(
        log(probability * valid_label_mass),
        log((1.0 - probability) * valid_label_mass),
        probability,
        valid_label_mass,
    )


def _records(
    role: ModelRole,
    probabilities: tuple[float, float],
) -> tuple[OutcomeEvaluationRecord, ...]:
    return tuple(
        completed_record(
            index,
            role,
            (f"nba-score-{index}", f"nba-score-{index}-side-swap"),
            ((1, index + 2), (1, index + 3)),
            (_orientation(probability), _orientation(1.0 - probability)),
        )
        for index, probability in enumerate(probabilities)
    )


def test_report_compares_adapter_base_and_venue_adjusted_elo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = (
        _answer(0, "team_wins", 0.7),
        _answer(1, "opponent_wins", 0.3),
    )
    sealed = SealedOutcomeEvaluation(
        manifest=_manifest(),
        prompt_pairs=(),
        base=_records("base", (0.6, 0.4)),
        adapter=_records("adapter", (0.8, 0.2)),
        seal={},
    )
    publication = PublicationProof("a" * 40, "origin", "url", "refs/heads/main")

    def constant_hash(_path: Path) -> str:
        return "f" * 64

    monkeypatch.setattr(scorer, "file_sha256", constant_hash)

    report = scorer.build_report(sealed, answers, publication)

    metrics = cast(dict[str, object], report["metrics"])
    adapter = cast(dict[str, object], metrics["adapter"])
    assert adapter["mean_brier"] == pytest.approx(0.04)
    assert report["paired_deltas"]


def test_main_never_opens_answers_when_publication_gate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer_accessed = False

    def reject_publication(_revision: str) -> PublicationProof:
        raise PublicationError("not published")

    def open_answers(_sealed: SealedOutcomeEvaluation) -> tuple[TrainingExample, ...]:
        nonlocal answer_accessed
        answer_accessed = True
        return ()

    def read_test_manifest(_path: Path) -> OutcomeEvaluationManifest:
        return _manifest()

    monkeypatch.setattr(scorer, "SCORES_PATH", tmp_path / "scores.json")
    monkeypatch.setattr(scorer, "read_manifest", read_test_manifest)
    monkeypatch.setattr(scorer, "require_published_raw_outputs", reject_publication)
    monkeypatch.setattr(scorer, "load_original_answers", open_answers)

    with pytest.raises(PublicationError, match="not published"):
        scorer.main()
    assert answer_accessed is False


def test_main_never_opens_answers_when_seal_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer_accessed = False
    publication = PublicationProof("a" * 40, "origin", "url", "refs/heads/main")

    def read_test_manifest(_path: Path) -> OutcomeEvaluationManifest:
        return _manifest()

    def accept_publication(_revision: str) -> PublicationProof:
        return publication

    def accept_bindings(_manifest: OutcomeEvaluationManifest) -> None:
        return None

    def reject_seal(_paths: EvaluationPaths) -> SealedOutcomeEvaluation:
        raise OutcomeEvaluationError("invalid seal")

    def open_answers(_sealed: SealedOutcomeEvaluation) -> tuple[TrainingExample, ...]:
        nonlocal answer_accessed
        answer_accessed = True
        return ()

    monkeypatch.setattr(scorer, "SCORES_PATH", tmp_path / "scores.json")
    monkeypatch.setattr(scorer, "read_manifest", read_test_manifest)
    monkeypatch.setattr(scorer, "require_published_raw_outputs", accept_publication)
    monkeypatch.setattr(scorer, "verify_manifest_bindings", accept_bindings)
    monkeypatch.setattr(scorer, "load_sealed_evaluation", reject_seal)
    monkeypatch.setattr(scorer, "load_original_answers", open_answers)

    with pytest.raises(OutcomeEvaluationError, match="invalid seal"):
        scorer.main()
    assert answer_accessed is False


def test_main_never_opens_answers_when_manifest_binding_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer_accessed = False
    publication = PublicationProof("a" * 40, "origin", "url", "refs/heads/main")

    def read_test_manifest(_path: Path) -> OutcomeEvaluationManifest:
        return _manifest()

    def accept_publication(_revision: str) -> PublicationProof:
        return publication

    def reject_bindings(_manifest: OutcomeEvaluationManifest) -> None:
        raise OutcomeEvaluationError("wrong source binding")

    def open_answers(_sealed: SealedOutcomeEvaluation) -> tuple[TrainingExample, ...]:
        nonlocal answer_accessed
        answer_accessed = True
        return ()

    monkeypatch.setattr(scorer, "SCORES_PATH", tmp_path / "scores.json")
    monkeypatch.setattr(scorer, "read_manifest", read_test_manifest)
    monkeypatch.setattr(scorer, "require_published_raw_outputs", accept_publication)
    monkeypatch.setattr(scorer, "verify_manifest_bindings", reject_bindings)
    monkeypatch.setattr(scorer, "load_original_answers", open_answers)

    with pytest.raises(OutcomeEvaluationError, match="wrong source binding"):
        scorer.main()
    assert answer_accessed is False


def test_main_never_opens_answers_when_journal_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answer_accessed = False
    publication = PublicationProof("a" * 40, "origin", "url", "refs/heads/main")
    sealed = SealedOutcomeEvaluation(
        manifest=_manifest(),
        prompt_pairs=(),
        base=_records("base", (0.6, 0.4)),
        adapter=_records("adapter", (0.8, 0.2)),
        seal={},
    )

    def read_test_manifest(_path: Path) -> OutcomeEvaluationManifest:
        return _manifest()

    def accept_publication(_revision: str) -> PublicationProof:
        return publication

    def accept_bindings(_manifest: OutcomeEvaluationManifest) -> None:
        return None

    def load_test_seal(_paths: EvaluationPaths) -> SealedOutcomeEvaluation:
        return sealed

    def reject_journal(
        _manifest: OutcomeEvaluationManifest,
        _base: tuple[OutcomeEvaluationRecord, ...],
        _adapter: tuple[OutcomeEvaluationRecord, ...],
    ) -> None:
        raise OutcomeEvaluationError("journal mismatch")

    def open_answers(_sealed: SealedOutcomeEvaluation) -> tuple[TrainingExample, ...]:
        nonlocal answer_accessed
        answer_accessed = True
        return ()

    monkeypatch.setattr(scorer, "SCORES_PATH", tmp_path / "scores.json")
    monkeypatch.setattr(scorer, "read_manifest", read_test_manifest)
    monkeypatch.setattr(scorer, "require_published_raw_outputs", accept_publication)
    monkeypatch.setattr(scorer, "verify_manifest_bindings", accept_bindings)
    monkeypatch.setattr(scorer, "load_sealed_evaluation", load_test_seal)
    monkeypatch.setattr(scorer, "require_journal_matches", reject_journal)
    monkeypatch.setattr(scorer, "load_original_answers", open_answers)

    with pytest.raises(OutcomeEvaluationError, match="journal mismatch"):
        scorer.main()
    assert answer_accessed is False


def test_answer_loader_rejects_prompt_content_misalignment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    originals = (
        _answer(0, "team_wins", 0.7),
        _answer(1, "opponent_wins", 0.3),
    )
    answers = tuple(
        example for original in originals for example in (original, side_swap_nba_example(original))
    )
    answers_path = tmp_path / "answers.jsonl"
    write_jsonl(answers, answers_path)
    prompt_pairs = tuple(
        (
            build_outcome_forecast_record(original.case),
            build_outcome_forecast_record(side_swap_nba_example(original).case),
        )
        for original in reversed(originals)
    )
    manifest = replace(_manifest(), source_answers_sha256=file_sha256(answers_path))
    sealed = SealedOutcomeEvaluation(manifest, prompt_pairs, (), (), {})
    monkeypatch.setattr(scorer, "ANSWERS_PATH", answers_path)

    with pytest.raises(OutcomeEvaluationError, match="prompts do not match"):
        scorer.load_original_answers(sealed)


def test_builder_never_constructs_or_hashes_the_answer_path() -> None:
    source = Path(builder.__file__).read_text(encoding="utf-8")

    assert "ANSWERS_PATH" not in source
    assert "file_sha256(ANSWERS" not in source
