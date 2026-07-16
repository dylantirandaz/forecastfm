"""Score outcome development only after raw outputs are sealed and published."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from examples.build_outcome_development_evaluation import (
    EXPECTED_REMOTE_URL,
    EXPERIMENT_PATH,
    MANIFEST_PATH,
    OUTPUT_DIRECTORY,
    PROMPTS_PATH,
    PROTOCOL_PATHS,
    SOURCE_MANIFEST_PATH,
    TRAINING_LOCK_PATH,
)
from examples.run_tinker_outcome_development import (
    ADAPTER_PATH,
    ATTEMPT_PATH,
    BASE_PATH,
    JOURNAL_PATH,
    PATHS,
    SEAL_PATH,
    require_journal_matches,
    verify_manifest_bindings,
)

from forecastfm.integrity import file_sha256
from forecastfm.models import TrainingExample
from forecastfm.nba_data import side_swap_nba_example
from forecastfm.outcome import TEAM_OUTCOME
from forecastfm.outcome_evaluation import (
    OutcomeEvaluationError,
    OutcomeEvaluationRecord,
    SealedOutcomeEvaluation,
    load_sealed_evaluation,
    read_manifest,
    write_json_exclusively,
)
from forecastfm.outcome_metrics import (
    DifficultySubsets,
    OutcomeMetrics,
    difficulty_subsets,
    paired_mean_delta,
    summarize_outcome_metrics,
)
from forecastfm.publication import (
    PublicationProof,
    require_paths_at_head,
    require_protocol_unchanged,
    require_published_head,
)
from forecastfm.serialization import read_jsonl
from forecastfm.tinker_data import build_outcome_forecast_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANSWERS_PATH = PROJECT_ROOT / "data" / "processed" / "outcome_v1" / "nba_development_answers.jsonl"
SCORES_PATH = OUTPUT_DIRECTORY / "scores.json"
PUBLISHED_ARTIFACTS = (
    SOURCE_MANIFEST_PATH,
    TRAINING_LOCK_PATH,
    EXPERIMENT_PATH,
    MANIFEST_PATH,
    PROMPTS_PATH,
    ATTEMPT_PATH,
    JOURNAL_PATH,
    BASE_PATH,
    ADAPTER_PATH,
    SEAL_PATH,
)


def main() -> None:
    """Prove publication, then open answers and write immutable scores."""
    if SCORES_PATH.exists():
        raise FileExistsError(f"refusing to replace outcome scores: {SCORES_PATH}")
    manifest = read_manifest(MANIFEST_PATH)
    publication = require_published_raw_outputs(manifest.protocol_revision)
    verify_manifest_bindings(manifest)
    sealed = load_sealed_evaluation(PATHS)
    require_journal_matches(manifest, sealed.base, sealed.adapter)
    answers = load_original_answers(sealed)
    report = build_report(sealed, answers, publication)
    write_json_exclusively(SCORES_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


def require_published_raw_outputs(protocol_revision: str) -> PublicationProof:
    """Prove exact sealed raw artifacts exist at authoritative origin/main."""
    proof = require_published_head(PROJECT_ROOT, EXPECTED_REMOTE_URL)
    require_paths_at_head(
        PROJECT_ROOT,
        proof.commit,
        (*PROTOCOL_PATHS, *PUBLISHED_ARTIFACTS),
    )
    require_protocol_unchanged(
        PROJECT_ROOT,
        protocol_revision,
        proof.commit,
        PROTOCOL_PATHS,
    )
    return proof


def build_report(
    sealed: SealedOutcomeEvaluation,
    answers: tuple[TrainingExample, ...],
    publication: PublicationProof,
) -> dict[str, object]:
    """Build proper scores, baselines, paired deltas, and diagnostics."""
    base_probabilities, base_failures = _probabilities_with_penalties(sealed.base, answers)
    adapter_probabilities, adapter_failures = _probabilities_with_penalties(
        sealed.adapter,
        answers,
    )
    elo_probabilities = tuple(
        answer.target.distribution.probability_for(TEAM_OUTCOME) for answer in answers
    )
    neutral_probabilities = tuple(
        answer.case.prior.probability_for(TEAM_OUTCOME) for answer in answers
    )
    metrics = {
        "base": summarize_outcome_metrics(answers, base_probabilities),
        "adapter": summarize_outcome_metrics(answers, adapter_probabilities),
        "venue_adjusted_fivethirtyeight_elo": summarize_outcome_metrics(answers, elo_probabilities),
        "neutral_elo_prior": summarize_outcome_metrics(answers, neutral_probabilities),
    }
    subsets = difficulty_subsets(answers)
    return {
        "schema_version": 1,
        "kind": "forecastfm_outcome_development_scores",
        "warning": "Historical development diagnostics are not prospective evidence.",
        "publication": asdict(publication),
        "commitments": {
            "evaluation_manifest_sha256": file_sha256(MANIFEST_PATH),
            "generation_seal_sha256": file_sha256(SEAL_PATH),
            "answers_sha256": sealed.manifest.source_answers_sha256,
        },
        "execution": {
            "game_count": sealed.manifest.game_count,
            "expected_logical_call_count": sealed.manifest.expected_total_logical_calls,
            "base_failure_count": len(base_failures),
            "adapter_failure_count": len(adapter_failures),
            "base_failure_ids": list(base_failures),
            "adapter_failure_ids": list(adapter_failures),
            "failed_row_policy": "worst-case realized-outcome probability",
            "transport_retry_note": sealed.manifest.transport_retry_note,
            "base_weight_digest": None,
            "provider_call_receipt": None,
            "base_weight_limitation": (
                "Tinker does not expose a catalog base-weight digest; resumed base sessions "
                "cannot be proven identical."
            ),
            "attempt_attestation_limitation": (
                "Tinker supplies no signed call receipt; unpublished local attempts can be "
                "suppressed or replaced."
            ),
        },
        "metrics": {name: _metrics_dict(value) for name, value in metrics.items()},
        "paired_deltas": _paired_deltas(metrics),
        "answer_free_diagnostics": {
            "completed_rows_only": True,
            "base": _diagnostics(sealed.base),
            "adapter": _diagnostics(sealed.adapter),
        },
        "difficulty_subsets": _difficulty_reports(
            answers,
            subsets,
            {
                "base": base_probabilities,
                "adapter": adapter_probabilities,
                "venue_adjusted_fivethirtyeight_elo": elo_probabilities,
                "neutral_elo_prior": neutral_probabilities,
            },
        ),
    }


def load_original_answers(
    sealed: SealedOutcomeEvaluation,
) -> tuple[TrainingExample, ...]:
    """First answer-file access: verify hash, pairs, and frozen ordered IDs."""
    if file_sha256(ANSWERS_PATH) != sealed.manifest.source_answers_sha256:
        raise OutcomeEvaluationError("development answers differ from the frozen commitment")
    rows = read_jsonl(ANSWERS_PATH)
    if len(rows) != sealed.manifest.orientation_count:
        raise OutcomeEvaluationError("development answer count differs from the manifest")
    originals: list[TrainingExample] = []
    for index in range(0, len(rows), 2):
        original = rows[index]
        swapped = rows[index + 1]
        if side_swap_nba_example(original) != swapped:
            raise OutcomeEvaluationError("development answers are not exact side-swap pairs")
        expected_prompts = (
            build_outcome_forecast_record(original.case),
            build_outcome_forecast_record(swapped.case),
        )
        if sealed.prompt_pairs[index // 2] != expected_prompts:
            raise OutcomeEvaluationError("frozen prompts do not match their answer cases")
        originals.append(original)
    question_ids = tuple(example.case.question.question_id for example in originals)
    if question_ids != sealed.manifest.question_ids:
        raise OutcomeEvaluationError("development answer order differs from the manifest")
    return tuple(originals)


def _probabilities_with_penalties(
    records: tuple[OutcomeEvaluationRecord, ...],
    answers: tuple[TrainingExample, ...],
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    probabilities: list[float] = []
    failures: list[str] = []
    for record, answer in zip(records, answers, strict=True):
        if record.status == "completed":
            probability = record.symmetric_team_probability
            if probability is None:
                raise OutcomeEvaluationError("completed record is missing its probability")
            probabilities.append(probability)
            continue
        failures.append(record.question_id)
        probabilities.append(0.0 if answer.realized_outcome == TEAM_OUTCOME else 1.0)
    return tuple(probabilities), tuple(failures)


def _metrics_dict(metrics: OutcomeMetrics) -> dict[str, object]:
    return cast(dict[str, object], asdict(metrics))


def _paired_deltas(metrics: dict[str, OutcomeMetrics]) -> dict[str, object]:
    comparisons = {
        "adapter_minus_base": (metrics["base"], metrics["adapter"]),
        "adapter_minus_venue_adjusted_elo": (
            metrics["venue_adjusted_fivethirtyeight_elo"],
            metrics["adapter"],
        ),
        "base_minus_venue_adjusted_elo": (
            metrics["venue_adjusted_fivethirtyeight_elo"],
            metrics["base"],
        ),
    }
    result: dict[str, object] = {}
    for name, (baseline, candidate) in comparisons.items():
        result[name] = {
            "log_loss": asdict(
                paired_mean_delta(
                    baseline.per_game_log_losses,
                    candidate.per_game_log_losses,
                )
            ),
            "brier": asdict(
                paired_mean_delta(
                    baseline.per_game_brier_scores,
                    candidate.per_game_brier_scores,
                )
            ),
            "accuracy_delta": candidate.accuracy - baseline.accuracy,
            "ece_delta": (
                candidate.expected_calibration_error - baseline.expected_calibration_error
            ),
        }
    return result


def _diagnostics(records: tuple[OutcomeEvaluationRecord, ...]) -> dict[str, object]:
    completed = tuple(record for record in records if record.status == "completed")
    gaps: list[float] = []
    masses: list[float] = []
    for record in completed:
        if (
            record.pre_average_side_swap_gap is None
            or record.original is None
            or record.swapped is None
        ):
            raise OutcomeEvaluationError("completed diagnostics are incomplete")
        gaps.append(record.pre_average_side_swap_gap)
        masses.extend((record.original.valid_label_mass, record.swapped.valid_label_mass))
    return {
        "completed_count": len(completed),
        "failed_count": len(records) - len(completed),
        "mean_pre_average_side_swap_gap": _optional_mean(tuple(gaps)),
        "max_pre_average_side_swap_gap": None if not gaps else max(gaps),
        "mean_valid_label_mass": _optional_mean(tuple(masses)),
        "minimum_valid_label_mass": None if not masses else min(masses),
    }


def _difficulty_reports(
    answers: tuple[TrainingExample, ...],
    subsets: DifficultySubsets,
    probabilities: dict[str, tuple[float, ...]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    named_subsets = (
        ("hard", subsets.hard),
        ("medium", subsets.medium),
        ("easy", subsets.easy),
    )
    for name, indices in named_subsets:
        examples = tuple(answers[index] for index in indices)
        result[name] = {
            "game_count": len(indices),
            "metrics": {
                model: (
                    None
                    if not indices
                    else _metrics_dict(
                        summarize_outcome_metrics(
                            examples,
                            tuple(values[index] for index in indices),
                        )
                    )
                )
                for model, values in probabilities.items()
            },
        }
    return result


def _optional_mean(values: tuple[float, ...]) -> float | None:
    return None if not values else sum(values) / len(values)


if __name__ == "__main__":
    main()
