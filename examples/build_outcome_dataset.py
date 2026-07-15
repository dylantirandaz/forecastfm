"""Build the chronological realized-winner dataset for ForecastFM outcome v1."""

import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from forecastfm.json_utils import (
    parse_json_object,
    require_object,
    require_string,
    required_field,
)
from forecastfm.models import TrainingExample
from forecastfm.nba_data import file_sha256, side_swap_nba_example
from forecastfm.outcome import (
    OPPONENT_LABEL,
    OPPONENT_OUTCOME,
    OUTCOME_INPUT_SCHEMA_VERSION,
    TEAM_LABEL,
    TEAM_OUTCOME,
)
from forecastfm.prompting import render_case
from forecastfm.serialization import read_jsonl, write_jsonl
from forecastfm.tinker_data import (
    write_outcome_forecast_jsonl,
    write_outcome_training_jsonl,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.json"
SOURCE_TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "nba_elo_train.jsonl"

OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "outcome_v1"
TRAIN_PATH = OUTPUT_DIRECTORY / "nba_train_outcome.jsonl"
DEVELOPMENT_ANSWERS_PATH = OUTPUT_DIRECTORY / "nba_development_answers.jsonl"
DEVELOPMENT_PROMPTS_PATH = OUTPUT_DIRECTORY / "nba_development_prompts.jsonl"
MANIFEST_PATH = OUTPUT_DIRECTORY / "manifest.json"

DEVELOPMENT_START = datetime(2007, 7, 1, tzinfo=UTC)
OUTCOME_EXCLUDED_CASES = {
    "nba-3d0aec4ec13259d1": "completion of a protested game that began on an earlier date",
}


def main() -> None:
    """Create a pre-2010 fit/development split without opening later splits."""
    _verify_source_training_file()
    source_examples = read_jsonl(SOURCE_TRAIN_PATH)
    examples = tuple(
        example
        for example in source_examples
        if example.case.question.question_id not in OUTCOME_EXCLUDED_CASES
    )
    excluded_count = len(source_examples) - len(examples)
    if excluded_count != len(OUTCOME_EXCLUDED_CASES):
        raise RuntimeError("outcome case exclusions do not match the pinned source")
    fit, unfiltered_development = _chronological_split(examples)
    development, removed_development_ids = remove_fit_prompt_overlaps(
        fit,
        unfiltered_development,
    )
    fit_prompt_audit = _augmented_prompt_audit(fit)
    development_prompt_audit = _augmented_prompt_audit(development)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_outcome_training_jsonl(_side_swap_pairs(fit), TRAIN_PATH)
    write_jsonl(_side_swap_pairs(development), DEVELOPMENT_ANSWERS_PATH)
    write_outcome_forecast_jsonl(
        (example.case for example in _side_swap_pairs(development)),
        DEVELOPMENT_PROMPTS_PATH,
    )

    original_label_counts = Counter(_require_outcome(example) for example in fit)
    manifest = {
        "schema_version": 1,
        "outcome_input_schema_version": OUTCOME_INPUT_SCHEMA_VERSION,
        "built_at": datetime.now(tz=UTC).isoformat(),
        "source": {
            "manifest_path": str(SOURCE_MANIFEST_PATH.relative_to(PROJECT_ROOT)),
            "manifest_sha256": file_sha256(SOURCE_MANIFEST_PATH),
            "training_path": str(SOURCE_TRAIN_PATH.relative_to(PROJECT_ROOT)),
            "training_sha256": file_sha256(SOURCE_TRAIN_PATH),
        },
        "split": {
            "method": "chronological cutoff inside the legacy pre-2010 training split",
            "source_rows": len(source_examples),
            "excluded_source_rows": excluded_count,
            "excluded_cases": OUTCOME_EXCLUDED_CASES,
            "development_start": DEVELOPMENT_START.isoformat(),
            "fit_original_rows": len(fit),
            "fit_rows_after_side_swap": len(fit) * 2,
            "development_original_rows_before_overlap_filter": len(unfiltered_development),
            "development_original_rows": len(development),
            "development_rows_after_side_swap": len(development) * 2,
            "development_pair_orbits_removed_for_fit_overlap": len(removed_development_ids),
            "removed_development_question_ids": list(removed_development_ids),
            "later_answer_files_read_by_outcome_builder": False,
            "legacy_validation_or_test_used_for_outcome_training": False,
            "historical_holdout_note": (
                "Legacy validation/test answers and aggregate metrics exist locally, but outcome "
                "v1 does not read or select on them. A future prospective cohort is required for "
                "a truly unseen claim."
            ),
        },
        "objective": {
            "loss": "cross-entropy on the realized winner label",
            "labels": {
                TEAM_OUTCOME: TEAM_LABEL,
                OPPONENT_OUTCOME: OPPONENT_LABEL,
            },
            "label_note": (
                "OTHER means opponent wins. Literal OPPONENT is not one token under the "
                "pinned Qwen3.5 tokenizer."
            ),
            "teacher_forecast_role": "stored baseline metadata; never an outcome training label",
            "original_fit_label_counts": dict(sorted(original_label_counts.items())),
            "augmented_fit_label_counts": {
                TEAM_OUTCOME: len(fit),
                OPPONENT_OUTCOME: len(fit),
            },
        },
        "side_swap": {
            "enabled": True,
            "each_original_has_one_swap": True,
            "adjacent_original_swap_pairs": True,
            "fit_prompt_audit": fit_prompt_audit,
            "development_prompt_audit": development_prompt_audit,
            "swapped_fields": ["prior", "venue", "teacher probability", "realized winner"],
        },
        "anti_cheating": {
            "realized_winner_location": "label field only",
            "messages_are_target_free": True,
            "teacher_probability_in_messages": False,
            "postgame_fields_in_messages": False,
            "augmented_fit_development_prompt_overlap": 0,
            "timestamp_note": (
                "Source game dates are 00:00 UTC date-granularity proxies, not publication "
                "or tipoff timestamps."
            ),
            "model_sees": ["anonymous question", "neutral Elo prior", "venue"],
            "warning": (
                "The inputs remain narrow. Outcome training can learn Elo and venue corrections, "
                "but richer forecasting requires more point-in-time pregame evidence."
            ),
        },
        "outputs": {
            path.name: file_sha256(path)
            for path in (TRAIN_PATH, DEVELOPMENT_ANSWERS_PATH, DEVELOPMENT_PROMPTS_PATH)
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"Built {len(fit) * 2:,} fit rows and {len(development) * 2:,} "
        "development rows after side swaps."
    )
    print(f"Manifest: {MANIFEST_PATH}")


def _verify_source_training_file() -> None:
    manifest = parse_json_object(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    outputs = require_object(required_field(manifest, "outputs"), "outputs")
    expected_hash = require_string(
        required_field(outputs, SOURCE_TRAIN_PATH.name),
        SOURCE_TRAIN_PATH.name,
    )
    if file_sha256(SOURCE_TRAIN_PATH) != expected_hash:
        raise RuntimeError("legacy training data differs from its pinned manifest")


def _chronological_split(
    examples: tuple[TrainingExample, ...],
) -> tuple[tuple[TrainingExample, ...], tuple[TrainingExample, ...]]:
    fit = tuple(
        example for example in examples if example.case.question.forecast_at < DEVELOPMENT_START
    )
    development = tuple(
        example for example in examples if example.case.question.forecast_at >= DEVELOPMENT_START
    )
    if not fit or not development:
        raise RuntimeError("outcome fit and development splits must both be non-empty")
    if max(example.case.question.forecast_at for example in fit) >= min(
        example.case.question.forecast_at for example in development
    ):
        raise RuntimeError("outcome development split is not strictly chronological")
    return fit, development


def _side_swap_pairs(examples: Iterable[TrainingExample]) -> Iterable[TrainingExample]:
    for example in examples:
        yield example
        yield side_swap_nba_example(example)


def remove_fit_prompt_overlaps(
    fit: tuple[TrainingExample, ...],
    development: tuple[TrainingExample, ...],
) -> tuple[tuple[TrainingExample, ...], tuple[str, ...]]:
    """Drop development orbits matching either orientation of a fit prompt."""
    fit_hashes = {
        _prompt_hash(example)
        for original in fit
        for example in (original, side_swap_nba_example(original))
    }
    kept: list[TrainingExample] = []
    removed_ids: list[str] = []
    for original in development:
        pair = (original, side_swap_nba_example(original))
        if any(_prompt_hash(example) in fit_hashes for example in pair):
            removed_ids.append(original.case.question.question_id)
        else:
            kept.append(original)
    kept_hashes = {
        _prompt_hash(example)
        for original in kept
        for example in (original, side_swap_nba_example(original))
    }
    if fit_hashes & kept_hashes:
        raise RuntimeError("outcome fit and development prompts overlap after side swaps")
    return tuple(kept), tuple(sorted(removed_ids))


def _prompt_hash(example: TrainingExample) -> str:
    return sha256(render_case(example.case).encode()).hexdigest()


def _augmented_prompt_audit(examples: tuple[TrainingExample, ...]) -> dict[str, int]:
    hashes = [
        _prompt_hash(example)
        for original in examples
        for example in (original, side_swap_nba_example(original))
    ]
    return {
        "rows": len(hashes),
        "unique_prompts": len(set(hashes)),
        "duplicate_prompt_rows": len(hashes) - len(set(hashes)),
    }


def _require_outcome(example: TrainingExample) -> str:
    if example.realized_outcome not in {TEAM_OUTCOME, OPPONENT_OUTCOME}:
        raise RuntimeError("NBA training row is missing a realized winner")
    return example.realized_outcome


if __name__ == "__main__":
    main()
