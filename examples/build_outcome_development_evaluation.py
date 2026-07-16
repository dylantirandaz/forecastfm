"""Freeze the full target-free outcome development evaluation."""

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from tinker_cookbook.tokenizer_utils import get_tokenizer

from forecastfm.integrity import canonical_sha256, file_sha256
from forecastfm.json_utils import (
    parse_json_object,
    require_object,
    require_string,
    required_field,
)
from forecastfm.outcome import TokenCodec, require_label_token_ids
from forecastfm.outcome_config import MAX_STEPS, OUTCOME_RENDERER_NAME
from forecastfm.outcome_evaluation import (
    OutcomeEvaluationManifest,
    write_attempt_marker,
    write_manifest,
    write_text_exclusively,
)
from forecastfm.outcome_run_lock import verify_outcome_training_lock
from forecastfm.publication import require_paths_at_head, require_published_head
from forecastfm.run_config import BASE_MODEL, require_tokenizer_snapshot
from forecastfm.run_lock import verify_experiment_lock
from forecastfm.scoring import MINIMUM_LOG_PROBABILITY
from forecastfm.tinker_data import (
    pair_outcome_forecast_records,
    read_outcome_forecast_jsonl,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "outcome_v1"
SOURCE_MANIFEST_PATH = SOURCE_DIRECTORY / "manifest.json"
SOURCE_PROMPTS_PATH = SOURCE_DIRECTORY / "nba_development_prompts.jsonl"
ANSWERS_FILENAME = "nba_development_answers.jsonl"
RUN_DIRECTORY = PROJECT_ROOT / "prospective" / "outcome_v1" / f"steps_{MAX_STEPS}"
TRAINING_LOCK_PATH = RUN_DIRECTORY / "training_lock.json"
EXPERIMENT_PATH = RUN_DIRECTORY / "experiment.json"
OUTPUT_DIRECTORY = PROJECT_ROOT / "evaluation" / "outcome_v1" / f"steps_{MAX_STEPS}"
PROMPTS_PATH = OUTPUT_DIRECTORY / "prompts.jsonl"
MANIFEST_PATH = OUTPUT_DIRECTORY / "manifest.json"
RAW_DIRECTORY = OUTPUT_DIRECTORY / "raw"
ATTEMPT_PATH = RAW_DIRECTORY / "attempt.json"
EXPECTED_REMOTE_URL = "https://github.com/dylantirandaz/forecastfm.git"
EXPECTED_GAME_COUNT = 2_612
TRANSPORT_RETRY_NOTE = (
    "Tinker 0.22.7 may retransmit one logical request with the same session and sequence ID."
)
PROTOCOL_PATHS = (
    PROJECT_ROOT / "pyproject.toml",
    PROJECT_ROOT / "uv.lock",
    PROJECT_ROOT / "src" / "forecastfm" / "__init__.py",
    PROJECT_ROOT / "src" / "forecastfm" / "calibration.py",
    PROJECT_ROOT / "src" / "forecastfm" / "integrity.py",
    PROJECT_ROOT / "src" / "forecastfm" / "json_utils.py",
    PROJECT_ROOT / "src" / "forecastfm" / "models.py",
    PROJECT_ROOT / "src" / "forecastfm" / "nba_data.py",
    PROJECT_ROOT / "src" / "forecastfm" / "outcome.py",
    PROJECT_ROOT / "src" / "forecastfm" / "outcome_config.py",
    PROJECT_ROOT / "src" / "forecastfm" / "outcome_evaluation.py",
    PROJECT_ROOT / "src" / "forecastfm" / "outcome_metrics.py",
    PROJECT_ROOT / "src" / "forecastfm" / "outcome_run_lock.py",
    PROJECT_ROOT / "src" / "forecastfm" / "prompting.py",
    PROJECT_ROOT / "src" / "forecastfm" / "publication.py",
    PROJECT_ROOT / "src" / "forecastfm" / "run_config.py",
    PROJECT_ROOT / "src" / "forecastfm" / "run_lock.py",
    PROJECT_ROOT / "src" / "forecastfm" / "scoring.py",
    PROJECT_ROOT / "src" / "forecastfm" / "serialization.py",
    PROJECT_ROOT / "src" / "forecastfm" / "tinker_data.py",
    PROJECT_ROOT / "src" / "forecastfm" / "tinker_screening.py",
    PROJECT_ROOT / "src" / "forecastfm" / "updating.py",
    PROJECT_ROOT / "examples" / "tinker_outcome_inference.py",
    Path(__file__).resolve(),
    PROJECT_ROOT / "examples" / "run_tinker_outcome_development.py",
    PROJECT_ROOT / "examples" / "score_outcome_development.py",
    PROJECT_ROOT / "examples" / "smoke_tinker_outcome_candidates.py",
)
SCORING_POLICY: dict[str, object] = {
    "unit": "one original game after side-swap averaging",
    "probability_formula": "(p_original + 1 - p_swapped) / 2",
    "primary": "mean_log_loss",
    "secondary": ["mean_brier", "accuracy", "ece_10"],
    "baselines": ["venue_adjusted_fivethirtyeight_elo", "neutral_elo_prior"],
    "answer_free_diagnostics": ["pre_average_side_swap_gap", "valid_label_mass"],
    "difficulty_subsets": {
        "hard": "max venue-adjusted Elo probability < 0.60",
        "medium": "0.60 <= max venue-adjusted Elo probability < 0.75",
        "easy": "max venue-adjusted Elo probability >= 0.75",
    },
    "paired_uncertainty": "normal_approximation_95_percent_ci",
    "failed_call_policy": {
        "generation": "one arm attempt; no answer-aware retry or row removal",
        "team_probability": "0 if team_wins; 1 if opponent_wins",
        "log_loss_probability_floor": MINIMUM_LOG_PROBABILITY,
    },
    "residual_limitations": [
        "Tinker does not expose a catalog base-weight digest; resumed base sessions may differ",
        "Tinker supplies no signed call receipt; unpublished local attempts can be suppressed",
        "historical development results cannot establish prospective forecasting skill",
    ],
    "historical_warning": "development results are contamination-prone, not prospective",
}


def main() -> None:
    """Publish-check the protocol, then copy and freeze target-free prompts."""
    proof = require_published_head(PROJECT_ROOT, EXPECTED_REMOTE_URL)
    require_paths_at_head(PROJECT_ROOT, proof.commit, PROTOCOL_PATHS)
    training_lock = verify_outcome_training_lock(PROJECT_ROOT, TRAINING_LOCK_PATH)
    experiment = verify_experiment_lock(TRAINING_LOCK_PATH, EXPERIMENT_PATH)
    source_manifest = _verify_source_manifest(training_lock)
    prompt_hash, answer_hash = source_hashes(source_manifest)
    if file_sha256(SOURCE_PROMPTS_PATH) != prompt_hash:
        raise RuntimeError("development prompts differ from the committed source manifest")
    pairs = pair_outcome_forecast_records(read_outcome_forecast_jsonl(SOURCE_PROMPTS_PATH))
    if len(pairs) != EXPECTED_GAME_COUNT:
        raise RuntimeError("development prompt pair count differs from the frozen protocol")
    question_ids = tuple(original["question_id"] for original, _swapped in pairs)
    token_ids = require_label_token_ids(
        cast(TokenCodec, get_tokenizer(str(require_tokenizer_snapshot())))
    )
    adapter_path = require_string(
        required_field(experiment, "adapter_sampler_path"),
        "adapter_sampler_path",
    )
    write_text_exclusively(
        PROMPTS_PATH,
        SOURCE_PROMPTS_PATH.read_text(encoding="utf-8"),
    )
    frozen_hash = file_sha256(PROMPTS_PATH)
    if frozen_hash != prompt_hash:
        raise RuntimeError("frozen prompt copy differs from the source")
    created_at = datetime.now(UTC)
    manifest = OutcomeEvaluationManifest(
        created_at=created_at.isoformat(),
        protocol_revision=proof.commit,
        source_manifest_sha256=file_sha256(SOURCE_MANIFEST_PATH),
        source_prompts_sha256=prompt_hash,
        source_answers_sha256=answer_hash,
        frozen_prompts_sha256=frozen_hash,
        training_lock_sha256=file_sha256(TRAINING_LOCK_PATH),
        experiment_sha256=file_sha256(EXPERIMENT_PATH),
        base_model=BASE_MODEL,
        adapter_sampler_path=adapter_path,
        renderer_name=OUTCOME_RENDERER_NAME,
        team_token_id=token_ids[0],
        opponent_token_id=token_ids[1],
        game_count=len(pairs),
        orientation_count=len(pairs) * 2,
        logical_calls_per_game_per_arm=4,
        expected_total_logical_calls=len(pairs) * 8,
        max_active_arms=1,
        application_retries=0,
        transport_retry_note=TRANSPORT_RETRY_NOTE,
        question_ids=question_ids,
        question_ids_sha256=canonical_sha256(list(question_ids)),
        scoring_policy=SCORING_POLICY,
    )
    write_manifest(MANIFEST_PATH, manifest)
    write_attempt_marker(ATTEMPT_PATH, MANIFEST_PATH, PROMPTS_PATH, created_at)
    print(f"Frozen {len(pairs)} target-free game pairs at {PROMPTS_PATH}.")
    print(f"Expected logical calls across both arms: {manifest.expected_total_logical_calls}.")


def _verify_source_manifest(training_lock: dict[str, object]) -> dict[str, object]:
    locked_data = require_object(required_field(training_lock, "data"), "data")
    locked_hash = require_string(
        required_field(locked_data, "manifest_sha256"),
        "manifest_sha256",
    )
    if file_sha256(SOURCE_MANIFEST_PATH) != locked_hash:
        raise RuntimeError("outcome source manifest differs from the training lock")
    return parse_json_object(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))


def source_hashes(source_manifest: dict[str, object]) -> tuple[str, str]:
    """Read the committed development prompt and answer digests."""
    outputs = require_object(required_field(source_manifest, "outputs"), "outputs")
    return (
        require_string(required_field(outputs, SOURCE_PROMPTS_PATH.name), SOURCE_PROMPTS_PATH.name),
        require_string(required_field(outputs, ANSWERS_FILENAME), ANSWERS_FILENAME),
    )


if __name__ == "__main__":
    main()
