"""Download and prepare real NBA forecast data for training and evaluation."""

import json
from datetime import UTC, datetime
from pathlib import Path

from forecastfm.calibration import expected_calibration_error, reliability_bins
from forecastfm.models import ResolvedForecast, TrainingExample
from forecastfm.nba_data import (
    DATE_AMBIGUOUS_GAME_IDS,
    ELO_HOME_ADVANTAGE,
    ELO_TARGET_TOLERANCE,
    LICENSE_URL,
    SOURCE_NBA_GAME_COUNT,
    SOURCE_PAGE,
    SOURCE_SHA256,
    SOURCE_URL,
    TRAIN_LAST_SEASON,
    VALIDATION_LAST_SEASON,
    audit_nba_splits,
    download_nba_elo,
    file_sha256,
    load_nba_splits,
)
from forecastfm.prompting import MODEL_INPUT_SCHEMA_VERSION
from forecastfm.scoring import summarize_scores
from forecastfm.serialization import write_jsonl
from forecastfm.tinker_data import write_forecast_jsonl, write_sft_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "nbaallelo.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = OUTPUT_DIRECTORY / "nba_elo_train.jsonl"
VALIDATION_ANSWERS_PATH = OUTPUT_DIRECTORY / "nba_elo_validation_answers.jsonl"
TEST_ANSWERS_PATH = OUTPUT_DIRECTORY / "nba_elo_test_answers.jsonl"
VALIDATION_PROMPTS_PATH = OUTPUT_DIRECTORY / "nba_elo_validation_prompts.jsonl"
TEST_PROMPTS_PATH = OUTPUT_DIRECTORY / "nba_elo_test_prompts.jsonl"
SFT_PATH = OUTPUT_DIRECTORY / "nba_elo_train_sft.jsonl"
MANIFEST_PATH = OUTPUT_DIRECTORY / "manifest.json"
OBSOLETE_PATHS = (
    OUTPUT_DIRECTORY / "nba_elo_validation.jsonl",
    OUTPUT_DIRECTORY / "nba_elo_test.jsonl",
)


def main() -> None:
    """Build chronological splits and a provenance manifest from the pinned source."""
    download_nba_elo(RAW_PATH)
    splits = load_nba_splits(RAW_PATH)
    expected_games = SOURCE_NBA_GAME_COUNT - len(DATE_AMBIGUOUS_GAME_IDS)
    if splits.source_game_count != expected_games:
        raise RuntimeError("Pinned NBA source did not produce the expected number of games")
    venue_counts = audit_nba_splits(splits)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for obsolete_path in OBSOLETE_PATHS:
        obsolete_path.unlink(missing_ok=True)

    write_jsonl(splits.train, TRAIN_PATH)
    write_jsonl(splits.validation, VALIDATION_ANSWERS_PATH)
    write_jsonl(splits.test, TEST_ANSWERS_PATH)
    write_forecast_jsonl((example.case for example in splits.validation), VALIDATION_PROMPTS_PATH)
    write_forecast_jsonl((example.case for example in splits.test), TEST_PROMPTS_PATH)
    write_sft_jsonl(splits.train, SFT_PATH)

    manifest = {
        "built_at": datetime.now(tz=UTC).isoformat(),
        "model_input_schema_version": MODEL_INPUT_SCHEMA_VERSION,
        "source": {
            "name": "FiveThirtyEight historical NBA Elo",
            "page": SOURCE_PAGE,
            "url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "license": "CC BY 4.0",
            "license_url": LICENSE_URL,
            "game_information_credit": "Basketball-Reference.com",
        },
        "transformation": {
            "filters": [
                "lg_id == NBA",
                "one deterministic hash-selected perspective per game",
                "one row per unique model-facing prompt across all splits",
            ],
            "postgame_fields_excluded_from_prompts": [
                "elo_n",
                "game_result",
                "opp_elo_n",
                "opp_pts",
                "pts",
                "win_equiv",
            ],
            "timestamp_note": (
                "Source dates are represented as 00:00 UTC date-granularity proxies; "
                "they are not publication or tipoff timestamps."
            ),
            "target": "FiveThirtyEight retrospective pregame Elo forecast",
            "prior": "Neutral-court probability computed from pregame Elo ratings",
            "elo_oracle": {
                "home_advantage_points": ELO_HOME_ADVANTAGE,
                "maximum_allowed_target_error": ELO_TARGET_TOLERANCE,
                "all_eligible_rows_passed": True,
            },
            "task_warning": (
                "The target is a deterministic venue adjustment of the Elo prior. "
                "This is formula distillation, not a broad forecasting benchmark."
            ),
        },
        "anti_cheating": {
            "audit_passed": True,
            "source_games": SOURCE_NBA_GAME_COUNT,
            "date_ambiguous_games_removed": sorted(DATE_AMBIGUOUS_GAME_IDS),
            "eligible_games": splits.source_game_count,
            "unique_prompt_games": len(splits.train) + len(splits.validation) + len(splits.test),
            "duplicate_prompt_games_removed": splits.duplicate_prompt_count,
            "selected_perspective_venues": venue_counts,
            "model_sees": ["anonymous question", "neutral Elo prior", "venue"],
            "model_does_not_see": [
                "absolute date",
                "data source",
                "game identity",
                "realized outcome",
                "team identity",
                "teacher target during evaluation",
            ],
            "evaluation_warning": (
                "Historical outcome scores cannot rule out pretraining contamination. "
                "Only a frozen prospective ledger can support a real forecasting claim."
            ),
        },
        "splits": {
            "train": {"through_season": TRAIN_LAST_SEASON, "rows": len(splits.train)},
            "validation": {
                "first_season": TRAIN_LAST_SEASON + 1,
                "through_season": VALIDATION_LAST_SEASON,
                "rows": len(splits.validation),
            },
            "test": {
                "first_season": VALIDATION_LAST_SEASON + 1,
                "rows": len(splits.test),
            },
        },
        "five_thirty_eight_baseline": {
            "train": _metrics(splits.train),
            "validation": _metrics(splits.validation),
            "test": _metrics(splits.test),
        },
        "outputs": {
            path.name: file_sha256(path)
            for path in (
                TRAIN_PATH,
                VALIDATION_ANSWERS_PATH,
                TEST_ANSWERS_PATH,
                VALIDATION_PROMPTS_PATH,
                TEST_PROMPTS_PATH,
                SFT_PATH,
            )
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"Built {len(splits.train):,} train, {len(splits.validation):,} validation, "
        f"and {len(splits.test):,} test games."
    )
    print(f"Manifest: {MANIFEST_PATH}")


def _metrics(examples: tuple[TrainingExample, ...]) -> dict[str, int | float]:
    forecasts = tuple(
        ResolvedForecast(
            question_id=example.case.question.question_id,
            forecast_at=example.case.question.forecast_at,
            distribution=example.target.distribution,
            realized_outcome=_require_outcome(example),
        )
        for example in examples
    )
    scores = summarize_scores(forecasts)
    calibration = reliability_bins(forecasts, positive_outcome="team_wins")
    return {
        "count": scores.count,
        "mean_brier": scores.mean_brier,
        "mean_log_loss": scores.mean_log_loss,
        "accuracy": scores.accuracy,
        "expected_calibration_error": expected_calibration_error(calibration),
    }


def _require_outcome(example: TrainingExample) -> str:
    if example.realized_outcome is None:
        raise ValueError("NBA examples must have a realized outcome")
    return example.realized_outcome


if __name__ == "__main__":
    main()
