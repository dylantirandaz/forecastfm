"""Build the real-data outcome-v2 artifact and its historical Elo gate."""

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from pathlib import Path

from forecastfm.elo_residual import (
    EloResidualFitConfig,
    EloResidualModel,
    EloResidualRow,
    fit_elo_residual,
)
from forecastfm.integrity import canonical_sha256
from forecastfm.models import TrainingExample
from forecastfm.nba_data import (
    DATE_AMBIGUOUS_GAME_IDS,
    LICENSE_URL,
    SOURCE_NBA_GAME_COUNT,
    SOURCE_PAGE,
    SOURCE_SHA256,
    SOURCE_URL,
    download_nba_elo,
    file_sha256,
)
from forecastfm.nba_v2 import (
    NBA_V2_DATA_LIMITATIONS,
    NBA_V2_FEATURE_NAMES,
    NbaV2Example,
    load_nba_v2_examples,
    side_swap_nba_v2_example,
)
from forecastfm.outcome import OUTCOME_INPUT_SCHEMA_VERSION, TEAM_OUTCOME
from forecastfm.outcome_v2_metrics import (
    FAILURE_REALIZED_PROBABILITY,
    BinaryForecast,
    BinaryProperScores,
    DatedBinaryCohortMember,
    MultiSeasonEvaluation,
    SeasonEvaluation,
    evaluate_multi_season,
)
from forecastfm.prompting import render_case
from forecastfm.serialization import write_jsonl
from forecastfm.tinker_data import (
    write_outcome_forecast_jsonl,
    write_outcome_training_jsonl,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "nbaallelo.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "outcome_v2"
TRAIN_PATH = OUTPUT_DIRECTORY / "nba_train_outcome.jsonl"
VALIDATION_ANSWERS_PATH = OUTPUT_DIRECTORY / "nba_validation_answers.jsonl"
VALIDATION_PROMPTS_PATH = OUTPUT_DIRECTORY / "nba_validation_prompts.jsonl"
TEST_ANSWERS_PATH = OUTPUT_DIRECTORY / "nba_historical_test_answers.jsonl"
TEST_PROMPTS_PATH = OUTPUT_DIRECTORY / "nba_historical_test_prompts.jsonl"
MANIFEST_PATH = OUTPUT_DIRECTORY / "manifest.json"

TRAIN_LAST_SEASON = 2009
VALIDATION_SEASONS = (2010, 2011, 2012)
HISTORICAL_TEST_SEASONS = (2013, 2014, 2015)
EXPECTED_TRAIN_COHORT = (
    51_359,
    "f94421c2fc21814b3d3219db92b90dd4438ef1829f72054c41e45029eef9cb71",
)
EXPECTED_VALIDATION_COHORT = (
    3_697,
    "09e82390fe4488a07946c8581afc88cbeac37b65d4f27d8c46e4ccd86184d0e9",
)
EXPECTED_HISTORICAL_TEST_COHORT = (
    3_944,
    "564126f5a1c95402eb580dd3f4d24f2c8f84f9ec20dd5ff607ac1d8a1769c543",
)

RICH_FIT_CONFIG = EloResidualFitConfig(steps=1_000, learning_rate=0.1, l2_penalty=0.01)
RECALIBRATION_FIT_CONFIG = EloResidualFitConfig(
    steps=1_000,
    learning_rate=0.1,
    l2_penalty=0.0,
)


@dataclass(frozen=True, slots=True)
class _Splits:
    train: tuple[NbaV2Example, ...]
    validation: tuple[NbaV2Example, ...]
    historical_test: tuple[NbaV2Example, ...]


@dataclass(frozen=True, slots=True)
class _FittedModels:
    rich: EloResidualModel
    recalibration: EloResidualModel
    feature_scales: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _Build:
    source_count: int
    kept_count: int
    removed_by_season: dict[int, int]
    splits: _Splits
    development_models: _FittedModels
    final_models: _FittedModels
    development_evaluations: tuple[
        MultiSeasonEvaluation,
        MultiSeasonEvaluation,
        MultiSeasonEvaluation,
    ]
    final_evaluations: tuple[
        MultiSeasonEvaluation,
        MultiSeasonEvaluation,
        MultiSeasonEvaluation,
    ]


def main() -> None:
    """Build screened records, simple baselines, and strict historical diagnostics."""
    download_nba_elo(RAW_PATH)
    source_examples = load_nba_v2_examples(RAW_PATH)
    expected_count = SOURCE_NBA_GAME_COUNT - len(DATE_AMBIGUOUS_GAME_IDS)
    if len(source_examples) != expected_count:
        raise RuntimeError("pinned NBA source produced an unexpected number of v2 games")

    examples, removed_by_season = deduplicate_prompt_orbits(source_examples)
    splits = split_examples(examples)
    _verify_split_contract(splits)
    development_models = _fit_models(splits.train)
    final_models = _fit_models((*splits.train, *splits.validation))

    development_evaluations = _evaluate_models(
        splits.validation,
        VALIDATION_SEASONS,
        development_models,
    )
    final_evaluations = _evaluate_models(
        splits.historical_test,
        HISTORICAL_TEST_SEASONS,
        final_models,
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    write_outcome_training_jsonl(_training_examples(_side_swap_pairs(splits.train)), TRAIN_PATH)
    _write_evaluation_records(
        splits.validation,
        VALIDATION_ANSWERS_PATH,
        VALIDATION_PROMPTS_PATH,
    )
    _write_evaluation_records(
        splits.historical_test,
        TEST_ANSWERS_PATH,
        TEST_PROMPTS_PATH,
    )

    build = _Build(
        source_count=len(source_examples),
        kept_count=len(examples),
        removed_by_season=removed_by_season,
        splits=splits,
        development_models=development_models,
        final_models=final_models,
        development_evaluations=development_evaluations,
        final_evaluations=final_evaluations,
    )
    manifest = _manifest(build)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rich_raw, _, rich_recalibrated = final_evaluations
    print(
        f"Built outcome v2 with {len(splits.train):,} train, "
        f"{len(splits.validation):,} validation, and "
        f"{len(splits.historical_test):,} historical-test games."
    )
    print(
        "Historical multi-season gate: "
        f"raw Elo={rich_raw.passes}, recalibrated Elo={rich_recalibrated.passes}."
    )
    print(f"Manifest: {MANIFEST_PATH}")


def _fit_models(
    examples: Sequence[NbaV2Example],
) -> _FittedModels:
    feature_scales = rms_feature_scales(examples)
    recalibration_rows = tuple(
        _model_row(example, feature_scales, recalibration_only=True) for example in examples
    )
    rich_rows = tuple(
        _model_row(example, feature_scales, recalibration_only=False) for example in examples
    )
    recalibration = fit_elo_residual(
        recalibration_rows,
        (NBA_V2_FEATURE_NAMES[0],),
        RECALIBRATION_FIT_CONFIG,
    )
    rich = fit_elo_residual(rich_rows, NBA_V2_FEATURE_NAMES, RICH_FIT_CONFIG)
    return _FittedModels(
        rich=rich,
        recalibration=recalibration,
        feature_scales=feature_scales,
    )


def _model_row(
    example: NbaV2Example,
    feature_scales: tuple[float, ...],
    *,
    recalibration_only: bool,
) -> EloResidualRow:
    scaled_features = scale_features(example, feature_scales)
    features = (scaled_features[0],) if recalibration_only else scaled_features
    return EloResidualRow(
        question_id=example.training_example.case.question.question_id,
        elo_probability=example.features.venue_adjusted_elo_probabilities[0],
        features=features,
        outcome=int(example.training_example.realized_outcome == TEAM_OUTCOME),
    )


def scale_features(
    example: NbaV2Example,
    feature_scales: tuple[float, ...],
) -> tuple[float, ...]:
    """Scale one oriented vector without centering or breaking side symmetry."""
    return tuple(
        value / scale for value, scale in zip(example.features.vector, feature_scales, strict=True)
    )


def rms_feature_scales(examples: Sequence[NbaV2Example]) -> tuple[float, ...]:
    """Calculate nonzero feature RMS values from the fitting cohort only."""
    if not examples:
        raise RuntimeError("cannot scale an empty NBA feature cohort")
    sums_of_squares = [0.0] * len(NBA_V2_FEATURE_NAMES)
    for example in examples:
        for index, value in enumerate(example.features.vector):
            sums_of_squares[index] += value * value
    scales = tuple(sqrt(total / len(examples)) for total in sums_of_squares)
    if any(scale == 0.0 for scale in scales):
        raise RuntimeError("NBA feature scaling found a constant feature")
    return scales


def _evaluate_models(
    examples: Sequence[NbaV2Example],
    seasons: tuple[int, ...],
    models: _FittedModels,
) -> tuple[MultiSeasonEvaluation, MultiSeasonEvaluation, MultiSeasonEvaluation]:
    raw_cohort: list[DatedBinaryCohortMember] = []
    recalibrated_cohort: list[DatedBinaryCohortMember] = []
    rich_forecasts: list[BinaryForecast] = []
    recalibrated_forecasts: list[BinaryForecast] = []
    for example in examples:
        elo_probability = example.features.venue_adjusted_elo_probabilities[0]
        scaled_features = scale_features(example, models.feature_scales)
        rich_probability = models.rich.predict_probability(elo_probability, scaled_features)
        recalibrated_probability = models.recalibration.predict_probability(
            elo_probability,
            (scaled_features[0],),
        )
        raw_cohort.append(_cohort_member(example, elo_probability))
        recalibrated_cohort.append(_cohort_member(example, recalibrated_probability))
        rich_forecasts.append(_forecast(example, rich_probability))
        recalibrated_forecasts.append(_forecast(example, recalibrated_probability))
    return (
        evaluate_multi_season(rich_forecasts, raw_cohort, seasons),
        evaluate_multi_season(recalibrated_forecasts, raw_cohort, seasons),
        evaluate_multi_season(rich_forecasts, recalibrated_cohort, seasons),
    )


def _forecast(
    example: NbaV2Example,
    model_probability: float,
) -> BinaryForecast:
    return BinaryForecast(
        question_id=example.training_example.case.question.question_id,
        team_probability=model_probability,
    )


def _cohort_member(
    example: NbaV2Example,
    baseline_probability: float,
) -> DatedBinaryCohortMember:
    training = example.training_example
    game_date = training.case.question.forecast_at.date()
    return DatedBinaryCohortMember(
        question_id=training.case.question.question_id,
        season=example.season,
        game_date=game_date,
        realized_team_win=training.realized_outcome == TEAM_OUTCOME,
        baseline_team_probability=baseline_probability,
    )


def deduplicate_prompt_orbits(
    examples: Sequence[NbaV2Example],
) -> tuple[tuple[NbaV2Example, ...], dict[int, int]]:
    """Keep the first chronological occurrence of each side-swap prompt orbit."""
    seen_orbits: set[tuple[str, str]] = set()
    kept: list[NbaV2Example] = []
    removed: Counter[int] = Counter()
    for example in examples:
        swapped = side_swap_nba_v2_example(example)
        hashes = sorted(
            (
                _prompt_hash(example.training_example),
                _prompt_hash(swapped.training_example),
            )
        )
        orbit = (hashes[0], hashes[1])
        if orbit in seen_orbits:
            removed[example.season] += 1
            continue
        seen_orbits.add(orbit)
        kept.append(example)
    return tuple(kept), dict(sorted(removed.items()))


def split_examples(examples: Sequence[NbaV2Example]) -> _Splits:
    """Apply the predeclared source-season boundaries."""
    train = tuple(example for example in examples if example.season <= TRAIN_LAST_SEASON)
    validation = tuple(example for example in examples if example.season in VALIDATION_SEASONS)
    historical_test = tuple(
        example for example in examples if example.season in HISTORICAL_TEST_SEASONS
    )
    if not train or not validation or not historical_test:
        raise RuntimeError("outcome-v2 chronological splits must all be non-empty")
    if {example.season for example in validation} != set(VALIDATION_SEASONS):
        raise RuntimeError("outcome-v2 validation seasons are incomplete")
    if {example.season for example in historical_test} != set(HISTORICAL_TEST_SEASONS):
        raise RuntimeError("outcome-v2 historical-test seasons are incomplete")
    return _Splits(train=train, validation=validation, historical_test=historical_test)


def _write_evaluation_records(
    examples: Sequence[NbaV2Example],
    answers_path: Path,
    prompts_path: Path,
) -> None:
    pairs = tuple(_side_swap_pairs(examples))
    write_jsonl(_training_examples(pairs), answers_path)
    write_outcome_forecast_jsonl(
        (example.training_example.case for example in pairs),
        prompts_path,
    )


def _side_swap_pairs(examples: Iterable[NbaV2Example]) -> Iterable[NbaV2Example]:
    for example in examples:
        yield example
        yield side_swap_nba_v2_example(example)


def _training_examples(examples: Iterable[NbaV2Example]) -> Iterable[TrainingExample]:
    return (example.training_example for example in examples)


def _prompt_hash(example: TrainingExample) -> str:
    return sha256(render_case(example.case).encode()).hexdigest()


def _manifest(build: _Build) -> dict[str, object]:
    final_rich_raw, _, final_rich_recalibration = build.final_evaluations
    output_paths = (
        TRAIN_PATH,
        VALIDATION_ANSWERS_PATH,
        VALIDATION_PROMPTS_PATH,
        TEST_ANSWERS_PATH,
        TEST_PROMPTS_PATH,
    )
    return {
        "schema_version": 1,
        "outcome_input_schema_version": OUTCOME_INPUT_SCHEMA_VERSION,
        "source": {
            "name": "FiveThirtyEight historical NBA Elo",
            "page": SOURCE_PAGE,
            "url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "license": "CC BY 4.0",
            "license_url": LICENSE_URL,
            "eligible_games": build.source_count,
        },
        "features": {
            "names": list(NBA_V2_FEATURE_NAMES),
            "scaling": "training-cohort root mean square without mean centering",
            "state_update": "whole-date batches using strictly earlier game dates",
            "season_reset": True,
            "side_swap_antisymmetric": True,
            "limitations": list(NBA_V2_DATA_LIMITATIONS),
        },
        "anti_cheating": {
            "target": "realized winner; never an input feature",
            "team_identity": "used only inside the feature state machine",
            "public_records_are_anonymous": True,
            "same_date_results_visible": False,
            "future_results_visible": False,
            "prompt_orbits_before_deduplication": build.source_count,
            "prompt_orbits_after_deduplication": build.kept_count,
            "removed_prompt_orbits_by_season": {
                str(season): count for season, count in build.removed_by_season.items()
            },
            "historical_warning": (
                "These answers already existed locally and may occur in model pretraining. "
                "This is a contamination-prone retrospective diagnostic, not a prospective claim."
            ),
        },
        "splits": {
            "train": {
                "through_season": TRAIN_LAST_SEASON,
                "original_games": len(build.splits.train),
                "side_swapped_training_rows": len(build.splits.train) * 2,
                "question_ids_sha256": _question_ids_sha256(build.splits.train),
                "full_cohort_sha256": cohort_sha256(build.splits.train),
            },
            "validation": {
                "seasons": list(VALIDATION_SEASONS),
                "original_games": len(build.splits.validation),
                "question_ids_sha256": _question_ids_sha256(build.splits.validation),
                "full_cohort_sha256": cohort_sha256(build.splits.validation),
            },
            "historical_test": {
                "seasons": list(HISTORICAL_TEST_SEASONS),
                "original_games": len(build.splits.historical_test),
                "question_ids_sha256": _question_ids_sha256(build.splits.historical_test),
                "full_cohort_sha256": cohort_sha256(build.splits.historical_test),
            },
        },
        "tabular_models": {
            "objective": "mean winner cross-entropy with an Elo log-odds offset",
            "intercept": False,
            "rich_fit_config": _fit_config_payload(RICH_FIT_CONFIG),
            "recalibration_fit_config": _fit_config_payload(RECALIBRATION_FIT_CONFIG),
            "development_fit_through_season": TRAIN_LAST_SEASON,
            "development": _models_payload(build.development_models),
            "final_fit_through_season": VALIDATION_SEASONS[-1],
            "final": _models_payload(build.final_models),
        },
        "evaluation": {
            "validation": _comparison_payloads(build.development_evaluations),
            "historical_test": _comparison_payloads(build.final_evaluations),
            "historical_gate_passes_raw_elo": final_rich_raw.passes,
            "historical_gate_passes_recalibrated_elo": final_rich_recalibration.passes,
            "valid_probability_contract": "strictly between zero and one",
            "failed_forecast_realized_probability": FAILURE_REALIZED_PROBABILITY,
            "failure_handling": (
                "Missing IDs and explicit malformed-output records remain in the denominator "
                "at the frozen worst-case realized probability."
            ),
            "full_outcome_v2_ready": False,
            "full_outcome_v2_missing": [
                "licensed point-in-time travel data",
                "licensed timestamped availability and expected-lineup data",
                "roster and player-level rolling metrics",
                "at least two prospectively frozen evaluation seasons",
            ],
        },
        "rl": {
            "ready": False,
            "gate": (
                "First clear raw and recalibrated Elo on every frozen season, then define real "
                "sequential evidence actions, retrieval costs, stopping, and a prospective cohort."
            ),
            "reward": "Elo-relative realized-outcome log score minus tool cost and optional KL",
            "paid_tinker_job_launched": False,
        },
        "outputs": {path.name: file_sha256(path) for path in output_paths},
    }


def _models_payload(models: _FittedModels) -> dict[str, object]:
    return {
        "feature_scales": list(models.feature_scales),
        "rich": _model_payload(models.rich),
        "recalibrated_elo": _model_payload(models.recalibration),
    }


def _model_payload(model: EloResidualModel) -> dict[str, object]:
    return {"feature_names": list(model.feature_names), "weights": list(model.weights)}


def _fit_config_payload(config: EloResidualFitConfig) -> dict[str, int | float]:
    return {
        "steps": config.steps,
        "learning_rate": config.learning_rate,
        "l2_penalty": config.l2_penalty,
    }


def _comparison_payloads(
    evaluations: tuple[
        MultiSeasonEvaluation,
        MultiSeasonEvaluation,
        MultiSeasonEvaluation,
    ],
) -> dict[str, object]:
    rich_raw, recalibration_raw, rich_recalibration = evaluations
    return {
        "rich_vs_raw_elo": _evaluation_payload(rich_raw, "raw_elo"),
        "recalibrated_vs_raw_elo": _evaluation_payload(recalibration_raw, "raw_elo"),
        "rich_vs_recalibrated_elo": _evaluation_payload(
            rich_recalibration,
            "training_only_recalibrated_elo",
        ),
    }


def _evaluation_payload(
    evaluation: MultiSeasonEvaluation,
    baseline_name: str,
) -> dict[str, object]:
    return {
        "baseline": baseline_name,
        "declared_seasons": list(evaluation.declared_seasons),
        "game_count": evaluation.game_count,
        "pooled_baseline_relative_log_score": evaluation.pooled_baseline_relative_log_score,
        "bootstrap": {
            "block_days": evaluation.bootstrap_block_days,
            "resamples": evaluation.bootstrap_resamples,
            "seed": evaluation.bootstrap_seed,
            "one_sided_alpha": evaluation.one_sided_alpha,
        },
        "seasons": [_season_payload(season) for season in evaluation.seasons],
        "passes_every_season": evaluation.passes,
    }


def _season_payload(evaluation: SeasonEvaluation) -> dict[str, object]:
    return {
        "season": evaluation.season,
        "game_count": evaluation.game_count,
        "calendar_block_count": evaluation.calendar_block_count,
        "model": _scores_payload(evaluation.model),
        "baseline": _scores_payload(evaluation.baseline),
        "baseline_relative_log_score": evaluation.mean_baseline_relative_log_score,
        "lower_one_sided_95": evaluation.lower_one_sided_95,
        "passes": evaluation.passes,
    }


def _scores_payload(scores: BinaryProperScores) -> dict[str, float]:
    return {
        "mean_log_loss": scores.mean_log_loss,
        "mean_brier": scores.mean_brier,
    }


def _question_ids_sha256(examples: Sequence[NbaV2Example]) -> str:
    question_ids = [example.training_example.case.question.question_id for example in examples]
    return canonical_sha256(question_ids)


def cohort_sha256(examples: Sequence[NbaV2Example]) -> str:
    """Hash frozen IDs, dates, outcomes, seasons, and raw Elo baselines."""
    records: list[dict[str, object]] = []
    for example in examples:
        training = example.training_example
        records.append(
            {
                "question_id": training.case.question.question_id,
                "season": example.season,
                "game_date": training.case.question.forecast_at.date().isoformat(),
                "realized_team_win": training.realized_outcome == TEAM_OUTCOME,
                "elo_team_probability": example.features.venue_adjusted_elo_probabilities[0],
            }
        )
    return canonical_sha256(records)


def _verify_split_contract(splits: _Splits) -> None:
    contracts = (
        ("train", splits.train, EXPECTED_TRAIN_COHORT),
        ("validation", splits.validation, EXPECTED_VALIDATION_COHORT),
        ("historical test", splits.historical_test, EXPECTED_HISTORICAL_TEST_COHORT),
    )
    for name, examples, (expected_count, expected_hash) in contracts:
        if len(examples) != expected_count or cohort_sha256(examples) != expected_hash:
            raise RuntimeError(f"outcome-v2 {name} cohort differs from its frozen contract")


if __name__ == "__main__":
    main()
