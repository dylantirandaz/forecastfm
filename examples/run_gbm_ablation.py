"""Run the gradient-boosting ablation of the private zero-cost prototype.

Rebuilds the exact private-prototype rows of :mod:`examples.run_private_prototype` and
replaces the logistic Elo-offset correction with a LightGBM binary classifier
(:mod:`forecastfm.nba_gbm`) that takes logit(Elo home probability) as a leading feature.
Fitting uses the 2021-22 through 2023-24 seasons; evaluation, baselines (raw replayed
margin-of-victory Elo and the training-only logit recalibration), and the declared gate
conjunction are identical to the logistic prototype. A disclosed ablation adds the two
local-only availability aggregates. Everything runs from local retained data; nothing is
uploaded.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from examples.run_private_prototype import (
    EVALUATION_SEASONS,
    INJURY_ARCHIVE,
    OPENED_EVALUATION_SEASONS,
    SEASON_FILES,
    TRAINING_SEASONS,
    build_schedule,
    elo_replay,
    load_season,
    season_rows,
)

from forecastfm.nba_evaluation_gate import (
    NbaRecalibrationRow,
    fit_training_only_logit_recalibrator,
)
from forecastfm.nba_feature_builder import load_injury_index
from forecastfm.nba_gbm import DEFAULT_GBM_PARAMS, GbmModel, fit_gbm
from forecastfm.nba_prototype_dataset import PrototypeGameRow
from forecastfm.nba_season_games import SeasonGame
from forecastfm.outcome_v2_config import outcome_v2_evaluation_policy
from forecastfm.outcome_v2_metrics import (
    BinaryForecast,
    DatedBinaryCohortMember,
    MultiSeasonEvaluation,
    evaluate_multi_season,
)

OUTPUT_DIR = Path("data/processed/private_prototype_gbm")


def main() -> int:
    """Build every season, fit on training seasons, and evaluate the declared seasons."""
    injury_snapshots = load_injury_index(INJURY_ARCHIVE)
    schedule = build_schedule(injury_snapshots)
    joined_by_season: dict[int, list[SeasonGame]] = {}
    notes: list[str] = []
    for season, path in SEASON_FILES.items():
        joined, season_notes = load_season(season, path, schedule)
        joined_by_season[season] = joined
        notes.extend(season_notes)
    replay = elo_replay(joined_by_season, notes)
    rows_by_season: dict[int, list[PrototypeGameRow]] = {}
    for season, joined in joined_by_season.items():
        rows, season_notes = season_rows(season, joined, replay, injury_snapshots)
        rows_by_season[season] = rows
        notes.extend(season_notes)
    training = [row for season in TRAINING_SEASONS for row in rows_by_season[season]]
    evaluation = [row for season in EVALUATION_SEASONS for row in rows_by_season[season]]
    report = _evaluate(training, evaluation, notes)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _evaluate(
    training: list[PrototypeGameRow],
    evaluation: list[PrototypeGameRow],
    notes: list[str],
) -> dict[str, object]:
    standard_model = fit_gbm(training, include_health=False, params=DEFAULT_GBM_PARAMS)
    health_training = [row for row in training if row.features_health is not None]
    health_model = fit_gbm(health_training, include_health=True, params=DEFAULT_GBM_PARAMS)
    health_evaluation = [row for row in evaluation if row.features_health is not None]
    policy = outcome_v2_evaluation_policy()
    recalibrator = fit_training_only_logit_recalibrator(
        [_recalibration_row(row) for row in training],
        policy=policy,
    )
    report: dict[str, object] = {
        "training_games": len(training),
        "evaluation_games": len(evaluation),
        "training_seasons": list(TRAINING_SEASONS),
        "evaluation_seasons": list(EVALUATION_SEASONS),
        "opened_evaluation_seasons": list(OPENED_EVALUATION_SEASONS),
        "health_training_games": len(health_training),
        "health_evaluation_games": len(health_evaluation),
        "notes_count": len(notes),
        "elo_recipe": {
            "name": "carryover_margin_of_victory",
            "initial_rating": 1500.0,
            "k_factor": 20.0,
            "rating_scale": 400.0,
            "home_advantage": 60.0,
            "carryover": 0.75,
        },
        "recalibration": {
            "intercept": recalibrator.intercept,
            "slope": recalibrator.slope,
        },
        "models": {
            "standard": _model_payload(standard_model),
            "health": _model_payload(health_model),
        },
    }
    variants: dict[str, object] = {}
    for name, model, rows in (
        ("standard", standard_model, evaluation),
        ("health", health_model, health_evaluation),
    ):
        include_health = name == "health"
        for baseline_name in ("raw_elo", "recalibrated_elo"):
            forecasts = [
                BinaryForecast(
                    row.question_id,
                    model.probability(row, include_health=include_health),
                )
                for row in rows
            ]
            cohort = [
                DatedBinaryCohortMember(
                    question_id=row.question_id,
                    season=row.season,
                    game_date=row.game_date,
                    realized_team_win=row.home_won,
                    baseline_team_probability=(
                        row.elo_home_probability
                        if baseline_name == "raw_elo"
                        else recalibrator.team_probability(row.elo_home_probability)
                    ),
                )
                for row in rows
            ]
            gate = evaluate_multi_season(forecasts, cohort, EVALUATION_SEASONS)
            variants[f"{name}_vs_{baseline_name}"] = _gate_payload(gate)
    report["variants"] = variants
    return report


def _model_payload(model: GbmModel) -> dict[str, object]:
    return {
        "feature_names": list(model.feature_names),
        "params": asdict(DEFAULT_GBM_PARAMS),
        "n_trees": model.booster.num_trees(),
    }


def _recalibration_row(row: PrototypeGameRow) -> NbaRecalibrationRow:
    return NbaRecalibrationRow(
        question_id=row.question_id,
        season=row.season,
        game_date=row.game_date,
        raw_elo_team_probability=row.elo_home_probability,
        realized_team_win=row.home_won,
    )


def _gate_payload(gate: MultiSeasonEvaluation) -> dict[str, object]:
    payload = asdict(gate)
    payload.pop("declared_seasons", None)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
