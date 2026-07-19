"""Run the private zero-cost prototype: free-data tabular corrections versus replayed Elo.

Fits the frozen no-intercept Elo-offset logistic correction on the 2021-22 and 2022-23 seasons
and evaluates it on the untouched 2023-24 and 2024-25 seasons under the predeclared conjunction:
positive mean baseline-relative log score and a positive one-sided 95% seven-day block-bootstrap
lower bound in every season, against both raw replayed Elo and a training-only recalibration.
A disclosed ablation adds the two local-only availability aggregates on the subset of games with
a pre-T-60 injury-report snapshot. Everything runs from local retained data; nothing is uploaded.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from forecastfm.elo_residual import (
    EloResidualFitConfig,
    EloResidualModel,
    EloResidualRow,
    fit_elo_residual,
)
from forecastfm.nba_arenas import EXCLUDED_CUP_FINALS
from forecastfm.nba_elo_replay import replay_nba_elo_states
from forecastfm.nba_evaluation_gate import (
    NbaRecalibrationRow,
    fit_training_only_logit_recalibrator,
)
from forecastfm.nba_feature_builder import (
    InjurySnapshot,
    build_game_features,
    load_injury_index,
    schedule_from_injury_index,
)
from forecastfm.nba_pbp import read_pbp_games
from forecastfm.nba_prototype_dataset import (
    PROTOTYPE_ELO_RECIPE,
    PrototypeGameRow,
    build_prototype_rows,
    build_replay_inputs,
    elo_ratings_by_game,
    feature_names,
    fit_rms_scales,
    to_residual_row,
)
from forecastfm.nba_rapm import fit_season_ratings
from forecastfm.nba_season_games import ScheduleEntry, join_season_games
from forecastfm.outcome_v2_config import outcome_v2_evaluation_policy
from forecastfm.outcome_v2_metrics import (
    BinaryForecast,
    DatedBinaryCohortMember,
    MultiSeasonEvaluation,
    evaluate_multi_season,
)

PBP_DIR = Path("data/raw/shufinskiy")
INJURY_ARCHIVE = Path("data/raw/nba_injury_reports")
OUTPUT_DIR = Path("data/processed/private_prototype")

SEASON_FILES = {
    2022: "nbastats_2021.csv",
    2023: "nbastats_2022.csv",
    2024: "nbastats_2023.csv",
    2025: "nbastats_2024.csv",
}
RAPM_PRIOR_FILES = {
    2020: "nbastats_2019.csv",
    2021: "nbastats_2020.csv",
    2022: "nbastats_2021.csv",
    2023: "nbastats_2022.csv",
    2024: "nbastats_2023.csv",
    2025: "nbastats_2024.csv",
}
TRAINING_SEASONS = (2022, 2023)
EVALUATION_SEASONS = (2024, 2025)
FIT_CONFIG = EloResidualFitConfig(steps=2_000, learning_rate=0.05, l2_penalty=0.01)


class _ScaledModel:
    """One fitted Elo-offset model with its frozen training-only RMS scales."""

    def __init__(self, model: EloResidualModel, scales: tuple[float, ...]) -> None:
        self.model = model
        self.scales = scales

    def probability(self, row: PrototypeGameRow, *, include_health: bool) -> float:
        residual = to_residual_row(row, include_health=include_health)
        scaled = tuple(
            value / scale for value, scale in zip(residual.features, self.scales, strict=True)
        )
        return self.model.predict_probability(residual.elo_probability, scaled)


def main() -> int:
    """Build every season, fit on training seasons, and evaluate the untouched seasons."""
    injury_index = load_injury_index(INJURY_ARCHIVE)
    schedule = [
        ScheduleEntry(
            game_date=day,
            away_abbreviation=away,
            home_abbreviation=home,
            tip_clock=clock,
        )
        for day, away, home, clock in schedule_from_injury_index(injury_index)
        if (day, away, home) not in EXCLUDED_CUP_FINALS
    ]
    rows_by_season: dict[int, list[PrototypeGameRow]] = {}
    notes: list[str] = []
    for season, filename in SEASON_FILES.items():
        season_rows, season_notes = _build_season(season, filename, schedule, injury_index)
        rows_by_season[season] = season_rows
        notes.extend(season_notes)
    training = [row for season in TRAINING_SEASONS for row in rows_by_season[season]]
    evaluation = [row for season in EVALUATION_SEASONS for row in rows_by_season[season]]
    report = _evaluate(training, evaluation, notes)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _build_season(
    season: int,
    filename: str,
    schedule: list[ScheduleEntry],
    injury_snapshots: list[InjurySnapshot],
) -> tuple[list[PrototypeGameRow], list[str]]:
    failures: list[str] = []
    games = list(read_pbp_games(PBP_DIR / filename, failures))
    season_schedule = [entry for entry in schedule if _entry_season(entry) == season]
    joined, join_notes = join_season_games(games, season_schedule)
    replay_rows, resolutions = build_replay_inputs(joined, f"shufinskiy:{filename}")
    states = list(replay_nba_elo_states(replay_rows, resolutions, PROTOTYPE_ELO_RECIPE))
    ratings = elo_ratings_by_game(states, joined)
    rapm_files = {
        label: PBP_DIR / name
        for label, name in RAPM_PRIOR_FILES.items()
        if (PBP_DIR / name).exists()
    }
    rapm = fit_season_ratings(rapm_files, season, failures=failures)
    features, feature_notes = build_game_features(
        joined,
        ratings,
        injury_snapshots,
        player_ratings=rapm.ratings,
    )
    rows = build_prototype_rows(joined, features, states)
    return rows, failures + join_notes + feature_notes


def _entry_season(entry: ScheduleEntry) -> int:
    return entry.game_date.year + 1 if entry.game_date.month >= 7 else entry.game_date.year


def _evaluate(
    training: list[PrototypeGameRow],
    evaluation: list[PrototypeGameRow],
    notes: list[str],
) -> dict[str, object]:
    standard_model = _fit_variant(training, include_health=False)
    health_training = [row for row in training if row.features_health is not None]
    health_model = _fit_variant(health_training, include_health=True)
    health_evaluation = [row for row in evaluation if row.features_health is not None]
    policy = outcome_v2_evaluation_policy()
    recalibrator = fit_training_only_logit_recalibrator(
        [_recalibration_row(row) for row in training],
        policy=policy,
    )
    report: dict[str, object] = {
        "training_games": len(training),
        "evaluation_games": len(evaluation),
        "health_training_games": len(health_training),
        "health_evaluation_games": len(health_evaluation),
        "notes_count": len(notes),
        "recalibration": {
            "intercept": recalibrator.intercept,
            "slope": recalibrator.slope,
        },
        "models": {
            "standard": _model_payload(standard_model, include_health=False),
            "health": _model_payload(health_model, include_health=True),
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
                    row.question_id, model.probability(row, include_health=include_health)
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


def _fit_variant(rows: list[PrototypeGameRow], *, include_health: bool) -> _ScaledModel:
    scales = fit_rms_scales(rows, include_health=include_health)
    residual_rows = [
        EloResidualRow(
            question_id=residual.question_id,
            elo_probability=residual.elo_probability,
            features=tuple(
                value / scale for value, scale in zip(residual.features, scales, strict=True)
            ),
            outcome=residual.outcome,
        )
        for residual in (to_residual_row(row, include_health=include_health) for row in rows)
    ]
    model = fit_elo_residual(
        residual_rows,
        feature_names(include_health=include_health),
        FIT_CONFIG,
    )
    return _ScaledModel(model, scales)


def _model_payload(model: _ScaledModel, *, include_health: bool) -> dict[str, object]:
    return {
        "feature_names": list(feature_names(include_health=include_health)),
        "weights": list(model.model.weights),
        "rms_scales": list(model.scales),
        "fit_config": {
            "steps": FIT_CONFIG.steps,
            "learning_rate": FIT_CONFIG.learning_rate,
            "l2_penalty": FIT_CONFIG.l2_penalty,
        },
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
