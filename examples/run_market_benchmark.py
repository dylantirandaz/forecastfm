"""Benchmark the frozen standard model against declared ESPN pregame market odds.

Rebuilds the private-prototype rows for the declared evaluation seasons (2025 opened,
2026 pristine), refits the frozen no-intercept Elo-offset logistic correction on the
2021-22 through 2023-24 training seasons exactly like ``run_private_prototype.py``
(same Elo replay, same training-only RMS scales, same training-only logit
recalibration), and evaluates the standard 11-feature model on the subset of evaluation
games with retained ESPN pregame moneyline odds. The declared baselines are the raw
carryover margin-of-victory Elo, the training-only recalibrated Elo, and the market
itself (de-vigged moneyline). Market odds are a DECLARED benchmark only — the project's
frozen rule forbids them as model features, so they never touch fitting. Coverage is
reported honestly: seasons without odds simply have no gate payloads, and the driver
still exits 0. Writes ``data/processed/market_benchmark/manifest.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from examples.run_private_prototype import (
    EVALUATION_SEASONS,
    FIT_CONFIG,
    INJURY_ARCHIVE,
    SEASON_FILES,
    TRAINING_SEASONS,
    build_schedule,
    elo_replay,
    load_season,
    season_rows,
)

from forecastfm.elo_residual import EloResidualModel, EloResidualRow, fit_elo_residual
from forecastfm.nba_evaluation_gate import (
    LogitRecalibrationModel,
    NbaRecalibrationRow,
    fit_training_only_logit_recalibrator,
)
from forecastfm.nba_feature_builder import load_injury_index
from forecastfm.nba_market import load_market_probabilities
from forecastfm.nba_prototype_dataset import (
    PrototypeGameRow,
    fit_rms_scales,
    to_residual_row,
)
from forecastfm.nba_rich import NBA_RICH_FEATURE_NAMES
from forecastfm.nba_season_games import SeasonGame
from forecastfm.outcome_v2_config import outcome_v2_evaluation_policy
from forecastfm.outcome_v2_metrics import (
    BinaryForecast,
    DatedBinaryCohortMember,
    MultiSeasonEvaluation,
    evaluate_multi_season,
)

ESPN_RAW_DIR = Path("data/raw/espn/raw")
ESPN_MANIFEST = Path("data/raw/espn/manifest.json")
OUTPUT_DIR = Path("data/processed/market_benchmark")

BASELINE_ARMS = ("raw_elo", "recalibrated_elo", "market")


class _ScaledModel:
    """One fitted Elo-offset model with its frozen training-only RMS scales."""

    def __init__(self, model: EloResidualModel, scales: tuple[float, ...]) -> None:
        self.model = model
        self.scales = scales

    def probability(self, row: PrototypeGameRow) -> float:
        residual = to_residual_row(row, include_health=False)
        scaled = tuple(
            value / scale for value, scale in zip(residual.features, self.scales, strict=True)
        )
        return self.model.predict_probability(residual.elo_probability, scaled)


def main() -> int:
    """Rebuild rows, refit the standard model, and score the market-benchmark cohort."""
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
        rows, _game_features, season_notes = season_rows(season, joined, replay, injury_snapshots)
        rows_by_season[season] = rows
        notes.extend(season_notes)
    training = [row for season in TRAINING_SEASONS for row in rows_by_season[season]]
    evaluation = [row for season in EVALUATION_SEASONS for row in rows_by_season[season]]
    market = load_market_probabilities(ESPN_RAW_DIR, ESPN_MANIFEST)
    report = _evaluate(training, evaluation, market, notes)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["headline"], indent=2))
    return 0


def _evaluate(
    training: list[PrototypeGameRow],
    evaluation: list[PrototypeGameRow],
    market: dict[int, float],
    notes: list[str],
) -> dict[str, object]:
    model = _fit_standard(training)
    policy = outcome_v2_evaluation_policy()
    recalibrator = fit_training_only_logit_recalibrator(
        [_recalibration_row(row) for row in training],
        policy=policy,
    )
    covered = [row for row in evaluation if row.game_id in market]
    covered_seasons = sorted({row.season for row in covered})
    arms = _evaluate_arms(model, recalibrator, covered, market, covered_seasons)
    report: dict[str, object] = {
        "benchmark": "espn_pregame_moneyline_devig",
        "benchmark_rule": "declared evaluation baseline only; never a model feature",
        "training_games": len(training),
        "training_seasons": list(TRAINING_SEASONS),
        "evaluation_seasons": list(EVALUATION_SEASONS),
        "coverage": _coverage_payload(evaluation, covered, market),
        "notes_count": len(notes),
        "recalibration": {
            "intercept": recalibrator.intercept,
            "slope": recalibrator.slope,
        },
        "model": _model_payload(model),
        "arms": arms,
        "headline": _headline(arms),
    }
    return report


def _evaluate_arms(
    model: _ScaledModel,
    recalibrator: LogitRecalibrationModel,
    covered: list[PrototypeGameRow],
    market: dict[int, float],
    covered_seasons: list[int],
) -> dict[str, object]:
    arms: dict[str, object] = {}
    if not covered:
        return arms
    model_forecasts = [BinaryForecast(row.question_id, model.probability(row)) for row in covered]
    for baseline in BASELINE_ARMS:
        cohort = [
            DatedBinaryCohortMember(
                question_id=row.question_id,
                season=row.season,
                game_date=row.game_date,
                realized_team_win=row.home_won,
                baseline_team_probability=_baseline_probability(
                    baseline, row, recalibrator, market
                ),
            )
            for row in covered
        ]
        gate = evaluate_multi_season(model_forecasts, cohort, covered_seasons)
        arms[f"standard_vs_{baseline}"] = _gate_payload(gate)
    market_forecasts = [BinaryForecast(row.question_id, market[row.game_id]) for row in covered]
    raw_cohort = [
        DatedBinaryCohortMember(
            question_id=row.question_id,
            season=row.season,
            game_date=row.game_date,
            realized_team_win=row.home_won,
            baseline_team_probability=row.elo_home_probability,
        )
        for row in covered
    ]
    arms["market_vs_raw_elo"] = _gate_payload(
        evaluate_multi_season(market_forecasts, raw_cohort, covered_seasons)
    )
    return arms


def _baseline_probability(
    baseline: str,
    row: PrototypeGameRow,
    recalibrator: LogitRecalibrationModel,
    market: dict[int, float],
) -> float:
    if baseline == "raw_elo":
        return row.elo_home_probability
    if baseline == "recalibrated_elo":
        return recalibrator.team_probability(row.elo_home_probability)
    return market[row.game_id]


def _headline(arms: dict[str, object]) -> dict[str, object]:
    per_season: list[dict[str, object]] = []
    market_arm = arms.get("market_vs_raw_elo")
    model_arm = arms.get("standard_vs_raw_elo")
    if isinstance(market_arm, dict) and isinstance(model_arm, dict):
        market_scores = _season_log_losses(cast(dict[str, object], market_arm))
        model_scores = _season_log_losses(cast(dict[str, object], model_arm))
        for season in sorted(market_scores):
            market_loss = market_scores[season]
            model_loss = model_scores[season]
            per_season.append(
                {
                    "season": season,
                    "market_log_loss": market_loss,
                    "model_log_loss": model_loss,
                    "market_minus_model": market_loss - model_loss,
                    "market_beats_model": market_loss < model_loss,
                }
            )
    return {
        "covered_seasons": [entry["season"] for entry in per_season],
        "market_log_loss": {str(e["season"]): e["market_log_loss"] for e in per_season},
        "model_log_loss": {str(e["season"]): e["model_log_loss"] for e in per_season},
        "per_season": per_season,
        "market_beats_model": bool(per_season)
        and all(entry["market_beats_model"] for entry in per_season),
    }


def _season_log_losses(arm: dict[str, object]) -> dict[int, float]:
    seasons = arm.get("seasons")
    losses: dict[int, float] = {}
    if not isinstance(seasons, list | tuple):
        return losses
    for season_value in cast(list[object], seasons):
        season = cast(dict[str, object], season_value)
        model_scores = cast(dict[str, object], season["model"])
        losses[int(cast(int, season["season"]))] = float(cast(float, model_scores["mean_log_loss"]))
    return losses


def _coverage_payload(
    evaluation: list[PrototypeGameRow],
    covered: list[PrototypeGameRow],
    market: dict[int, float],
) -> dict[str, object]:
    by_season: dict[str, object] = {}
    for season in EVALUATION_SEASONS:
        season_rows_all = [row for row in evaluation if row.season == season]
        season_covered = [row for row in covered if row.season == season]
        by_season[str(season)] = {
            "evaluation_games": len(season_rows_all),
            "with_market_odds": len(season_covered),
        }
    return {
        "evaluation_games": len(evaluation),
        "evaluation_games_with_market_odds": len(covered),
        "manifest_games_with_odds": len(market),
        "by_season": by_season,
    }


def _fit_standard(rows: list[PrototypeGameRow]) -> _ScaledModel:
    scales = fit_rms_scales(rows, include_health=False)
    residual_rows = [
        EloResidualRow(
            question_id=residual.question_id,
            elo_probability=residual.elo_probability,
            features=tuple(
                value / scale for value, scale in zip(residual.features, scales, strict=True)
            ),
            outcome=residual.outcome,
        )
        for residual in (to_residual_row(row, include_health=False) for row in rows)
    ]
    model = fit_elo_residual(residual_rows, tuple(NBA_RICH_FEATURE_NAMES), FIT_CONFIG)
    return _ScaledModel(model, scales)


def _model_payload(model: _ScaledModel) -> dict[str, object]:
    return {
        "feature_names": list(NBA_RICH_FEATURE_NAMES),
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
