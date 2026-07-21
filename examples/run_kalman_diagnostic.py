"""Run the DARKO-lite Kalman diagnostic: Kalman versus RAPM as the player-value input.

Rebuilds the private-prototype pipeline exactly (same injury-archive schedule, same carryover
MOV Elo replay including the 2020-2021 double-count in warmup, same projected-variant feature
schema, same frozen fit config, same 2022-2024 training and 2025-2026 evaluation split) but
replaces the causal RAPM ``player_ratings`` input to ``build_game_features`` with the
Kalman-filtered name-keyed ratings from :mod:`forecastfm.nba_kalman`. The filter runs over
the 2022-2026 seasons in tipoff order; each season resets to that season's causal RAPM fit
(``fit_season_ratings`` over the three strictly earlier seasons only). The predeclared
verdict: ``kalman_beats_rapm`` holds when the pooled (2025, 2026) projected-variant log loss
with Kalman ratings is at least 0.001 below the pooled RAPM projected log loss, where the
RAPM per-season values are cited as constants from
``data/processed/private_prototype/manifest.json`` (``variants.projected_vs_raw_elo`` model
``mean_log_loss``). Both evaluation seasons are opened; this is a diagnostic, not a gate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

from examples.run_private_prototype import (
    EVALUATION_SEASONS,
    FIT_CONFIG,
    INJURY_ARCHIVE,
    RAPM_PRIOR_FILES,
    SEASON_FILES,
    TRAINING_SEASONS,
    build_schedule,
    elo_replay,
    load_season,
)

from forecastfm.elo_residual import EloResidualModel, EloResidualRow, fit_elo_residual
from forecastfm.json_utils import require_float
from forecastfm.nba_feature_builder import (
    GameFeatures,
    InjurySnapshot,
    PlayerValueInputs,
    build_game_features,
    load_injury_index,
)
from forecastfm.nba_kalman import KalmanConfig, KalmanRatings, ratings_at
from forecastfm.nba_mov_elo import MovEloReplay
from forecastfm.nba_prototype_dataset import (
    PrototypeGameRow,
    build_prototype_rows,
    fit_rms_scales,
    to_residual_row,
)
from forecastfm.nba_rapm import fit_season_ratings
from forecastfm.nba_rich import NBA_RICH_FEATURE_NAMES
from forecastfm.nba_season_games import SeasonGame
from forecastfm.outcome_v2_metrics import (
    BinaryForecast,
    DatedBinaryCohortMember,
    evaluate_multi_season,
)

OUTPUT_DIR = Path("data/processed/kalman_diagnostic")
KALMAN_SEASONS = (2022, 2023, 2024, 2025, 2026)
PROJECTED_FEATURE_NAMES = (*NBA_RICH_FEATURE_NAMES, "projected_rotation_value")
VERDICT_MARGIN = 0.001

# RAPM projected-variant log losses, copied from
# data/processed/private_prototype/manifest.json (variants.projected_vs_raw_elo seasons'
# model.mean_log_loss); the predeclared comparison point for the Kalman input.
RAPM_PROJECTED_LOG_LOSS: dict[int, float] = {
    2025: 0.6052788362661518,
    2026: 0.6038951954199013,
}


def main() -> int:
    """Build Kalman ratings, rebuild the projected variant, and score the verdict."""
    notes: list[str] = []
    injury_snapshots = load_injury_index(INJURY_ARCHIVE)
    schedule = build_schedule(injury_snapshots)
    joined_by_season: dict[int, list[SeasonGame]] = {}
    for season, path in SEASON_FILES.items():
        joined, season_notes = load_season(season, path, schedule)
        joined_by_season[season] = joined
        notes.extend(season_notes)
    replay = elo_replay(joined_by_season, notes)
    kalman_games = {season: joined_by_season[season] for season in KALMAN_SEASONS}
    kalman = ratings_at(kalman_games, _kalman_priors(notes))
    rows_by_season: dict[int, list[PrototypeGameRow]] = {}
    game_features: dict[int, GameFeatures] = {}
    for season in KALMAN_SEASONS:
        rows, features, season_notes = _season_rows(
            season, joined_by_season[season], replay, injury_snapshots, kalman
        )
        rows_by_season[season] = rows
        game_features.update(features)
        notes.extend(season_notes)
    training = [row for season in TRAINING_SEASONS for row in rows_by_season[season]]
    evaluation = [row for season in EVALUATION_SEASONS for row in rows_by_season[season]]
    projected_training = [_projected_row(row, game_features) for row in training]
    projected_evaluation = [_projected_row(row, game_features) for row in evaluation]
    model, scales = _fit_projected(projected_training)
    per_season = {
        season: _evaluate_season(model, scales, projected_evaluation, season)
        for season in EVALUATION_SEASONS
    }
    report = _report(model, scales, per_season, notes)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    return 0


def _kalman_priors(notes: list[str]) -> dict[int, dict[int, float]]:
    """Fit each Kalman season's causal RAPM prior from strictly earlier seasons only."""
    priors: dict[int, dict[int, float]] = {}
    for season in KALMAN_SEASONS:
        failures: list[str] = []
        priors[season] = fit_season_ratings(RAPM_PRIOR_FILES, season, failures=failures).ratings
        notes.extend(failures)
    return priors


def _season_rows(
    season: int,
    joined: list[SeasonGame],
    replay: MovEloReplay,
    injury_snapshots: list[InjurySnapshot],
    kalman: KalmanRatings,
) -> tuple[list[PrototypeGameRow], dict[int, GameFeatures], list[str]]:
    """Build one season's rows with the Kalman name-keyed ratings as the player-value input."""
    features, notes = build_game_features(
        joined,
        replay.ratings,
        injury_snapshots,
        player_values=PlayerValueInputs(by_game=kalman.by_name),
    )
    rows = build_prototype_rows(joined, features, replay.home_probabilities)
    return rows, {entry.game_id: entry for entry in features}, notes


def _projected_row(
    row: PrototypeGameRow,
    game_features: Mapping[int, GameFeatures],
) -> PrototypeGameRow:
    """Append the projected-rotation difference (home minus away, 0.0 when unknown)."""
    projected = game_features[row.game_id].projected_rotation
    delta = 0.0 if projected is None else projected[1] - projected[0]
    return replace(row, features_standard=(*row.features_standard, delta))


def _fit_projected(rows: list[PrototypeGameRow]) -> tuple[EloResidualModel, tuple[float, ...]]:
    """Fit the frozen Elo-offset logistic on projected rows with training-only RMS scales."""
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
    return fit_elo_residual(residual_rows, PROJECTED_FEATURE_NAMES, FIT_CONFIG), scales


def _evaluate_season(
    model: EloResidualModel,
    scales: tuple[float, ...],
    rows: list[PrototypeGameRow],
    season: int,
) -> dict[str, object]:
    """Score one evaluation season against the raw-Elo baseline cohort."""
    season_rows = [row for row in rows if row.season == season]
    forecasts = [
        BinaryForecast(row.question_id, _probability(model, scales, row)) for row in season_rows
    ]
    cohort = [
        DatedBinaryCohortMember(
            question_id=row.question_id,
            season=row.season,
            game_date=row.game_date,
            realized_team_win=row.home_won,
            baseline_team_probability=row.elo_home_probability,
        )
        for row in season_rows
    ]
    gate = evaluate_multi_season(forecasts, cohort, (season,))
    scores = gate.seasons[0]
    return {
        "games": scores.game_count,
        "log_loss": scores.model.mean_log_loss,
        "brier": scores.model.mean_brier,
        "rapm_log_loss": RAPM_PROJECTED_LOG_LOSS[season],
        "delta_vs_rapm": RAPM_PROJECTED_LOG_LOSS[season] - scores.model.mean_log_loss,
    }


def _probability(
    model: EloResidualModel, scales: tuple[float, ...], row: PrototypeGameRow
) -> float:
    residual = to_residual_row(row, include_health=False)
    scaled = tuple(value / scale for value, scale in zip(residual.features, scales, strict=True))
    return model.predict_probability(residual.elo_probability, scaled)


def _report(
    model: EloResidualModel,
    scales: tuple[float, ...],
    per_season: dict[int, dict[str, object]],
    notes: list[str],
) -> dict[str, object]:
    """Assemble the manifest with the predeclared pooled verdict."""
    total_games = sum(
        int(require_float(season["games"], "games")) for season in per_season.values()
    )
    pooled_kalman = (
        sum(
            require_float(season["log_loss"], "log_loss")
            * int(require_float(season["games"], "games"))
            for season in per_season.values()
        )
        / total_games
    )
    pooled_rapm = (
        sum(
            RAPM_PROJECTED_LOG_LOSS[season] * int(require_float(entry["games"], "games"))
            for season, entry in per_season.items()
        )
        / total_games
    )
    verdict = {
        "kalman_beats_rapm": pooled_kalman < pooled_rapm - VERDICT_MARGIN,
        "rule": (
            "kalman pooled projected log loss < rapm pooled projected log loss - "
            f"{VERDICT_MARGIN} over (2025, 2026)"
        ),
        "pooled_kalman_log_loss": pooled_kalman,
        "pooled_rapm_log_loss": pooled_rapm,
        "pooled_delta_rapm_minus_kalman": pooled_rapm - pooled_kalman,
        "margin": VERDICT_MARGIN,
    }
    return {
        "schema_version": 1,
        "kalman_config": asdict(KalmanConfig()),
        "training_seasons": list(TRAINING_SEASONS),
        "evaluation_seasons": list(EVALUATION_SEASONS),
        "feature_names": list(PROJECTED_FEATURE_NAMES),
        "fit_config": {
            "steps": FIT_CONFIG.steps,
            "learning_rate": FIT_CONFIG.learning_rate,
            "l2_penalty": FIT_CONFIG.l2_penalty,
        },
        "models": {
            "kalman_projected": {
                "per_season": {str(season): entry for season, entry in per_season.items()},
                "pooled_log_loss": pooled_kalman,
                "weights": list(model.weights),
                "rms_scales": list(scales),
            },
            "rapm_projected": {
                "per_season": {
                    str(season): {"log_loss": value}
                    for season, value in RAPM_PROJECTED_LOG_LOSS.items()
                },
                "pooled_log_loss": pooled_rapm,
                "source": (
                    "data/processed/private_prototype/manifest.json "
                    "variants.projected_vs_raw_elo model.mean_log_loss"
                ),
            },
        },
        "verdict": verdict,
        "notes_count": len(notes),
    }


if __name__ == "__main__":
    raise SystemExit(main())
