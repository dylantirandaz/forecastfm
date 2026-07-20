"""Run the sequential revision-stream diagnostic: injury-news momentum beyond static T-60.

For each horizon in (360, 120, 60, 15) minutes before tipoff, builds the revision-delta rows of
``nba_revision`` for the 2021-22 through 2025-26 seasons, fits the frozen no-intercept
Elo-offset logistic correction (training-only uncentered RMS scales, same FIT_CONFIG as the main
driver) on the training seasons, and evaluates per-season log loss on 2024-25 (opened) and
2025-26 (pristine) against the raw carryover margin-of-victory Elo replay. The predeclared
verdicts are: (i) the delta-feature model at T-15 beats the same model at T-60, and (ii) the
T-60 delta-feature model beats the main driver's static T-60 standard model, each requiring a
lower log loss in every evaluation season. Everything runs from local retained data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

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

from forecastfm.elo_residual import EloResidualRow, fit_elo_residual
from forecastfm.nba_feature_builder import InjurySnapshot, load_injury_index
from forecastfm.nba_rapm import fit_season_ratings_by_name
from forecastfm.nba_revision import (
    REVISION_FEATURE_NAMES,
    REVISION_HORIZON_MINUTES,
    RevisionGame,
    RevisionGameRow,
    build_revision_rows,
)
from forecastfm.nba_season_games import SeasonGame
from forecastfm.outcome_v2_metrics import (
    BinaryForecast,
    DatedBinaryCohortMember,
    evaluate_multi_season,
)

OUTPUT_DIR = Path("data/processed/revision_model")

# Static T-60 standard-model log losses, copied from
# data/processed/private_prototype/manifest.json (variants.standard_vs_raw_elo seasons'
# model.mean_log_loss); the predeclared comparison point for the T-60 delta model.
STATIC_T60_LOG_LOSS: dict[int, float] = {
    2025: 0.6076334469097109,
    2026: 0.6067189701472464,
}


def main() -> int:
    """Build all horizons, fit per horizon on training seasons, and evaluate 2025/2026."""
    notes: list[str] = []
    injury_snapshots = load_injury_index(INJURY_ARCHIVE)
    schedule = build_schedule(injury_snapshots)
    joined_by_season: dict[int, list[SeasonGame]] = {}
    for season, path in SEASON_FILES.items():
        joined, season_notes = load_season(season, path, schedule)
        joined_by_season[season] = joined
        notes.extend(season_notes)
    replay = elo_replay(joined_by_season, notes)
    rows, coverage = _build_all_rows(
        joined_by_season, injury_snapshots, replay.home_probabilities, notes
    )
    horizons_report = {
        horizon: _evaluate_horizon(horizon, rows) for horizon in REVISION_HORIZON_MINUTES
    }
    report = {
        "schema_version": 1,
        "feature_names": list(REVISION_FEATURE_NAMES),
        "horizons_minutes": list(REVISION_HORIZON_MINUTES),
        "training_seasons": list(TRAINING_SEASONS),
        "evaluation_seasons": list(EVALUATION_SEASONS),
        "coverage": coverage,
        "horizons": horizons_report,
        "verdict": _verdicts(horizons_report),
        "notes_count": len(notes),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    return 0


def _build_all_rows(
    joined_by_season: dict[int, list[SeasonGame]],
    injury_snapshots: list[InjurySnapshot],
    home_probabilities: dict[int, float],
    notes: list[str],
) -> tuple[list[RevisionGameRow], dict[str, object]]:
    rows: list[RevisionGameRow] = []
    coverage: dict[str, object] = {}
    for season, joined in sorted(joined_by_season.items()):
        failures: list[str] = []
        rapm = fit_season_ratings_by_name(RAPM_PRIOR_FILES, season, failures=failures)
        notes.extend(failures)
        games = [
            RevisionGame(
                game_id=game.game_id,
                season=game.season_label,
                game_date=game.game_date,
                tipoff=game.tipoff,
                away_abbreviation=game.away_abbreviation,
                home_abbreviation=game.home_abbreviation,
                home_won=game.home_won,
            )
            for game in joined
        ]
        result = build_revision_rows(games, injury_snapshots, home_probabilities, rapm)
        rows.extend(result.rows)
        coverage[str(season)] = {
            "games": result.games_total,
            "rows": len(result.rows),
            "skipped_game_horizons": result.skipped,
        }
    return rows, coverage


def _evaluate_horizon(horizon: int, rows: list[RevisionGameRow]) -> dict[str, object]:
    horizon_rows = [row for row in rows if row.horizon_minutes == horizon]
    training = [row for row in horizon_rows if row.season in TRAINING_SEASONS]
    evaluation = [row for row in horizon_rows if row.season in EVALUATION_SEASONS]
    scales = _rms_scales([row.features for row in training])
    model = fit_elo_residual(
        [_scaled_row(row, scales) for row in training],
        REVISION_FEATURE_NAMES,
        FIT_CONFIG,
    )
    forecasts = [
        BinaryForecast(
            row.question_id,
            model.predict_probability(
                row.elo_home_probability,
                tuple(value / scale for value, scale in zip(row.features, scales, strict=True)),
            ),
        )
        for row in evaluation
    ]
    cohort = [
        DatedBinaryCohortMember(
            question_id=row.question_id,
            season=row.season,
            game_date=row.game_date,
            realized_team_win=row.home_won,
            baseline_team_probability=row.elo_home_probability,
        )
        for row in evaluation
    ]
    gate = evaluate_multi_season(forecasts, cohort, EVALUATION_SEASONS)
    seasons: dict[str, object] = {}
    for season_eval in gate.seasons:
        seasons[str(season_eval.season)] = {
            "games": season_eval.game_count,
            "model_log_loss": season_eval.model.mean_log_loss,
            "baseline_raw_elo_log_loss": season_eval.baseline.mean_log_loss,
        }
    return {
        "training_games": len(training),
        "weights": list(model.weights),
        "rms_scales": list(scales),
        "seasons": seasons,
    }


def _verdicts(horizons_report: dict[int, dict[str, object]]) -> dict[str, object]:
    t15 = cast(dict[str, dict[str, float]], horizons_report[15]["seasons"])
    t60 = cast(dict[str, dict[str, float]], horizons_report[60]["seasons"])
    t15_losses = {season: t15[str(season)]["model_log_loss"] for season in EVALUATION_SEASONS}
    t60_losses = {season: t60[str(season)]["model_log_loss"] for season in EVALUATION_SEASONS}
    return {
        "t15_beats_t60": {
            "definition": "T-15 delta model log loss below T-60 delta model in every season",
            "value": all(t15_losses[s] < t60_losses[s] for s in EVALUATION_SEASONS),
            "t15_model_log_loss": t15_losses,
            "t60_model_log_loss": t60_losses,
        },
        "t60_delta_beats_static": {
            "definition": "T-60 delta model log loss below the static T-60 standard model",
            "value": all(
                t60_losses[season] < STATIC_T60_LOG_LOSS[season] for season in EVALUATION_SEASONS
            ),
            "t60_delta_model_log_loss": t60_losses,
            "static_t60_log_loss": STATIC_T60_LOG_LOSS,
        },
    }


def _rms_scales(vectors: list[tuple[float, ...]]) -> tuple[float, ...]:
    """Compute uncentered RMS scales over original training rows only (main-driver rule)."""
    width = len(vectors[0])
    scales: list[float] = []
    for index in range(width):
        mean_square = sum(vector[index] ** 2 for vector in vectors) / len(vectors)
        scales.append(mean_square**0.5 if mean_square > 0.0 else 1.0)
    return tuple(scales)


def _scaled_row(row: RevisionGameRow, scales: tuple[float, ...]) -> EloResidualRow:
    return EloResidualRow(
        question_id=row.question_id,
        elo_probability=row.elo_home_probability,
        features=tuple(value / scale for value, scale in zip(row.features, scales, strict=True)),
        outcome=1 if row.home_won else 0,
    )


if __name__ == "__main__":
    raise SystemExit(main())
