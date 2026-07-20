"""Run the private zero-cost prototype: free-data tabular corrections versus replayed Elo.

Fits the frozen no-intercept Elo-offset logistic correction on the 2021-22 through 2023-24
seasons and evaluates on 2024-25 (opened, corroborating) and 2025-26 (the only pristine
untouched season, from ESPN-derived play-by-play with synthetic non-official game IDs). The
baseline is the disclosed carryover margin-of-victory Elo replay; the recalibration is fitted
on training seasons only. The gate is the predeclared conjunction: positive mean
baseline-relative log score and a positive one-sided 95% seven-day block-bootstrap lower bound
in every declared season, against both baselines. A disclosed ablation adds the two local-only
availability aggregates. A second disclosed prototype variant, ``projected``, appends the
projected-rotation value difference (home minus away, zero when no pre-T-60 snapshot exists)
to the frozen standard features; it is disabled when ``--exclude-families`` masks the
standard schema. Everything runs from local retained data; nothing is uploaded.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from forecastfm.elo_residual import (
    EloResidualFitConfig,
    EloResidualModel,
    EloResidualRow,
    fit_elo_residual,
)
from forecastfm.nba_arenas import EXCLUDED_CUP_FINALS, is_neutral_site
from forecastfm.nba_evaluation_gate import (
    NbaRecalibrationRow,
    fit_training_only_logit_recalibrator,
)
from forecastfm.nba_feature_builder import (
    GameFeatures,
    InjurySnapshot,
    build_game_features,
    load_injury_index,
    schedule_from_injury_index,
)
from forecastfm.nba_mov_elo import EloGameResult, MovEloReplay, replay_mov_elo
from forecastfm.nba_pbp import PbpGame, read_pbp_games
from forecastfm.nba_prototype_dataset import (
    PrototypeGameRow,
    build_prototype_rows,
    fit_rms_scales,
    to_residual_row,
)
from forecastfm.nba_rapm import fit_season_ratings_by_name
from forecastfm.nba_rich import NBA_LOCAL_HEALTH_FEATURE_NAMES, NBA_RICH_FEATURE_NAMES
from forecastfm.nba_season_games import ScheduleEntry, SeasonGame, join_season_games
from forecastfm.outcome_v2_config import outcome_v2_evaluation_policy
from forecastfm.outcome_v2_metrics import (
    BinaryForecast,
    DatedBinaryCohortMember,
    MultiSeasonEvaluation,
    evaluate_multi_season,
)

DATA_RAW = Path("data/raw")
INJURY_ARCHIVE = DATA_RAW / "nba_injury_reports"
OUTPUT_DIR = Path("data/processed/private_prototype")

SEASON_FILES = {
    2022: DATA_RAW / "shufinskiy/nbastats_2021.csv",
    2023: DATA_RAW / "shufinskiy/nbastats_2022.csv",
    2024: DATA_RAW / "shufinskiy/nbastats_2023.csv",
    2025: DATA_RAW / "shufinskiy/nbastats_2024.csv",
    2026: DATA_RAW / "espn/espn_2025.csv",
}
WARMUP_FILES = {
    2020: DATA_RAW / "shufinskiy/nbastats_2019.csv",
    2021: DATA_RAW / "shufinskiy/nbastats_2020.csv",
}
RAPM_PRIOR_FILES = {
    2020: DATA_RAW / "shufinskiy/nbastats_2019.csv",
    2021: DATA_RAW / "shufinskiy/nbastats_2020.csv",
    2022: DATA_RAW / "shufinskiy/nbastats_2021.csv",
    2023: DATA_RAW / "shufinskiy/nbastats_2022.csv",
    2024: DATA_RAW / "shufinskiy/nbastats_2023.csv",
    2025: DATA_RAW / "shufinskiy/nbastats_2024.csv",
}
TRAINING_SEASONS = (2022, 2023, 2024)
EVALUATION_SEASONS = (2025, 2026)
OPENED_EVALUATION_SEASONS = (2025,)
FIT_CONFIG = EloResidualFitConfig(steps=2_000, learning_rate=0.05, l2_penalty=0.01)

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "rest": ("rest_days", "back_to_back", "games_last_7", "road_games_last_7"),
    "travel": ("travel_miles", "travel_time_zones"),
    "continuity": ("roster_continuity", "expected_lineup_continuity"),
    "team_form": ("rolling_team_net_rating",),
    "player_value": ("rolling_player_value",),
    "schedule_strength": ("schedule_strength",),
}


@dataclass(frozen=True, slots=True)
class _RunConfig:
    excluded_families: frozenset[str]
    output_dir: Path
    skip_health: bool


def _excluded_names(config: _RunConfig) -> frozenset[str]:
    names: set[str] = set()
    for family in config.excluded_families:
        names.update(FEATURE_FAMILIES[family])
    return frozenset(names)


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


def main(argv: Sequence[str] | None = None) -> int:
    """Build every season, fit on training seasons, and evaluate the declared seasons."""
    config = _parse_arguments(argv)
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
    game_features: dict[int, GameFeatures] = {}
    for season, joined in joined_by_season.items():
        rows, season_features, season_notes = season_rows(season, joined, replay, injury_snapshots)
        rows_by_season[season] = rows
        game_features.update(season_features)
        notes.extend(season_notes)
    training = [row for season in TRAINING_SEASONS for row in rows_by_season[season]]
    evaluation = [row for season in EVALUATION_SEASONS for row in rows_by_season[season]]
    excluded = _excluded_names(config)
    if excluded:
        training = _mask_rows(training, excluded)
        evaluation = _mask_rows(evaluation, excluded)
    report = _evaluate(training, evaluation, game_features, notes, config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


def _mask_rows(
    rows: list[PrototypeGameRow],
    excluded: frozenset[str],
) -> list[PrototypeGameRow]:
    indices = [index for index, name in enumerate(NBA_RICH_FEATURE_NAMES) if name not in excluded]
    return [
        replace(
            row,
            features_standard=tuple(row.features_standard[index] for index in indices),
        )
        for row in rows
    ]


def _parse_arguments(argv: Sequence[str] | None) -> _RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exclude-families",
        default="",
        help="comma-separated feature families to drop: " + ",".join(sorted(FEATURE_FAMILIES)),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-health", action="store_true")
    namespace = parser.parse_args(argv)
    families = frozenset(family for family in str(namespace.exclude_families).split(",") if family)
    unknown = families - frozenset(FEATURE_FAMILIES)
    if unknown:
        raise RuntimeError(f"unknown feature families: {sorted(unknown)}")
    return _RunConfig(
        excluded_families=families,
        output_dir=namespace.output_dir,
        skip_health=bool(namespace.skip_health),
    )


def build_schedule(snapshots: list[InjurySnapshot]) -> list[ScheduleEntry]:
    """Build the injury-archive schedule minus excluded cup finals."""
    return [
        ScheduleEntry(
            game_date=day,
            away_abbreviation=away,
            home_abbreviation=home,
            tip_clock=clock,
        )
        for day, away, home, clock in schedule_from_injury_index(snapshots)
        if (day, away, home) not in EXCLUDED_CUP_FINALS
    ]


def load_season(
    season: int,
    path: Path,
    schedule: list[ScheduleEntry],
) -> tuple[list[SeasonGame], list[str]]:
    """Read one season of play-by-play and join it to the schedule."""
    failures: list[str] = []
    games = list(read_pbp_games(path, failures))
    season_schedule = [entry for entry in schedule if _entry_season(entry) == season]
    joined, join_notes = join_season_games(games, season_schedule)
    return joined, failures + join_notes


def _entry_season(entry: ScheduleEntry) -> int:
    return entry.game_date.year + 1 if entry.game_date.month >= 7 else entry.game_date.year


def elo_replay(
    joined_by_season: dict[int, list[SeasonGame]],
    notes: list[str],
) -> MovEloReplay:
    """Replay the carryover margin-of-victory Elo over warmup and model seasons."""
    sequences: list[list[EloGameResult]] = []
    for path in (WARMUP_FILES[season] for season in sorted(WARMUP_FILES)):
        failures: list[str] = []
        games = list(read_pbp_games(path, failures))
        notes.extend(failures)
        sequences.append([_elo_result_from_pbp(game) for game in games])
    sequences.extend(
        [_elo_result_from_season_game(game) for game in joined_by_season[season]]
        for season in sorted(joined_by_season)
    )
    return replay_mov_elo(sequences)


def _elo_result_from_pbp(game: PbpGame) -> EloGameResult:
    return EloGameResult(
        game_id=game.game_id,
        home_abbreviation=game.home_abbreviation,
        away_abbreviation=game.away_abbreviation,
        home_score=game.home_score,
        away_score=game.away_score,
        neutral=False,
    )


def _elo_result_from_season_game(game: SeasonGame) -> EloGameResult:
    return EloGameResult(
        game_id=game.game_id,
        home_abbreviation=game.home_abbreviation,
        away_abbreviation=game.away_abbreviation,
        home_score=game.home_score,
        away_score=game.away_score,
        neutral=is_neutral_site(game.game_date, game.away_abbreviation, game.home_abbreviation),
    )


def season_rows(
    season: int,
    joined: list[SeasonGame],
    replay: MovEloReplay,
    injury_snapshots: list[InjurySnapshot],
) -> tuple[list[PrototypeGameRow], dict[int, GameFeatures], list[str]]:
    """Build prototype rows for one season from RAPM priors and replayed Elo.

    Also returns the per-game ``GameFeatures`` keyed by game id so disclosed prototype
    variants outside the frozen schema (for example projected rotation) can be derived.
    """
    failures: list[str] = []
    rapm = fit_season_ratings_by_name(RAPM_PRIOR_FILES, season, failures=failures)
    features, notes = build_game_features(
        joined,
        replay.ratings,
        injury_snapshots,
        player_ratings=rapm,
    )
    rows = build_prototype_rows(joined, features, replay.home_probabilities)
    return rows, {entry.game_id: entry for entry in features}, failures + notes


def _evaluate(
    training: list[PrototypeGameRow],
    evaluation: list[PrototypeGameRow],
    game_features: Mapping[int, GameFeatures],
    notes: list[str],
    config: _RunConfig,
) -> dict[str, object]:
    excluded = _excluded_names(config)
    standard_names = tuple(name for name in NBA_RICH_FEATURE_NAMES if name not in excluded)
    health_names = standard_names + NBA_LOCAL_HEALTH_FEATURE_NAMES
    standard_model = _fit_variant(training, standard_names, include_health=False)
    health_training = [row for row in training if row.features_health is not None]
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
        "excluded_feature_families": sorted(config.excluded_families),
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
        "models": {},
    }
    variants: dict[str, object] = {}
    model_variants: list[tuple[str, _ScaledModel, list[PrototypeGameRow], tuple[str, ...]]] = [
        ("standard", standard_model, evaluation, standard_names)
    ]
    models_report: dict[str, object] = {"standard": _model_payload(standard_model, standard_names)}
    if not config.skip_health:
        health_model = _fit_variant(health_training, health_names, include_health=True)
        models_report["health"] = _model_payload(health_model, health_names)
        model_variants.append(("health", health_model, health_evaluation, health_names))
    if excluded:
        report["projected_variant"] = (
            "skipped: --exclude-families masks the frozen standard features, "
            "so the projected variant is disabled"
        )
    else:
        projected_names = (*standard_names, "projected_rotation_value")
        projected_training = [_projected_row(row, game_features) for row in training]
        projected_evaluation = [_projected_row(row, game_features) for row in evaluation]
        projected_model = _fit_variant(projected_training, projected_names, include_health=False)
        models_report["projected"] = _model_payload(projected_model, projected_names)
        model_variants.append(("projected", projected_model, projected_evaluation, projected_names))
    report["models"] = models_report
    for name, model, rows, _names in model_variants:
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


def _projected_row(
    row: PrototypeGameRow,
    game_features: Mapping[int, GameFeatures],
) -> PrototypeGameRow:
    """Append the projected-rotation difference (home minus away, 0.0 when unknown)."""
    projected = game_features[row.game_id].projected_rotation
    delta = 0.0 if projected is None else projected[1] - projected[0]
    return replace(row, features_standard=(*row.features_standard, delta))


def _fit_variant(
    rows: list[PrototypeGameRow],
    names: tuple[str, ...],
    *,
    include_health: bool,
) -> _ScaledModel:
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
    model = fit_elo_residual(residual_rows, names, FIT_CONFIG)
    return _ScaledModel(model, scales)


def _model_payload(model: _ScaledModel, names: tuple[str, ...]) -> dict[str, object]:
    return {
        "feature_names": list(names),
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
