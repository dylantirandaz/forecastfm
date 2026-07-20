"""T-15 forecast-horizon diagnostic: how much of the model-vs-market gap is horizon.

Only 2025-26 games from 2025-12-22 onward are in scope. For each game and each horizon H in
(60, 30, 15) minutes the diagnostic selects the latest injury-report snapshot at or before
tipoff minus H that still contains the matchup (the ``_pregame_report`` rule with a
parameterized cutoff) and recomputes the two health aggregates plus the projected-rotation
value from that snapshot with the identical math as ``build_game_features`` (binary
Out/Doubtful rule, RAPM-priced median expected minutes). One Elo-offset logistic model per
horizon is fitted on the standard 11 features (horizon-independent, computed once) plus the
horizon-specific snapshot triple, trained on 2025-12-22 through 2026-02-15 and tested on
2026-02-16 through 2026-04-12 with training-only RMS scales and L2 0.01. The predeclared
verdict is whether T-15 closes at least half of the T-60-to-market log-loss gap on the
identical test cohort. Everything runs from local retained data; nothing is uploaded.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, timedelta
from math import log
from pathlib import Path

from examples.run_private_prototype import (
    FIT_CONFIG,
    INJURY_ARCHIVE,
    RAPM_PRIOR_FILES,
    SEASON_FILES,
    build_schedule,
    elo_replay,
    load_season,
)

from forecastfm.elo_residual import EloResidualModel, EloResidualRow, fit_elo_residual
from forecastfm.nba_feature_builder import (
    GameFeatures,
    InjurySnapshot,
    _HealthContext,
    _side_health,
    _side_projected_rotation,
    build_game_features,
    load_injury_index,
)
from forecastfm.nba_injury_report import matchup_teams
from forecastfm.nba_prototype_dataset import PrototypeGameRow, build_prototype_rows
from forecastfm.nba_rapm import fit_season_ratings_by_name
from forecastfm.nba_rich import NBA_LOCAL_HEALTH_FEATURE_NAMES, NBA_RICH_FEATURE_NAMES
from forecastfm.nba_season_games import SeasonGame
from forecastfm.nba_team_history import GameContext, NbaTeamHistory

SEASON = 2026
SCOPE_START = date(2025, 12, 22)
TRAIN_END = date(2026, 2, 15)
TEST_START = date(2026, 2, 16)
TEST_END = date(2026, 4, 12)
HORIZONS_MINUTES = (60, 30, 15)
MARKET_LOG_LOSS = 0.57106
T60_STANDARD_LOG_LOSS = 0.60672
T60_PROJECTED_LOG_LOSS = 0.60390
OUTPUT_DIR = Path("data/processed/t15_diagnostic")

FEATURE_NAMES = (
    NBA_RICH_FEATURE_NAMES + NBA_LOCAL_HEALTH_FEATURE_NAMES + ("projected_rotation_value",)
)
CADENCE_NOTE = (
    "the retained archive in scope holds one report every two hours at :30 ET (11:30 "
    "through 21:30), not the anticipated 15-minute cadence, so T-15 and T-30 select the "
    "same snapshot for standard :00/:30 tipoffs and T-60 differs only for tipoffs in the "
    "hour immediately after a report"
)
CONSTANTS_COMMENT = (
    "the market constant covers a 971-game 2026 pickcenter subset and the T-60 constants "
    "cover full-season 2026 models trained on 2022-2026, so all three constants describe "
    "different cohorts than this diagnostic and are context only; the verdict compares "
    "horizons fitted and tested on identical cohorts"
)

type PregameReport = tuple[tuple[tuple[float, float], tuple[float, float]], tuple[float, float]]


@dataclass(frozen=True, slots=True)
class HorizonInputs:
    """Shared inputs for horizon-parameterized snapshot feature derivation."""

    snapshots: tuple[InjurySnapshot, ...]
    player_ratings: Mapping[str, float]
    notes: list[str]


@dataclass(frozen=True, slots=True)
class HorizonModel:
    """One fitted horizon model with its frozen training-only RMS scales."""

    model: EloResidualModel
    scales: tuple[float, ...]

    def probability(self, row: PrototypeGameRow, features: tuple[float, ...]) -> float:
        """Predict the home win probability for one row and horizon feature vector."""
        scaled = tuple(value / scale for value, scale in zip(features, self.scales, strict=True))
        return self.model.predict_probability(row.elo_home_probability, scaled)


@dataclass(frozen=True, slots=True)
class HorizonResult:
    """One horizon's fitted model and train/test log losses."""

    horizon_minutes: int
    train_log_loss: float
    test_log_loss: float
    missing_snapshot_games: int
    model: HorizonModel


@dataclass(frozen=True, slots=True)
class _ScopeData:
    rows: list[PrototypeGameRow]
    game_features: dict[int, GameFeatures]
    derived: dict[int, dict[int, PregameReport | None]]
    notes: list[str]


def select_snapshot(
    snapshots: Sequence[InjurySnapshot],
    game: SeasonGame,
    horizon_minutes: int,
) -> InjurySnapshot | None:
    """Select the latest snapshot at or before tipoff minus the horizon containing the game.

    This is the ``_pregame_report`` selection rule with the T-60 cutoff replaced by a
    parameterized horizon: report time at or before the cutoff, and at least one row for
    the game's own date and matchup, so a later report that no longer lists the game falls
    back to an earlier one that does.
    """
    cutoff = game.tipoff - timedelta(minutes=horizon_minutes)
    teams = (game.away_abbreviation, game.home_abbreviation)
    return next(
        (
            snapshot
            for snapshot in reversed(snapshots)
            if snapshot.report_time.astimezone(UTC) <= cutoff
            and any(
                row.game_date == game.game_date and matchup_teams(row.matchup) == teams
                for row in snapshot.rows
            )
        ),
        None,
    )


def pregame_report_at_horizon(
    game: SeasonGame,
    inputs: HorizonInputs,
    away_history: NbaTeamHistory,
    home_history: NbaTeamHistory,
    horizon_minutes: int,
) -> PregameReport | None:
    """Compute horizon-specific health and projected-rotation aggregates for one game.

    Identical math to ``_pregame_report`` (binary Out/Doubtful rule, RAPM-priced median
    expected minutes, projected rotation value) at a parameterized snapshot horizon.
    """
    selected = select_snapshot(inputs.snapshots, game, horizon_minutes)
    if selected is None:
        inputs.notes.append(
            f"game {game.game_id} has no report snapshot at or before its "
            f"T-{horizon_minutes} cutoff"
        )
        return None
    context = _HealthContext(list(inputs.snapshots), inputs.notes, inputs.player_ratings)
    away = _side_health(selected.rows, game.away_abbreviation, away_history, game, context)
    home = _side_health(selected.rows, game.home_abbreviation, home_history, game, context)
    projected = (
        _side_projected_rotation(
            selected.rows, game.away_abbreviation, away_history, game, context
        ),
        _side_projected_rotation(
            selected.rows, game.home_abbreviation, home_history, game, context
        ),
    )
    return (away, home), projected


def compute_horizon_features(
    games: Sequence[SeasonGame],
    elo_ratings: Mapping[tuple[int, str], float],
    inputs: HorizonInputs,
) -> dict[int, dict[int, PregameReport | None]]:
    """Re-run the per-team history state over one season and derive per-horizon features.

    Mirrors ``build_game_features``' strictly-pregame ordering exactly (derive from history,
    then record the game), so the T-60 derivation reproduces the pipeline's health and
    projected aggregates; the caller cross-checks that identity. Only games on or after
    ``SCOPE_START`` are derived.
    """
    histories: dict[str, NbaTeamHistory] = {}
    derived: dict[int, dict[int, PregameReport | None]] = {}
    for game in games:
        away_history = histories.setdefault(
            game.away_abbreviation, NbaTeamHistory(game.away_abbreviation)
        )
        home_history = histories.setdefault(
            game.home_abbreviation, NbaTeamHistory(game.home_abbreviation)
        )
        if game.game_date >= SCOPE_START:
            derived[game.game_id] = {
                horizon: pregame_report_at_horizon(
                    game, inputs, away_history, home_history, horizon
                )
                for horizon in HORIZONS_MINUTES
            }
        away_context = GameContext(game.game_date, game.tipoff, False, game.arena)
        home_context = GameContext(game.game_date, game.tipoff, True, game.arena)
        away_elo = elo_ratings[(game.game_id, game.away_abbreviation)]
        home_elo = elo_ratings[(game.game_id, game.home_abbreviation)]
        away_history.record_game(game.pbp, away_context, home_elo)
        home_history.record_game(game.pbp, home_context, away_elo)
    return derived


def horizon_feature_vector(
    row: PrototypeGameRow,
    report: PregameReport | None,
) -> tuple[float, ...]:
    """Assemble the horizon vector: standard 11 plus the snapshot-derived triple.

    The health aggregates and the projected-rotation difference are home minus away; games
    with no snapshot at the horizon contribute zeros, the same zero-fill policy as the
    pipeline's projected variant.
    """
    if report is None:
        return (*row.features_standard, 0.0, 0.0, 0.0)
    (away, home), projected = report
    return (
        *row.features_standard,
        home[0] - away[0],
        home[1] - away[1],
        projected[1] - projected[0],
    )


def rms_scales(vectors: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
    """Compute uncentered RMS scales over training vectors only (1.0 for zero columns)."""
    width = len(vectors[0])
    scales: list[float] = []
    for index in range(width):
        mean_square = sum(vector[index] ** 2 for vector in vectors) / len(vectors)
        scales.append(mean_square**0.5 if mean_square > 0.0 else 1.0)
    return tuple(scales)


def fit_horizon_model(
    rows: Sequence[PrototypeGameRow],
    reports: Mapping[int, PregameReport | None],
) -> HorizonModel:
    """Fit one horizon's Elo-offset logistic model on the given training rows."""
    vectors = [horizon_feature_vector(row, reports[row.game_id]) for row in rows]
    scales = rms_scales(vectors)
    residual_rows = [
        EloResidualRow(
            question_id=row.question_id,
            elo_probability=row.elo_home_probability,
            features=tuple(value / scale for value, scale in zip(vector, scales, strict=True)),
            outcome=1 if row.home_won else 0,
        )
        for row, vector in zip(rows, vectors, strict=True)
    ]
    return HorizonModel(fit_elo_residual(residual_rows, FEATURE_NAMES, FIT_CONFIG), scales)


def mean_log_loss(
    model: HorizonModel,
    rows: Sequence[PrototypeGameRow],
    reports: Mapping[int, PregameReport | None],
) -> float:
    """Compute mean binary log loss of one horizon model over the given rows."""
    total = 0.0
    for row in rows:
        probability = model.probability(row, horizon_feature_vector(row, reports[row.game_id]))
        total += log(probability) if row.home_won else log(1.0 - probability)
    return -total / len(rows)


def main() -> int:
    """Build the diagnostic, write the manifest, and print the verdict."""
    report = run_diagnostic()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    return 0


def run_diagnostic() -> dict[str, object]:
    """Run the full horizon diagnostic and return the manifest payload."""
    scope = _load_scope()
    train_rows, test_rows, dropped = _split_rows(scope.rows)
    results = [
        _evaluate_horizon(horizon, train_rows, test_rows, scope.derived)
        for horizon in HORIZONS_MINUTES
    ]
    return _manifest(scope, train_rows, test_rows, dropped, results)


def _load_scope() -> _ScopeData:
    notes: list[str] = []
    snapshots = load_injury_index(INJURY_ARCHIVE)
    schedule = build_schedule(snapshots)
    joined_by_season: dict[int, list[SeasonGame]] = {}
    for season, path in SEASON_FILES.items():
        joined, season_notes = load_season(season, path, schedule)
        joined_by_season[season] = joined
        notes.extend(season_notes)
    replay = elo_replay(joined_by_season, notes)
    rapm = fit_season_ratings_by_name(RAPM_PRIOR_FILES, SEASON, failures=notes)
    games = joined_by_season[SEASON]
    features, builder_notes = build_game_features(
        games, replay.ratings, snapshots, player_ratings=rapm
    )
    notes.extend(builder_notes)
    rows = build_prototype_rows(games, features, replay.home_probabilities)
    inputs = HorizonInputs(snapshots=tuple(snapshots), player_ratings=rapm, notes=notes)
    derived = compute_horizon_features(games, replay.ratings, inputs)
    return _ScopeData(
        rows=rows,
        game_features={entry.game_id: entry for entry in features},
        derived=derived,
        notes=notes,
    )


def _split_rows(
    rows: list[PrototypeGameRow],
) -> tuple[list[PrototypeGameRow], list[PrototypeGameRow], int]:
    scoped = [row for row in rows if row.game_date >= SCOPE_START]
    train = [row for row in scoped if row.game_date <= TRAIN_END]
    test = [row for row in scoped if TEST_START <= row.game_date <= TEST_END]
    return train, test, len(scoped) - len(train) - len(test)


def _evaluate_horizon(
    horizon: int,
    train_rows: list[PrototypeGameRow],
    test_rows: list[PrototypeGameRow],
    derived: Mapping[int, Mapping[int, PregameReport | None]],
) -> HorizonResult:
    reports = {game_id: horizons[horizon] for game_id, horizons in derived.items()}
    model = fit_horizon_model(train_rows, reports)
    return HorizonResult(
        horizon_minutes=horizon,
        train_log_loss=mean_log_loss(model, train_rows, reports),
        test_log_loss=mean_log_loss(model, test_rows, reports),
        missing_snapshot_games=sum(1 for report in reports.values() if report is None),
        model=model,
    )


def _report_values(
    health: tuple[tuple[float, float], tuple[float, float]] | None,
    projected: tuple[float, float] | None,
) -> tuple[float, ...] | None:
    if health is None or projected is None:
        return None
    return (*health[0], *health[1], *projected)


def _t60_consistency(
    game_features: Mapping[int, GameFeatures],
    derived: Mapping[int, Mapping[int, PregameReport | None]],
) -> dict[str, object]:
    """Cross-check that the T-60 re-derivation reproduces the pipeline's T-60 aggregates."""
    checked = 0
    mismatches: list[int] = []
    max_abs_diff = 0.0
    for game_id, horizons in derived.items():
        report = horizons[60]
        expected = game_features[game_id]
        expected_values = _report_values(expected.health, expected.projected_rotation)
        actual_values = None if report is None else _report_values(report[0], report[1])
        checked += 1
        if expected_values != actual_values:
            if expected_values is None or actual_values is None:
                mismatches.append(game_id)
                continue
            diffs = [
                abs(actual - reference)
                for actual, reference in zip(actual_values, expected_values, strict=True)
            ]
            max_abs_diff = max([max_abs_diff, *diffs])
            mismatches.append(game_id)
    return {
        "checked_games": checked,
        "exact_matches": checked - len(mismatches),
        "mismatch_game_ids": mismatches,
        "max_abs_diff": max_abs_diff,
    }


def _manifest(
    scope: _ScopeData,
    train_rows: list[PrototypeGameRow],
    test_rows: list[PrototypeGameRow],
    dropped: int,
    results: list[HorizonResult],
) -> dict[str, object]:
    models = {f"t{result.horizon_minutes}": _model_payload(result) for result in results}
    by_horizon = {result.horizon_minutes: result for result in results}
    return {
        "schema": "forecastfm.t15_diagnostic/v1",
        "purpose": (
            "measure how much of the model-vs-market log-loss gap is pure forecast "
            "horizon, by refitting the projected feature set with injury snapshots "
            "selected at T-60, T-30, and T-15 on identical cohorts"
        ),
        "season": SEASON,
        "scope": {
            "start_date": SCOPE_START.isoformat(),
            "end_date": TEST_END.isoformat(),
            "train_window": {
                "start": SCOPE_START.isoformat(),
                "end": TRAIN_END.isoformat(),
                "games": len(train_rows),
            },
            "test_window": {
                "start": TEST_START.isoformat(),
                "end": TEST_END.isoformat(),
                "games": len(test_rows),
            },
            "games_after_end_date_dropped": dropped,
            "archive_cadence_note": CADENCE_NOTE,
        },
        "horizons_minutes": list(HORIZONS_MINUTES),
        "feature_names": list(FEATURE_NAMES),
        "feature_note": (
            "11 standard features (horizon-independent, computed once) plus the "
            "horizon-specific unavailable_rotation_minutes and "
            "unavailable_rotation_value aggregates and the projected_rotation_value "
            "difference, each home minus away and zero-filled when no snapshot exists "
            "at the horizon"
        ),
        "t60_consistency": _t60_consistency(scope.game_features, scope.derived),
        "models": models,
        "comparison_constants": {
            "market_log_loss": MARKET_LOG_LOSS,
            "t60_standard_log_loss": T60_STANDARD_LOG_LOSS,
            "t60_projected_log_loss": T60_PROJECTED_LOG_LOSS,
            "comment": CONSTANTS_COMMENT,
        },
        "verdict": _verdict(by_horizon[60].test_log_loss, by_horizon[15].test_log_loss),
        "notes_count": len(scope.notes),
    }


def _model_payload(result: HorizonResult) -> dict[str, object]:
    return {
        "horizon_minutes": result.horizon_minutes,
        "train_log_loss": result.train_log_loss,
        "test_log_loss": result.test_log_loss,
        "missing_snapshot_games": result.missing_snapshot_games,
        "feature_names": list(FEATURE_NAMES),
        "weights": list(result.model.model.weights),
        "rms_scales": list(result.model.scales),
        "fit_config": {
            "steps": FIT_CONFIG.steps,
            "learning_rate": FIT_CONFIG.learning_rate,
            "l2_penalty": FIT_CONFIG.l2_penalty,
        },
    }


def _verdict(t60_test_log_loss: float, t15_test_log_loss: float) -> dict[str, object]:
    """Evaluate the predeclared half-gap rule on identical train/test cohorts."""
    actual_improvement = t60_test_log_loss - t15_test_log_loss
    required_improvement = 0.5 * (t60_test_log_loss - MARKET_LOG_LOSS)
    return {
        "t15_closes_half_gap": actual_improvement >= required_improvement,
        "t60_projected_test_ll": t60_test_log_loss,
        "t15_test_ll": t15_test_log_loss,
        "market_log_loss": MARKET_LOG_LOSS,
        "actual_improvement": actual_improvement,
        "required_improvement": required_improvement,
        "rule": (
            "t15_closes_half_gap = (t60_projected_test_ll - t15_test_ll) >= "
            "0.5 * (t60_projected_test_ll - 0.57106)"
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
