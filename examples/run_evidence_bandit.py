"""Run the offline contextual bandit over four frozen NBA forecast arms.

Rebuilds the exact prototype rows with the public helpers from
``examples.run_private_prototype``, reconstructs each arm's home-win probabilities from
frozen manifests (raw MOV Elo; schedule-only logistic from ``data/processed/arm_schedule_only``;
standard-11 and projected-12 from ``data/processed/private_prototype``), fits the softmax
mixture selector on the 2022-2024 training rows, and evaluates on 2025-2026.

Disclosure: both evaluation seasons are OPENED (2025 corroborating, 2026 pristine but later
inspected), so every evaluation number here is diagnostic only; no new claim is made on them.
The verdict block is predeclared: ``selector_beats_best_static`` holds when the mixture's
pooled evaluation log loss beats the best static arm by more than 0.002.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
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

from forecastfm.elo_residual import probability_from_logit, probability_logit
from forecastfm.json_utils import (
    parse_json_object,
    require_float,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.nba_bandit import (
    BANDIT_CONTEXT_NAMES,
    MISSING_CONTEXT_POLICY,
    UNAVAILABLE_MINUTES_BUCKETS,
    UNAVAILABLE_MINUTES_CONTEXT_INDEX,
    VERDICT_MARGIN,
    BanditEvaluation,
    BanditGame,
    BanditSelector,
    SelectionWeightBucket,
    bandit_context,
    evaluate_bandit,
    fit_bandit_selector,
    selection_weight_distribution,
    unavailable_minutes_bucket,
)
from forecastfm.nba_feature_builder import GameFeatures, load_injury_index
from forecastfm.nba_prototype_dataset import PrototypeGameRow
from forecastfm.nba_rich import NBA_RICH_FEATURE_NAMES
from forecastfm.nba_season_games import SeasonGame
from forecastfm.outcome_v2_metrics import (
    BinaryForecast,
    DatedBinaryCohortMember,
    evaluate_multi_season,
)

PRIVATE_MANIFEST = Path("data/processed/private_prototype/manifest.json")
SCHEDULE_ONLY_MANIFEST = Path("data/processed/arm_schedule_only/manifest.json")
OUTPUT_DIR = Path("data/processed/evidence_bandit")
ARM_RAW_ELO = "raw_elo"
ARM_SCHEDULE_ONLY = "schedule_only"
ARM_STANDARD = "standard"
ARM_PROJECTED = "projected"
ARM_NAMES = (ARM_RAW_ELO, ARM_SCHEDULE_ONLY, ARM_STANDARD, ARM_PROJECTED)


class _FrozenArm:
    """One frozen Elo-offset logistic arm read from a prototype manifest."""

    def __init__(
        self,
        feature_names: tuple[str, ...],
        weights: tuple[float, ...],
        rms_scales: tuple[float, ...],
    ) -> None:
        self.feature_names = feature_names
        self.weights = weights
        self.rms_scales = rms_scales

    def probability(self, elo_probability: float, features: tuple[float, ...]) -> float:
        residual = sum(
            weight * (value / scale)
            for weight, value, scale in zip(self.weights, features, self.rms_scales, strict=True)
        )
        return probability_from_logit(probability_logit(elo_probability) + residual)


def main(argv: Sequence[str] | None = None) -> int:
    """Rebuild rows, fit the mixture selector, and write the bandit manifest."""
    namespace = _parse_arguments(argv)
    rows_by_season, game_features = _rebuild_rows()
    training_rows = [row for season in TRAINING_SEASONS for row in rows_by_season[season]]
    evaluation_rows = [row for season in EVALUATION_SEASONS for row in rows_by_season[season]]
    evaluation_rows = _drop_metric_incompatible_rows(evaluation_rows)
    arms = _load_arms(namespace.private_manifest, namespace.schedule_manifest)
    training = [_bandit_game(row, arms, game_features) for row in training_rows]
    evaluation = [_bandit_game(row, arms, game_features) for row in evaluation_rows]
    selector = fit_bandit_selector(training, ARM_NAMES)
    report = _report(selector, training, evaluation, evaluation_rows)
    namespace.output_dir.mkdir(parents=True, exist_ok=True)
    (namespace.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--private-manifest", type=Path, default=PRIVATE_MANIFEST)
    parser.add_argument("--schedule-manifest", type=Path, default=SCHEDULE_ONLY_MANIFEST)
    return parser.parse_args(argv)


def _rebuild_rows() -> tuple[dict[int, list[PrototypeGameRow]], dict[int, GameFeatures]]:
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
    return rows_by_season, game_features


def _drop_metric_incompatible_rows(rows: list[PrototypeGameRow]) -> list[PrototypeGameRow]:
    return [
        row
        for row in rows
        if (row.game_date.year + 1 if row.game_date.month >= 7 else row.game_date.year)
        == row.season
    ]


def _load_arms(
    private_manifest: Path,
    schedule_manifest: Path,
) -> dict[str, _FrozenArm]:
    schedule_only = _manifest_arm(schedule_manifest, "standard")
    if not set(schedule_only.feature_names) <= set(NBA_RICH_FEATURE_NAMES):
        raise RuntimeError("schedule-only arm names must be a subset of the standard schema")
    standard = _manifest_arm(private_manifest, "standard")
    if standard.feature_names != NBA_RICH_FEATURE_NAMES:
        raise RuntimeError("standard arm names differ from the frozen rich schema")
    projected = _manifest_arm(private_manifest, "projected")
    if projected.feature_names != (*NBA_RICH_FEATURE_NAMES, "projected_rotation_value"):
        raise RuntimeError("projected arm names differ from standard plus projected rotation")
    return {
        ARM_SCHEDULE_ONLY: schedule_only,
        ARM_STANDARD: standard,
        ARM_PROJECTED: projected,
    }


def _manifest_arm(path: Path, model_name: str) -> _FrozenArm:
    manifest = parse_json_object(path.read_text(encoding="utf-8"))
    models = require_object(required_field(manifest, "models"), "models")
    payload = require_object(required_field(models, model_name), f"models.{model_name}")
    names = tuple(
        require_string(item, "feature_names")
        for item in require_list(required_field(payload, "feature_names"), "feature_names")
    )
    weights = tuple(
        require_float(item, "weights")
        for item in require_list(required_field(payload, "weights"), "weights")
    )
    scales = tuple(
        require_float(item, "rms_scales")
        for item in require_list(required_field(payload, "rms_scales"), "rms_scales")
    )
    if not (len(names) == len(weights) == len(scales)):
        raise RuntimeError(f"arm {model_name} in {path} has misaligned payload widths")
    return _FrozenArm(names, weights, scales)


def _projected_delta(row: PrototypeGameRow, game_features: Mapping[int, GameFeatures]) -> float:
    projected = game_features[row.game_id].projected_rotation
    return 0.0 if projected is None else projected[1] - projected[0]


def _arm_probabilities(
    row: PrototypeGameRow,
    arms: Mapping[str, _FrozenArm],
    game_features: Mapping[int, GameFeatures],
) -> tuple[float, ...]:
    standard_indices = {name: index for index, name in enumerate(NBA_RICH_FEATURE_NAMES)}
    schedule_features = tuple(
        row.features_standard[standard_indices[name]]
        for name in arms[ARM_SCHEDULE_ONLY].feature_names
    )
    projected_features = (*row.features_standard, _projected_delta(row, game_features))
    return (
        row.elo_home_probability,
        arms[ARM_SCHEDULE_ONLY].probability(row.elo_home_probability, schedule_features),
        arms[ARM_STANDARD].probability(row.elo_home_probability, row.features_standard),
        arms[ARM_PROJECTED].probability(row.elo_home_probability, projected_features),
    )


def _bandit_game(
    row: PrototypeGameRow,
    arms: Mapping[str, _FrozenArm],
    game_features: Mapping[int, GameFeatures],
) -> BanditGame:
    context = bandit_context(
        elo_probability=row.elo_home_probability,
        rest_days_difference=row.features_standard[0],
        travel_miles_difference=row.features_standard[4],
        unavailable_rotation_minutes_difference=(
            None if row.features_health is None else row.features_health[0]
        ),
        projected_rotation_difference=_projected_delta(row, game_features)
        if game_features[row.game_id].projected_rotation is not None
        else None,
    )
    return BanditGame(
        question_id=row.question_id,
        season=row.season,
        context=context,
        arm_probabilities=_arm_probabilities(row, arms, game_features),
        outcome=1 if row.home_won else 0,
    )


def _report(
    selector: BanditSelector,
    training: list[BanditGame],
    evaluation: list[BanditGame],
    evaluation_rows: list[PrototypeGameRow],
) -> dict[str, object]:
    training_eval = evaluate_bandit(selector, training)
    evaluation_eval = evaluate_bandit(selector, evaluation)
    by_season = {
        season: evaluate_bandit(selector, [game for game in evaluation if game.season == season])
        for season in EVALUATION_SEASONS
    }
    verdict = _verdict(evaluation_eval, by_season)
    best_static = verdict["best_static_arm"]
    return {
        "training_seasons": list(TRAINING_SEASONS),
        "evaluation_seasons": list(EVALUATION_SEASONS),
        "opened_evaluation_seasons": list(OPENED_EVALUATION_SEASONS),
        "disclosure": (
            "Both evaluation seasons are opened; all evaluation numbers are diagnostics only "
            "and support no new claim."
        ),
        "missing_context_policy": MISSING_CONTEXT_POLICY,
        "context_names": list(BANDIT_CONTEXT_NAMES),
        "arm_names": list(ARM_NAMES),
        "arm_sources": {
            ARM_RAW_ELO: "row.elo_home_probability (replayed carryover MOV Elo, no model)",
            ARM_SCHEDULE_ONLY: str(SCHEDULE_ONLY_MANIFEST) + " models.standard",
            ARM_STANDARD: str(PRIVATE_MANIFEST) + " models.standard",
            ARM_PROJECTED: str(PRIVATE_MANIFEST) + " models.projected",
        },
        "selector": _selector_payload(selector),
        "training": _evaluation_payload(training_eval),
        "training_by_season": {
            str(season): _evaluation_payload(
                evaluate_bandit(selector, [game for game in training if game.season == season])
            )
            for season in TRAINING_SEASONS
        },
        "evaluation": _evaluation_payload(evaluation_eval),
        "evaluation_by_season": {
            str(season): _evaluation_payload(season_eval)
            for season, season_eval in by_season.items()
        },
        "selection_weight_buckets": {
            "bucketed_by": BANDIT_CONTEXT_NAMES[UNAVAILABLE_MINUTES_CONTEXT_INDEX],
            "buckets": list(UNAVAILABLE_MINUTES_BUCKETS),
            "training": _bucket_payloads(selector, training),
            "evaluation": _bucket_payloads(selector, evaluation),
        },
        "gates": {
            "mixture_vs_raw_elo": _gate_payload(selector, evaluation, evaluation_rows, 0),
            "mixture_vs_best_static_arm": _gate_payload(
                selector, evaluation, evaluation_rows, ARM_NAMES.index(str(best_static))
            ),
        },
        "verdict": verdict,
    }


def _selector_payload(selector: BanditSelector) -> dict[str, object]:
    return {
        "scales": list(selector.scales),
        "theta": [list(row) for row in selector.theta],
    }


def _evaluation_payload(evaluation: BanditEvaluation) -> dict[str, object]:
    return {
        "game_count": evaluation.game_count,
        "arm_log_losses": dict(evaluation.arm_log_losses),
        "oracle_log_loss": evaluation.oracle_log_loss,
        "mixture_log_loss": evaluation.mixture_log_loss,
    }


def _bucket_payloads(
    selector: BanditSelector,
    games: list[BanditGame],
) -> list[dict[str, object]]:
    buckets = selection_weight_distribution(
        selector,
        games,
        lambda game: unavailable_minutes_bucket(game.context[UNAVAILABLE_MINUTES_CONTEXT_INDEX]),
    )
    return [_bucket_payload(bucket) for bucket in buckets]


def _bucket_payload(bucket: SelectionWeightBucket) -> dict[str, object]:
    return {
        "bucket": bucket.bucket,
        "game_count": bucket.game_count,
        "mean_weights": dict(bucket.mean_weights),
    }


def _verdict(
    evaluation: BanditEvaluation,
    by_season: Mapping[int, BanditEvaluation],
) -> dict[str, object]:
    best_arm, best_loss = min(evaluation.arm_log_losses, key=lambda pair: pair[1])
    per_season = {
        str(season): {
            "best_static_log_loss": dict(season_eval.arm_log_losses)[best_arm],
            "mixture_log_loss": season_eval.mixture_log_loss,
        }
        for season, season_eval in by_season.items()
    }
    return {
        "rule": (
            "selector_beats_best_static := mixture pooled evaluation log loss < best static "
            f"arm pooled evaluation log loss - {VERDICT_MARGIN}"
        ),
        "margin": VERDICT_MARGIN,
        "best_static_arm": best_arm,
        "pooled": {
            "best_static_log_loss": best_loss,
            "mixture_log_loss": evaluation.mixture_log_loss,
            "advantage": best_loss - evaluation.mixture_log_loss,
        },
        "by_season": per_season,
        "selector_beats_best_static": evaluation.mixture_log_loss < best_loss - VERDICT_MARGIN,
    }


def _gate_payload(
    selector: BanditSelector,
    evaluation: list[BanditGame],
    evaluation_rows: list[PrototypeGameRow],
    baseline_arm_index: int,
) -> dict[str, object]:
    rows_by_id = {row.question_id: row for row in evaluation_rows}
    forecasts = [
        BinaryForecast(
            game.question_id,
            selector.forecast(game.context, game.arm_probabilities),
        )
        for game in evaluation
    ]
    cohort = [
        DatedBinaryCohortMember(
            question_id=game.question_id,
            season=game.season,
            game_date=rows_by_id[game.question_id].game_date,
            realized_team_win=bool(game.outcome),
            baseline_team_probability=game.arm_probabilities[baseline_arm_index],
        )
        for game in evaluation
    ]
    gate = evaluate_multi_season(forecasts, cohort, EVALUATION_SEASONS)
    payload = asdict(gate)
    payload.pop("declared_seasons", None)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
