"""Tests for the sealed, dependency-free rich NBA baseline."""

import json
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from inspect import signature
from pathlib import Path

import pytest

from forecastfm.elo_residual import EloResidualFitConfig
from forecastfm.integrity import bytes_sha256, canonical_json
from forecastfm.nba_feature_rows import NbaRichFeatureRow
from forecastfm.nba_resolutions import NbaResolution
from forecastfm.nba_rich import NBA_RICH_FEATURE_NAMES, NbaRichFeatures
from forecastfm.nba_rich_baseline import (
    MAXIMUM_SIDE_SWAP_GAP,
    NbaRichBaselineError,
    NbaRichBaselineForecastLock,
    NbaRichBaselineModel,
    build_nba_rich_baseline_forecast_lock,
    fit_nba_rich_baseline,
    predict_nba_rich_baseline,
    read_nba_rich_baseline_forecast_lock,
    read_nba_rich_baseline_model,
    write_nba_rich_baseline_forecast_lock,
    write_nba_rich_baseline_model,
)

FIT_CONFIG = EloResidualFitConfig(steps=200, learning_rate=0.2, l2_penalty=0.01)


def _row(
    question_id: str,
    season: int,
    day: int,
    signal: float,
) -> NbaRichFeatureRow:
    cutoff = datetime(season - 1, 10, day, 22, tzinfo=UTC)
    features = NbaRichFeatures.from_vector((signal, *(0.0 for _ in range(10))))
    return NbaRichFeatureRow(
        question_id=question_id,
        source_game_id=f"source-{question_id}",
        team_id=f"team-{question_id}",
        opponent_id=f"opponent-{question_id}",
        site="neutral",
        season=season,
        forecast_cutoff=cutoff,
        scheduled_tipoff=cutoff + timedelta(hours=1),
        elo_team_win_probability=0.5,
        elo_opponent_win_probability=0.5,
        elo_available_at=cutoff - timedelta(hours=1),
        elo_state_sha256=bytes_sha256(f"elo:{question_id}".encode()),
        rich_features=features,
        evidence_bundle_sha256=bytes_sha256(f"evidence:{question_id}".encode()),
        input_available_at=cutoff - timedelta(minutes=30),
    )


def _resolution(row: NbaRichFeatureRow, *, team_won: bool) -> NbaResolution:
    return NbaResolution(
        question_id=row.question_id,
        source_game_id=row.source_game_id,
        team_id=row.team_id,
        opponent_id=row.opponent_id,
        site=row.site,
        team_score=110 if team_won else 100,
        opponent_score=100 if team_won else 110,
        resolved_at=row.scheduled_tipoff + timedelta(hours=3),
        source_id=f"final:{row.question_id}",
        snapshot_metadata_sha256=bytes_sha256(f"final:{row.question_id}".encode()),
    )


def _training_data() -> tuple[tuple[NbaRichFeatureRow, ...], tuple[NbaResolution, ...]]:
    rows = (
        _row("train-positive-1", 2025, 1, 1.0),
        _row("train-negative-1", 2025, 2, -1.0),
        _row("train-positive-2", 2025, 3, 2.0),
        _row("train-negative-2", 2025, 4, -2.0),
    )
    resolutions = tuple(
        _resolution(row, team_won=row.rich_features.vector[0] > 0.0) for row in rows
    )
    return rows, resolutions


def _model() -> NbaRichBaselineModel:
    rows, resolutions = _training_data()
    return fit_nba_rich_baseline(rows, resolutions, FIT_CONFIG)


def _evaluation_rows() -> tuple[NbaRichFeatureRow, ...]:
    return (
        _row("eval-positive", 2026, 1, 0.5),
        _row("eval-negative", 2026, 2, -0.5),
    )


def test_fit_uses_resolved_winners_and_audits_train_only_scaling() -> None:
    rows, resolutions = _training_data()

    model = fit_nba_rich_baseline(rows, resolutions, FIT_CONFIG)
    reversed_answers = tuple(
        _resolution(row, team_won=not resolution.team_won)
        for row, resolution in zip(rows, resolutions, strict=True)
    )
    reversed_model = fit_nba_rich_baseline(rows, reversed_answers, FIT_CONFIG)

    assert model.weights[0] > 0.0
    assert reversed_model.weights[0] < 0.0
    assert model.rms_scales[0] == pytest.approx(5.0**0.5 / 2.0**0.5)
    assert model.zero_rms_feature_names == NBA_RICH_FEATURE_NAMES[1:]
    assert model.rms_scales[1:] == (1.0,) * 10
    assert model.training_row_count == 4
    assert model.training_seasons == (2025,)
    assert (
        model.training_feature_rows_jsonl_sha256
        == reversed_model.training_feature_rows_jsonl_sha256
    )
    assert (
        model.training_resolutions_jsonl_sha256 != reversed_model.training_resolutions_jsonl_sha256
    )
    assert model.model_sha256 != reversed_model.model_sha256


def test_fit_requires_exact_chronological_id_and_causal_answer_alignment() -> None:
    rows, resolutions = _training_data()

    with pytest.raises(NbaRichBaselineError, match="chronological"):
        fit_nba_rich_baseline(tuple(reversed(rows)), tuple(reversed(resolutions)), FIT_CONFIG)
    with pytest.raises(NbaRichBaselineError, match="IDs or order"):
        fit_nba_rich_baseline(rows, (resolutions[1], resolutions[0], *resolutions[2:]), FIT_CONFIG)
    with pytest.raises(NbaRichBaselineError, match="requires one resolution"):
        fit_nba_rich_baseline(rows, resolutions[:-1], FIT_CONFIG)

    wrong_identity = replace(resolutions[0], opponent_id="wrong-opponent")
    with pytest.raises(NbaRichBaselineError, match="identity differs"):
        fit_nba_rich_baseline(rows, (wrong_identity, *resolutions[1:]), FIT_CONFIG)

    early = replace(resolutions[0], resolved_at=rows[0].scheduled_tipoff)
    with pytest.raises(NbaRichBaselineError, match="postdate"):
        fit_nba_rich_baseline(rows, (early, *resolutions[1:]), FIT_CONFIG)

    wrong_season = replace(rows[0], season=2023)
    with pytest.raises(NbaRichBaselineError, match="season disagrees"):
        fit_nba_rich_baseline((wrong_season, *rows[1:]), resolutions, FIT_CONFIG)


def test_prediction_is_answer_free_later_season_and_side_swap_symmetric() -> None:
    model = _model()
    evaluation_rows = _evaluation_rows()

    forecasts = predict_nba_rich_baseline(model, evaluation_rows)

    assert tuple(signature(predict_nba_rich_baseline).parameters) == (
        "model",
        "evaluation_rows",
    )
    assert tuple(forecast.question_id for forecast in forecasts) == (
        "eval-positive",
        "eval-negative",
    )
    assert forecasts[0].team_probability is not None
    assert forecasts[1].team_probability is not None
    assert forecasts[0].team_probability > 0.5
    assert forecasts[1].team_probability < 0.5

    original = model.predict_probability(evaluation_rows[0])
    swapped = model.predict_probability(evaluation_rows[0].side_swap())
    assert abs(original - (1.0 - swapped)) <= MAXIMUM_SIDE_SWAP_GAP
    assert forecasts[0].team_probability == pytest.approx((original + 1.0 - swapped) / 2.0)

    same_season = (_row("same-season", 2025, 5, 0.5),)
    with pytest.raises(NbaRichBaselineError, match="strictly later"):
        predict_nba_rich_baseline(model, same_season)


def test_prediction_rejects_a_material_side_swap_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    row = _evaluation_rows()[0]
    model_type = type(model)
    original_predict = model_type.predict_probability

    def asymmetric_predict(self: object, feature_row: NbaRichFeatureRow) -> float:
        probability = original_predict(model, feature_row)
        if feature_row.question_id.endswith("-side-swap"):
            return probability + 1e-4
        return probability

    monkeypatch.setattr(model_type, "predict_probability", asymmetric_predict)

    with pytest.raises(NbaRichBaselineError, match="side-swap gap"):
        predict_nba_rich_baseline(model, (row,))


def test_model_io_is_canonical_create_only_and_detects_tampering(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / "baseline.json"

    digest = write_nba_rich_baseline_model(path, model)

    assert digest == model.model_sha256
    assert path.read_bytes() == model.canonical_bytes
    assert read_nba_rich_baseline_model(path, digest) == model
    with pytest.raises(NbaRichBaselineError, match="already exists"):
        write_nba_rich_baseline_model(path, model)
    with pytest.raises(FrozenInstanceError):
        model.weights = (0.0,) * len(model.weights)  # type: ignore[misc]

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["weights"][0] = payload["weights"][0] + 0.1
    path.write_text(canonical_json(payload), encoding="utf-8")
    with pytest.raises(NbaRichBaselineError, match="digest changed"):
        read_nba_rich_baseline_model(path, digest)

    path.write_bytes(model.canonical_bytes + b"\n")
    with pytest.raises(NbaRichBaselineError, match="canonical JSON bytes"):
        read_nba_rich_baseline_model(path)


def test_forecast_lock_binds_deterministic_answer_free_inference_and_io(
    tmp_path: Path,
) -> None:
    model = _model()
    rows = _evaluation_rows()
    forecasts = predict_nba_rich_baseline(model, rows)

    lock = build_nba_rich_baseline_forecast_lock(model, rows, forecasts)

    assert lock.model_sha256 == model.model_sha256
    assert lock.training_seasons == (2025,)
    assert lock.evaluation_seasons == (2026,)
    assert lock.forecast_count == 2
    assert len(lock.evaluation_feature_rows_jsonl_sha256) == 64
    assert len(lock.evaluation_question_ids_sha256) == 64
    assert len(lock.forecast_jsonl_sha256) == 64

    changed = replace(forecasts[0], team_probability=0.51)
    with pytest.raises(NbaRichBaselineError, match="deterministic model inference"):
        build_nba_rich_baseline_forecast_lock(model, rows, (changed, forecasts[1]))
    with pytest.raises(NbaRichBaselineError, match="IDs or order"):
        build_nba_rich_baseline_forecast_lock(model, rows, tuple(reversed(forecasts)))
    with pytest.raises(NbaRichBaselineError, match="strictly later"):
        replace(lock, evaluation_seasons=(2025,))

    path = tmp_path / "forecast-lock.json"
    digest = write_nba_rich_baseline_forecast_lock(path, lock)
    assert digest == lock.lock_sha256
    assert read_nba_rich_baseline_forecast_lock(path, digest) == lock
    with pytest.raises(NbaRichBaselineError, match="already exists"):
        write_nba_rich_baseline_forecast_lock(path, lock)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["forecast_count"] = 3
    path.write_text(canonical_json(payload), encoding="utf-8")
    with pytest.raises(NbaRichBaselineError, match="digest changed"):
        read_nba_rich_baseline_forecast_lock(path, digest)


def test_model_rejects_hidden_or_malformed_zero_rms_audit_fields(tmp_path: Path) -> None:
    model = _model()
    payload = model.canonical_payload()
    scaling = payload["rms_scaling"]
    assert isinstance(scaling, dict)
    scaling["zero_rms_feature_names"] = []
    path = tmp_path / "hidden-zero-rms.json"
    path.write_text(canonical_json(payload), encoding="utf-8")

    with pytest.raises(NbaRichBaselineError, match="digest changed"):
        read_nba_rich_baseline_model(path, model.model_sha256)

    scaling["zero_rms_feature_names"] = [NBA_RICH_FEATURE_NAMES[1]]
    scaling["scales"][1] = 2.0
    path.write_text(canonical_json(payload), encoding="utf-8")
    with pytest.raises(NbaRichBaselineError, match="invalid NBA rich baseline model"):
        read_nba_rich_baseline_model(path)


def test_forecast_lock_requires_positive_count_and_exact_hashes() -> None:
    with pytest.raises(NbaRichBaselineError, match="forecast_count"):
        NbaRichBaselineForecastLock(
            model_sha256="a" * 64,
            evaluation_feature_rows_jsonl_sha256="b" * 64,
            evaluation_question_ids_sha256="c" * 64,
            training_seasons=(2025,),
            evaluation_seasons=(2026,),
            forecast_jsonl_sha256="d" * 64,
            forecast_count=0,
        )
