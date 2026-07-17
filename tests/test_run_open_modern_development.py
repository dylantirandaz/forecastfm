"""Tests for the write-once open-modern validation runner."""

import csv
import inspect
import json
from dataclasses import replace
from datetime import date
from hashlib import sha256
from math import log
from pathlib import Path

import pytest
from examples import run_open_modern_development as runner

from forecastfm.json_utils import require_object, required_field
from forecastfm.open_modern import DEVELOPMENT_COLUMNS
from forecastfm.open_modern_features import (
    OpenModernCausalFeatures,
    OpenModernFeatureRow,
    OpenModernInputGame,
)
from forecastfm.open_modern_model import OpenModernResolvedRow, fit_open_modern_validation


def _development_row(
    game_id: str,
    outcomes: tuple[str, str],
) -> tuple[str, ...]:
    values = {
        "game_id": game_id,
        "season": "2020",
        "date": "2019-10-01",
        "team1": "Hawks",
        "team2": "Celtics",
        "prob1": "0.6",
        "prob1_outcome": outcomes[0],
        "prob2": "0.4",
        "prob2_outcome": outcomes[1],
    }
    return tuple(values[column] for column in DEVELOPMENT_COLUMNS)


def _write_development(path: Path, rows: tuple[tuple[str, ...], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(DEVELOPMENT_COLUMNS)
        writer.writerows(rows)


def _pin_development(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "OPEN_MODERN_DEVELOPMENT_SHA256",
        sha256(path.read_bytes()).hexdigest(),
    )


def _resolved_row(season: int, index: int, outcome: int) -> OpenModernResolvedRow:
    probability = 0.4 + 0.02 * index
    return OpenModernResolvedRow(
        question_id=f"game-{season}-{index}",
        season=season,
        game_date=date(season, 1, index + 1),
        source_probability=probability,
        features=(
            log(probability / (1.0 - probability)),
            float(index + 1),
            float(index % 2 * 2 - 1),
            float(index + 2),
            float(index - 2) / 10.0,
            float(index + 3),
            float(index - 1),
        ),
        outcome=outcome,
    )


def _experiment_rows() -> tuple[OpenModernResolvedRow, ...]:
    rows: list[OpenModernResolvedRow] = []
    for season_index, season in enumerate(range(2016, 2021)):
        rows.append(_resolved_row(season, season_index + 1, season_index % 2))
        rows.append(_resolved_row(season, season_index + 2, (season_index + 1) % 2))
    return tuple(rows)


def _pin_synthetic_splits(
    rows: tuple[OpenModernResolvedRow, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_ids = [row.question_id for row in rows if row.season < 2020]
    validation_ids = [row.question_id for row in rows if row.season == 2020]
    monkeypatch.setattr(runner, "EXPECTED_TRAIN_COUNT", len(training_ids))
    monkeypatch.setattr(runner, "EXPECTED_VALIDATION_COUNT", len(validation_ids))
    monkeypatch.setattr(
        runner,
        "EXPECTED_TRAIN_IDS_SHA256",
        runner.canonical_sha256(training_ids),
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_VALIDATION_IDS_SHA256",
        runner.canonical_sha256(validation_ids),
    )


def test_label_loader_uses_actual_complementary_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "development.csv"
    _write_development(
        path,
        (
            _development_row("game-1", ("1", "0")),
            _development_row("game-2", ("0", "1")),
        ),
    )
    _pin_development(path, monkeypatch)

    labels = runner.load_development_labels(path)

    assert [(label.question_id, label.outcome) for label in labels] == [
        ("game-1", 1),
        ("game-2", 0),
    ]


def test_label_loader_rejects_noncomplementary_or_duplicate_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "development.csv"
    _write_development(path, (_development_row("game-1", ("1", "1")),))
    _pin_development(path, monkeypatch)
    with pytest.raises(runner.OpenModernDevelopmentError, match="complementary"):
        runner.load_development_labels(path)

    _write_development(
        path,
        (
            _development_row("game-1", ("1", "0")),
            _development_row("game-1", ("0", "1")),
        ),
    )
    _pin_development(path, monkeypatch)
    with pytest.raises(runner.OpenModernDevelopmentError, match="IDs must be unique"):
        runner.load_development_labels(path)


def test_label_loader_parses_the_pinned_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "development.csv"
    replacement = tmp_path / "replacement.csv"
    _write_development(path, (_development_row("game-1", ("1", "0")),))
    _write_development(replacement, (_development_row("game-1", ("0", "1")),))
    _pin_development(path, monkeypatch)
    original_read_bytes = Path.read_bytes

    def replace_after_read(candidate: Path) -> bytes:
        payload = original_read_bytes(candidate)
        if candidate == path:
            candidate.write_bytes(original_read_bytes(replacement))
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)

    labels = runner.load_development_labels(path)

    assert labels == (runner.DevelopmentLabel("game-1", 1),)
    assert original_read_bytes(path) == original_read_bytes(replacement)


def test_resolved_builder_finishes_features_before_loading_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    game = OpenModernInputGame(
        "game-1",
        2020,
        date(2019, 10, 1),
        "Hawks",
        "Celtics",
        0.6,
        0.4,
    )
    features = OpenModernCausalFeatures(log(1.5), 1.0, 1.0, 1.0, 0.1, 1.0, 1.0)
    feature_row = OpenModernFeatureRow(
        "game-1",
        2020,
        game.game_date,
        game.team1,
        game.team2,
        features,
    )

    def load_inputs(
        _path: Path,
        *,
        seal_path: Path,
        protocol_path: Path,
        exposure_path: Path,
    ) -> tuple[OpenModernInputGame, ...]:
        del seal_path, protocol_path, exposure_path
        events.append("inputs")
        return (game,)

    def load_raptor(_path: Path, *, max_allowed_season: int) -> dict[tuple[int, str], float]:
        assert max_allowed_season == 2019
        events.append("raptor")
        return {}

    def build_features(
        _games: tuple[OpenModernInputGame, ...],
        _raptor: dict[tuple[int, str], float],
    ) -> tuple[OpenModernFeatureRow, ...]:
        events.append("features")
        return (feature_row,)

    def load_labels(_path: Path) -> tuple[runner.DevelopmentLabel, ...]:
        events.append("labels")
        return (runner.DevelopmentLabel("game-1", 1),)

    def reverify(*_paths: Path) -> None:
        events.append("reverified")

    monkeypatch.setattr(runner, "load_open_modern_feature_inputs", load_inputs)
    monkeypatch.setattr(runner, "load_prior_season_raptor", load_raptor)
    monkeypatch.setattr(runner, "build_open_modern_features", build_features)
    monkeypatch.setattr(runner, "load_development_labels", load_labels)
    monkeypatch.setattr(runner, "require_open_modern_development", reverify)

    rows = runner.build_resolved_development_rows(
        tmp_path,
        tmp_path,
        tmp_path,
        tmp_path,
        tmp_path,
    )

    assert events == ["inputs", "raptor", "features", "labels", "reverified"]
    assert rows[0].outcome == 1


def test_validation_lock_binds_models_and_keeps_holdout_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _experiment_rows()
    result = fit_open_modern_validation(rows)
    _pin_synthetic_splits(rows, monkeypatch)

    lock = runner.build_validation_lock(rows, result, "a" * 40)
    serialized = json.dumps(lock, sort_keys=True)
    contracts = require_object(required_field(lock, "contracts"), "contracts")
    validation = require_object(required_field(lock, "validation"), "validation")
    holdout = require_object(required_field(lock, "holdout"), "holdout")

    assert required_field(contracts, "causal_features_sha256")
    assert required_field(contracts, "model_contract_sha256")
    forecast = require_object(required_field(validation, "forecast"), "forecast")
    assert required_field(forecast, "candidate_id") == result.forecast.spec.candidate_id
    assert "candidates" not in validation
    assert "selected_candidate_id" not in validation
    assert "fixed chronological full-file pass" in str(required_field(holdout, "inference_policy"))
    assert required_field(holdout, "predictions_written") is False
    assert required_field(holdout, "answers_opened_for_scoring") is False
    assert required_field(holdout, "scored") is False
    assert "test_inputs.csv" not in serialized
    assert "prob1_outcome" not in serialized


def test_validation_lock_rejects_wrong_split_order_or_result_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _experiment_rows()
    result = fit_open_modern_validation(rows)
    _pin_synthetic_splits(rows, monkeypatch)

    with pytest.raises(runner.OpenModernDevelopmentError, match="training IDs"):
        runner.build_validation_lock(tuple(reversed(rows)), result, "a" * 40)

    wrong_forecast = replace(result.forecast, spec=result.recalibration.spec)
    with pytest.raises(runner.OpenModernDevelopmentError, match="forecast specification"):
        runner.build_validation_lock(rows, replace(result, forecast=wrong_forecast), "a" * 40)

    with pytest.raises(runner.OpenModernDevelopmentError, match="internally inconsistent"):
        runner.build_validation_lock(
            rows,
            replace(result, advances_to_holdout=not result.advances_to_holdout),
            "a" * 40,
        )


def test_validation_lock_writer_is_finite_and_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "lock.json"
    runner.write_validation_lock(path, {"value": 1.0})

    with pytest.raises(runner.OpenModernDevelopmentError, match="already exists"):
        runner.write_validation_lock(path, {"value": 1.0})
    with pytest.raises(runner.OpenModernDevelopmentError, match="non-finite"):
        runner.write_validation_lock(tmp_path / "bad.json", {"value": float("nan")})
    unserializable = tmp_path / "unserializable.json"
    with pytest.raises(runner.OpenModernDevelopmentError, match="not serializable"):
        runner.write_validation_lock(unserializable, {"value": object()})
    assert not unserializable.exists()
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_runner_source_defines_no_holdout_or_raw_answer_path() -> None:
    source = inspect.getsource(runner)

    assert "TEST_INPUTS_PATH" not in source
    assert "nba_games.csv" not in source
    assert "test_answers" not in source
