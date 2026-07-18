"""Tests for the fixed rich-baseline fitting and answer-free prediction CLIs."""

import ast
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Never, cast

import pytest
from examples import fit_nba_rich_baseline as fit_cli
from examples import predict_nba_rich_baseline as predict_cli

from forecastfm.elo_residual import EloResidualFitConfig
from forecastfm.integrity import file_sha256
from forecastfm.nba_feature_rows import NbaRichFeatureRow
from forecastfm.nba_resolutions import NbaResolution
from forecastfm.nba_rich_baseline import (
    NbaRichBaselineForecastLock,
    NbaRichBaselineModel,
)
from forecastfm.outcome_v2_metrics import BinaryForecast


@dataclass(frozen=True, slots=True)
class _ModelMarker:
    training_row_count: int = 2
    zero_rms_feature_names: tuple[str, ...] = ()
    training_feature_rows_jsonl_sha256: str = "a" * 64
    training_resolutions_jsonl_sha256: str = "b" * 64


@dataclass(slots=True)
class _Recorder:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]]

    def returning(self, name: str, value: object) -> Callable[..., object]:
        def call(*args: object, **kwargs: object) -> object:
            self.calls.append((name, args, kwargs))
            return value

        return call


def _lock() -> NbaRichBaselineForecastLock:
    return NbaRichBaselineForecastLock(
        model_sha256="a" * 64,
        evaluation_feature_rows_jsonl_sha256="b" * 64,
        evaluation_question_ids_sha256="c" * 64,
        training_seasons=(2025,),
        evaluation_seasons=(2026,),
        forecast_jsonl_sha256="d" * 64,
        forecast_count=1,
    )


def _unexpected_read(*_args: object, **_kwargs: object) -> Never:
    raise AssertionError("an input was read after an existing output was detected")


def _prediction_binding_checker() -> Callable[[NbaRichBaselineForecastLock], None]:
    return cast(
        Callable[[NbaRichBaselineForecastLock], None],
        predict_cli.__dict__["_require_exact_file_bindings"],
    )


def _training_binding_checker() -> Callable[[NbaRichBaselineModel], None]:
    return cast(
        Callable[[NbaRichBaselineModel], None],
        fit_cli.__dict__["_require_exact_training_files"],
    )


def test_fit_uses_fixed_training_inputs_then_writes_and_reverifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fit_cli, "MODEL_PATH", tmp_path / "model.json")
    rows = cast(tuple[NbaRichFeatureRow, ...], (object(),))
    resolutions = cast(tuple[NbaResolution, ...], (object(),))
    model = cast(NbaRichBaselineModel, _ModelMarker())
    snapshot_index = object()
    config = EloResidualFitConfig(steps=1)
    digest = "a" * 64
    recorder = _Recorder([])
    patches = {
        "read_nba_feature_rows_jsonl": recorder.returning("read_rows", rows),
        "load_snapshot_pack": recorder.returning("read_snapshots", snapshot_index),
        "read_nba_resolutions_jsonl": recorder.returning("read_resolutions", resolutions),
        "outcome_v2_rich_baseline_fit_config": recorder.returning("config", config),
        "fit_nba_rich_baseline": recorder.returning("fit", model),
        "_require_exact_training_files": recorder.returning("bind", None),
        "write_nba_rich_baseline_model": recorder.returning("write", digest),
        "read_nba_rich_baseline_model": recorder.returning("reverify", model),
    }
    for name, replacement in patches.items():
        monkeypatch.setattr(fit_cli, name, replacement)

    fit_cli.main()

    assert recorder.calls == [
        ("read_rows", (fit_cli.FEATURE_ROWS_PATH,), {}),
        ("read_snapshots", (fit_cli.SNAPSHOT_PACK_PATH,), {}),
        ("read_resolutions", (fit_cli.RESOLUTIONS_PATH,), {"snapshot_index": snapshot_index}),
        ("config", (), {}),
        ("fit", (rows, resolutions, config), {}),
        ("bind", (model,), {}),
        ("write", (fit_cli.MODEL_PATH, model), {}),
        ("reverify", (fit_cli.MODEL_PATH, digest), {}),
    ]


def test_prediction_uses_only_fixed_answer_free_inputs_and_reverifies_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = cast(NbaRichBaselineModel, _ModelMarker())
    rows = cast(tuple[NbaRichFeatureRow, ...], (object(),))
    forecasts = (BinaryForecast("evaluation-game", 0.6),)
    forecast_lock = _lock()
    lock_digest = "e" * 64
    recorder = _Recorder([])
    patches = {
        "_require_unused_outputs": recorder.returning("unused", None),
        "read_nba_rich_baseline_model": recorder.returning("read_model", model),
        "read_nba_feature_rows_jsonl": recorder.returning("read_rows", rows),
        "predict_nba_rich_baseline": recorder.returning("predict", forecasts),
        "build_nba_rich_baseline_forecast_lock": recorder.returning("build", forecast_lock),
        "write_nba_evaluation_forecasts_jsonl": recorder.returning("write_forecasts", None),
        "read_nba_evaluation_forecasts_jsonl": recorder.returning("reverify_forecasts", forecasts),
        "write_nba_rich_baseline_forecast_lock": recorder.returning("write_lock", lock_digest),
        "read_nba_rich_baseline_forecast_lock": recorder.returning("reverify_lock", forecast_lock),
        "_require_exact_file_bindings": recorder.returning("bind", None),
    }
    for name, replacement in patches.items():
        monkeypatch.setattr(predict_cli, name, replacement)

    predict_cli.main()

    assert recorder.calls == [
        ("unused", (), {}),
        ("read_model", (predict_cli.MODEL_PATH,), {}),
        ("read_rows", (predict_cli.EVALUATION_FEATURE_ROWS_PATH,), {}),
        ("predict", (model, rows), {}),
        ("build", (model, rows, forecasts), {}),
        ("write_forecasts", (predict_cli.FORECASTS_PATH, forecasts), {}),
        ("reverify_forecasts", (predict_cli.FORECASTS_PATH,), {}),
        ("write_lock", (predict_cli.FORECAST_LOCK_PATH, forecast_lock), {}),
        ("reverify_lock", (predict_cli.FORECAST_LOCK_PATH, lock_digest), {}),
        ("bind", (forecast_lock,), {}),
    ]


def test_prediction_ast_has_no_answer_resolution_or_fit_dependencies() -> None:
    tree = ast.parse(Path(predict_cli.__file__).read_text(encoding="utf-8"))
    referenced = (
        {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        | {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        | {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        | {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    )

    assert not {
        name
        for name in referenced
        if any(fragment in name.lower() for fragment in ("answer", "resolution", "fit"))
    }


def test_existing_fit_output_fails_before_training_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.json"
    model_path.write_text("already frozen", encoding="utf-8")
    monkeypatch.setattr(fit_cli, "MODEL_PATH", model_path)
    monkeypatch.setattr(fit_cli, "read_nba_feature_rows_jsonl", _unexpected_read)

    with pytest.raises(FileExistsError, match="already exists"):
        fit_cli.main()


@pytest.mark.parametrize("existing_output", ["forecasts", "lock"])
def test_existing_prediction_output_fails_before_answer_free_reads(
    existing_output: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forecasts_path = tmp_path / "forecasts.jsonl"
    lock_path = tmp_path / "forecast-lock.json"
    selected = forecasts_path if existing_output == "forecasts" else lock_path
    selected.write_text("already frozen", encoding="utf-8")
    monkeypatch.setattr(predict_cli, "FORECASTS_PATH", forecasts_path)
    monkeypatch.setattr(predict_cli, "FORECAST_LOCK_PATH", lock_path)
    monkeypatch.setattr(predict_cli, "read_nba_rich_baseline_model", _unexpected_read)

    with pytest.raises(FileExistsError, match="already exists"):
        predict_cli.main()


def test_exact_file_binding_helper_checks_all_three_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model.json"
    rows_path = tmp_path / "evaluation-rows.jsonl"
    forecasts_path = tmp_path / "forecasts.jsonl"
    model_path.write_bytes(b"model")
    rows_path.write_bytes(b"rows")
    forecasts_path.write_bytes(b"forecasts")
    monkeypatch.setattr(predict_cli, "MODEL_PATH", model_path)
    monkeypatch.setattr(predict_cli, "EVALUATION_FEATURE_ROWS_PATH", rows_path)
    monkeypatch.setattr(predict_cli, "FORECASTS_PATH", forecasts_path)
    lock = replace(
        _lock(),
        model_sha256=file_sha256(model_path),
        evaluation_feature_rows_jsonl_sha256=file_sha256(rows_path),
        forecast_jsonl_sha256=file_sha256(forecasts_path),
    )

    checker = _prediction_binding_checker()
    checker(lock)

    with pytest.raises(RuntimeError, match="exact model file"):
        checker(replace(lock, model_sha256="f" * 64))
    with pytest.raises(RuntimeError, match="exact evaluation rows"):
        checker(replace(lock, evaluation_feature_rows_jsonl_sha256="f" * 64))
    with pytest.raises(RuntimeError, match="exact forecast file"):
        checker(replace(lock, forecast_jsonl_sha256="f" * 64))


def test_exact_training_file_helper_checks_both_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows_path = tmp_path / "training-rows.jsonl"
    resolutions_path = tmp_path / "training-resolutions.jsonl"
    rows_path.write_bytes(b"rows")
    resolutions_path.write_bytes(b"resolutions")
    monkeypatch.setattr(fit_cli, "FEATURE_ROWS_PATH", rows_path)
    monkeypatch.setattr(fit_cli, "RESOLUTIONS_PATH", resolutions_path)
    marker = _ModelMarker(
        training_feature_rows_jsonl_sha256=file_sha256(rows_path),
        training_resolutions_jsonl_sha256=file_sha256(resolutions_path),
    )

    checker = _training_binding_checker()
    checker(cast(NbaRichBaselineModel, marker))

    with pytest.raises(RuntimeError, match="exact training feature-row file"):
        checker(
            cast(
                NbaRichBaselineModel,
                replace(marker, training_feature_rows_jsonl_sha256="f" * 64),
            )
        )
    with pytest.raises(RuntimeError, match="exact training resolution file"):
        checker(
            cast(
                NbaRichBaselineModel,
                replace(marker, training_resolutions_jsonl_sha256="f" * 64),
            )
        )
