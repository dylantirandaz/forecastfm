"""Tests for immutable training and experiment locks."""

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from forecastfm.integrity import file_sha256
from forecastfm.prompting import MODEL_INPUT_SCHEMA_VERSION
from forecastfm.run_lock import (
    DATA_MANIFEST_PATH,
    LOCKED_CODE_PATHS,
    TRAINING_DATA_PATH,
    RunLockError,
    build_experiment_lock,
    build_training_lock,
    verify_experiment_lock,
    verify_training_lock,
    write_new_lock,
)

REVISION = "a" * 40
CREATED_AT = datetime(2026, 7, 11, 12, tzinfo=UTC)


def _make_project(path: Path) -> None:
    for code_path in LOCKED_CODE_PATHS:
        full_path = path / code_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"fixture for {code_path}\n", encoding="utf-8")
    (path / "uv.lock").write_text("fixture lock\n", encoding="utf-8")

    training_path = path / TRAINING_DATA_PATH
    training_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.write_text('{"messages": []}\n', encoding="utf-8")
    manifest = {
        "model_input_schema_version": MODEL_INPUT_SCHEMA_VERSION,
        "outputs": {training_path.name: file_sha256(training_path)},
    }
    manifest_path = path / DATA_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_training_lock_verifies_exact_working_inputs(tmp_path: Path) -> None:
    _make_project(tmp_path)
    path = tmp_path / "training_lock.json"
    write_new_lock(path, build_training_lock(tmp_path, REVISION, CREATED_AT))

    record = verify_training_lock(tmp_path, path)

    assert record["code_revision"] == REVISION


def test_training_lock_rejects_changed_prompt_code(tmp_path: Path) -> None:
    _make_project(tmp_path)
    path = tmp_path / "training_lock.json"
    write_new_lock(path, build_training_lock(tmp_path, REVISION, CREATED_AT))
    (tmp_path / "src/forecastfm/prompting.py").write_text("changed\n", encoding="utf-8")

    with pytest.raises(RunLockError, match="differs"):
        verify_training_lock(tmp_path, path)


def test_experiment_lock_binds_sampler_and_training_lock(tmp_path: Path) -> None:
    _make_project(tmp_path)
    training_path = tmp_path / "training_lock.json"
    experiment_path = tmp_path / "experiment.json"
    write_new_lock(training_path, build_training_lock(tmp_path, REVISION, CREATED_AT))
    experiment = build_experiment_lock(
        training_path,
        "tinker://run/sampler_weights/final",
        {"sampler_path": "tinker://run/sampler_weights/final"},
        CREATED_AT,
    )
    write_new_lock(experiment_path, experiment)

    verified = verify_experiment_lock(training_path, experiment_path)

    assert verified["adapter_sampler_path"] == "tinker://run/sampler_weights/final"


def test_experiment_lock_rejects_replaced_training_lock(tmp_path: Path) -> None:
    _make_project(tmp_path)
    training_path = tmp_path / "training_lock.json"
    experiment_path = tmp_path / "experiment.json"
    write_new_lock(training_path, build_training_lock(tmp_path, REVISION, CREATED_AT))
    write_new_lock(
        experiment_path,
        build_experiment_lock(training_path, "tinker://sampler", {}, CREATED_AT),
    )
    with training_path.open("a", encoding="utf-8") as file:
        file.write("\n")

    with pytest.raises(RunLockError, match="different training lock"):
        verify_experiment_lock(training_path, experiment_path)


def test_locks_cannot_be_accidentally_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "lock.json"
    write_new_lock(path, {"first": True})

    with pytest.raises(FileExistsError):
        write_new_lock(path, {"second": True})


def test_training_lock_requires_utc_creation_time(tmp_path: Path) -> None:
    _make_project(tmp_path)
    central_time = timezone(-timedelta(hours=6))

    with pytest.raises(RunLockError, match="must use UTC"):
        build_training_lock(tmp_path, REVISION, datetime(2026, 7, 11, 12, tzinfo=central_time))
