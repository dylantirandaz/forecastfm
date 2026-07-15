"""Tests for the immutable realized-outcome training lock."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from forecastfm.integrity import file_sha256
from forecastfm.outcome import OUTCOME_INPUT_SCHEMA_VERSION
from forecastfm.outcome_run_lock import (
    OUTCOME_DATA_MANIFEST_PATH,
    OUTCOME_LOCKED_CODE_PATHS,
    OUTCOME_TRAINING_DATA_PATH,
    build_outcome_training_lock,
    verify_outcome_training_lock,
)
from forecastfm.run_lock import RunLockError, write_new_lock

REVISION = "b" * 40
CREATED_AT = datetime(2026, 7, 15, 12, tzinfo=UTC)


def _make_project(path: Path) -> None:
    for code_path in OUTCOME_LOCKED_CODE_PATHS:
        full_path = path / code_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"fixture for {code_path}\n", encoding="utf-8")
    (path / "uv.lock").write_text("fixture lock\n", encoding="utf-8")

    training_path = path / OUTCOME_TRAINING_DATA_PATH
    training_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.write_text(
        '{"label":"TEAM","messages":[],"question_id":"q"}\n',
        encoding="utf-8",
    )
    manifest = {
        "outcome_input_schema_version": OUTCOME_INPUT_SCHEMA_VERSION,
        "outputs": {training_path.name: file_sha256(training_path)},
    }
    manifest_path = path / OUTCOME_DATA_MANIFEST_PATH
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_outcome_training_lock_verifies_exact_inputs(tmp_path: Path) -> None:
    _make_project(tmp_path)
    path = tmp_path / "outcome_training_lock.json"
    write_new_lock(path, build_outcome_training_lock(tmp_path, REVISION, CREATED_AT))

    record = verify_outcome_training_lock(tmp_path, path)

    assert record["code_revision"] == REVISION
    assert record["kind"] == "forecastfm_outcome_training_lock"


def test_outcome_training_lock_rejects_changed_objective_code(tmp_path: Path) -> None:
    _make_project(tmp_path)
    path = tmp_path / "outcome_training_lock.json"
    write_new_lock(path, build_outcome_training_lock(tmp_path, REVISION, CREATED_AT))
    (tmp_path / "src/forecastfm/outcome.py").write_text("changed\n", encoding="utf-8")

    with pytest.raises(RunLockError, match="differs"):
        verify_outcome_training_lock(tmp_path, path)


def test_outcome_training_lock_rejects_changed_model_config(tmp_path: Path) -> None:
    _make_project(tmp_path)
    path = tmp_path / "outcome_training_lock.json"
    write_new_lock(path, build_outcome_training_lock(tmp_path, REVISION, CREATED_AT))
    (tmp_path / "src/forecastfm/run_config.py").write_text("changed\n", encoding="utf-8")

    with pytest.raises(RunLockError, match="differs"):
        verify_outcome_training_lock(tmp_path, path)
