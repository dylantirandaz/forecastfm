"""Tests for the paid-run safety checks."""

import json
from pathlib import Path

import pytest
from examples import train_tinker_sft

from forecastfm.nba_data import file_sha256
from forecastfm.prompting import MODEL_INPUT_SCHEMA_VERSION
from forecastfm.run_lock import RunLockError


def _accept_training_lock(_project_root: Path, _path: Path) -> dict[str, object]:
    return {}


def _accept_tokenizer_snapshot() -> Path:
    return Path("/pinned/tokenizer")


def _configure_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    schema_version: int = MODEL_INPUT_SCHEMA_VERSION,
    expected_hash: str | None = None,
) -> None:
    data_path = tmp_path / "nba_elo_train_sft.jsonl"
    data_path.write_text('{"messages": []}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "model_input_schema_version": schema_version,
        "outputs": {data_path.name: expected_hash or file_sha256(data_path)},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(train_tinker_sft, "DATA_PATH", data_path)
    monkeypatch.setattr(train_tinker_sft, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(train_tinker_sft, "verify_training_lock", _accept_training_lock)
    monkeypatch.setattr(
        train_tinker_sft,
        "require_tokenizer_snapshot",
        _accept_tokenizer_snapshot,
    )
    monkeypatch.setenv("TINKER_API_KEY", "test-key")


def test_runner_accepts_current_verified_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_artifacts(tmp_path, monkeypatch)

    train_tinker_sft.require_prerequisites()


def test_runner_rejects_stale_prompt_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_artifacts(tmp_path, monkeypatch, schema_version=MODEL_INPUT_SCHEMA_VERSION - 1)

    with pytest.raises(RuntimeError, match="stale"):
        train_tinker_sft.require_prerequisites()


def test_runner_rejects_modified_training_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_artifacts(tmp_path, monkeypatch, expected_hash="0" * 64)

    with pytest.raises(RuntimeError, match="hash"):
        train_tinker_sft.require_prerequisites()


def test_runner_rejects_unsealed_training_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_artifacts(tmp_path, monkeypatch)

    def reject_training_lock(_project_root: Path, _path: Path) -> dict[str, object]:
        raise RunLockError("training lock differs")

    monkeypatch.setattr(train_tinker_sft, "verify_training_lock", reject_training_lock)

    with pytest.raises(RunLockError, match="differs"):
        train_tinker_sft.require_prerequisites()
