"""Tests for binding a completed Tinker checkpoint to its training lock."""

import json
from pathlib import Path

import pytest
from examples import freeze_outcome_experiment

from forecastfm.checkpoints import read_final_checkpoint
from forecastfm.run_lock import verify_experiment_lock

RUN_PREFIX = "tinker://run:train:0"


def _checkpoint(name: str, prefix: str = RUN_PREFIX) -> dict[str, object]:
    suffix = "final" if name == "final" else name
    return {
        "name": name,
        "state_path": f"{prefix}/weights/{suffix}",
        "sampler_path": f"{prefix}/sampler_weights/{suffix}",
    }


def _write_checkpoints(path: Path, *records: dict[str, object]) -> None:
    path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records),
        encoding="utf-8",
    )


def test_read_final_checkpoint_accepts_periodic_records_before_final(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.jsonl"
    _write_checkpoints(path, _checkpoint("step_16"), _checkpoint("final"))

    record = read_final_checkpoint(path)

    assert record["sampler_path"] == f"{RUN_PREFIX}/sampler_weights/final"


def test_read_final_checkpoint_rejects_a_nonfinal_last_record(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.jsonl"
    _write_checkpoints(path, _checkpoint("final"), _checkpoint("step_32"))

    with pytest.raises(RuntimeError, match="must be the last"):
        read_final_checkpoint(path)


def test_read_final_checkpoint_rejects_duplicate_final_records(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.jsonl"
    _write_checkpoints(path, _checkpoint("final"), _checkpoint("final"))

    with pytest.raises(RuntimeError, match="exactly one"):
        read_final_checkpoint(path)


def test_read_final_checkpoint_rejects_mismatched_run_paths(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.jsonl"
    record = _checkpoint("final")
    record["sampler_path"] = "tinker://other:train:0/sampler_weights/final"
    _write_checkpoints(path, record)

    with pytest.raises(RuntimeError, match="same run"):
        read_final_checkpoint(path)


def test_outcome_freezer_binds_the_step_specific_training_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_lock_path = tmp_path / "training_lock.json"
    checkpoint_path = tmp_path / "checkpoints.jsonl"
    output_path = tmp_path / "experiment.json"
    training_lock_path.write_text("frozen training lock\n", encoding="utf-8")
    _write_checkpoints(checkpoint_path, _checkpoint("final"))
    monkeypatch.setattr(freeze_outcome_experiment, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(freeze_outcome_experiment, "TRAINING_LOCK_PATH", training_lock_path)
    monkeypatch.setattr(freeze_outcome_experiment, "CHECKPOINT_LOG_PATH", checkpoint_path)
    monkeypatch.setattr(freeze_outcome_experiment, "OUTPUT_PATH", output_path)

    def accept_outcome_lock(_root: Path, _path: Path) -> dict[str, object]:
        return {}

    monkeypatch.setattr(
        freeze_outcome_experiment,
        "verify_outcome_training_lock",
        accept_outcome_lock,
    )

    freeze_outcome_experiment.main()

    record = verify_experiment_lock(training_lock_path, output_path)
    assert record["adapter_sampler_path"] == f"{RUN_PREFIX}/sampler_weights/final"
