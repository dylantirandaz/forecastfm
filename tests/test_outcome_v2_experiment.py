"""Tests for the immutable outcome-v2 post-training experiment seal."""

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

from forecastfm.integrity import bytes_sha256, canonical_json
from forecastfm.outcome_v2_experiment import (
    OutcomeV2ExperimentError,
    OutcomeV2ExperimentLock,
    build_outcome_v2_experiment_lock,
    read_outcome_v2_experiment_lock,
    verify_outcome_v2_experiment_lock,
    write_outcome_v2_experiment_lock,
)
from forecastfm.outcome_v2_preflight import OutcomeV2Preflight, PreparedOutcomeV2Run
from forecastfm.outcome_v2_run import build_outcome_v2_run_lock

PROJECT_ROOT = Path(__file__).parents[1]
REVISION = "a" * 40
CREATED_AT = datetime(2026, 7, 17, 18, 30, tzinfo=UTC)
STATE_PATH = "tinker://run/state/final"
SAMPLER_PATH = "tinker://run/sampler/final"
TRAINING_BYTES = b'{"label":"TEAM","question_id":"game-1"}\n'


def _prepared() -> PreparedOutcomeV2Run:
    proof = OutcomeV2Preflight(
        manifest_sha256="1" * 64,
        action_at=datetime(2026, 7, 17, 18, tzinfo=UTC),
        action_time_source="internal_paid_preparation",
        untouched_evaluation_seasons=(2024, 2025),
        training_sha256=bytes_sha256(TRAINING_BYTES),
        feature_rows_sha256="2" * 64,
        snapshot_pack_sha256="3" * 64,
        evidence_bundles_sha256="4" * 64,
        elo_states_sha256="5" * 64,
        elo_replay_sha256="6" * 64,
        seasons_sha256="7" * 64,
        resolutions_sha256="8" * 64,
        rights_lock_sha256="9" * 64,
        evaluation_feature_rows_sha256="0" * 64,
        evaluation_elo_replay_sha256="a" * 64,
        evaluation_elo_states_sha256="b" * 64,
        evaluation_resolutions_sha256="c" * 64,
        calibration_sha256="0" * 64,
        rich_baseline_model_sha256="e" * 64,
        rich_baseline_forecast_lock_sha256="f" * 64,
        evaluation_report_sha256="d" * 64,
        row_count=14,
        pair_count=7,
        batch_size=14,
    )
    return PreparedOutcomeV2Run(proof, TRAINING_BYTES)


def _write_run_lock(path: Path) -> bytes:
    lock = build_outcome_v2_run_lock(PROJECT_ROOT, _prepared(), REVISION)
    path.write_bytes(lock.canonical_bytes)
    return lock.canonical_bytes


def test_builder_is_deterministic_and_binds_only_permanent_identifiers(
    tmp_path: Path,
) -> None:
    run_lock_path = tmp_path / "run-lock.json"
    run_lock_bytes = _write_run_lock(run_lock_path)

    first = build_outcome_v2_experiment_lock(
        run_lock_path,
        STATE_PATH,
        SAMPLER_PATH,
        CREATED_AT,
    )
    second = build_outcome_v2_experiment_lock(
        run_lock_path,
        STATE_PATH,
        SAMPLER_PATH,
        CREATED_AT,
    )
    record = first.to_record()

    assert first == second
    assert first.canonical_bytes == canonical_json(record).encode("utf-8")
    assert set(record) == {
        "schema_version",
        "kind",
        "status",
        "created_at",
        "outcome_v2_run_lock_sha256",
        "state_path",
        "sampler_path",
    }
    assert record["kind"] == "forecastfm_outcome_v2_experiment_lock"
    assert record["status"] == "ready_for_prospective_forecasts"
    assert record["created_at"] == "2026-07-17T18:30:00Z"
    assert record["outcome_v2_run_lock_sha256"] == bytes_sha256(run_lock_bytes)
    assert record["state_path"] == STATE_PATH
    assert record["sampler_path"] == SAMPLER_PATH
    assert str(tmp_path) not in first.canonical_bytes.decode("utf-8")


def test_write_read_and_verify_are_exact_and_create_only(tmp_path: Path) -> None:
    run_lock_path = tmp_path / "run-lock.json"
    _write_run_lock(run_lock_path)
    lock = build_outcome_v2_experiment_lock(
        run_lock_path,
        STATE_PATH,
        SAMPLER_PATH,
        CREATED_AT,
    )
    path = tmp_path / "sealed/experiment.json"

    written_hash = write_outcome_v2_experiment_lock(path, lock)

    assert path.read_bytes() == lock.canonical_bytes
    assert written_hash == lock.sha256
    assert read_outcome_v2_experiment_lock(path) == lock
    assert verify_outcome_v2_experiment_lock(run_lock_path, path) == lock
    with pytest.raises(FileExistsError):
        write_outcome_v2_experiment_lock(path, lock)


def test_verifier_rejects_changed_referenced_run_lock(tmp_path: Path) -> None:
    run_lock_path = tmp_path / "run-lock.json"
    original = _write_run_lock(run_lock_path)
    lock = build_outcome_v2_experiment_lock(
        run_lock_path,
        STATE_PATH,
        SAMPLER_PATH,
        CREATED_AT,
    )
    path = tmp_path / "experiment.json"
    write_outcome_v2_experiment_lock(path, lock)
    run_lock_path.write_bytes(original + b"\n")

    with pytest.raises(OutcomeV2ExperimentError, match="run lock bytes changed"):
        verify_outcome_v2_experiment_lock(run_lock_path, path)


def test_builder_requires_a_valid_outcome_v2_run_lock(tmp_path: Path) -> None:
    run_lock_path = tmp_path / "run-lock.json"
    run_lock_path.write_bytes(b"not a run lock")

    with pytest.raises(OutcomeV2ExperimentError, match="run lock is invalid"):
        build_outcome_v2_experiment_lock(
            run_lock_path,
            STATE_PATH,
            SAMPLER_PATH,
            CREATED_AT,
        )


def test_reader_rejects_noncanonical_and_unexpected_fields(tmp_path: Path) -> None:
    run_lock_path = tmp_path / "run-lock.json"
    _write_run_lock(run_lock_path)
    lock = build_outcome_v2_experiment_lock(
        run_lock_path,
        STATE_PATH,
        SAMPLER_PATH,
        CREATED_AT,
    )
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(lock.canonical_bytes + b"\n")

    with pytest.raises(OutcomeV2ExperimentError, match="canonical JSON"):
        read_outcome_v2_experiment_lock(noncanonical)

    record = lock.to_record()
    record["unexpected"] = True
    with pytest.raises(OutcomeV2ExperimentError, match="structure"):
        OutcomeV2ExperimentLock(canonical_json(record).encode("utf-8"))


def test_strict_record_validation_rejects_tampered_identity_and_hash(tmp_path: Path) -> None:
    run_lock_path = tmp_path / "run-lock.json"
    _write_run_lock(run_lock_path)
    lock = build_outcome_v2_experiment_lock(
        run_lock_path,
        STATE_PATH,
        SAMPLER_PATH,
        CREATED_AT,
    )

    wrong_status = lock.to_record()
    wrong_status["status"] = "awaiting_sampler"
    with pytest.raises(OutcomeV2ExperimentError, match="not ready"):
        OutcomeV2ExperimentLock(canonical_json(wrong_status).encode("utf-8"))

    wrong_hash = lock.to_record()
    wrong_hash["outcome_v2_run_lock_sha256"] = "A" * 64
    with pytest.raises(OutcomeV2ExperimentError, match="lowercase SHA-256"):
        OutcomeV2ExperimentLock(canonical_json(wrong_hash).encode("utf-8"))


@pytest.mark.parametrize(
    "path",
    ["", "tinker://", "https://example.com/model", "tinker://run path", "tinker://run#x"],
)
def test_builder_rejects_nonpermanent_tinker_paths(tmp_path: Path, path: str) -> None:
    run_lock_path = tmp_path / "run-lock.json"
    _write_run_lock(run_lock_path)

    with pytest.raises(OutcomeV2ExperimentError, match=r"tinker://|fragment"):
        build_outcome_v2_experiment_lock(
            run_lock_path,
            path,
            SAMPLER_PATH,
            CREATED_AT,
        )


def test_builder_rejects_identical_state_and_sampler_paths(tmp_path: Path) -> None:
    run_lock_path = tmp_path / "run-lock.json"
    _write_run_lock(run_lock_path)

    with pytest.raises(OutcomeV2ExperimentError, match="must differ"):
        build_outcome_v2_experiment_lock(
            run_lock_path,
            STATE_PATH,
            STATE_PATH,
            CREATED_AT,
        )


@pytest.mark.parametrize(
    "created_at",
    [
        CREATED_AT.replace(tzinfo=None),
        datetime(2026, 7, 17, 18, tzinfo=timezone(-timedelta(hours=5))),
    ],
)
def test_builder_requires_utc_creation_time(tmp_path: Path, created_at: datetime) -> None:
    run_lock_path = tmp_path / "run-lock.json"
    _write_run_lock(run_lock_path)

    with pytest.raises(OutcomeV2ExperimentError, match="UTC datetime"):
        build_outcome_v2_experiment_lock(
            run_lock_path,
            STATE_PATH,
            SAMPLER_PATH,
            created_at,
        )


def test_missing_files_and_secret_free_source_fail_cleanly(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(OutcomeV2ExperimentError, match="cannot read"):
        read_outcome_v2_experiment_lock(missing)

    source = (PROJECT_ROOT / "src/forecastfm/outcome_v2_experiment.py").read_text(encoding="utf-8")
    assert "import tinker" not in source
    assert "api_key" not in source.lower()


def test_lock_rejects_mutable_bytes(tmp_path: Path) -> None:
    run_lock_path = tmp_path / "run-lock.json"
    _write_run_lock(run_lock_path)
    lock = build_outcome_v2_experiment_lock(
        run_lock_path,
        STATE_PATH,
        SAMPLER_PATH,
        CREATED_AT,
    )

    with pytest.raises(OutcomeV2ExperimentError, match="immutable bytes"):
        OutcomeV2ExperimentLock(cast(bytes, bytearray(lock.canonical_bytes)))
