"""Tests for the Tinker-free outcome-v2 paid-run entrypoint."""

import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import pytest
from examples import train_tinker_outcome_v2_sft as entrypoint

from forecastfm.outcome_v2_experiment import OutcomeV2ExperimentLock
from forecastfm.outcome_v2_preflight import OutcomeV2PreflightError, PreparedOutcomeV2Run
from forecastfm.outcome_v2_run import OutcomeV2RunError, OutcomeV2RunLock
from forecastfm.publication import PublicationProof


@dataclass(frozen=True, slots=True)
class _FakeLock:
    sha256: str


@dataclass(frozen=True, slots=True)
class _FakePaidResult:
    state_path: str
    sampler_path: str


@dataclass(slots=True)
class _MainHarness:
    events: list[str]
    prepared: PreparedOutcomeV2Run
    run_lock: OutcomeV2RunLock

    def outputs(self) -> None:
        self.events.append("outputs")

    def key(self) -> str:
        self.events.append("key")
        return "local-secret"

    def packages(self) -> None:
        self.events.append("packages")

    def publication(self, _project_root: Path, remote_url: str) -> PublicationProof:
        self.events.append("publication")
        return PublicationProof("c" * 40, "origin", remote_url, "refs/heads/main")

    def commit(self, _revision: str) -> tuple[PreparedOutcomeV2Run, OutcomeV2RunLock]:
        self.events.append("commit")
        return self.prepared, self.run_lock

    def paid(
        self,
        _prepared: PreparedOutcomeV2Run,
        _digest: str,
    ) -> entrypoint.PaidRunResult:
        self.events.append("paid")
        return _FakePaidResult("tinker://state", "tinker://sampler")

    def load(self) -> entrypoint.PaidRuntime:
        self.events.append("load")
        return self.paid

    def reverify(
        self,
        _prepared: PreparedOutcomeV2Run,
        _revision: str,
        _digest: str,
    ) -> None:
        self.events.append("reverify")

    def seal(
        self,
        _result: entrypoint.PaidRunResult,
        _created_at: datetime,
    ) -> OutcomeV2ExperimentLock:
        self.events.append("seal")
        return _experiment_marker()


def _prepared_marker() -> PreparedOutcomeV2Run:
    return cast(PreparedOutcomeV2Run, object())


def _lock_marker(digest: str = "a" * 64) -> OutcomeV2RunLock:
    return cast(OutcomeV2RunLock, _FakeLock(digest))


def _experiment_marker(digest: str = "d" * 64) -> OutcomeV2ExperimentLock:
    return cast(OutcomeV2ExperimentLock, _FakeLock(digest))


def test_entrypoint_has_no_eager_paid_runtime_import() -> None:
    source_path = Path(entrypoint.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "tinker" not in imported_modules
    assert "tinker_cookbook" not in imported_modules
    assert "examples.tinker_outcome_v2_runtime" not in imported_modules


def test_artifact_layout_covers_every_manifest_bound_path() -> None:
    artifacts = entrypoint.outcome_v2_artifacts()
    paths = artifacts.sealed_paths()

    assert paths
    baseline_paths = {artifacts.rich_baseline_forecast_lock_path}
    assert artifacts.rich_baseline_model_path.parent == entrypoint.DATA_DIRECTORY
    assert artifacts.rich_baseline_forecast_lock_path.parent == entrypoint.BASELINE_DIRECTORY
    assert all(
        path.parent == entrypoint.DATA_DIRECTORY
        for path in paths.values()
        if path not in baseline_paths
    )
    assert artifacts.agreement_path == entrypoint.AGREEMENT_PATH
    assert entrypoint.TRAINING_PATH.parent == entrypoint.DATA_DIRECTORY


def test_checked_in_manifest_blocks_before_a_run_lock_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "training_lock.json"
    monkeypatch.setattr(entrypoint, "RUN_LOCK_PATH", lock_path)

    with pytest.raises(OutcomeV2PreflightError, match="full_outcome_v2_ready is false"):
        entrypoint.commit_outcome_v2_paid_run("a" * 40)

    assert not lock_path.exists()


def test_commit_writes_then_reverifies_before_returning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    prepared = _prepared_marker()
    lock = _lock_marker()

    def fake_prepare(
        manifest_path: Path,
        training_path: Path,
        artifacts: object,
    ) -> PreparedOutcomeV2Run:
        assert manifest_path == entrypoint.MANIFEST_PATH
        assert training_path == entrypoint.TRAINING_PATH
        assert artifacts is not None
        events.append("prepare")
        return prepared

    def fake_build(
        project_root: Path,
        actual_prepared: PreparedOutcomeV2Run,
        revision: str,
    ) -> OutcomeV2RunLock:
        assert project_root == entrypoint.PROJECT_ROOT
        assert actual_prepared is prepared
        assert revision == "b" * 40
        events.append("build")
        return lock

    def fake_write(path: Path, actual_lock: OutcomeV2RunLock) -> str:
        assert path == entrypoint.RUN_LOCK_PATH
        assert actual_lock is lock
        events.append("write")
        return lock.sha256

    def fake_verify(
        project_root: Path,
        path: Path,
        actual_prepared: PreparedOutcomeV2Run,
        revision: str,
    ) -> OutcomeV2RunLock:
        assert project_root == entrypoint.PROJECT_ROOT
        assert path == entrypoint.RUN_LOCK_PATH
        assert actual_prepared is prepared
        assert revision == "b" * 40
        events.append("verify")
        return lock

    monkeypatch.setattr(entrypoint, "prepare_outcome_v2_sft_run", fake_prepare)
    monkeypatch.setattr(entrypoint, "build_outcome_v2_run_lock", fake_build)
    monkeypatch.setattr(entrypoint, "write_outcome_v2_run_lock", fake_write)
    monkeypatch.setattr(entrypoint, "verify_outcome_v2_run_lock", fake_verify)

    result = entrypoint.commit_outcome_v2_paid_run("b" * 40)

    assert result == (prepared, lock)
    assert events == ["prepare", "build", "write", "verify"]


def test_commit_rejects_a_reverification_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared_marker()
    written = _lock_marker("a" * 64)
    verified = _lock_marker("b" * 64)

    def fake_prepare(
        _manifest_path: Path,
        _training_path: Path,
        _artifacts: object,
    ) -> PreparedOutcomeV2Run:
        return prepared

    def fake_build(
        _project_root: Path,
        _prepared: PreparedOutcomeV2Run,
        _revision: str,
    ) -> OutcomeV2RunLock:
        return written

    def fake_write(_path: Path, _lock: OutcomeV2RunLock) -> str:
        return written.sha256

    def fake_verify(
        _project_root: Path,
        _path: Path,
        _prepared: PreparedOutcomeV2Run,
        _revision: str,
    ) -> OutcomeV2RunLock:
        return verified

    monkeypatch.setattr(entrypoint, "prepare_outcome_v2_sft_run", fake_prepare)
    monkeypatch.setattr(entrypoint, "build_outcome_v2_run_lock", fake_build)
    monkeypatch.setattr(entrypoint, "write_outcome_v2_run_lock", fake_write)
    monkeypatch.setattr(entrypoint, "verify_outcome_v2_run_lock", fake_verify)

    with pytest.raises(OutcomeV2RunError, match="could not be re-verified"):
        entrypoint.commit_outcome_v2_paid_run("b" * 40)


def test_main_loads_paid_runtime_only_after_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    harness = _MainHarness(events, _prepared_marker(), _lock_marker())

    monkeypatch.setattr(entrypoint, "_require_unused_output_paths", harness.outputs)
    monkeypatch.setattr(entrypoint, "_local_api_key", harness.key)
    monkeypatch.setattr(entrypoint, "_require_pinned_packages", harness.packages)
    monkeypatch.setattr(entrypoint, "require_published_head", harness.publication)
    monkeypatch.setattr(entrypoint, "commit_outcome_v2_paid_run", harness.commit)
    monkeypatch.setattr(entrypoint, "_load_paid_runtime", harness.load)
    monkeypatch.setattr(entrypoint, "require_committed_run_unchanged", harness.reverify)
    monkeypatch.setattr(entrypoint, "seal_outcome_v2_experiment", harness.seal)
    monkeypatch.delenv("TINKER_API_KEY", raising=False)

    entrypoint.main()

    assert events == [
        "outputs",
        "key",
        "packages",
        "publication",
        "commit",
        "load",
        "reverify",
        "paid",
        "seal",
    ]
    assert entrypoint.os.environ["TINKER_API_KEY"] == "local-secret"
