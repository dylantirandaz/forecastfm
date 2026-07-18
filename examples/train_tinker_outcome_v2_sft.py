"""Commit a verified outcome-v2 run before loading its paid runtime."""

import importlib
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

from forecastfm.local_config import read_tinker_api_key
from forecastfm.nba_evaluation_gate import NbaEvaluationGateArtifacts
from forecastfm.outcome_v2_config import (
    ELO_REPLAY_FILENAME,
    ELO_STATES_FILENAME,
    EVALUATION_ANSWERS_FILENAME,
    EVALUATION_COHORT_FILENAME,
    EVALUATION_ELO_REPLAY_FILENAME,
    EVALUATION_ELO_STATES_FILENAME,
    EVALUATION_FEATURE_ROWS_FILENAME,
    EVALUATION_FORECASTS_FILENAME,
    EVALUATION_REPORT_FILENAME,
    EVALUATION_RESOLUTIONS_FILENAME,
    EVIDENCE_BUNDLES_FILENAME,
    FEATURE_ROWS_FILENAME,
    MAX_STEPS,
    RECALIBRATION_FILENAME,
    RESOLUTIONS_FILENAME,
    RICH_BASELINE_FORECAST_LOCK_FILENAME,
    RICH_BASELINE_MODEL_FILENAME,
    RIGHTS_LOCK_FILENAME,
    SEASONS_FILENAME,
    SNAPSHOT_PACK_FILENAME,
    TRAINING_FILENAME,
)
from forecastfm.outcome_v2_experiment import (
    OutcomeV2ExperimentError,
    OutcomeV2ExperimentLock,
    build_outcome_v2_experiment_lock,
    verify_outcome_v2_experiment_lock,
    write_outcome_v2_experiment_lock,
)
from forecastfm.outcome_v2_preflight import (
    OutcomeV2Artifacts,
    PreparedOutcomeV2Run,
    prepare_outcome_v2_sft_run,
)
from forecastfm.outcome_v2_run import (
    OutcomeV2RunError,
    OutcomeV2RunLock,
    build_outcome_v2_run_lock,
    verify_outcome_v2_run_lock,
    write_outcome_v2_run_lock,
)
from forecastfm.publication import require_published_head
from forecastfm.run_config import require_pinned_tinker_packages

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/dylantirandaz/forecastfm.git"
DATA_DIRECTORY = PROJECT_ROOT / "data" / "processed" / "outcome_v2"
BASELINE_DIRECTORY = PROJECT_ROOT / "prospective" / "outcome_v2" / "rich_baseline"
MANIFEST_PATH = DATA_DIRECTORY / "manifest.json"
TRAINING_PATH = DATA_DIRECTORY / TRAINING_FILENAME
AGREEMENT_PATH = PROJECT_ROOT / "data" / "private" / "outcome_v2" / "source_agreement.bin"
RUN_LOCK_PATH = PROJECT_ROOT / "prospective" / "outcome_v2" / "training_lock.json"
EXPERIMENT_LOCK_PATH = PROJECT_ROOT / "prospective" / "outcome_v2" / "experiment.json"
LOCAL_ENV_PATH = PROJECT_ROOT / ".env"


class PaidRunResult(Protocol):
    """Late-bound immutable remote paths returned by the paid runtime."""

    @property
    def state_path(self) -> str:
        """Return the permanent resumable state path."""
        ...

    @property
    def sampler_path(self) -> str:
        """Return the permanent inference sampler path."""
        ...


type PaidRuntime = Callable[[PreparedOutcomeV2Run, str], PaidRunResult]


def outcome_v2_artifacts() -> OutcomeV2Artifacts:
    """Return the one conventional local layout for every sealed artifact."""
    evaluation = NbaEvaluationGateArtifacts(
        cohort_path=DATA_DIRECTORY / EVALUATION_COHORT_FILENAME,
        answers_path=DATA_DIRECTORY / EVALUATION_ANSWERS_FILENAME,
        forecasts_path=DATA_DIRECTORY / EVALUATION_FORECASTS_FILENAME,
        calibration_path=DATA_DIRECTORY / RECALIBRATION_FILENAME,
        supplied_report_path=DATA_DIRECTORY / EVALUATION_REPORT_FILENAME,
    )
    return OutcomeV2Artifacts(
        feature_rows_path=DATA_DIRECTORY / FEATURE_ROWS_FILENAME,
        snapshot_pack_path=DATA_DIRECTORY / SNAPSHOT_PACK_FILENAME,
        evidence_bundles_path=DATA_DIRECTORY / EVIDENCE_BUNDLES_FILENAME,
        elo_states_path=DATA_DIRECTORY / ELO_STATES_FILENAME,
        elo_replay_path=DATA_DIRECTORY / ELO_REPLAY_FILENAME,
        seasons_path=DATA_DIRECTORY / SEASONS_FILENAME,
        resolutions_path=DATA_DIRECTORY / RESOLUTIONS_FILENAME,
        rights_lock_path=DATA_DIRECTORY / RIGHTS_LOCK_FILENAME,
        agreement_path=AGREEMENT_PATH,
        evaluation=evaluation,
        evaluation_feature_rows_path=DATA_DIRECTORY / EVALUATION_FEATURE_ROWS_FILENAME,
        evaluation_elo_replay_path=DATA_DIRECTORY / EVALUATION_ELO_REPLAY_FILENAME,
        evaluation_elo_states_path=DATA_DIRECTORY / EVALUATION_ELO_STATES_FILENAME,
        evaluation_resolutions_path=DATA_DIRECTORY / EVALUATION_RESOLUTIONS_FILENAME,
        rich_baseline_model_path=DATA_DIRECTORY / RICH_BASELINE_MODEL_FILENAME,
        rich_baseline_forecast_lock_path=(
            BASELINE_DIRECTORY / RICH_BASELINE_FORECAST_LOCK_FILENAME
        ),
    )


def commit_outcome_v2_paid_run(code_revision: str) -> tuple[PreparedOutcomeV2Run, OutcomeV2RunLock]:
    """Prepare exact bytes, create the run lock once, and verify it from disk."""
    prepared = prepare_outcome_v2_sft_run(
        MANIFEST_PATH,
        TRAINING_PATH,
        outcome_v2_artifacts(),
    )
    lock = build_outcome_v2_run_lock(PROJECT_ROOT, prepared, code_revision)
    written_sha256 = write_outcome_v2_run_lock(RUN_LOCK_PATH, lock)
    verified = verify_outcome_v2_run_lock(
        PROJECT_ROOT,
        RUN_LOCK_PATH,
        prepared,
        code_revision,
    )
    if written_sha256 != verified.sha256:
        raise OutcomeV2RunError("written outcome-v2 run lock could not be re-verified")
    return prepared, verified


def require_committed_run_unchanged(
    prepared: PreparedOutcomeV2Run,
    code_revision: str,
    expected_sha256: str,
) -> None:
    """Re-verify code, config, bytes, and lock after the paid runtime is imported."""
    verified = verify_outcome_v2_run_lock(
        PROJECT_ROOT,
        RUN_LOCK_PATH,
        prepared,
        code_revision,
    )
    if verified.sha256 != expected_sha256:
        raise OutcomeV2RunError("outcome-v2 run changed while loading its paid runtime")


def seal_outcome_v2_experiment(
    result: PaidRunResult,
    created_at: datetime,
) -> OutcomeV2ExperimentLock:
    """Create and re-verify the post-training state and sampler seal."""
    lock = build_outcome_v2_experiment_lock(
        RUN_LOCK_PATH,
        result.state_path,
        result.sampler_path,
        created_at,
    )
    written_sha256 = write_outcome_v2_experiment_lock(EXPERIMENT_LOCK_PATH, lock)
    verified = verify_outcome_v2_experiment_lock(RUN_LOCK_PATH, EXPERIMENT_LOCK_PATH)
    if written_sha256 != verified.sha256:
        raise OutcomeV2ExperimentError("written outcome-v2 experiment could not be re-verified")
    return verified


def main() -> None:
    """Fail closed locally, then load the module capable of paid calls."""
    _require_unused_output_paths()
    api_key = _local_api_key()
    _require_pinned_packages()
    publication = require_published_head(PROJECT_ROOT, REPOSITORY_URL)
    prepared, lock = commit_outcome_v2_paid_run(publication.commit)
    os.environ["TINKER_API_KEY"] = api_key

    print(f"Committed and verified outcome-v2 run lock: {lock.sha256}")
    print(f"Starting the frozen {MAX_STEPS}-step paid run.")
    paid_runtime = _load_paid_runtime()
    require_committed_run_unchanged(prepared, publication.commit, lock.sha256)
    result = paid_runtime(prepared, lock.sha256)
    experiment = seal_outcome_v2_experiment(result, datetime.now(UTC))
    print(f"Final state path: {result.state_path}")
    print(f"Final sampler path: {result.sampler_path}")
    print(f"Sealed outcome-v2 experiment: {experiment.sha256}")


def _require_unused_output_paths() -> None:
    for path in (RUN_LOCK_PATH, EXPERIMENT_LOCK_PATH):
        if path.exists():
            raise FileExistsError(f"outcome-v2 output already exists: {path}")


def _local_api_key() -> str:
    value = os.environ.get("TINKER_API_KEY")
    if value is not None and value.strip():
        return value
    return read_tinker_api_key(LOCAL_ENV_PATH)


_require_pinned_packages = require_pinned_tinker_packages


def _load_paid_runtime() -> PaidRuntime:
    module = importlib.import_module("examples.tinker_outcome_v2_runtime")
    function = getattr(module, "run_paid", None)
    if not callable(function):
        raise RuntimeError("outcome-v2 paid runtime does not expose run_paid")
    return cast(PaidRuntime, function)


if __name__ == "__main__":
    main()
