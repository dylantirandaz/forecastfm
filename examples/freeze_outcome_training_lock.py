"""Freeze the committed ForecastFM outcome-v1 training inputs."""

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from forecastfm.integrity import file_sha256
from forecastfm.outcome_config import MAX_STEPS
from forecastfm.outcome_run_lock import (
    OUTCOME_DATA_MANIFEST_PATH,
    OUTCOME_LOCKED_CODE_PATHS,
    build_outcome_training_lock,
)
from forecastfm.run_lock import write_new_lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    PROJECT_ROOT / "prospective" / "outcome_v1" / f"steps_{MAX_STEPS}" / "training_lock.json"
)


def git_output(*arguments: str) -> str:
    """Run a read-only Git query in the project repository."""
    result = subprocess.run(
        ("git", *arguments),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    """Create the outcome lock from a clean committed code snapshot."""
    required_paths = (*OUTCOME_LOCKED_CODE_PATHS, OUTCOME_DATA_MANIFEST_PATH, Path("uv.lock"))
    status = git_output(
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *(str(path) for path in required_paths),
    )
    if status:
        raise RuntimeError(
            "outcome code and manifest must be tracked and committed before freezing"
        )
    revision = git_output("rev-parse", "HEAD")
    record = build_outcome_training_lock(PROJECT_ROOT, revision, datetime.now(UTC))
    write_new_lock(OUTPUT_PATH, record)
    print(f"Created {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Outcome training lock SHA-256: {file_sha256(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
