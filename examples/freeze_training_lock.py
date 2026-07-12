"""Freeze the exact code, data, prompt, model reference, and run settings."""

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from forecastfm.integrity import file_sha256
from forecastfm.run_lock import build_training_lock, write_new_lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "prospective" / "training_lock.json"


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
    """Create the pretraining lock from a clean committed code snapshot."""
    if git_output("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked files have uncommitted changes; commit them before freezing")
    revision = git_output("rev-parse", "HEAD")
    record = build_training_lock(PROJECT_ROOT, revision, datetime.now(UTC))
    write_new_lock(OUTPUT_PATH, record)
    print(f"Created {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Training lock SHA-256: {file_sha256(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
