"""Verify the prospective ledger against its frozen experiment and optional public head."""

import sys
from pathlib import Path

from forecastfm.integrity import file_sha256
from forecastfm.ledger import audit_ledger
from forecastfm.run_lock import verify_experiment_lock, verify_training_lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROSPECTIVE_PATH = PROJECT_ROOT / "prospective"
TRAINING_LOCK_PATH = PROSPECTIVE_PATH / "training_lock.json"
EXPERIMENT_LOCK_PATH = PROSPECTIVE_PATH / "experiment.json"
LEDGER_PATH = PROSPECTIVE_PATH / "ledger.jsonl"


def expected_head(arguments: list[str]) -> str | None:
    """Accept zero arguments or one externally published ledger head."""
    if len(arguments) > 1:
        raise RuntimeError("usage: verify_prospective_ledger.py [expected-head-sha256]")
    return arguments[0] if arguments else None


def main() -> None:
    """Validate every local commitment and print the current ledger head."""
    verify_training_lock(PROJECT_ROOT, TRAINING_LOCK_PATH)
    verify_experiment_lock(TRAINING_LOCK_PATH, EXPERIMENT_LOCK_PATH)
    audit = audit_ledger(
        LEDGER_PATH,
        expected_head=expected_head(sys.argv[1:]),
        expected_experiment_sha256=file_sha256(EXPERIMENT_LOCK_PATH),
    )
    print(f"Verified {audit.event_count} events across {audit.cohort_count} cohorts.")
    print(f"Ledger head SHA-256: {audit.head_sha256}")
    if audit.unresolved_cohort_ids:
        print(f"Unresolved cohorts: {', '.join(audit.unresolved_cohort_ids)}")


if __name__ == "__main__":
    main()
