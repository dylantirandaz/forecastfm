"""Readable settings for the eventual outcome-v2 historical SFT run."""

from forecastfm.outcome import OUTCOME_INPUT_SCHEMA_VERSION

MANIFEST_SCHEMA_VERSION = 1
TRAINING_FILENAME = "nba_train_outcome.jsonl"

# The frozen historical artifact has 51,359 pairs, so 14 covers every row exactly.
BATCH_SIZE = 14
DROP_LAST = False


def outcome_v2_sft_settings() -> dict[str, object]:
    """Return the small, vendor-neutral settings checked before SFT."""
    return {
        "artifact": "contamination_prone_historical_diagnostic",
        "input_schema_version": OUTCOME_INPUT_SCHEMA_VERSION,
        "batch_size": BATCH_SIZE,
        "drop_last": DROP_LAST,
        "requires_full_outcome_v2_ready": True,
    }
