"""Readable settings for the eventual outcome-v2 historical SFT run."""

from forecastfm.outcome import OUTCOME_INPUT_SCHEMA_VERSION

MANIFEST_SCHEMA_VERSION = 1
TRAINING_FILENAME = "nba_train_outcome.jsonl"
FEATURE_ROWS_FILENAME = "nba_train_feature_rows.jsonl"
SNAPSHOT_PACK_FILENAME = "nba_train_snapshots.jsonl"
EVIDENCE_BUNDLES_FILENAME = "nba_train_evidence.jsonl"
ELO_STATES_FILENAME = "nba_train_elo_states.jsonl"
SEASONS_FILENAME = "nba_train_seasons.json"
RESOLUTIONS_FILENAME = "nba_train_resolutions.jsonl"
RIGHTS_LOCK_FILENAME = "nba_rights_approval_lock.json"
QUESTION_TEXT = "Will the listed team defeat its opponent in this NBA game?"
RESOLUTION_RULE = "Resolve to the team with the higher final score."

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
        "snapshot_pack_filename": SNAPSHOT_PACK_FILENAME,
        "evidence_bundles_filename": EVIDENCE_BUNDLES_FILENAME,
        "elo_states_filename": ELO_STATES_FILENAME,
        "seasons_filename": SEASONS_FILENAME,
        "resolutions_filename": RESOLUTIONS_FILENAME,
        "rights_lock_filename": RIGHTS_LOCK_FILENAME,
        "requires_full_outcome_v2_ready": True,
    }
