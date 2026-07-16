"""Tests for the offline outcome-v2 SFT readiness gate."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from forecastfm.integrity import canonical_sha256, file_sha256
from forecastfm.json_utils import parse_json_object, require_list, require_object, required_field
from forecastfm.nba_rich import NBA_RICH_SCHEMA_SHA256, NBA_RICH_SCHEMA_VERSION
from forecastfm.outcome import (
    OPPONENT_LABEL,
    OPPONENT_OUTCOME,
    OUTCOME_SYSTEM_PROMPT,
    TEAM_LABEL,
    TEAM_OUTCOME,
)
from forecastfm.outcome_v2_config import BATCH_SIZE, DROP_LAST, outcome_v2_sft_settings
from forecastfm.outcome_v2_preflight import (
    OutcomeV2PreflightError,
    require_outcome_v2_sft_ready,
)

PROJECT_ROOT = Path(__file__).parents[1]
CHECKED_IN_MANIFEST = PROJECT_ROOT / "data/processed/outcome_v2/manifest.json"


@dataclass(frozen=True, slots=True)
class _ManifestOptions:
    ready: bool = True
    input_schema_version: int = 2
    third_party_processing: str = "allowed"
    tinker_processing: str = "allowed"
    player_health_included: bool = False
    training_hash: str | None = None
    id_hash: str | None = None
    feature_schema_hash: str = NBA_RICH_SCHEMA_SHA256


_DEFAULT_MANIFEST_OPTIONS = _ManifestOptions()


def _record(question_id: str, label: str, *, swapped: bool = False) -> dict[str, object]:
    team_probability = 0.4 if swapped else 0.6
    feature_value = -1.0 if swapped else 1.0
    user_prompt = {
        "evidence": [f'Pregame numeric feature: {{"signal":{feature_value}}}'],
        "outcomes": [TEAM_OUTCOME, OPPONENT_OUTCOME],
        "prior": {
            TEAM_OUTCOME: team_probability,
            OPPONENT_OUTCOME: 1.0 - team_probability,
        },
        "question": "Will the listed team win?",
        "resolution_rule": "Use the final score.",
    }
    return {
        "question_id": question_id,
        "messages": [
            {"role": "system", "content": OUTCOME_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt, sort_keys=True)},
        ],
        "label": label,
    }


def _write_training(
    path: Path,
    *,
    pair_count: int = 7,
    swapped_suffix: str = "-side-swap",
) -> tuple[str, ...]:
    original_ids = tuple(f"game-{index}" for index in range(pair_count))
    rows: list[dict[str, object]] = []
    for question_id in original_ids:
        rows.append(_record(question_id, TEAM_LABEL))
        rows.append(
            _record(
                f"{question_id}{swapped_suffix}",
                OPPONENT_LABEL,
                swapped=True,
            )
        )
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )
    return original_ids


def _write_manifest(
    path: Path,
    training_path: Path,
    original_ids: tuple[str, ...],
    *,
    options: _ManifestOptions = _DEFAULT_MANIFEST_OPTIONS,
) -> None:
    row_count = len(original_ids) * 2
    manifest: dict[str, object] = {
        "schema_version": 1,
        "outcome_input_schema_version": options.input_schema_version,
        "evaluation": {"full_outcome_v2_ready": options.ready},
        "features": {
            "full_schema": {
                "version": NBA_RICH_SCHEMA_VERSION,
                "sha256": options.feature_schema_hash,
                "current_artifact_contains_full_schema": True,
            }
        },
        "upload_rights": {
            "third_party_processing": options.third_party_processing,
            "tinker_processing": options.tinker_processing,
            "player_health_included": options.player_health_included,
        },
        "outputs": {
            training_path.name: options.training_hash or file_sha256(training_path),
        },
        "splits": {
            "train": {
                "original_games": len(original_ids),
                "side_swapped_training_rows": row_count,
                "question_ids_sha256": options.id_hash or canonical_sha256(list(original_ids)),
            }
        },
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _ready_fixture(tmp_path: Path, *, pair_count: int = 7) -> tuple[Path, Path]:
    training_path = tmp_path / "nba_train_outcome.jsonl"
    manifest_path = tmp_path / "manifest.json"
    original_ids = _write_training(training_path, pair_count=pair_count)
    _write_manifest(manifest_path, training_path, original_ids)
    return manifest_path, training_path


def test_config_keeps_every_historical_artifact_row() -> None:
    settings = outcome_v2_sft_settings()

    assert BATCH_SIZE == 14
    assert DROP_LAST is False
    assert settings["batch_size"] == 14
    assert settings["drop_last"] is False


def test_checked_in_manifest_fails_closed_before_file_access() -> None:
    missing_training_path = PROJECT_ROOT / "does-not-exist/nba_train_outcome.jsonl"

    with pytest.raises(OutcomeV2PreflightError, match="full_outcome_v2_ready is false"):
        require_outcome_v2_sft_ready(CHECKED_IN_MANIFEST, missing_training_path)


def test_valid_artifact_passes_without_tinker(tmp_path: Path) -> None:
    manifest_path, training_path = _ready_fixture(tmp_path)

    result = require_outcome_v2_sft_ready(manifest_path, training_path)

    assert result.training_sha256 == file_sha256(training_path)
    assert result.row_count == 14
    assert result.pair_count == 7
    assert result.batch_size == 14


def test_readiness_is_checked_before_upload_rights(tmp_path: Path) -> None:
    training_path = tmp_path / "nba_train_outcome.jsonl"
    manifest_path = tmp_path / "manifest.json"
    original_ids = _write_training(training_path)
    options = _ManifestOptions(ready=False, third_party_processing="unknown")
    _write_manifest(manifest_path, training_path, original_ids, options=options)

    with pytest.raises(OutcomeV2PreflightError, match="full_outcome_v2_ready is false"):
        require_outcome_v2_sft_ready(manifest_path, training_path)


def test_unknown_upload_rights_fail_closed(tmp_path: Path) -> None:
    training_path = tmp_path / "nba_train_outcome.jsonl"
    manifest_path = tmp_path / "manifest.json"
    original_ids = _write_training(training_path)
    _write_manifest(
        manifest_path,
        training_path,
        original_ids,
        options=_ManifestOptions(third_party_processing="unknown"),
    )

    with pytest.raises(OutcomeV2PreflightError, match="third_party_processing"):
        require_outcome_v2_sft_ready(manifest_path, training_path)


def test_richer_feature_schema_hash_is_frozen(tmp_path: Path) -> None:
    training_path = tmp_path / "nba_train_outcome.jsonl"
    manifest_path = tmp_path / "manifest.json"
    original_ids = _write_training(training_path)
    _write_manifest(
        manifest_path,
        training_path,
        original_ids,
        options=_ManifestOptions(feature_schema_hash="0" * 64),
    )

    with pytest.raises(OutcomeV2PreflightError, match="feature schema hash differs"):
        require_outcome_v2_sft_ready(manifest_path, training_path)


def test_tinker_upload_rights_must_be_explicitly_allowed(tmp_path: Path) -> None:
    training_path = tmp_path / "nba_train_outcome.jsonl"
    manifest_path = tmp_path / "manifest.json"
    original_ids = _write_training(training_path)
    _write_manifest(
        manifest_path,
        training_path,
        original_ids,
        options=_ManifestOptions(tinker_processing="unknown"),
    )

    with pytest.raises(OutcomeV2PreflightError, match="tinker_processing"):
        require_outcome_v2_sft_ready(manifest_path, training_path)


def test_standard_tinker_training_rejects_player_health_lineage(tmp_path: Path) -> None:
    training_path = tmp_path / "nba_train_outcome.jsonl"
    manifest_path = tmp_path / "manifest.json"
    original_ids = _write_training(training_path)
    _write_manifest(
        manifest_path,
        training_path,
        original_ids,
        options=_ManifestOptions(player_health_included=True),
    )

    with pytest.raises(OutcomeV2PreflightError, match="cannot include player health"):
        require_outcome_v2_sft_ready(manifest_path, training_path)


def test_training_rows_are_health_screened_again_at_preflight(tmp_path: Path) -> None:
    training_path = tmp_path / "nba_train_outcome.jsonl"
    manifest_path = tmp_path / "manifest.json"
    original_ids = _write_training(training_path)
    text = training_path.read_text(encoding="utf-8").replace("signal", "injury_signal")
    training_path.write_text(text, encoding="utf-8")
    _write_manifest(manifest_path, training_path, original_ids)

    with pytest.raises(OutcomeV2PreflightError, match="health-language screen"):
        require_outcome_v2_sft_ready(manifest_path, training_path)


def test_input_schema_version_must_match_outcome_v2(tmp_path: Path) -> None:
    training_path = tmp_path / "nba_train_outcome.jsonl"
    manifest_path = tmp_path / "manifest.json"
    original_ids = _write_training(training_path)
    _write_manifest(
        manifest_path,
        training_path,
        original_ids,
        options=_ManifestOptions(input_schema_version=1),
    )

    with pytest.raises(OutcomeV2PreflightError, match="input schema version"):
        require_outcome_v2_sft_ready(manifest_path, training_path)


def test_training_hash_must_match_manifest(tmp_path: Path) -> None:
    training_path = tmp_path / "nba_train_outcome.jsonl"
    manifest_path = tmp_path / "manifest.json"
    original_ids = _write_training(training_path)
    _write_manifest(
        manifest_path,
        training_path,
        original_ids,
        options=_ManifestOptions(training_hash="0" * 64),
    )

    with pytest.raises(OutcomeV2PreflightError, match="SHA-256 does not match"):
        require_outcome_v2_sft_ready(manifest_path, training_path)


def test_side_swap_rows_must_be_adjacent_and_exact(tmp_path: Path) -> None:
    training_path = tmp_path / "nba_train_outcome.jsonl"
    manifest_path = tmp_path / "manifest.json"
    original_ids = _write_training(training_path, swapped_suffix="-wrong")
    _write_manifest(manifest_path, training_path, original_ids)

    with pytest.raises(OutcomeV2PreflightError, match="exact adjacent side-swap pairs"):
        require_outcome_v2_sft_ready(manifest_path, training_path)


def test_side_swap_prompt_content_must_be_an_exact_complement(tmp_path: Path) -> None:
    training_path = tmp_path / "nba_train_outcome.jsonl"
    manifest_path = tmp_path / "manifest.json"
    original_ids = _write_training(training_path)
    rows = [
        parse_json_object(line) for line in training_path.read_text(encoding="utf-8").splitlines()
    ]
    original_messages = require_list(required_field(rows[0], "messages"), "messages")
    swapped_messages = require_list(required_field(rows[1], "messages"), "messages")
    original_user = require_object(original_messages[1], "original user")
    swapped_user = require_object(swapped_messages[1], "swapped user")
    swapped_user["content"] = required_field(original_user, "content")
    swapped_messages[1] = swapped_user
    rows[1]["messages"] = swapped_messages
    training_path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )
    _write_manifest(manifest_path, training_path, original_ids)

    with pytest.raises(OutcomeV2PreflightError, match="not exact complements"):
        require_outcome_v2_sft_ready(manifest_path, training_path)


def test_original_id_order_must_match_manifest(tmp_path: Path) -> None:
    training_path = tmp_path / "nba_train_outcome.jsonl"
    manifest_path = tmp_path / "manifest.json"
    original_ids = _write_training(training_path)
    _write_manifest(
        manifest_path,
        training_path,
        original_ids,
        options=_ManifestOptions(id_hash="0" * 64),
    )

    with pytest.raises(OutcomeV2PreflightError, match="IDs or their order"):
        require_outcome_v2_sft_ready(manifest_path, training_path)


def test_batch_size_cannot_drop_training_rows(tmp_path: Path) -> None:
    manifest_path, training_path = _ready_fixture(tmp_path, pair_count=6)

    with pytest.raises(OutcomeV2PreflightError, match="not divisible by batch size 14"):
        require_outcome_v2_sft_ready(manifest_path, training_path)
