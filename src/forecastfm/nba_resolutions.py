"""Sealed NBA final scores with independent snapshot and label validation.

This module does not parse opaque provider payloads. A licensed connector must derive
the two scores from the exact snapshot bytes before constructing ``NbaResolution``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from forecastfm.integrity import canonical_json
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_string,
    required_field,
)
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_evidence import NbaEvidenceBundle
from forecastfm.nba_snapshot_pack import (
    NbaSnapshotIndex,
    SnapshotPackError,
    snapshot_metadata_sha256,
)
from forecastfm.outcome import OPPONENT_LABEL, TEAM_LABEL
from forecastfm.tinker_data import OutcomeTrainingRecord

NBA_RESOLUTION_SCHEMA_VERSION = 2

type NbaResolutionSite = Literal["home", "away", "neutral"]

_HASH_CHARACTERS = frozenset("0123456789abcdef")
_ALLOWED_SITES = frozenset({"home", "away", "neutral"})
_RECORD_KEYS = {
    "schema_version",
    "question_id",
    "source_game_id",
    "team_id",
    "opponent_id",
    "site",
    "team_score",
    "opponent_score",
    "resolved_at",
    "source_id",
    "snapshot_metadata_sha256",
}


class NbaResolutionError(ValueError):
    """Raised when a sealed NBA resolution violates its integrity contract."""


@dataclass(frozen=True, slots=True)
class NbaResolution:
    """One claimed final score structurally linked to retained snapshot metadata."""

    question_id: str
    source_game_id: str
    team_id: str
    opponent_id: str
    site: NbaResolutionSite
    team_score: int
    opponent_score: int
    resolved_at: datetime
    source_id: str
    snapshot_metadata_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.question_id, "question_id")
        if self.question_id.endswith(SIDE_SWAP_SUFFIX):
            raise NbaResolutionError("resolution question_id must identify an original game")
        _require_text(self.source_game_id, "source_game_id")
        _require_text(self.team_id, "team_id")
        _require_text(self.opponent_id, "opponent_id")
        if self.team_id == self.opponent_id:
            raise NbaResolutionError("resolution team_id and opponent_id must differ")
        if self.site not in _ALLOWED_SITES:
            raise NbaResolutionError("resolution site must be home, away, or neutral")
        _require_score(self.team_score, "team_score")
        _require_score(self.opponent_score, "opponent_score")
        if self.team_score == self.opponent_score:
            raise NbaResolutionError("NBA resolution scores cannot be tied")
        _require_utc(self.resolved_at, "resolved_at")
        _require_text(self.source_id, "source_id")
        _require_sha256(self.snapshot_metadata_sha256, "snapshot_metadata_sha256")

    @property
    def team_won(self) -> bool:
        """Return whether the listed team won according to the sealed score."""
        return self.team_score > self.opponent_score


def write_nba_resolutions_jsonl(
    path: Path,
    resolutions: Iterable[NbaResolution],
    *,
    snapshot_index: NbaSnapshotIndex,
) -> None:
    """Create a canonical resolution file after validating snapshot linkage."""
    checked = _validate_resolution_collection(tuple(resolutions), snapshot_index)
    try:
        with path.open("x", encoding="utf-8", newline="") as file:
            file.write(_canonical_jsonl(checked))
    except FileExistsError as error:
        raise NbaResolutionError("NBA resolution JSONL already exists") from error
    except OSError as error:
        raise NbaResolutionError("cannot write NBA resolution JSONL") from error


def read_nba_resolutions_jsonl(
    path: Path,
    *,
    snapshot_index: NbaSnapshotIndex,
) -> tuple[NbaResolution, ...]:
    """Load canonical resolutions and rebind each one to its latest source snapshot."""
    try:
        value = path.read_bytes()
    except OSError as error:
        raise NbaResolutionError("cannot read NBA resolution JSONL") from error
    return read_nba_resolutions_jsonl_bytes(value, snapshot_index=snapshot_index)


def read_nba_resolutions_jsonl_bytes(
    value: bytes,
    *,
    snapshot_index: NbaSnapshotIndex,
) -> tuple[NbaResolution, ...]:
    """Load canonical resolution bytes and bind them to source snapshots."""
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NbaResolutionError("cannot read NBA resolution JSONL") from error

    resolutions: list[NbaResolution] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            resolutions.append(_resolution_from_payload(parse_json_object(line)))
        except ValueError as error:
            message = f"invalid NBA resolution at line {line_number}"
            raise NbaResolutionError(message) from error

    checked = _validate_resolution_collection(tuple(resolutions), snapshot_index)
    if text != _canonical_jsonl(checked):
        raise NbaResolutionError("NBA resolutions must use canonical JSONL encoding")
    return checked


def validate_outcome_training_labels(
    bundles: Sequence[NbaEvidenceBundle],
    resolutions: Sequence[NbaResolution],
    records: Sequence[OutcomeTrainingRecord],
    *,
    snapshot_index: NbaSnapshotIndex,
    action_at: datetime,
) -> None:
    """Verify that each original/swap label comes only from its sealed final score."""
    _require_utc(action_at, "action_at")
    checked_resolutions = _validate_resolution_collection(tuple(resolutions), snapshot_index)
    if any(resolution.resolved_at > action_at for resolution in checked_resolutions):
        raise NbaResolutionError("resolution cannot postdate the protected action")
    checked_bundles = tuple(bundles)
    _require_bundle_alignment(checked_bundles, checked_resolutions)
    if len(records) != 2 * len(checked_resolutions):
        raise NbaResolutionError("training rows do not contain one complete pair per resolution")

    for index, (bundle, resolution) in enumerate(
        zip(checked_bundles, checked_resolutions, strict=True)
    ):
        _require_training_pair(
            bundle.game.question_id,
            resolution,
            records[2 * index],
            records[2 * index + 1],
        )


def _validate_resolution_collection(
    resolutions: tuple[NbaResolution, ...],
    snapshot_index: NbaSnapshotIndex,
) -> tuple[NbaResolution, ...]:
    if not resolutions:
        raise NbaResolutionError("NBA resolution JSONL must not be empty")
    question_ids = tuple(resolution.question_id for resolution in resolutions)
    source_game_ids = tuple(resolution.source_game_id for resolution in resolutions)
    if len(set(question_ids)) != len(question_ids):
        raise NbaResolutionError("NBA resolutions contain a duplicate question_id")
    if len(set(source_game_ids)) != len(source_game_ids):
        raise NbaResolutionError("NBA resolutions contain a duplicate source_game_id")
    for resolution in resolutions:
        _require_latest_snapshot(resolution, snapshot_index)
    return resolutions


def _require_latest_snapshot(
    resolution: NbaResolution,
    snapshot_index: NbaSnapshotIndex,
) -> None:
    try:
        snapshot = snapshot_index.latest_eligible(resolution.source_id, resolution.resolved_at)
    except SnapshotPackError as error:
        raise NbaResolutionError("cannot select one latest resolution snapshot") from error
    if snapshot is None:
        raise NbaResolutionError("resolution has no source snapshot available by resolved_at")
    expected_sha256 = snapshot_metadata_sha256(snapshot.metadata)
    if resolution.snapshot_metadata_sha256 != expected_sha256:
        raise NbaResolutionError("resolution is not bound to the latest eligible snapshot")


def _require_bundle_alignment(
    bundles: tuple[NbaEvidenceBundle, ...],
    resolutions: tuple[NbaResolution, ...],
) -> None:
    bundle_ids = tuple(bundle.game.question_id for bundle in bundles)
    resolution_ids = tuple(resolution.question_id for resolution in resolutions)
    if bundle_ids != resolution_ids:
        raise NbaResolutionError("resolution IDs or order differ from the original bundles")
    for bundle, resolution in zip(bundles, resolutions, strict=True):
        if bundle.game.source_game_id != resolution.source_game_id:
            raise NbaResolutionError("resolution source_game_id differs from its original bundle")
        if (
            resolution.team_id,
            resolution.opponent_id,
            resolution.site,
        ) != (
            bundle.game.team_id,
            bundle.game.opponent_id,
            bundle.game.site,
        ):
            raise NbaResolutionError("resolution orientation differs from its original bundle")
        if resolution.resolved_at <= bundle.game.scheduled_tipoff:
            raise NbaResolutionError("resolution must occur after the frozen scheduled tipoff")


def _require_training_pair(
    question_id: str,
    resolution: NbaResolution,
    original: OutcomeTrainingRecord,
    swapped: OutcomeTrainingRecord,
) -> None:
    if original["question_id"] != question_id:
        raise NbaResolutionError("original training row differs from resolution order")
    if swapped["question_id"] != f"{question_id}{SIDE_SWAP_SUFFIX}":
        raise NbaResolutionError("side-swap training row differs from resolution order")
    original_label = TEAM_LABEL if resolution.team_won else OPPONENT_LABEL
    swapped_label = OPPONENT_LABEL if resolution.team_won else TEAM_LABEL
    if original["label"] != original_label or swapped["label"] != swapped_label:
        raise NbaResolutionError("training labels do not match the sealed score winner")


def _canonical_jsonl(resolutions: Sequence[NbaResolution]) -> str:
    return "".join(f"{canonical_json(_resolution_to_payload(item))}\n" for item in resolutions)


def _resolution_to_payload(resolution: NbaResolution) -> dict[str, object]:
    return {
        "schema_version": NBA_RESOLUTION_SCHEMA_VERSION,
        "question_id": resolution.question_id,
        "source_game_id": resolution.source_game_id,
        "team_id": resolution.team_id,
        "opponent_id": resolution.opponent_id,
        "site": resolution.site,
        "team_score": resolution.team_score,
        "opponent_score": resolution.opponent_score,
        "resolved_at": _utc_text(resolution.resolved_at),
        "source_id": resolution.source_id,
        "snapshot_metadata_sha256": resolution.snapshot_metadata_sha256,
    }


def _resolution_from_payload(payload: Mapping[str, object]) -> NbaResolution:
    require_exact_keys(payload, _RECORD_KEYS, "NBA resolution")
    version = _integer_field(payload, "schema_version")
    if version != NBA_RESOLUTION_SCHEMA_VERSION:
        raise JsonFormatError(f"unsupported NBA resolution schema version: {version}")
    return NbaResolution(
        question_id=_string_field(payload, "question_id"),
        source_game_id=_string_field(payload, "source_game_id"),
        team_id=_string_field(payload, "team_id"),
        opponent_id=_string_field(payload, "opponent_id"),
        site=_site_field(payload),
        team_score=_integer_field(payload, "team_score"),
        opponent_score=_integer_field(payload, "opponent_score"),
        resolved_at=_time_field(payload, "resolved_at"),
        source_id=_string_field(payload, "source_id"),
        snapshot_metadata_sha256=_string_field(payload, "snapshot_metadata_sha256"),
    )


def _string_field(payload: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(payload, field_name), field_name)


def _integer_field(payload: Mapping[str, object], field_name: str) -> int:
    value = required_field(payload, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise JsonFormatError(f"{field_name} must be an integer")
    return value


def _site_field(payload: Mapping[str, object]) -> NbaResolutionSite:
    value = _string_field(payload, "site")
    if value == "home":
        return "home"
    if value == "away":
        return "away"
    if value == "neutral":
        return "neutral"
    raise JsonFormatError("site must be home, away, or neutral")


def _time_field(payload: Mapping[str, object], field_name: str) -> datetime:
    text = _string_field(payload, field_name)
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise JsonFormatError(f"{field_name} must be an ISO 8601 datetime") from error
    _require_utc(value, field_name)
    return value.astimezone(UTC)


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise NbaResolutionError(f"{field_name} must not be empty")


def _require_score(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NbaResolutionError(f"{field_name} must be a nonnegative integer")


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in _HASH_CHARACTERS for character in value):
        raise NbaResolutionError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise NbaResolutionError(f"{field_name} must be in UTC")


def _utc_text(value: datetime) -> str:
    _require_utc(value, "datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
