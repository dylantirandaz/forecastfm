"""Canonical I/O and causal validation for derived NBA evidence bundles.

This module only preserves records already derived from selected raw snapshots. It does
not interpret opaque provider payload bytes. A production connector must derive every
``NbaEvidenceRecord`` from those bytes, then pass the records through this boundary.
Categorical question ``outcomes`` are inputs; realized outcomes, scores, labels, targets,
and postgame results are deliberately absent from the schema.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from forecastfm.integrity import canonical_json
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_float,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.ledger import CohortGame
from forecastfm.models import ForecastQuestion
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_evidence import (
    CaptureMethod,
    EvidenceKind,
    NbaEvidenceBundle,
    NbaEvidenceRecord,
    Permission,
    Sensitivity,
    SourceRights,
    SourceSnapshot,
    evidence_bundle_sha256,
)
from forecastfm.nba_feature_rows import NbaRichFeatureRow
from forecastfm.nba_rich import tinker_rich_features_from_bundle
from forecastfm.nba_snapshot_pack import NbaSnapshotIndex

NBA_EVIDENCE_JSONL_SCHEMA_VERSION = 2
_BUNDLE_KEYS = {
    "schema_version",
    "evidence_bundle_sha256",
    "game",
    "question",
    "sources",
    "records",
}
_GAME_KEYS = {
    "question_id",
    "source_game_id",
    "matchup",
    "outcomes",
    "forecast_deadline",
    "scheduled_tipoff",
}
_QUESTION_KEYS = {
    "question_id",
    "text",
    "resolution_rule",
    "resolution_source",
    "outcomes",
    "forecast_at",
    "resolves_at",
}
_SOURCE_KEYS = {
    "source_id",
    "rights_scope",
    "source_url",
    "payload_sha256",
    "snapshot_metadata_sha256",
    "published_at",
    "retrieved_at",
    "capture_method",
    "sensitivity",
    "archive_version_id",
    "archive_attestation_sha256",
    "rights",
}
_RIGHTS_KEYS = {
    "license_name",
    "terms_url",
    "terms_sha256",
    "rights_as_of",
    "local_processing",
    "third_party_processing",
    "tinker_processing",
    "redistribution",
}
_RECORD_KEYS = {
    "record_id",
    "kind",
    "feature_name",
    "team_value",
    "opponent_value",
    "source_ids",
    "available_at",
}

type JsonObject = dict[str, object]


class NbaEvidenceIoError(ValueError):
    """Raised when serialized NBA evidence violates its causal contract."""


def write_nba_evidence_bundles_jsonl(
    path: Path,
    bundles: Iterable[NbaEvidenceBundle],
    *,
    snapshot_index: NbaSnapshotIndex,
) -> None:
    """Write nonempty original bundles after binding sources to the snapshot index."""
    checked = _validate_bundle_collection(tuple(bundles), snapshot_index)
    try:
        with path.open("x", encoding="utf-8", newline="") as file:
            file.write(_canonical_jsonl(checked))
    except FileExistsError as error:
        raise NbaEvidenceIoError("NBA evidence-bundle JSONL already exists") from error
    except OSError as error:
        raise NbaEvidenceIoError("cannot write NBA evidence-bundle JSONL") from error


def read_nba_evidence_bundles_jsonl(
    path: Path,
    *,
    snapshot_index: NbaSnapshotIndex,
) -> tuple[NbaEvidenceBundle, ...]:
    """Read canonical original bundles and bind every source to its latest snapshot."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise NbaEvidenceIoError("cannot read NBA evidence-bundle JSONL") from error

    bundles: list[NbaEvidenceBundle] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            bundles.append(_bundle_from_payload(parse_json_object(line)))
        except ValueError as error:
            message = f"invalid NBA evidence bundle at line {line_number}"
            raise NbaEvidenceIoError(message) from error

    checked = _validate_bundle_collection(tuple(bundles), snapshot_index)
    if text != _canonical_jsonl(checked):
        raise NbaEvidenceIoError("NBA evidence bundles must use canonical JSONL encoding")
    return checked


def validate_tinker_feature_rows_from_bundles(
    bundles: Sequence[NbaEvidenceBundle],
    rows: Sequence[NbaRichFeatureRow],
    frozen_seasons: Mapping[str, object],
    *,
    action_at: datetime,
) -> None:
    """Recompute ordinary Tinker features and verify exact row and season alignment."""
    _require_utc(action_at, "action_at")
    bundle_ids = _require_original_question_ids(tuple(bundles))
    row_ids = tuple(row.question_id for row in rows)
    if row_ids != bundle_ids:
        raise NbaEvidenceIoError("feature-row IDs and order must exactly match evidence bundles")
    checked_seasons = _require_frozen_seasons(bundle_ids, frozen_seasons)

    for bundle, row in zip(bundles, rows, strict=True):
        features = tinker_rich_features_from_bundle(bundle, action_at=action_at)
        _validate_feature_row(bundle, row, features.vector, checked_seasons[row.question_id])


def _validate_feature_row(
    bundle: NbaEvidenceBundle,
    row: NbaRichFeatureRow,
    feature_vector: tuple[float, ...],
    expected_season: int,
) -> None:
    game = bundle.game
    if row.season != expected_season:
        raise NbaEvidenceIoError("feature-row season differs from the frozen season mapping")
    if row.forecast_cutoff != game.forecast_deadline:
        raise NbaEvidenceIoError("feature-row cutoff differs from its evidence bundle")
    if row.scheduled_tipoff != game.scheduled_tipoff:
        raise NbaEvidenceIoError("feature-row tipoff differs from its evidence bundle")
    if row.evidence_bundle_sha256 != evidence_bundle_sha256(bundle):
        raise NbaEvidenceIoError("feature-row evidence digest differs from its evidence bundle")
    if row.rich_features.vector != feature_vector:
        raise NbaEvidenceIoError("feature-row vector differs from recomputed Tinker features")
    latest_input_at = max(
        row.elo_available_at,
        *(record.available_at for record in bundle.records),
    )
    if row.input_available_at != latest_input_at:
        raise NbaEvidenceIoError("feature-row input availability differs from its causal inputs")


def _require_frozen_seasons(
    question_ids: tuple[str, ...],
    frozen_seasons: Mapping[str, object],
) -> dict[str, int]:
    if set(frozen_seasons) != set(question_ids):
        raise NbaEvidenceIoError("frozen season mapping must exactly cover evidence bundles")
    checked: dict[str, int] = {}
    for question_id in question_ids:
        season = frozen_seasons[question_id]
        if isinstance(season, bool) or not isinstance(season, int) or season <= 0:
            raise NbaEvidenceIoError("frozen seasons must be positive integers")
        checked[question_id] = season
    return checked


def _validate_bundle_collection(
    bundles: tuple[NbaEvidenceBundle, ...],
    snapshot_index: NbaSnapshotIndex,
) -> tuple[NbaEvidenceBundle, ...]:
    _require_original_question_ids(bundles)
    for bundle in bundles:
        _validate_sources_against_index(bundle, snapshot_index)
    return bundles


def _require_original_question_ids(bundles: tuple[NbaEvidenceBundle, ...]) -> tuple[str, ...]:
    if not bundles:
        raise NbaEvidenceIoError("NBA evidence-bundle JSONL must not be empty")
    question_ids = tuple(bundle.game.question_id for bundle in bundles)
    if any(question_id.endswith(SIDE_SWAP_SUFFIX) for question_id in question_ids):
        raise NbaEvidenceIoError("NBA evidence-bundle JSONL may contain only original bundles")
    if len(set(question_ids)) != len(question_ids):
        raise NbaEvidenceIoError("NBA evidence-bundle JSONL contains a duplicate question ID")
    return question_ids


def _validate_sources_against_index(
    bundle: NbaEvidenceBundle,
    snapshot_index: NbaSnapshotIndex,
) -> None:
    cutoff = bundle.game.forecast_deadline
    for source in bundle.sources:
        latest = snapshot_index.latest_eligible(source.source_id, cutoff)
        if latest is None:
            message = f"no eligible snapshot for source {source.source_id}"
            raise NbaEvidenceIoError(message)
        if source != latest.to_source_snapshot():
            message = f"bundle source is not the latest eligible snapshot: {source.source_id}"
            raise NbaEvidenceIoError(message)


def _canonical_jsonl(bundles: Sequence[NbaEvidenceBundle]) -> str:
    return "".join(f"{canonical_json(_bundle_to_payload(bundle))}\n" for bundle in bundles)


def _bundle_to_payload(bundle: NbaEvidenceBundle) -> JsonObject:
    return {
        "schema_version": NBA_EVIDENCE_JSONL_SCHEMA_VERSION,
        "evidence_bundle_sha256": evidence_bundle_sha256(bundle),
        "game": _game_to_payload(bundle.game),
        "question": _question_to_payload(bundle.question),
        "sources": [_source_to_payload(source) for source in bundle.sources],
        "records": [_record_to_payload(record) for record in bundle.records],
    }


def _game_to_payload(game: CohortGame) -> JsonObject:
    return {
        "question_id": game.question_id,
        "source_game_id": game.source_game_id,
        "matchup": game.matchup,
        "outcomes": list(game.outcomes),
        "forecast_deadline": _utc_text(game.forecast_deadline),
        "scheduled_tipoff": _utc_text(game.scheduled_tipoff),
    }


def _question_to_payload(question: ForecastQuestion) -> JsonObject:
    return {
        "question_id": question.question_id,
        "text": question.text,
        "resolution_rule": question.resolution_rule,
        "resolution_source": question.resolution_source,
        "outcomes": list(question.outcomes),
        "forecast_at": _utc_text(question.forecast_at),
        "resolves_at": _utc_text(question.resolves_at),
    }


def _source_to_payload(source: SourceSnapshot) -> JsonObject:
    return {
        "source_id": source.source_id,
        "rights_scope": source.rights_scope,
        "source_url": source.source_url,
        "payload_sha256": source.payload_sha256,
        "snapshot_metadata_sha256": source.snapshot_metadata_sha256,
        "published_at": _utc_text(source.published_at),
        "retrieved_at": _utc_text(source.retrieved_at),
        "capture_method": source.capture_method,
        "sensitivity": source.sensitivity,
        "archive_version_id": source.archive_version_id,
        "archive_attestation_sha256": source.archive_attestation_sha256,
        "rights": _rights_to_payload(source.rights),
    }


def _rights_to_payload(rights: SourceRights) -> JsonObject:
    return {
        "license_name": rights.license_name,
        "terms_url": rights.terms_url,
        "terms_sha256": rights.terms_sha256,
        "rights_as_of": _utc_text(rights.rights_as_of),
        "local_processing": rights.local_processing,
        "third_party_processing": rights.third_party_processing,
        "tinker_processing": rights.tinker_processing,
        "redistribution": rights.redistribution,
    }


def _record_to_payload(record: NbaEvidenceRecord) -> JsonObject:
    return {
        "record_id": record.record_id,
        "kind": record.kind,
        "feature_name": record.feature_name,
        "team_value": record.team_value,
        "opponent_value": record.opponent_value,
        "source_ids": list(record.source_ids),
        "available_at": _utc_text(record.available_at),
    }


def _bundle_from_payload(payload: Mapping[str, object]) -> NbaEvidenceBundle:
    require_exact_keys(payload, _BUNDLE_KEYS, "NBA evidence bundle")
    version = required_field(payload, "schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise JsonFormatError("schema_version must be an integer")
    if version != NBA_EVIDENCE_JSONL_SCHEMA_VERSION:
        raise JsonFormatError(f"unsupported NBA evidence schema version: {version}")
    bundle = NbaEvidenceBundle(
        game=_game_from_payload(_object_field(payload, "game")),
        question=_question_from_payload(_object_field(payload, "question")),
        sources=tuple(
            _source_from_payload(require_object(item, "source"))
            for item in _list_field(payload, "sources")
        ),
        records=tuple(
            _record_from_payload(require_object(item, "record"))
            for item in _list_field(payload, "records")
        ),
    )
    stored_sha256 = _string_field(payload, "evidence_bundle_sha256")
    if stored_sha256 != evidence_bundle_sha256(bundle):
        raise JsonFormatError("evidence_bundle_sha256 does not match reconstructed evidence")
    return bundle


def _game_from_payload(payload: Mapping[str, object]) -> CohortGame:
    require_exact_keys(payload, _GAME_KEYS, "game")
    return CohortGame(
        question_id=_string_field(payload, "question_id"),
        source_game_id=_string_field(payload, "source_game_id"),
        matchup=_string_field(payload, "matchup"),
        outcomes=_string_tuple(payload, "outcomes"),
        forecast_deadline=_time_field(payload, "forecast_deadline"),
        scheduled_tipoff=_time_field(payload, "scheduled_tipoff"),
    )


def _question_from_payload(payload: Mapping[str, object]) -> ForecastQuestion:
    require_exact_keys(payload, _QUESTION_KEYS, "question")
    return ForecastQuestion(
        question_id=_string_field(payload, "question_id"),
        text=_string_field(payload, "text"),
        resolution_rule=_string_field(payload, "resolution_rule"),
        resolution_source=_string_field(payload, "resolution_source"),
        outcomes=_string_tuple(payload, "outcomes"),
        forecast_at=_time_field(payload, "forecast_at"),
        resolves_at=_time_field(payload, "resolves_at"),
    )


def _source_from_payload(payload: Mapping[str, object]) -> SourceSnapshot:
    require_exact_keys(payload, _SOURCE_KEYS, "source")
    return SourceSnapshot(
        source_id=_string_field(payload, "source_id"),
        rights_scope=_string_field(payload, "rights_scope"),
        source_url=_string_field(payload, "source_url"),
        payload_sha256=_string_field(payload, "payload_sha256"),
        snapshot_metadata_sha256=_string_field(payload, "snapshot_metadata_sha256"),
        published_at=_time_field(payload, "published_at"),
        retrieved_at=_time_field(payload, "retrieved_at"),
        capture_method=_capture_method(required_field(payload, "capture_method")),
        sensitivity=_sensitivity(required_field(payload, "sensitivity")),
        rights=_rights_from_payload(_object_field(payload, "rights")),
        archive_version_id=_optional_string_field(payload, "archive_version_id"),
        archive_attestation_sha256=_optional_string_field(
            payload,
            "archive_attestation_sha256",
        ),
    )


def _rights_from_payload(payload: Mapping[str, object]) -> SourceRights:
    require_exact_keys(payload, _RIGHTS_KEYS, "rights")
    return SourceRights(
        license_name=_string_field(payload, "license_name"),
        terms_url=_string_field(payload, "terms_url"),
        terms_sha256=_string_field(payload, "terms_sha256"),
        rights_as_of=_time_field(payload, "rights_as_of"),
        local_processing=_permission_field(payload, "local_processing"),
        third_party_processing=_permission_field(payload, "third_party_processing"),
        tinker_processing=_permission_field(payload, "tinker_processing"),
        redistribution=_permission_field(payload, "redistribution"),
    )


def _record_from_payload(payload: Mapping[str, object]) -> NbaEvidenceRecord:
    require_exact_keys(payload, _RECORD_KEYS, "record")
    return NbaEvidenceRecord(
        record_id=_string_field(payload, "record_id"),
        kind=_evidence_kind(required_field(payload, "kind")),
        feature_name=_string_field(payload, "feature_name"),
        team_value=require_float(required_field(payload, "team_value"), "team_value"),
        opponent_value=require_float(
            required_field(payload, "opponent_value"),
            "opponent_value",
        ),
        source_ids=_string_tuple(payload, "source_ids"),
        available_at=_time_field(payload, "available_at"),
    )


def _object_field(payload: Mapping[str, object], field_name: str) -> JsonObject:
    return require_object(required_field(payload, field_name), field_name)


def _list_field(payload: Mapping[str, object], field_name: str) -> list[object]:
    return require_list(required_field(payload, field_name), field_name)


def _string_field(payload: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(payload, field_name), field_name)


def _optional_string_field(
    payload: Mapping[str, object],
    field_name: str,
) -> str | None:
    value = required_field(payload, field_name)
    if value is None:
        return None
    return require_string(value, field_name)


def _string_tuple(payload: Mapping[str, object], field_name: str) -> tuple[str, ...]:
    return tuple(
        require_string(item, f"{field_name}[{index}]")
        for index, item in enumerate(_list_field(payload, field_name))
    )


def _time_field(payload: Mapping[str, object], field_name: str) -> datetime:
    text = _string_field(payload, field_name)
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise JsonFormatError(f"{field_name} must be an ISO 8601 datetime") from error
    _require_utc(value, field_name)
    return value.astimezone(UTC)


def _permission_field(payload: Mapping[str, object], field_name: str) -> Permission:
    value = _string_field(payload, field_name)
    match value:
        case "allowed" | "prohibited" | "unknown":
            return value
        case _:
            raise JsonFormatError(f"unsupported permission: {value}")


def _capture_method(value: object) -> CaptureMethod:
    text = require_string(value, "capture_method")
    match text:
        case "live" | "provider_versioned_archive":
            return text
        case _:
            raise JsonFormatError(f"unsupported capture_method: {text}")


def _sensitivity(value: object) -> Sensitivity:
    text = require_string(value, "sensitivity")
    match text:
        case "ordinary" | "player_health":
            return text
        case _:
            raise JsonFormatError(f"unsupported sensitivity: {text}")


def _evidence_kind(value: object) -> EvidenceKind:
    text = require_string(value, "kind")
    match text:
        case (
            "injury"
            | "expected_lineup"
            | "rest"
            | "travel"
            | "back_to_back"
            | "roster"
            | "team_metric"
            | "player_metric"
            | "schedule_strength"
        ):
            return text
        case _:
            raise JsonFormatError(f"unsupported evidence kind: {text}")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise NbaEvidenceIoError(f"{field_name} must be in UTC")


def _utc_text(value: datetime) -> str:
    _require_utc(value, "datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
