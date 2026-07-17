"""Rights-aware, point-in-time NBA evidence for prospective forecasts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Literal, Protocol

from forecastfm.integrity import canonical_json, canonical_sha256
from forecastfm.ledger import CohortGame
from forecastfm.models import EvidenceCard, ForecastQuestion
from forecastfm.tinker_screening import require_text_health_screen_passes

type Permission = Literal["allowed", "prohibited", "unknown"]
type EvidenceKind = Literal[
    "injury",
    "expected_lineup",
    "rest",
    "travel",
    "back_to_back",
    "roster",
    "team_metric",
    "player_metric",
    "schedule_strength",
]
type Sensitivity = Literal["ordinary", "player_health"]
type CaptureMethod = Literal["live", "provider_versioned_archive"]
type PermissionField = Literal[
    "local_processing",
    "third_party_processing",
    "tinker_processing",
    "redistribution",
]

_PERMISSIONS = frozenset({"allowed", "prohibited", "unknown"})
_EVIDENCE_KINDS = frozenset(
    {
        "injury",
        "expected_lineup",
        "rest",
        "travel",
        "back_to_back",
        "roster",
        "team_metric",
        "player_metric",
        "schedule_strength",
    }
)
_SENSITIVITIES = frozenset({"ordinary", "player_health"})
_CAPTURE_METHODS = frozenset({"live", "provider_versioned_archive"})
_HASH_CHARACTERS = frozenset("0123456789abcdef")
_FEATURE_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class NbaEvidenceError(ValueError):
    """Raised when NBA evidence violates timing, rights, or lineage rules."""


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise NbaEvidenceError(f"{field_name} must not be empty")


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in _HASH_CHARACTERS for character in value):
        raise NbaEvidenceError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise NbaEvidenceError(f"{field_name} must be in UTC")


def _utc_text(value: datetime) -> str:
    _require_utc(value, "datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class SourceRights:
    """A frozen, fail-closed summary of one source's usage rights."""

    license_name: str
    terms_url: str
    terms_sha256: str
    rights_as_of: datetime
    local_processing: Permission
    third_party_processing: Permission
    tinker_processing: Permission
    redistribution: Permission

    def __post_init__(self) -> None:
        _require_text(self.license_name, "license_name")
        _require_text(self.terms_url, "terms_url")
        _require_sha256(self.terms_sha256, "terms_sha256")
        _require_utc(self.rights_as_of, "rights_as_of")
        permissions = (
            self.local_processing,
            self.third_party_processing,
            self.tinker_processing,
            self.redistribution,
        )
        if any(permission not in _PERMISSIONS for permission in permissions):
            raise NbaEvidenceError("source permissions must be allowed, prohibited, or unknown")


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """One immutable raw-source snapshot retained outside model-facing data."""

    source_id: str
    rights_scope: str
    source_url: str
    payload_sha256: str
    snapshot_metadata_sha256: str
    published_at: datetime
    retrieved_at: datetime
    capture_method: CaptureMethod
    sensitivity: Sensitivity
    rights: SourceRights
    archive_version_id: str | None = None
    archive_attestation_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.source_id, "source_id")
        _require_text(self.rights_scope, "rights_scope")
        _require_text(self.source_url, "source_url")
        _require_sha256(self.payload_sha256, "payload_sha256")
        _require_sha256(self.snapshot_metadata_sha256, "snapshot_metadata_sha256")
        _require_utc(self.published_at, "published_at")
        _require_utc(self.retrieved_at, "retrieved_at")
        if self.published_at > self.retrieved_at:
            raise NbaEvidenceError("published_at cannot be after retrieved_at")
        if self.capture_method not in _CAPTURE_METHODS:
            raise NbaEvidenceError("unsupported capture_method")
        if self.sensitivity not in _SENSITIVITIES:
            raise NbaEvidenceError("unsupported source sensitivity")
        if self.capture_method == "provider_versioned_archive":
            if self.archive_version_id is None:
                raise NbaEvidenceError("provider archives require an immutable version ID")
            _require_text(self.archive_version_id, "archive_version_id")
            if self.archive_attestation_sha256 is None:
                raise NbaEvidenceError("provider archives require an attestation digest")
            _require_sha256(self.archive_attestation_sha256, "archive_attestation_sha256")
        elif self.archive_version_id is not None or self.archive_attestation_sha256 is not None:
            raise NbaEvidenceError("live captures cannot carry provider archive attestations")

    def historical_available_at(self) -> datetime:
        """Return when this snapshot could first support a point-in-time feature."""
        if self.capture_method == "provider_versioned_archive":
            return self.published_at
        return self.retrieved_at


@dataclass(frozen=True, slots=True)
class NbaEvidenceRecord:
    """One oriented numeric feature with complete upstream source lineage."""

    record_id: str
    kind: EvidenceKind
    feature_name: str
    team_value: float
    opponent_value: float
    source_ids: tuple[str, ...]
    available_at: datetime

    def __post_init__(self) -> None:
        _require_text(self.record_id, "record_id")
        if _FEATURE_NAME_PATTERN.fullmatch(self.feature_name) is None:
            raise NbaEvidenceError("feature_name must be lowercase snake case")
        if not isfinite(self.team_value) or not isfinite(self.opponent_value):
            raise NbaEvidenceError("evidence values must be finite")
        _require_utc(self.available_at, "available_at")
        if self.kind not in _EVIDENCE_KINDS:
            raise NbaEvidenceError("unsupported evidence kind")
        if not self.source_ids:
            raise NbaEvidenceError("an evidence record must reference at least one source")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise NbaEvidenceError("an evidence record cannot repeat a source ID")
        if self.source_ids != tuple(sorted(self.source_ids)):
            raise NbaEvidenceError("evidence source IDs must be ordered")
        for source_id in self.source_ids:
            _require_text(source_id, "source_id")

    @property
    def difference(self) -> float:
        """Return the exactly oriented team-minus-opponent value."""
        difference = self.team_value - self.opponent_value
        if difference == 0.0:
            return 0.0
        return difference

    def side_swap(self) -> NbaEvidenceRecord:
        """Exchange team and opponent values without changing source lineage."""
        return NbaEvidenceRecord(
            record_id=self.record_id,
            kind=self.kind,
            feature_name=self.feature_name,
            team_value=self.opponent_value,
            opponent_value=self.team_value,
            source_ids=self.source_ids,
            available_at=self.available_at,
        )


@dataclass(frozen=True, slots=True)
class NbaEvidenceBundle:
    """The complete evidence and provenance frozen for one NBA forecast."""

    game: CohortGame
    question: ForecastQuestion
    sources: tuple[SourceSnapshot, ...]
    records: tuple[NbaEvidenceRecord, ...]

    def __post_init__(self) -> None:
        _validate_bindings(self)
        source_map = _source_map(self.sources)
        _validate_records(self.records, source_map, self.game.forecast_deadline)


class NbaEvidenceConnector(Protocol):
    """Interface implemented by a buyer-owned or explicitly licensed feed."""

    def collect(
        self,
        game: CohortGame,
        question: ForecastQuestion,
    ) -> NbaEvidenceBundle:
        """Collect and freeze all evidence available at the forecast deadline."""
        ...


def local_evidence_cards(
    bundle: NbaEvidenceBundle,
    *,
    action_at: datetime,
) -> tuple[EvidenceCard, ...]:
    """Build local model inputs after requiring explicit local-processing rights."""
    _require_permission(bundle, "local_processing", action_at)
    return _evidence_cards(bundle)


def tinker_evidence_cards(
    bundle: NbaEvidenceBundle,
    *,
    action_at: datetime,
) -> tuple[EvidenceCard, ...]:
    """Build standard Tinker inputs after rights and health-lineage checks."""
    _require_tinker_processing(bundle, action_at)
    return _evidence_cards(bundle)


def require_redistribution_allowed(
    bundle: NbaEvidenceBundle,
    *,
    action_at: datetime,
) -> None:
    """Require explicit permission before distributing a derived evidence pack."""
    _require_permission(bundle, "redistribution", action_at)


def require_prospective_capture(bundle: NbaEvidenceBundle) -> None:
    """Reject retrospective archives when making a prospective timing claim."""
    if any(source.capture_method != "live" for source in bundle.sources):
        raise NbaEvidenceError("prospective evidence requires live source capture")


def evidence_bundle_sha256(bundle: NbaEvidenceBundle) -> str:
    """Hash exact evidence, timing, rights, and lineage deterministically."""
    return canonical_sha256(
        {
            "game": {
                "question_id": bundle.game.question_id,
                "source_game_id": bundle.game.source_game_id,
                "matchup": bundle.game.matchup,
                "outcomes": list(bundle.game.outcomes),
                "forecast_deadline": _utc_text(bundle.game.forecast_deadline),
                "scheduled_tipoff": _utc_text(bundle.game.scheduled_tipoff),
            },
            "question": {
                "question_id": bundle.question.question_id,
                "text": bundle.question.text,
                "resolution_rule": bundle.question.resolution_rule,
                "resolution_source": bundle.question.resolution_source,
                "outcomes": list(bundle.question.outcomes),
                "forecast_at": _utc_text(bundle.question.forecast_at),
                "resolves_at": _utc_text(bundle.question.resolves_at),
            },
            "sources": [_source_payload(source) for source in bundle.sources],
            "records": [
                {
                    "record_id": record.record_id,
                    "kind": record.kind,
                    "feature_name": record.feature_name,
                    "team_value": record.team_value,
                    "opponent_value": record.opponent_value,
                    "source_ids": list(record.source_ids),
                    "available_at": _utc_text(record.available_at),
                }
                for record in bundle.records
            ],
        }
    )


def local_numeric_feature_vector(
    bundle: NbaEvidenceBundle,
    feature_names: tuple[str, ...],
    *,
    action_at: datetime,
) -> tuple[float, ...]:
    """Build an aligned local-model vector after checking source rights."""
    _require_permission(bundle, "local_processing", action_at)
    return _numeric_feature_vector(bundle, feature_names)


def tinker_numeric_feature_vector(
    bundle: NbaEvidenceBundle,
    feature_names: tuple[str, ...],
    *,
    action_at: datetime,
) -> tuple[float, ...]:
    """Build an aligned Tinker vector after rights and health checks."""
    _require_tinker_processing(bundle, action_at)
    return _numeric_feature_vector(bundle, feature_names)


def _numeric_feature_vector(
    bundle: NbaEvidenceBundle,
    feature_names: tuple[str, ...],
) -> tuple[float, ...]:
    valid_names = all(_FEATURE_NAME_PATTERN.fullmatch(name) is not None for name in feature_names)
    if not valid_names or len(set(feature_names)) != len(feature_names):
        raise NbaEvidenceError("feature schema names must be valid and unique")
    values = {record.feature_name: record.difference for record in bundle.records}
    if set(values) != set(feature_names):
        raise NbaEvidenceError("evidence features do not match the predeclared schema")
    return tuple(values[name] for name in feature_names)


def _validate_bindings(bundle: NbaEvidenceBundle) -> None:
    question = bundle.question
    game = bundle.game
    _require_utc(question.forecast_at, "question.forecast_at")
    _require_utc(question.resolves_at, "question.resolves_at")
    if question.question_id != game.question_id:
        raise NbaEvidenceError("question and cohort game IDs must match")
    if question.outcomes != game.outcomes:
        raise NbaEvidenceError("question and cohort game outcomes must match")
    if question.forecast_at != game.forecast_deadline:
        raise NbaEvidenceError("question cutoff must equal the cohort forecast deadline")
    if question.resolves_at < game.scheduled_tipoff:
        raise NbaEvidenceError("question resolution cannot precede scheduled tipoff")
    if not bundle.sources:
        raise NbaEvidenceError("an evidence bundle must contain at least one source")
    if not bundle.records:
        raise NbaEvidenceError("an evidence bundle must contain at least one record")


def _source_map(sources: Sequence[SourceSnapshot]) -> dict[str, SourceSnapshot]:
    source_map = {source.source_id: source for source in sources}
    if len(source_map) != len(sources):
        raise NbaEvidenceError("source IDs must be unique")
    if tuple(sources) != tuple(sorted(sources, key=lambda source: source.source_id)):
        raise NbaEvidenceError("sources must be ordered by source ID")
    return source_map


def _validate_records(
    records: Sequence[NbaEvidenceRecord],
    source_map: dict[str, SourceSnapshot],
    cutoff: datetime,
) -> None:
    _validate_record_identity(records)
    referenced_source_ids: set[str] = set()
    for record in records:
        sources = _record_sources(record, source_map)
        referenced_source_ids.update(record.source_ids)
        expected_available_at = max(source.historical_available_at() for source in sources)
        if record.available_at < expected_available_at:
            raise NbaEvidenceError("record availability cannot predate its latest source")
        if record.available_at > cutoff:
            raise NbaEvidenceError("evidence cannot be newer than the forecast deadline")
        if record.kind == "injury" and not any(
            source.sensitivity == "player_health" for source in sources
        ):
            raise NbaEvidenceError("injury evidence must retain player-health lineage")
    if referenced_source_ids != set(source_map):
        raise NbaEvidenceError("every source snapshot must be referenced by evidence")


def _validate_record_identity(records: Sequence[NbaEvidenceRecord]) -> None:
    record_ids = [record.record_id for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise NbaEvidenceError("record IDs must be unique")
    feature_names = [record.feature_name for record in records]
    if len(set(feature_names)) != len(feature_names):
        raise NbaEvidenceError("feature names must be unique within a bundle")
    ordered_records = tuple(sorted(records, key=lambda item: (item.available_at, item.record_id)))
    if tuple(records) != ordered_records:
        raise NbaEvidenceError("evidence records must be ordered by availability and record ID")


def _record_sources(
    record: NbaEvidenceRecord,
    source_map: dict[str, SourceSnapshot],
) -> tuple[SourceSnapshot, ...]:
    try:
        return tuple(source_map[source_id] for source_id in record.source_ids)
    except KeyError as error:
        raise NbaEvidenceError("evidence record references an unknown source ID") from error


def _require_permission(
    bundle: NbaEvidenceBundle,
    field_name: PermissionField,
    action_at: datetime,
) -> None:
    _require_utc(action_at, "action_at")
    if any(record.available_at > action_at for record in bundle.records):
        raise NbaEvidenceError("evidence availability cannot be after the protected action")
    for source in bundle.sources:
        if source.rights.rights_as_of > action_at:
            raise NbaEvidenceError("rights_as_of cannot be after the protected action")
        if source.retrieved_at > action_at:
            raise NbaEvidenceError("source retrieval cannot be after the protected action")
        permission = _permission(source.rights, field_name)
        if permission != "allowed":
            raise NbaEvidenceError(
                f"{field_name} must be explicitly allowed for source {source.source_id}"
            )


def _require_tinker_processing(bundle: NbaEvidenceBundle, action_at: datetime) -> None:
    _require_permission(bundle, "third_party_processing", action_at)
    _require_permission(bundle, "tinker_processing", action_at)
    if any(source.sensitivity == "player_health" for source in bundle.sources):
        raise NbaEvidenceError("standard Tinker exports cannot contain player-health lineage")
    require_text_health_screen_passes(
        text
        for record in bundle.records
        for text in (record.feature_name, *(_source_urls(record, bundle.sources)))
    )


def _permission(rights: SourceRights, field_name: PermissionField) -> Permission:
    if field_name == "local_processing":
        return rights.local_processing
    if field_name == "third_party_processing":
        return rights.third_party_processing
    if field_name == "tinker_processing":
        return rights.tinker_processing
    return rights.redistribution


def _evidence_cards(bundle: NbaEvidenceBundle) -> tuple[EvidenceCard, ...]:
    sources = {source.source_id: source.source_url for source in bundle.sources}
    return tuple(
        EvidenceCard(
            text="Pregame numeric feature: "
            + canonical_json({record.feature_name: record.difference}),
            source=", ".join(sources[source_id] for source_id in record.source_ids),
            available_at=record.available_at,
        )
        for record in bundle.records
    )


def _source_urls(
    record: NbaEvidenceRecord,
    sources: Sequence[SourceSnapshot],
) -> tuple[str, ...]:
    urls = {source.source_id: source.source_url for source in sources}
    return tuple(urls[source_id] for source_id in record.source_ids)


def _source_payload(source: SourceSnapshot) -> dict[str, object]:
    rights = source.rights
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
        "rights": {
            "license_name": rights.license_name,
            "terms_url": rights.terms_url,
            "terms_sha256": rights.terms_sha256,
            "rights_as_of": _utc_text(rights.rights_as_of),
            "local_processing": rights.local_processing,
            "third_party_processing": rights.third_party_processing,
            "tinker_processing": rights.tinker_processing,
            "redistribution": rights.redistribution,
        },
    }
