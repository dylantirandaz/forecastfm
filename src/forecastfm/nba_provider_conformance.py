"""Bounded conformance checks for a licensed NBA provider connector.

This module verifies an archive sample against an externally reviewed exact
inventory.  It deliberately does not authenticate the reviewer, the provider,
or the connector implementation; callers bind those trust decisions by digest.
Snapshot metadata has no independent archive-capture timestamp, so that decoded
field is externally reviewed and only bounded by publication and local retrieval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

from forecastfm.integrity import canonical_sha256, file_sha256
from forecastfm.ledger import Cohort, CohortGame, cohort_sha256
from forecastfm.nba_elo_replay import NbaEloReplayRow
from forecastfm.nba_snapshot_pack import (
    NbaSnapshot,
    NbaSnapshotIndex,
    load_snapshot_pack,
    snapshot_metadata_sha256,
)

NBA_PROVIDER_CONFORMANCE_SCHEMA_VERSION = 1
NBA_PROVIDER_CONFORMANCE_PROOF_SCOPE = (
    "bounded sample matches reviewed inventory; reviewer, vendor, connector, complete-provider "
    "coverage, and trusted-timestamp authentication require separate evidence"
)

type NbaProviderCaseKind = Literal[
    "late_scratch",
    "lineup_change",
    "roster_transaction",
    "reschedule",
    "other",
]
type NbaProviderChangeKind = Literal["upsert", "correction", "deletion"]
type NbaProviderEntityKind = Literal["injury", "lineup", "roster", "schedule", "other"]
type NbaProviderCutoffState = Literal["T-6h", "T-60", "T-15"]

_HASH_CHARACTERS = frozenset("0123456789abcdef")
_CASE_KINDS = frozenset(
    {"late_scratch", "lineup_change", "roster_transaction", "reschedule", "other"}
)
_CHANGE_KINDS = frozenset({"upsert", "correction", "deletion"})
_ENTITY_KINDS = frozenset({"injury", "lineup", "roster", "schedule", "other"})
_CUTOFF_OFFSETS: Mapping[NbaProviderCutoffState, timedelta] = {
    "T-6h": timedelta(hours=6),
    "T-60": timedelta(minutes=60),
    "T-15": timedelta(minutes=15),
}
_CUTOFF_ORDER: Mapping[NbaProviderCutoffState, int] = {
    "T-6h": 0,
    "T-60": 1,
    "T-15": 2,
}
_REQUIRED_CASES = frozenset({"late_scratch", "lineup_change", "roster_transaction", "reschedule"})
_REQUIRED_CHANGES = frozenset({"correction", "deletion"})


class NbaProviderConformanceError(ValueError):
    """Raised when a provider sample cannot earn a conformance pass."""


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise NbaProviderConformanceError(f"{field_name} must not be empty")


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in _HASH_CHARACTERS for character in value):
        raise NbaProviderConformanceError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise NbaProviderConformanceError(f"{field_name} must be in UTC")


def _utc_text(value: datetime) -> str:
    _require_utc(value, "datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class NbaProviderRevision:
    """Provider fields decoded from one retained raw revision."""

    source_id: str
    version: str
    entity_id: str
    entity_kind: NbaProviderEntityKind
    case_kind: NbaProviderCaseKind
    change_kind: NbaProviderChangeKind
    provider_schema_version: str
    provider_api_version: str
    effective_at: datetime
    provider_published_at: datetime
    archive_captured_at: datetime

    def __post_init__(self) -> None:
        for field_name in (
            "source_id",
            "version",
            "entity_id",
            "provider_schema_version",
            "provider_api_version",
        ):
            _require_text(getattr(self, field_name), field_name)
        if self.entity_kind not in _ENTITY_KINDS:
            raise NbaProviderConformanceError("unsupported entity_kind")
        if self.case_kind not in _CASE_KINDS:
            raise NbaProviderConformanceError("unsupported case_kind")
        if self.change_kind not in _CHANGE_KINDS:
            raise NbaProviderConformanceError("unsupported change_kind")
        _require_utc(self.effective_at, "effective_at")
        _require_utc(self.provider_published_at, "provider_published_at")
        _require_utc(self.archive_captured_at, "archive_captured_at")
        if self.archive_captured_at < self.provider_published_at:
            raise NbaProviderConformanceError(
                "archive_captured_at cannot precede provider_published_at"
            )

    def canonical_payload(self) -> dict[str, object]:
        """Return every reviewed field decoded from the raw provider revision."""
        return {
            "source_id": self.source_id,
            "version": self.version,
            "entity_id": self.entity_id,
            "entity_kind": self.entity_kind,
            "case_kind": self.case_kind,
            "change_kind": self.change_kind,
            "provider_schema_version": self.provider_schema_version,
            "provider_api_version": self.provider_api_version,
            "effective_at": _utc_text(self.effective_at),
            "provider_published_at": _utc_text(self.provider_published_at),
            "archive_captured_at": _utc_text(self.archive_captured_at),
        }


@dataclass(frozen=True, slots=True)
class NbaProviderCutoffExpectation:
    """One reviewed expected revision at a named pre-event cutoff."""

    state: NbaProviderCutoffState
    source_id: str
    cutoff: datetime
    scheduled_tipoff: datetime
    expected_version: str

    def __post_init__(self) -> None:
        if self.state not in _CUTOFF_OFFSETS:
            raise NbaProviderConformanceError("unsupported cutoff state")
        _require_text(self.source_id, "source_id")
        _require_text(self.expected_version, "expected_version")
        _require_utc(self.cutoff, "cutoff")
        _require_utc(self.scheduled_tipoff, "scheduled_tipoff")
        if self.scheduled_tipoff - self.cutoff != _CUTOFF_OFFSETS[self.state]:
            raise NbaProviderConformanceError(f"cutoff does not match {self.state}")

    def canonical_payload(self) -> dict[str, object]:
        """Return the exact cutoff expectation covered by the inventory digest."""
        return {
            "state": self.state,
            "source_id": self.source_id,
            "cutoff": _utc_text(self.cutoff),
            "scheduled_tipoff": _utc_text(self.scheduled_tipoff),
            "expected_version": self.expected_version,
        }


@dataclass(frozen=True, slots=True)
class NbaDerivedScheduleGame:
    """One raw-derived schedule game, reusing the sealed Elo replay row."""

    replay_row: NbaEloReplayRow
    season_type: str
    venue_id: str
    source_id: str
    snapshot_metadata_sha256: str

    def __post_init__(self) -> None:
        _require_text(self.season_type, "season_type")
        _require_text(self.venue_id, "venue_id")
        _require_text(self.source_id, "source_id")
        _require_sha256(self.snapshot_metadata_sha256, "snapshot_metadata_sha256")

    def canonical_payload(self) -> dict[str, object]:
        """Return the derived schedule and its exact snapshot lineage."""
        return {
            "replay_row": self.replay_row.canonical_payload(),
            "season_type": self.season_type,
            "venue_id": self.venue_id,
            "source_id": self.source_id,
            "snapshot_metadata_sha256": self.snapshot_metadata_sha256,
        }


@dataclass(frozen=True, slots=True)
class NbaProviderInventory:
    """Exact external inventory required for a bounded conformance decision."""

    revisions: tuple[NbaProviderRevision, ...]
    cutoff_expectations: tuple[NbaProviderCutoffExpectation, ...]
    cutoff_exempt_source_ids: tuple[str, ...]
    schedule_games: tuple[NbaDerivedScheduleGame, ...]
    known_schedule_gap_ids: tuple[str, ...]
    archive_attestation_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.archive_attestation_sha256, "archive_attestation_sha256")
        revision_keys = _require_revision_inventory(self.revisions)
        schedule_sources = frozenset(
            revision.source_id for revision in self.revisions if revision.entity_kind == "schedule"
        )
        _require_cutoff_matrix(
            self.cutoff_expectations,
            self.cutoff_exempt_source_ids,
            revision_keys,
            schedule_sources,
        )
        _require_schedule_inventory(self.schedule_games, self.known_schedule_gap_ids)

    def canonical_payload(self) -> dict[str, object]:
        """Return the normalized bytes-equivalent inventory payload."""
        return {
            "schema_version": NBA_PROVIDER_CONFORMANCE_SCHEMA_VERSION,
            "revisions": [revision.canonical_payload() for revision in self.revisions],
            "cutoff_expectations": [
                expectation.canonical_payload() for expectation in self.cutoff_expectations
            ],
            "cutoff_exempt_source_ids": list(self.cutoff_exempt_source_ids),
            "schedule_games": [game.canonical_payload() for game in self.schedule_games],
            "known_schedule_gap_ids": list(self.known_schedule_gap_ids),
            "archive_attestation_sha256": self.archive_attestation_sha256,
        }

    @property
    def sha256(self) -> str:
        """Return the canonical digest supplied by an independent reviewer."""
        return canonical_sha256(self.canonical_payload())


class NbaProviderConnector(Protocol):
    """Vendor adapter required by the provider-neutral conformance layer."""

    def decode_revision(self, snapshot: NbaSnapshot, /) -> NbaProviderRevision:
        """Decode provider fields directly from one retained raw payload."""
        ...

    def derive_schedule(
        self,
        snapshots: tuple[NbaSnapshot, ...],
        /,
    ) -> tuple[NbaDerivedScheduleGame, ...]:
        """Derive schedule games only from the supplied eligible snapshots."""
        ...


@dataclass(frozen=True, slots=True)
class NbaProviderConformanceRequest:
    """All external bindings needed for one conformance decision."""

    snapshot_pack_path: Path
    inventory: NbaProviderInventory
    reviewed_inventory_sha256: str
    connector_id: str
    connector_sha256: str
    replay_rows: tuple[NbaEloReplayRow, ...]
    cohort: Cohort

    def __post_init__(self) -> None:
        _require_sha256(self.reviewed_inventory_sha256, "reviewed_inventory_sha256")
        _require_text(self.connector_id, "connector_id")
        _require_sha256(self.connector_sha256, "connector_sha256")
        if not self.replay_rows:
            raise NbaProviderConformanceError("replay_rows must not be empty")


@dataclass(frozen=True, slots=True)
class NbaProviderConformanceReport:
    """Digest-bound evidence that every bounded conformance check passed."""

    inventory_sha256: str
    connector_id: str
    connector_sha256: str
    snapshot_pack_sha256: str
    cutoff_selection_sha256: str
    schedule_derivation_sha256: str
    replay_rows_sha256: str
    schedule_season_types: tuple[str, ...]
    cohort_sha256: str
    revision_count: int
    schedule_game_count: int
    status: Literal["passed"] = "passed"

    def canonical_payload(self) -> dict[str, object]:
        """Return the complete deterministic report payload."""
        return {
            "schema_version": NBA_PROVIDER_CONFORMANCE_SCHEMA_VERSION,
            "status": self.status,
            "proof_scope": NBA_PROVIDER_CONFORMANCE_PROOF_SCOPE,
            "inventory_sha256": self.inventory_sha256,
            "connector_id": self.connector_id,
            "connector_sha256": self.connector_sha256,
            "snapshot_pack_sha256": self.snapshot_pack_sha256,
            "cutoff_selection_sha256": self.cutoff_selection_sha256,
            "schedule_derivation_sha256": self.schedule_derivation_sha256,
            "replay_rows_sha256": self.replay_rows_sha256,
            "schedule_season_types": list(self.schedule_season_types),
            "cohort_sha256": self.cohort_sha256,
            "revision_count": self.revision_count,
            "schedule_game_count": self.schedule_game_count,
        }

    @property
    def report_sha256(self) -> str:
        """Return the canonical digest of this passing report."""
        return canonical_sha256(self.canonical_payload())


def require_nba_provider_conformance(
    request: NbaProviderConformanceRequest,
    connector: NbaProviderConnector,
) -> NbaProviderConformanceReport:
    """Require exact inventory, timing, correction, schedule, and cohort agreement."""
    if request.reviewed_inventory_sha256 != request.inventory.sha256:
        raise NbaProviderConformanceError(
            "inventory differs from the independently reviewed digest"
        )
    index = load_snapshot_pack(request.snapshot_pack_path)
    pack_sha256 = file_sha256(request.snapshot_pack_path)
    if request.cohort.schedule_snapshot_sha256 != pack_sha256:
        raise NbaProviderConformanceError("cohort does not bind the exact snapshot pack")
    revisions = _decode_revisions(index, request.inventory, connector)
    cutoff_sha256 = _validate_cutoff_selections(index, request.inventory.cutoff_expectations)
    schedule_snapshots = _select_schedule_snapshots(index, revisions, request.cohort)
    derived_games = connector.derive_schedule(schedule_snapshots)
    schedule_sha256 = _validate_schedule(
        derived_games,
        schedule_snapshots,
        request.inventory,
        request.replay_rows,
        request.cohort,
    )
    return NbaProviderConformanceReport(
        inventory_sha256=request.inventory.sha256,
        connector_id=request.connector_id,
        connector_sha256=request.connector_sha256,
        snapshot_pack_sha256=pack_sha256,
        cutoff_selection_sha256=cutoff_sha256,
        schedule_derivation_sha256=schedule_sha256,
        replay_rows_sha256=canonical_sha256(
            [row.canonical_payload() for row in request.replay_rows]
        ),
        schedule_season_types=tuple(
            sorted({game.season_type for game in request.inventory.schedule_games})
        ),
        cohort_sha256=cohort_sha256(request.cohort),
        revision_count=len(revisions),
        schedule_game_count=len(derived_games),
    )


def _revision_sort_key(revision: NbaProviderRevision) -> tuple[str, str]:
    return (revision.source_id, revision.version)


def _require_revision_inventory(
    revisions: tuple[NbaProviderRevision, ...],
) -> frozenset[tuple[str, str]]:
    if not revisions:
        raise NbaProviderConformanceError("revision inventory must not be empty")
    if revisions != tuple(sorted(revisions, key=_revision_sort_key)):
        raise NbaProviderConformanceError("revision inventory must be canonically sorted")
    keys = tuple(_revision_sort_key(revision) for revision in revisions)
    if len(keys) != len(set(keys)):
        raise NbaProviderConformanceError("revision inventory keys must be unique")
    entity_shapes = {
        revision.source_id: (revision.entity_id, revision.entity_kind) for revision in revisions
    }
    if any(
        entity_shapes[revision.source_id] != (revision.entity_id, revision.entity_kind)
        for revision in revisions
    ):
        raise NbaProviderConformanceError("one source_id must have one entity identity and kind")
    return frozenset(keys)


def _cutoff_sort_key(
    expectation: NbaProviderCutoffExpectation,
) -> tuple[str, int]:
    return (expectation.source_id, _CUTOFF_ORDER[expectation.state])


def _require_cutoff_matrix(
    expectations: tuple[NbaProviderCutoffExpectation, ...],
    exempt_source_ids: tuple[str, ...],
    revision_keys: frozenset[tuple[str, str]],
    schedule_sources: frozenset[str],
) -> None:
    if not expectations:
        raise NbaProviderConformanceError("cutoff inventory must not be empty")
    if expectations != tuple(sorted(expectations, key=_cutoff_sort_key)):
        raise NbaProviderConformanceError("cutoff expectations must be canonically sorted")
    keys = tuple((item.source_id, item.state) for item in expectations)
    if len(keys) != len(set(keys)):
        raise NbaProviderConformanceError("cutoff expectations must be unique")
    for expectation in expectations:
        if (expectation.source_id, expectation.expected_version) not in revision_keys:
            raise NbaProviderConformanceError("cutoff version is absent from revision inventory")
    expected_states = frozenset(_CUTOFF_OFFSETS)
    states_by_source = {
        source_id: frozenset(item.state for item in expectations if item.source_id == source_id)
        for source_id, _state in keys
    }
    if any(states != expected_states for states in states_by_source.values()):
        raise NbaProviderConformanceError("every cutoff source must cover T-6h, T-60, and T-15")
    tipoffs_by_source = {
        source_id: frozenset(
            item.scheduled_tipoff for item in expectations if item.source_id == source_id
        )
        for source_id in states_by_source
    }
    if any(len(tipoffs) != 1 for tipoffs in tipoffs_by_source.values()):
        raise NbaProviderConformanceError("one cutoff source must use one scheduled_tipoff")
    _require_cutoff_source_coverage(
        frozenset(states_by_source),
        exempt_source_ids,
        revision_keys,
        schedule_sources,
    )


def _require_cutoff_source_coverage(
    matrix_sources: frozenset[str],
    exempt_source_ids: tuple[str, ...],
    revision_keys: frozenset[tuple[str, str]],
    schedule_sources: frozenset[str],
) -> None:
    for source_id in exempt_source_ids:
        _require_text(source_id, "cutoff-exempt source_id")
    if exempt_source_ids != tuple(sorted(set(exempt_source_ids))):
        raise NbaProviderConformanceError("cutoff-exempt source IDs must be unique and sorted")
    exempt_sources = frozenset(exempt_source_ids)
    revision_sources = frozenset(source_id for source_id, _version in revision_keys)
    if matrix_sources.intersection(exempt_sources):
        raise NbaProviderConformanceError("a cutoff source cannot also be exempt")
    if not exempt_sources.issubset(schedule_sources):
        raise NbaProviderConformanceError("only reviewed schedule sources may be cutoff-exempt")
    if matrix_sources.union(exempt_sources) != revision_sources:
        raise NbaProviderConformanceError(
            "every revision source must have cutoff tests or an explicit exemption"
        )


def _require_schedule_inventory(
    games: tuple[NbaDerivedScheduleGame, ...],
    gap_ids: tuple[str, ...],
) -> None:
    if not games:
        raise NbaProviderConformanceError("schedule inventory must not be empty")
    game_ids = tuple(game.replay_row.source_game_id for game in games)
    if game_ids != tuple(sorted(set(game_ids))):
        raise NbaProviderConformanceError("schedule games must have unique, sorted IDs")
    for gap_id in gap_ids:
        _require_text(gap_id, "known schedule gap ID")
    if gap_ids != tuple(sorted(set(gap_ids))):
        raise NbaProviderConformanceError("known schedule gap IDs must be unique and sorted")
    if set(game_ids).intersection(gap_ids):
        raise NbaProviderConformanceError("a schedule game cannot also be a known gap")


def _decode_revisions(
    index: NbaSnapshotIndex,
    inventory: NbaProviderInventory,
    connector: NbaProviderConnector,
) -> tuple[NbaProviderRevision, ...]:
    snapshots = tuple(index)
    actual_keys = frozenset((item.metadata.source_id, item.metadata.version) for item in snapshots)
    expected_by_key = {
        (revision.source_id, revision.version): revision for revision in inventory.revisions
    }
    if actual_keys != frozenset(expected_by_key):
        raise NbaProviderConformanceError("snapshot pack differs from the exact revision inventory")
    decoded: list[NbaProviderRevision] = []
    entities: dict[str, str] = {}
    for snapshot in snapshots:
        _require_archive_binding(snapshot, inventory.archive_attestation_sha256)
        revision = connector.decode_revision(snapshot)
        _require_revision_matches_snapshot(revision, snapshot)
        key = (revision.source_id, revision.version)
        if revision != expected_by_key[key]:
            raise NbaProviderConformanceError("decoded revision differs from reviewed inventory")
        previous_entity = entities.setdefault(revision.source_id, revision.entity_id)
        if previous_entity != revision.entity_id:
            raise NbaProviderConformanceError("one source_id decoded to multiple entity IDs")
        decoded.append(revision)
    _require_sample_coverage(decoded)
    return tuple(decoded)


def _require_archive_binding(snapshot: NbaSnapshot, attestation_sha256: str) -> None:
    metadata = snapshot.metadata
    if metadata.capture_method != "provider_versioned_archive":
        raise NbaProviderConformanceError("conformance samples must use provider archives")
    if metadata.archive_attestation_sha256 != attestation_sha256:
        raise NbaProviderConformanceError("snapshot has an unreviewed archive attestation")


def _require_revision_matches_snapshot(
    revision: NbaProviderRevision,
    snapshot: NbaSnapshot,
) -> None:
    metadata = snapshot.metadata
    if (revision.source_id, revision.version) != (metadata.source_id, metadata.version):
        raise NbaProviderConformanceError("raw revision identity differs from snapshot metadata")
    if revision.effective_at != metadata.effective_at:
        raise NbaProviderConformanceError("raw effective_at differs from snapshot metadata")
    if revision.provider_published_at != metadata.provider_published_at:
        raise NbaProviderConformanceError("raw publication time differs from snapshot metadata")
    if revision.archive_captured_at > metadata.retrieved_at:
        raise NbaProviderConformanceError("raw archive capture is after local retrieval")


def _require_sample_coverage(revisions: Sequence[NbaProviderRevision]) -> None:
    case_kinds = frozenset(revision.case_kind for revision in revisions)
    change_kinds = frozenset(revision.change_kind for revision in revisions)
    missing_cases = sorted(_REQUIRED_CASES - case_kinds)
    missing_changes = sorted(_REQUIRED_CHANGES - change_kinds)
    if missing_cases:
        raise NbaProviderConformanceError(
            f"provider sample lacks required cases: {', '.join(missing_cases)}"
        )
    if missing_changes:
        raise NbaProviderConformanceError(
            f"provider sample lacks required change markers: {', '.join(missing_changes)}"
        )


def _validate_cutoff_selections(
    index: NbaSnapshotIndex,
    expectations: Sequence[NbaProviderCutoffExpectation],
) -> str:
    selections: list[dict[str, object]] = []
    for expectation in expectations:
        snapshot = index.latest_eligible(expectation.source_id, expectation.cutoff)
        if snapshot is None:
            raise NbaProviderConformanceError("no snapshot is eligible at a reviewed cutoff")
        if snapshot.metadata.version != expectation.expected_version:
            raise NbaProviderConformanceError(
                "cutoff selected a revision other than the reviewed one"
            )
        selections.append(
            {
                **expectation.canonical_payload(),
                "snapshot_metadata_sha256": snapshot_metadata_sha256(snapshot.metadata),
            }
        )
    return canonical_sha256(selections)


def _select_schedule_snapshots(
    index: NbaSnapshotIndex,
    revisions: Sequence[NbaProviderRevision],
    cohort: Cohort,
) -> tuple[NbaSnapshot, ...]:
    source_ids = sorted(
        {revision.source_id for revision in revisions if revision.entity_kind == "schedule"}
    )
    if not source_ids:
        raise NbaProviderConformanceError("provider sample has no schedule source")
    snapshots: list[NbaSnapshot] = []
    for source_id in source_ids:
        snapshot = index.latest_eligible(source_id, cohort.schedule_retrieved)
        if snapshot is None:
            raise NbaProviderConformanceError("no schedule revision was eligible when sealed")
        snapshots.append(snapshot)
    return tuple(snapshots)


def _validate_schedule(
    derived_games: Sequence[NbaDerivedScheduleGame],
    schedule_snapshots: Sequence[NbaSnapshot],
    inventory: NbaProviderInventory,
    replay_rows: Sequence[NbaEloReplayRow],
    cohort: Cohort,
) -> str:
    if inventory.known_schedule_gap_ids:
        raise NbaProviderConformanceError("known schedule gaps prevent a completeness pass")
    if not derived_games:
        raise NbaProviderConformanceError("connector derived no schedule games")
    selected = {snapshot.metadata.source_id: snapshot for snapshot in schedule_snapshots}
    _require_schedule_lineage(derived_games, selected)
    derived_by_id = _unique_derived_games(derived_games)
    _require_inventory_games(derived_by_id, inventory.schedule_games)
    _require_replay_rows(derived_by_id, replay_rows)
    _require_cohort_games(derived_by_id, cohort.games)
    payload = [derived_by_id[game_id].canonical_payload() for game_id in sorted(derived_by_id)]
    return canonical_sha256(payload)


def _require_schedule_lineage(
    games: Sequence[NbaDerivedScheduleGame],
    selected: Mapping[str, NbaSnapshot],
) -> None:
    if {game.source_id for game in games} != set(selected):
        raise NbaProviderConformanceError("every selected schedule source must derive a game")
    for game in games:
        snapshot = selected.get(game.source_id)
        if snapshot is None:
            raise NbaProviderConformanceError("derived game cites an unselected schedule source")
        expected_sha256 = snapshot_metadata_sha256(snapshot.metadata)
        if game.snapshot_metadata_sha256 != expected_sha256:
            raise NbaProviderConformanceError("derived game has incorrect snapshot lineage")


def _unique_derived_games(
    games: Sequence[NbaDerivedScheduleGame],
) -> dict[str, NbaDerivedScheduleGame]:
    by_id = {game.replay_row.source_game_id: game for game in games}
    if len(by_id) != len(games):
        raise NbaProviderConformanceError("connector derived duplicate source_game_id values")
    question_ids = [game.replay_row.question_id for game in games]
    if len(set(question_ids)) != len(question_ids):
        raise NbaProviderConformanceError("connector derived duplicate question_id values")
    return by_id


def _require_inventory_games(
    derived_by_id: Mapping[str, NbaDerivedScheduleGame],
    inventory_games: Sequence[NbaDerivedScheduleGame],
) -> None:
    inventory_by_id = {game.replay_row.source_game_id: game for game in inventory_games}
    if set(derived_by_id) != set(inventory_by_id):
        raise NbaProviderConformanceError("derived schedule differs from the reviewed inventory")
    for game_id, derived in derived_by_id.items():
        if derived != inventory_by_id[game_id]:
            raise NbaProviderConformanceError("derived schedule facts differ from the inventory")


def _require_replay_rows(
    derived_by_id: Mapping[str, NbaDerivedScheduleGame],
    replay_rows: Sequence[NbaEloReplayRow],
) -> None:
    replay_by_id = {row.source_game_id: row for row in replay_rows}
    if len(replay_by_id) != len(replay_rows) or set(replay_by_id) != set(derived_by_id):
        raise NbaProviderConformanceError("replay rows differ from the derived schedule set")
    for game_id, derived in derived_by_id.items():
        if replay_by_id[game_id] != derived.replay_row:
            raise NbaProviderConformanceError("replay row differs from raw-derived schedule data")


def _require_cohort_games(
    derived_by_id: Mapping[str, NbaDerivedScheduleGame],
    cohort_games: Sequence[CohortGame],
) -> None:
    cohort_by_id = {game.source_game_id: game for game in cohort_games}
    if set(cohort_by_id) != set(derived_by_id):
        raise NbaProviderConformanceError("cohort games differ from the derived schedule set")
    for game_id, derived in derived_by_id.items():
        cohort_game = cohort_by_id[game_id]
        row = derived.replay_row
        if (
            cohort_game.question_id,
            cohort_game.forecast_deadline,
            cohort_game.scheduled_tipoff,
        ) != (row.question_id, row.forecast_cutoff, row.scheduled_tipoff):
            raise NbaProviderConformanceError(
                "cohort timing or identity differs from schedule data"
            )
