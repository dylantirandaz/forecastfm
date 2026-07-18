"""Tests for the bounded NBA provider conformance layer."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from forecastfm.integrity import canonical_json, canonical_sha256, file_sha256
from forecastfm.ledger import Cohort, CohortGame
from forecastfm.nba_elo_replay import NbaEloReplayRow
from forecastfm.nba_evidence import SourceRights
from forecastfm.nba_provider_conformance import (
    NBA_PROVIDER_CONFORMANCE_PROOF_SCOPE,
    NbaDerivedScheduleGame,
    NbaProviderCaseKind,
    NbaProviderChangeKind,
    NbaProviderConformanceError,
    NbaProviderConformanceRequest,
    NbaProviderConnector,
    NbaProviderCutoffExpectation,
    NbaProviderEntityKind,
    NbaProviderInventory,
    NbaProviderRevision,
    require_nba_provider_conformance,
)
from forecastfm.nba_snapshot_pack import (
    NbaSnapshot,
    NbaSnapshotMetadata,
    snapshot_metadata_sha256,
    write_snapshot_pack,
)

TIPOFF = datetime(2026, 10, 11, 2, tzinfo=UTC)
ATTESTATION_SHA256 = "a" * 64
CONNECTOR_SHA256 = "c" * 64


@dataclass(frozen=True, slots=True)
class FakeConnector:
    revisions_by_payload: Mapping[bytes, NbaProviderRevision]
    schedule_by_payload: Mapping[bytes, tuple[NbaDerivedScheduleGame, ...]]

    def decode_revision(self, snapshot: NbaSnapshot, /) -> NbaProviderRevision:
        return self.revisions_by_payload[snapshot.payload]

    def derive_schedule(
        self,
        snapshots: tuple[NbaSnapshot, ...],
        /,
    ) -> tuple[NbaDerivedScheduleGame, ...]:
        return tuple(
            game for snapshot in snapshots for game in self.schedule_by_payload[snapshot.payload]
        )


@dataclass(frozen=True, slots=True)
class Scenario:
    request: NbaProviderConformanceRequest
    connector: FakeConnector
    snapshots: tuple[NbaSnapshot, ...]


@dataclass(frozen=True, slots=True)
class RevisionSpec:
    source_id: str
    version: str
    entity_kind: NbaProviderEntityKind
    case_kind: NbaProviderCaseKind
    change_kind: NbaProviderChangeKind
    published_at: datetime


def _rights() -> SourceRights:
    return SourceRights(
        license_name="Licensed test archive",
        terms_url="https://provider.test/terms",
        terms_sha256="b" * 64,
        rights_as_of=TIPOFF - timedelta(days=30),
        local_processing="allowed",
        third_party_processing="allowed",
        tinker_processing="allowed",
        redistribution="prohibited",
    )


def _revision_snapshot(
    spec: RevisionSpec,
) -> tuple[NbaSnapshot, NbaProviderRevision]:
    payload = canonical_json({"source_id": spec.source_id, "version": spec.version}).encode("utf-8")
    retrieved_at = spec.published_at + timedelta(minutes=2)
    snapshot = NbaSnapshot(
        metadata=NbaSnapshotMetadata(
            source_id=spec.source_id,
            rights_scope="provider-test:nba",
            source_url=f"https://provider.test/archive/{spec.source_id}/{spec.version}",
            version=spec.version,
            effective_at=spec.published_at,
            provider_published_at=spec.published_at,
            retrieved_at=retrieved_at,
            available_at=spec.published_at,
            capture_method="provider_versioned_archive",
            sensitivity="ordinary",
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            archive_attestation_sha256=ATTESTATION_SHA256,
            rights=_rights(),
        ),
        payload=payload,
    )
    revision = NbaProviderRevision(
        source_id=spec.source_id,
        version=spec.version,
        entity_id=spec.source_id,
        entity_kind=spec.entity_kind,
        case_kind=spec.case_kind,
        change_kind=spec.change_kind,
        provider_schema_version="2026-10",
        provider_api_version="v3",
        effective_at=spec.published_at,
        provider_published_at=spec.published_at,
        archive_captured_at=spec.published_at + timedelta(minutes=1),
    )
    return snapshot, revision


def _schedule_rows() -> tuple[NbaEloReplayRow, ...]:
    return (
        NbaEloReplayRow(
            question_id="nba:g1",
            source_game_id="g1",
            season=2026,
            team_id="BOS",
            opponent_id="NYK",
            site="home",
            forecast_cutoff=TIPOFF - timedelta(minutes=60),
            scheduled_tipoff=TIPOFF,
        ),
        NbaEloReplayRow(
            question_id="nba:g2",
            source_game_id="g2",
            season=2026,
            team_id="LAL",
            opponent_id="GSW",
            site="neutral",
            forecast_cutoff=TIPOFF + timedelta(days=1, minutes=-60),
            scheduled_tipoff=TIPOFF + timedelta(days=1),
        ),
    )


def _make_scenario(tmp_path: Path) -> Scenario:
    revision_pairs = (
        _revision_snapshot(
            RevisionSpec(
                "lineup:game-1",
                "v1",
                "lineup",
                "lineup_change",
                "upsert",
                TIPOFF - timedelta(hours=7),
            )
        ),
        _revision_snapshot(
            RevisionSpec(
                "lineup:game-1",
                "v2",
                "lineup",
                "late_scratch",
                "correction",
                TIPOFF - timedelta(minutes=45),
            )
        ),
        _revision_snapshot(
            RevisionSpec(
                "lineup:game-1",
                "v3",
                "lineup",
                "late_scratch",
                "deletion",
                TIPOFF - timedelta(minutes=10),
            )
        ),
        _revision_snapshot(
            RevisionSpec(
                "roster:player-1",
                "v1",
                "roster",
                "roster_transaction",
                "upsert",
                TIPOFF - timedelta(days=1),
            )
        ),
        _revision_snapshot(
            RevisionSpec(
                "schedule:2026",
                "v1",
                "schedule",
                "reschedule",
                "correction",
                TIPOFF - timedelta(hours=9),
            )
        ),
    )
    snapshots = tuple(pair[0] for pair in revision_pairs)
    revisions = tuple(pair[1] for pair in revision_pairs)
    schedule_snapshot = snapshots[-1]
    replay_rows = _schedule_rows()
    schedule_games = (
        NbaDerivedScheduleGame(
            replay_row=replay_rows[0],
            season_type="regular",
            venue_id="td-garden",
            source_id=schedule_snapshot.metadata.source_id,
            snapshot_metadata_sha256=snapshot_metadata_sha256(schedule_snapshot.metadata),
        ),
        NbaDerivedScheduleGame(
            replay_row=replay_rows[1],
            season_type="regular",
            venue_id="t-mobile-arena",
            source_id=schedule_snapshot.metadata.source_id,
            snapshot_metadata_sha256=snapshot_metadata_sha256(schedule_snapshot.metadata),
        ),
    )
    inventory = NbaProviderInventory(
        revisions=revisions,
        cutoff_expectations=(
            NbaProviderCutoffExpectation(
                state="T-6h",
                source_id="lineup:game-1",
                cutoff=TIPOFF - timedelta(hours=6),
                scheduled_tipoff=TIPOFF,
                expected_version="v1",
            ),
            NbaProviderCutoffExpectation(
                state="T-60",
                source_id="lineup:game-1",
                cutoff=TIPOFF - timedelta(minutes=60),
                scheduled_tipoff=TIPOFF,
                expected_version="v1",
            ),
            NbaProviderCutoffExpectation(
                state="T-15",
                source_id="lineup:game-1",
                cutoff=TIPOFF - timedelta(minutes=15),
                scheduled_tipoff=TIPOFF,
                expected_version="v2",
            ),
            NbaProviderCutoffExpectation(
                state="T-6h",
                source_id="roster:player-1",
                cutoff=TIPOFF - timedelta(hours=6),
                scheduled_tipoff=TIPOFF,
                expected_version="v1",
            ),
            NbaProviderCutoffExpectation(
                state="T-60",
                source_id="roster:player-1",
                cutoff=TIPOFF - timedelta(minutes=60),
                scheduled_tipoff=TIPOFF,
                expected_version="v1",
            ),
            NbaProviderCutoffExpectation(
                state="T-15",
                source_id="roster:player-1",
                cutoff=TIPOFF - timedelta(minutes=15),
                scheduled_tipoff=TIPOFF,
                expected_version="v1",
            ),
        ),
        cutoff_exempt_source_ids=("schedule:2026",),
        schedule_games=schedule_games,
        known_schedule_gap_ids=(),
        archive_attestation_sha256=ATTESTATION_SHA256,
    )
    pack_path = tmp_path / "provider-sample.jsonl"
    write_snapshot_pack(snapshots, pack_path)
    cohort = Cohort(
        cohort_id="provider-conformance-sample",
        experiment_sha256="d" * 64,
        schedule_source="licensed-provider",
        schedule_snapshot_sha256=file_sha256(pack_path),
        schedule_retrieved=TIPOFF - timedelta(hours=8),
        inclusion_rule="all reviewed games in the bounded sample",
        games=tuple(
            CohortGame(
                question_id=row.question_id,
                source_game_id=row.source_game_id,
                team_id=row.team_id,
                opponent_id=row.opponent_id,
                site=row.site,
                matchup=f"{row.team_id} vs {row.opponent_id}",
                outcomes=("TEAM", "OPPONENT"),
                forecast_deadline=row.forecast_cutoff,
                scheduled_tipoff=row.scheduled_tipoff,
            )
            for row in replay_rows
        ),
    )
    request = NbaProviderConformanceRequest(
        snapshot_pack_path=pack_path,
        inventory=inventory,
        reviewed_inventory_sha256=inventory.sha256,
        connector_id="licensed-provider-v3",
        connector_sha256=CONNECTOR_SHA256,
        replay_rows=replay_rows,
        cohort=cohort,
    )
    connector = FakeConnector(
        revisions_by_payload={
            snapshot.payload: revision
            for snapshot, revision in zip(snapshots, revisions, strict=True)
        },
        schedule_by_payload={schedule_snapshot.payload: schedule_games},
    )
    return Scenario(request=request, connector=connector, snapshots=snapshots)


def _with_revisions(
    scenario: Scenario,
    revisions: tuple[NbaProviderRevision, ...],
) -> tuple[NbaProviderConformanceRequest, FakeConnector]:
    inventory = replace(scenario.request.inventory, revisions=revisions)
    request = replace(
        scenario.request,
        inventory=inventory,
        reviewed_inventory_sha256=inventory.sha256,
    )
    connector = replace(
        scenario.connector,
        revisions_by_payload={
            snapshot.payload: revision
            for snapshot, revision in zip(scenario.snapshots, revisions, strict=True)
        },
    )
    return request, connector


def test_exact_reviewed_sample_passes_and_report_is_deterministic(tmp_path: Path) -> None:
    scenario = _make_scenario(tmp_path)

    first = require_nba_provider_conformance(scenario.request, scenario.connector)
    second = require_nba_provider_conformance(scenario.request, scenario.connector)

    assert first == second
    assert first.status == "passed"
    assert first.canonical_payload()["proof_scope"] == NBA_PROVIDER_CONFORMANCE_PROOF_SCOPE
    assert first.inventory_sha256 == scenario.request.inventory.sha256
    assert first.replay_rows_sha256 == canonical_sha256(
        [row.canonical_payload() for row in scenario.request.replay_rows]
    )
    assert first.schedule_season_types == ("regular",)
    assert first.revision_count == 5
    assert first.schedule_game_count == 2
    changed_binding = replace(scenario.request, connector_sha256="e" * 64)
    changed_report = require_nba_provider_conformance(changed_binding, scenario.connector)
    assert changed_report.report_sha256 != first.report_sha256


def test_inventory_requires_an_independently_supplied_exact_digest(tmp_path: Path) -> None:
    scenario = _make_scenario(tmp_path)
    request = replace(scenario.request, reviewed_inventory_sha256="0" * 64)

    with pytest.raises(NbaProviderConformanceError, match="independently reviewed"):
        require_nba_provider_conformance(request, scenario.connector)


def test_pack_must_contain_every_reviewed_revision(tmp_path: Path) -> None:
    scenario = _make_scenario(tmp_path)
    incomplete_path = tmp_path / "incomplete.jsonl"
    write_snapshot_pack(scenario.snapshots[:-1], incomplete_path)
    cohort = replace(
        scenario.request.cohort,
        schedule_snapshot_sha256=file_sha256(incomplete_path),
    )
    request = replace(
        scenario.request,
        snapshot_pack_path=incomplete_path,
        cohort=cohort,
    )

    with pytest.raises(NbaProviderConformanceError, match="exact revision inventory"):
        require_nba_provider_conformance(request, scenario.connector)


def test_conformance_sample_must_use_the_reviewed_provider_archive(tmp_path: Path) -> None:
    scenario = _make_scenario(tmp_path)
    original = scenario.snapshots[0]
    live_metadata = replace(
        original.metadata,
        available_at=original.metadata.retrieved_at,
        capture_method="live",
        archive_attestation_sha256=None,
    )
    live_snapshot = NbaSnapshot(metadata=live_metadata, payload=original.payload)
    snapshots = (live_snapshot, *scenario.snapshots[1:])
    path = tmp_path / "live-sample.jsonl"
    write_snapshot_pack(snapshots, path)
    cohort = replace(scenario.request.cohort, schedule_snapshot_sha256=file_sha256(path))
    request = replace(scenario.request, snapshot_pack_path=path, cohort=cohort)

    with pytest.raises(NbaProviderConformanceError, match="must use provider archives"):
        require_nba_provider_conformance(request, scenario.connector)


def test_decoded_full_revision_must_equal_external_inventory(tmp_path: Path) -> None:
    scenario = _make_scenario(tmp_path)
    target = scenario.request.inventory.revisions[1]
    fabricated = replace(target, provider_schema_version="fabricated")
    revisions_by_payload = dict(scenario.connector.revisions_by_payload)
    revisions_by_payload[scenario.snapshots[1].payload] = fabricated
    connector = replace(scenario.connector, revisions_by_payload=revisions_by_payload)

    with pytest.raises(NbaProviderConformanceError, match="reviewed inventory"):
        require_nba_provider_conformance(scenario.request, connector)


def test_reviewed_revision_still_must_match_pack_metadata(tmp_path: Path) -> None:
    scenario = _make_scenario(tmp_path)
    revisions = list(scenario.request.inventory.revisions)
    revisions[1] = replace(
        revisions[1],
        provider_published_at=revisions[1].provider_published_at + timedelta(seconds=1),
    )
    request, connector = _with_revisions(scenario, tuple(revisions))

    with pytest.raises(NbaProviderConformanceError, match="publication time"):
        require_nba_provider_conformance(request, connector)


def test_sample_requires_correction_deletion_and_core_cases(tmp_path: Path) -> None:
    scenario = _make_scenario(tmp_path)
    revisions = list(scenario.request.inventory.revisions)
    revisions[2] = replace(revisions[2], change_kind="upsert")
    request, connector = _with_revisions(scenario, tuple(revisions))

    with pytest.raises(NbaProviderConformanceError, match="change markers: deletion"):
        require_nba_provider_conformance(request, connector)

    revisions = list(scenario.request.inventory.revisions)
    revisions[-1] = replace(revisions[-1], case_kind="other")
    request, connector = _with_revisions(scenario, tuple(revisions))
    with pytest.raises(NbaProviderConformanceError, match="required cases: reschedule"):
        require_nba_provider_conformance(request, connector)


def test_cutoff_matrix_rejects_future_revision_leakage(tmp_path: Path) -> None:
    scenario = _make_scenario(tmp_path)
    expectations = list(scenario.request.inventory.cutoff_expectations)
    expectations[1] = replace(expectations[1], expected_version="v2")
    inventory = replace(scenario.request.inventory, cutoff_expectations=tuple(expectations))
    request = replace(
        scenario.request,
        inventory=inventory,
        reviewed_inventory_sha256=inventory.sha256,
    )

    with pytest.raises(NbaProviderConformanceError, match="other than the reviewed one"):
        require_nba_provider_conformance(request, scenario.connector)


@pytest.mark.parametrize(
    ("exempt_source_ids", "message"),
    [
        ((), "every revision source"),
        (("lineup:game-1", "schedule:2026"), "cannot also be exempt"),
        (("schedule:2026", "unknown:feed"), "only reviewed schedule sources"),
    ],
)
def test_cutoff_exemptions_are_exact_and_schedule_only(
    tmp_path: Path,
    exempt_source_ids: tuple[str, ...],
    message: str,
) -> None:
    scenario = _make_scenario(tmp_path)

    with pytest.raises(NbaProviderConformanceError, match=message):
        replace(
            scenario.request.inventory,
            cutoff_exempt_source_ids=exempt_source_ids,
        )


def test_one_cutoff_matrix_cannot_mix_event_tipoffs(tmp_path: Path) -> None:
    scenario = _make_scenario(tmp_path)
    expectations = list(scenario.request.inventory.cutoff_expectations)
    expectations[2] = replace(
        expectations[2],
        cutoff=expectations[2].cutoff + timedelta(days=1),
        scheduled_tipoff=expectations[2].scheduled_tipoff + timedelta(days=1),
    )

    with pytest.raises(NbaProviderConformanceError, match="one scheduled_tipoff"):
        replace(
            scenario.request.inventory,
            cutoff_expectations=tuple(expectations),
        )


def test_schedule_must_equal_external_facts_not_only_local_replay(tmp_path: Path) -> None:
    scenario = _make_scenario(tmp_path)
    schedule_games = list(scenario.request.inventory.schedule_games)
    changed_row = replace(schedule_games[0].replay_row, team_id="MIA")
    schedule_games[0] = replace(schedule_games[0], replay_row=changed_row)
    connector = replace(
        scenario.connector,
        schedule_by_payload={scenario.snapshots[-1].payload: tuple(schedule_games)},
    )

    with pytest.raises(NbaProviderConformanceError, match="schedule facts"):
        require_nba_provider_conformance(scenario.request, connector)


def test_schedule_lineage_completeness_replay_and_cohort_are_fail_closed(
    tmp_path: Path,
) -> None:
    scenario = _make_scenario(tmp_path)
    schedule_games = scenario.request.inventory.schedule_games
    bad_lineage = (replace(schedule_games[0], snapshot_metadata_sha256="f" * 64), schedule_games[1])
    connector = replace(
        scenario.connector,
        schedule_by_payload={scenario.snapshots[-1].payload: bad_lineage},
    )
    with pytest.raises(NbaProviderConformanceError, match="incorrect snapshot lineage"):
        require_nba_provider_conformance(scenario.request, connector)

    inventory = replace(scenario.request.inventory, known_schedule_gap_ids=("missing-g3",))
    request = replace(
        scenario.request,
        inventory=inventory,
        reviewed_inventory_sha256=inventory.sha256,
    )
    with pytest.raises(NbaProviderConformanceError, match="known schedule gaps"):
        require_nba_provider_conformance(request, scenario.connector)

    connector = replace(
        scenario.connector,
        schedule_by_payload={scenario.snapshots[-1].payload: schedule_games[:1]},
    )
    with pytest.raises(NbaProviderConformanceError, match="reviewed inventory"):
        require_nba_provider_conformance(scenario.request, connector)

    replay_rows = list(scenario.request.replay_rows)
    replay_rows[0] = replace(replay_rows[0], opponent_id="MIA")
    request = replace(scenario.request, replay_rows=tuple(replay_rows))
    with pytest.raises(NbaProviderConformanceError, match="replay row differs"):
        require_nba_provider_conformance(request, scenario.connector)

    cohort = replace(scenario.request.cohort, games=scenario.request.cohort.games[:1])
    request = replace(scenario.request, cohort=cohort)
    with pytest.raises(NbaProviderConformanceError, match="cohort games differ"):
        require_nba_provider_conformance(request, scenario.connector)


def test_cohort_must_bind_the_exact_snapshot_pack(tmp_path: Path) -> None:
    scenario = _make_scenario(tmp_path)
    cohort = replace(scenario.request.cohort, schedule_snapshot_sha256="0" * 64)
    request = replace(scenario.request, cohort=cohort)

    with pytest.raises(NbaProviderConformanceError, match="exact snapshot pack"):
        require_nba_provider_conformance(request, scenario.connector)


def test_connector_protocol_remains_structural(tmp_path: Path) -> None:
    scenario = _make_scenario(tmp_path)
    connector: NbaProviderConnector = scenario.connector

    assert require_nba_provider_conformance(scenario.request, connector).status == "passed"
