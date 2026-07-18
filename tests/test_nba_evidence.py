"""Tests for rights-aware, point-in-time NBA evidence."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from typing import cast

import pytest

from forecastfm.ledger import CohortGame
from forecastfm.models import EvidenceCard, ForecastQuestion
from forecastfm.nba_evidence import (
    CaptureMethod,
    EvidenceKind,
    NbaEvidenceBundle,
    NbaEvidenceError,
    NbaEvidenceRecord,
    Sensitivity,
    SourceRights,
    SourceSnapshot,
    evidence_bundle_sha256,
    local_evidence_cards,
    local_numeric_feature_vector,
    require_prospective_capture,
    require_redistribution_allowed,
    tinker_evidence_cards,
    tinker_numeric_feature_vector,
)

RIGHTS_AS_OF = datetime(2026, 9, 1, tzinfo=UTC)
PUBLISHED_AT = datetime(2026, 10, 1, 14, tzinfo=UTC)
RETRIEVED_AT = datetime(2026, 10, 1, 15, tzinfo=UTC)
FORECAST_DEADLINE = datetime(2026, 10, 1, 16, tzinfo=UTC)
TIPOFF = datetime(2026, 10, 1, 17, tzinfo=UTC)
ACTION_AT = datetime(2026, 10, 1, 16, 30, tzinfo=UTC)


def make_rights() -> SourceRights:
    return SourceRights(
        license_name="Test commercial feed agreement",
        terms_url="https://provider.test/terms",
        terms_sha256="a" * 64,
        rights_as_of=RIGHTS_AS_OF,
        local_processing="allowed",
        third_party_processing="allowed",
        tinker_processing="allowed",
        redistribution="allowed",
    )


def make_source(
    *,
    source_id: str = "team-feed-1",
    capture_method: CaptureMethod = "live",
    sensitivity: Sensitivity = "ordinary",
    retrieved_at: datetime = RETRIEVED_AT,
    rights: SourceRights | None = None,
) -> SourceSnapshot:
    archive_version_id = None
    archive_attestation_sha256 = None
    if capture_method == "provider_versioned_archive":
        archive_version_id = f"version-{source_id}"
        archive_attestation_sha256 = "c" * 64
    return SourceSnapshot(
        source_id=source_id,
        rights_scope="provider-test:nba:team-metrics",
        source_url=f"https://provider.test/snapshots/{source_id}",
        payload_sha256="b" * 64,
        snapshot_metadata_sha256="d" * 64,
        published_at=PUBLISHED_AT,
        retrieved_at=retrieved_at,
        capture_method=capture_method,
        sensitivity=sensitivity,
        rights=rights or make_rights(),
        archive_version_id=archive_version_id,
        archive_attestation_sha256=archive_attestation_sha256,
    )


def make_game() -> CohortGame:
    return CohortGame(
        question_id="nba-game-1",
        source_game_id="provider-game-1",
        team_id="Team",
        opponent_id="Opponent",
        site="neutral",
        matchup="Team vs Opponent",
        outcomes=("listed_team_wins", "opponent_wins"),
        forecast_deadline=FORECAST_DEADLINE,
        scheduled_tipoff=TIPOFF,
    )


def make_question(
    *,
    question_id: str = "nba-game-1",
    forecast_at: datetime = FORECAST_DEADLINE,
) -> ForecastQuestion:
    return ForecastQuestion(
        question_id=question_id,
        text="Will the listed team win?",
        resolution_rule="Use the official final score.",
        resolution_source="https://provider.test/results/provider-game-1",
        outcomes=("listed_team_wins", "opponent_wins"),
        forecast_at=forecast_at,
        resolves_at=TIPOFF + timedelta(hours=3),
    )


def make_record(
    *,
    kind: EvidenceKind = "team_metric",
    feature_name: str = "trailing_net_rating",
    team_value: float = 4.1,
    source_ids: tuple[str, ...] = ("team-feed-1",),
    available_at: datetime = RETRIEVED_AT,
) -> NbaEvidenceRecord:
    return NbaEvidenceRecord(
        record_id="team-form-1",
        kind=kind,
        feature_name=feature_name,
        team_value=team_value,
        opponent_value=0.0,
        source_ids=source_ids,
        available_at=available_at,
    )


def make_bundle(
    *,
    game: CohortGame | None = None,
    question: ForecastQuestion | None = None,
    sources: tuple[SourceSnapshot, ...] | None = None,
    records: tuple[NbaEvidenceRecord, ...] | None = None,
) -> NbaEvidenceBundle:
    return NbaEvidenceBundle(
        game=game or make_game(),
        question=question or make_question(),
        sources=sources or (make_source(),),
        records=records or (make_record(),),
    )


def test_allowed_bundle_builds_local_and_tinker_cards() -> None:
    bundle = make_bundle()
    expected = (
        EvidenceCard(
            text='Pregame numeric feature: {"trailing_net_rating":4.1}',
            source="https://provider.test/snapshots/team-feed-1",
            available_at=RETRIEVED_AT,
        ),
    )

    assert local_evidence_cards(bundle, action_at=ACTION_AT) == expected
    assert tinker_evidence_cards(bundle, action_at=ACTION_AT) == expected
    assert bundle.records[0].difference == 4.1
    assert local_numeric_feature_vector(
        bundle,
        ("trailing_net_rating",),
        action_at=ACTION_AT,
    ) == (4.1,)
    assert tinker_numeric_feature_vector(
        bundle,
        ("trailing_net_rating",),
        action_at=ACTION_AT,
    ) == (4.1,)
    assert bundle.records[0].side_swap().difference == -4.1
    assert bundle.records[0].side_swap().side_swap() == bundle.records[0]
    assert require_redistribution_allowed(bundle, action_at=ACTION_AT) is None
    assert require_prospective_capture(bundle) is None

    with pytest.raises(NbaEvidenceError, match="predeclared schema"):
        local_numeric_feature_vector(bundle, ("travel_miles",), action_at=ACTION_AT)


def test_evidence_timestamps_must_be_utc_and_causally_ordered() -> None:
    central = timezone(timedelta(hours=-6))

    with pytest.raises(NbaEvidenceError, match="rights_as_of must be in UTC"):
        replace(make_rights(), rights_as_of=RIGHTS_AS_OF.astimezone(central))
    with pytest.raises(NbaEvidenceError, match="published_at must be in UTC"):
        replace(make_source(), published_at=PUBLISHED_AT.astimezone(central))
    with pytest.raises(NbaEvidenceError, match="available_at must be in UTC"):
        replace(make_record(), available_at=RETRIEVED_AT.astimezone(central))
    with pytest.raises(NbaEvidenceError, match="cannot be after"):
        replace(make_source(), published_at=RETRIEVED_AT + timedelta(minutes=1))


def test_bundle_binds_question_id_cutoff_and_source_availability() -> None:
    with pytest.raises(NbaEvidenceError, match="IDs must match"):
        make_bundle(question=make_question(question_id="different-game"))

    with pytest.raises(NbaEvidenceError, match="cutoff must equal"):
        make_bundle(question=make_question(forecast_at=FORECAST_DEADLINE - timedelta(minutes=1)))

    reversed_outcomes = tuple(reversed(make_question().outcomes))
    with pytest.raises(NbaEvidenceError, match="outcomes must match"):
        make_bundle(question=replace(make_question(), outcomes=reversed_outcomes))

    central = timezone(timedelta(hours=-6))
    with pytest.raises(NbaEvidenceError, match=r"question\.forecast_at must be in UTC"):
        make_bundle(question=make_question(forecast_at=FORECAST_DEADLINE.astimezone(central)))

    before_retrieval = RETRIEVED_AT - timedelta(seconds=1)
    with pytest.raises(NbaEvidenceError, match="cannot predate its latest source"):
        make_bundle(records=(replace(make_record(), available_at=before_retrieval),))

    late = FORECAST_DEADLINE + timedelta(seconds=1)
    with pytest.raises(NbaEvidenceError, match="newer than the forecast deadline"):
        make_bundle(
            sources=(make_source(retrieved_at=late),),
            records=(make_record(available_at=late),),
        )


def test_unknown_or_prohibited_rights_fail_closed() -> None:
    local_unknown = replace(make_rights(), local_processing="unknown")
    with pytest.raises(NbaEvidenceError, match="local_processing must be explicitly allowed"):
        local_evidence_cards(
            make_bundle(sources=(make_source(rights=local_unknown),)),
            action_at=ACTION_AT,
        )
    with pytest.raises(NbaEvidenceError, match="local_processing must be explicitly allowed"):
        local_numeric_feature_vector(
            make_bundle(sources=(make_source(rights=local_unknown),)),
            ("trailing_net_rating",),
            action_at=ACTION_AT,
        )

    third_party_unknown = replace(make_rights(), third_party_processing="unknown")
    with pytest.raises(NbaEvidenceError, match="third_party_processing"):
        tinker_evidence_cards(
            make_bundle(sources=(make_source(rights=third_party_unknown),)),
            action_at=ACTION_AT,
        )

    tinker_prohibited = replace(make_rights(), tinker_processing="prohibited")
    with pytest.raises(NbaEvidenceError, match="tinker_processing"):
        tinker_evidence_cards(
            make_bundle(sources=(make_source(rights=tinker_prohibited),)),
            action_at=ACTION_AT,
        )

    redistribution_unknown = replace(make_rights(), redistribution="unknown")
    with pytest.raises(NbaEvidenceError, match="redistribution"):
        require_redistribution_allowed(
            make_bundle(sources=(make_source(rights=redistribution_unknown),)),
            action_at=ACTION_AT,
        )


def test_protected_action_uses_only_prior_rights_and_retrievals() -> None:
    bundle = make_bundle()
    central = timezone(timedelta(hours=-6))

    with pytest.raises(NbaEvidenceError, match="action_at must be in UTC"):
        local_evidence_cards(bundle, action_at=ACTION_AT.astimezone(central))

    future_rights = replace(make_rights(), rights_as_of=ACTION_AT + timedelta(minutes=1))
    future_rights_bundle = make_bundle(sources=(make_source(rights=future_rights),))
    with pytest.raises(NbaEvidenceError, match="rights_as_of cannot be after"):
        local_evidence_cards(future_rights_bundle, action_at=ACTION_AT)

    archive_source = make_source(
        capture_method="provider_versioned_archive",
        retrieved_at=ACTION_AT + timedelta(minutes=1),
    )
    archive_bundle = make_bundle(
        sources=(archive_source,),
        records=(make_record(available_at=PUBLISHED_AT),),
    )
    with pytest.raises(NbaEvidenceError, match="retrieval cannot be after"):
        local_evidence_cards(archive_bundle, action_at=ACTION_AT)

    derived_later = RETRIEVED_AT + timedelta(minutes=45)
    later_bundle = make_bundle(records=(make_record(available_at=derived_later),))
    with pytest.raises(NbaEvidenceError, match="availability cannot be after"):
        local_evidence_cards(
            later_bundle,
            action_at=RETRIEVED_AT + timedelta(minutes=30),
        )


def test_tinker_rejects_health_lineage_even_when_text_looks_innocuous() -> None:
    health_source = make_source(sensitivity="player_health")
    innocuous_record = make_record(
        kind="expected_lineup",
        feature_name="rotation_continuity",
        team_value=0.73,
    )
    bundle = make_bundle(sources=(health_source,), records=(innocuous_record,))

    assert "rotation_continuity" in local_evidence_cards(bundle, action_at=ACTION_AT)[0].text
    with pytest.raises(NbaEvidenceError, match="player-health lineage"):
        tinker_evidence_cards(bundle, action_at=ACTION_AT)
    with pytest.raises(NbaEvidenceError, match="player-health lineage"):
        tinker_numeric_feature_vector(
            bundle,
            ("rotation_continuity",),
            action_at=ACTION_AT,
        )


def test_tinker_accepts_explicitly_licensed_ordinary_lineup_aggregate() -> None:
    record = make_record(
        kind="expected_lineup",
        feature_name="expected_lineup_continuity",
        team_value=0.8,
    )

    cards = tinker_evidence_cards(make_bundle(records=(record,)), action_at=ACTION_AT)

    assert cards[0].text == ('Pregame numeric feature: {"expected_lineup_continuity":0.8}')


def test_tinker_applies_lexical_screen_after_lineage_gate() -> None:
    record = make_record(feature_name="injury_adjustment")

    with pytest.raises(ValueError, match="injury"):
        tinker_evidence_cards(make_bundle(records=(record,)), action_at=ACTION_AT)


def test_injury_record_cannot_hide_missing_health_lineage() -> None:
    with pytest.raises(NbaEvidenceError, match="must retain player-health lineage"):
        make_bundle(records=(make_record(kind="injury"),))


def test_provider_archive_and_live_capture_have_different_availability() -> None:
    retrieved_after_cutoff = FORECAST_DEADLINE + timedelta(minutes=15)
    archived_source = make_source(
        capture_method="provider_versioned_archive",
        retrieved_at=retrieved_after_cutoff,
    )
    archived_record = make_record(available_at=PUBLISHED_AT)

    archived_bundle = make_bundle(
        sources=(archived_source,),
        records=(archived_record,),
    )

    assert archived_source.historical_available_at() == PUBLISHED_AT
    assert (
        local_evidence_cards(archived_bundle, action_at=ACTION_AT)[0].available_at == PUBLISHED_AT
    )
    with pytest.raises(NbaEvidenceError, match="requires live source capture"):
        require_prospective_capture(archived_bundle)
    with pytest.raises(NbaEvidenceError, match="require an attestation digest"):
        replace(archived_source, archive_attestation_sha256=None)

    live_source = make_source(retrieved_at=retrieved_after_cutoff)
    assert live_source.historical_available_at() == retrieved_after_cutoff
    with pytest.raises(NbaEvidenceError, match="newer than the forecast deadline"):
        make_bundle(
            sources=(live_source,),
            records=(make_record(available_at=retrieved_after_cutoff),),
        )


def test_duplicate_and_missing_source_references_are_rejected() -> None:
    with pytest.raises(NbaEvidenceError, match="cannot repeat a source ID"):
        make_record(source_ids=("team-feed-1", "team-feed-1"))

    with pytest.raises(NbaEvidenceError, match="source IDs must be unique"):
        make_bundle(sources=(make_source(), make_source()))

    other_source = make_source(source_id="other-feed")
    with pytest.raises(NbaEvidenceError, match="sources must be ordered"):
        make_bundle(sources=(make_source(), other_source))

    with pytest.raises(NbaEvidenceError, match="source IDs must be ordered"):
        make_record(source_ids=("team-feed-1", "other-feed"))

    with pytest.raises(NbaEvidenceError, match="unknown source ID"):
        make_bundle(records=(make_record(source_ids=("missing-feed",)),))

    with pytest.raises(NbaEvidenceError, match="every source snapshot must be referenced"):
        make_bundle(sources=(other_source, make_source()))

    duplicate_record = make_record()
    with pytest.raises(NbaEvidenceError, match="record IDs must be unique"):
        make_bundle(records=(duplicate_record, duplicate_record))


def test_evidence_values_require_canonical_floats() -> None:
    with pytest.raises(NbaEvidenceError, match="team_value must be a finite float"):
        replace(make_record(), team_value=cast(float, 1))
    with pytest.raises(NbaEvidenceError, match="team_value cannot use negative zero"):
        replace(make_record(), team_value=-0.0)


def test_bundle_hash_is_deterministic_and_sensitive_to_bound_inputs() -> None:
    bundle = make_bundle()
    original = evidence_bundle_sha256(bundle)

    changed_record = replace(bundle.records[0], team_value=4.2)
    changed_evidence = replace(bundle, records=(changed_record,))
    changed_rights = replace(bundle.sources[0].rights, redistribution="unknown")
    changed_source = replace(bundle.sources[0], rights=changed_rights)
    changed_lineage = replace(bundle, sources=(changed_source,))
    changed_snapshot_identity = replace(
        bundle,
        sources=(replace(bundle.sources[0], snapshot_metadata_sha256="e" * 64),),
    )
    changed_matchup = replace(
        bundle,
        game=replace(
            bundle.game,
            team_id="Changed",
            matchup="Changed vs Opponent",
        ),
    )
    changed_question = replace(bundle, question=replace(bundle.question, text="Changed question"))

    assert len(original) == 64
    assert evidence_bundle_sha256(bundle) == original
    assert evidence_bundle_sha256(changed_evidence) != original
    assert evidence_bundle_sha256(changed_lineage) != original
    assert evidence_bundle_sha256(changed_snapshot_identity) != original
    assert evidence_bundle_sha256(changed_matchup) != original
    assert evidence_bundle_sha256(changed_question) != original
