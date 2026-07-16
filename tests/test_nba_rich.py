"""Tests for the stable richer NBA feature schema."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import copysign

import pytest

from forecastfm.ledger import CohortGame
from forecastfm.models import ForecastQuestion
from forecastfm.nba_evidence import (
    NbaEvidenceBundle,
    NbaEvidenceError,
    NbaEvidenceRecord,
    SourceRights,
    SourceSnapshot,
)
from forecastfm.nba_rich import (
    NBA_LOCAL_HEALTH_FEATURE_SPECS,
    NBA_RICH_FEATURE_NAMES,
    NBA_RICH_FEATURE_SPECS,
    NBA_RICH_SCHEMA_SHA256,
    local_health_feature_vector,
    local_rich_features_from_bundle,
    tinker_rich_features_from_bundle,
)

CUTOFF = datetime(2026, 10, 1, 16, tzinfo=UTC)
RETRIEVED = CUTOFF - timedelta(hours=1)


def _bundle() -> NbaEvidenceBundle:
    rights = SourceRights(
        license_name="Test agreement",
        terms_url="https://provider.test/terms",
        terms_sha256="a" * 64,
        rights_as_of=RETRIEVED - timedelta(days=1),
        local_processing="allowed",
        third_party_processing="allowed",
        tinker_processing="allowed",
        redistribution="unknown",
    )
    source = SourceSnapshot(
        source_id="licensed-feed",
        source_url="https://provider.test/snapshot",
        payload_sha256="b" * 64,
        published_at=RETRIEVED - timedelta(minutes=1),
        retrieved_at=RETRIEVED,
        capture_method="live",
        sensitivity="ordinary",
        rights=rights,
    )
    records = tuple(
        NbaEvidenceRecord(
            record_id=f"feature-{index:02d}",
            kind=spec.kind,
            feature_name=spec.name,
            team_value=float(index),
            opponent_value=0.5,
            source_ids=(source.source_id,),
            available_at=RETRIEVED,
        )
        for index, spec in enumerate(
            reversed(NBA_RICH_FEATURE_SPECS),
            start=1,
        )
    )
    game = CohortGame(
        question_id="game-1",
        source_game_id="source-game-1",
        matchup="Team vs Opponent",
        outcomes=("team", "opponent"),
        forecast_deadline=CUTOFF,
        scheduled_tipoff=CUTOFF + timedelta(hours=1),
    )
    question = ForecastQuestion(
        question_id=game.question_id,
        text="Will the listed team win?",
        resolution_rule="Use the final score.",
        resolution_source="https://provider.test/result",
        outcomes=game.outcomes,
        forecast_at=CUTOFF,
        resolves_at=CUTOFF + timedelta(hours=4),
    )
    return NbaEvidenceBundle(
        game=game,
        question=question,
        sources=(source,),
        records=records,
    )


def test_bundle_aggregates_in_predeclared_feature_order() -> None:
    bundle = _bundle()
    features = local_rich_features_from_bundle(bundle, action_at=CUTOFF)

    assert tuple(features.as_dict()) == NBA_RICH_FEATURE_NAMES
    assert features.rest_days_difference == 10.5
    assert features.schedule_strength_difference == 0.5
    assert tinker_rich_features_from_bundle(bundle, action_at=CUTOFF) == features
    assert len(NBA_RICH_SCHEMA_SHA256) == 64


def test_rich_feature_side_swap_is_an_exact_involution() -> None:
    features = local_rich_features_from_bundle(_bundle(), action_at=CUTOFF)
    swapped = features.side_swap()

    assert swapped.vector == tuple(0.0 if value == 0.0 else -value for value in features.vector)
    assert swapped.side_swap() == features

    zeroed = replace(features, rest_days_difference=0.0)
    assert copysign(1.0, zeroed.side_swap().rest_days_difference) == 1.0


def test_rich_feature_kinds_are_predeclared() -> None:
    bundle = _bundle()
    wrong_kind = replace(bundle.records[0], kind="team_metric")

    with pytest.raises(NbaEvidenceError, match="kinds do not match"):
        local_rich_features_from_bundle(
            replace(bundle, records=(wrong_kind, *bundle.records[1:])),
            action_at=CUTOFF,
        )


def test_health_features_have_an_explicit_local_only_vector() -> None:
    bundle = _bundle()
    health_source = replace(bundle.sources[0], sensitivity="player_health")
    records = tuple(
        NbaEvidenceRecord(
            record_id=f"health-{index:02d}",
            kind=spec.kind,
            feature_name=spec.name,
            team_value=float(index),
            opponent_value=0.0,
            source_ids=(health_source.source_id,),
            available_at=RETRIEVED,
        )
        for index, spec in enumerate(NBA_LOCAL_HEALTH_FEATURE_SPECS, start=1)
    )
    health_bundle = replace(bundle, sources=(health_source,), records=records)

    assert local_health_feature_vector(health_bundle, action_at=CUTOFF) == (1.0, 2.0)
