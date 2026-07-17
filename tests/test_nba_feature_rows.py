"""Tests for immutable target-free richer NBA feature rows."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import copysign
from pathlib import Path
from typing import cast

import pytest

from forecastfm.integrity import canonical_json
from forecastfm.json_utils import require_list, require_object, required_field
from forecastfm.ledger import CohortGame
from forecastfm.models import ForecastQuestion
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_evidence import (
    NbaEvidenceBundle,
    NbaEvidenceError,
    NbaEvidenceRecord,
    SourceRights,
    SourceSnapshot,
    evidence_bundle_sha256,
)
from forecastfm.nba_feature_rows import (
    NBA_PRIMARY_STATE_ID,
    NbaEloPriorInput,
    NbaFeatureRowError,
    build_local_rich_feature_row,
    build_tinker_rich_feature_row,
    read_nba_feature_rows_jsonl,
    write_nba_feature_rows_jsonl,
)
from forecastfm.nba_rich import (
    NBA_RICH_FEATURE_NAMES,
    NBA_RICH_FEATURE_SPECS,
    NBA_RICH_SCHEMA_SHA256,
)

CUTOFF = datetime(2026, 10, 21, 22, tzinfo=UTC)
TIPOFF = CUTOFF + timedelta(hours=1)
EVIDENCE_AT = CUTOFF - timedelta(hours=2)
ELO_AT = CUTOFF - timedelta(hours=1)
ACTION_AT = CUTOFF + timedelta(days=1)
ELO_STATE_SHA256 = "c" * 64


def _elo(
    probability: float = 0.61,
    *,
    available_at: datetime = ELO_AT,
    state_sha256: str = ELO_STATE_SHA256,
) -> NbaEloPriorInput:
    return NbaEloPriorInput(
        team_win_probability=probability,
        available_at=available_at,
        state_sha256=state_sha256,
    )


def _bundle() -> NbaEvidenceBundle:
    rights = SourceRights(
        license_name="Test agreement",
        terms_url="https://provider.test/terms",
        terms_sha256="a" * 64,
        rights_as_of=EVIDENCE_AT - timedelta(days=1),
        local_processing="allowed",
        third_party_processing="allowed",
        tinker_processing="allowed",
        redistribution="unknown",
    )
    source = SourceSnapshot(
        source_id="licensed-feed",
        rights_scope="provider-test:nba:metrics",
        source_url="https://provider.test/snapshot",
        payload_sha256="b" * 64,
        snapshot_metadata_sha256="c" * 64,
        published_at=EVIDENCE_AT - timedelta(minutes=1),
        retrieved_at=EVIDENCE_AT,
        capture_method="live",
        sensitivity="ordinary",
        rights=rights,
    )
    records = tuple(
        NbaEvidenceRecord(
            record_id=f"feature-{index:02d}",
            kind=spec.kind,
            feature_name=spec.name,
            team_value=spec.minimum,
            opponent_value=spec.maximum,
            source_ids=(source.source_id,),
            available_at=EVIDENCE_AT,
        )
        for index, spec in enumerate(NBA_RICH_FEATURE_SPECS)
    )
    game = CohortGame(
        question_id="nba-rich-1",
        source_game_id="provider-game-1",
        matchup="Team vs Opponent",
        outcomes=("team", "opponent"),
        forecast_deadline=CUTOFF,
        scheduled_tipoff=TIPOFF,
    )
    question = ForecastQuestion(
        question_id=game.question_id,
        text="Will the listed team win?",
        resolution_rule="Use the official final score.",
        resolution_source="https://provider.test/final",
        outcomes=game.outcomes,
        forecast_at=CUTOFF,
        resolves_at=TIPOFF + timedelta(hours=3),
    )
    return NbaEvidenceBundle(game=game, question=question, sources=(source,), records=records)


def test_builders_bind_the_exact_target_free_inputs() -> None:
    bundle = _bundle()
    local = build_local_rich_feature_row(
        bundle,
        season=2027,
        elo=_elo(),
        action_at=ACTION_AT,
    )
    tinker = build_tinker_rich_feature_row(
        bundle,
        season=2027,
        elo=_elo(),
        action_at=ACTION_AT,
    )

    assert local == tinker
    assert local.question_id == bundle.game.question_id
    assert local.season == 2027
    assert local.forecast_cutoff == CUTOFF
    assert local.scheduled_tipoff == TIPOFF
    assert local.elo_team_win_probability == 0.61
    assert local.elo_opponent_win_probability == 1.0 - 0.61
    assert local.elo_available_at == ELO_AT
    assert local.elo_state_sha256 == ELO_STATE_SHA256
    assert local.evidence_bundle_sha256 == evidence_bundle_sha256(bundle)
    assert local.input_available_at == ELO_AT
    assert tuple(name for name, _value in local.feature_items) == NBA_RICH_FEATURE_NAMES
    assert tuple(value for _name, value in local.feature_items) == local.rich_features.vector
    payload = local.canonical_payload()
    assert payload["elo_state_sha256"] == ELO_STATE_SHA256
    assert payload["state_id"] == NBA_PRIMARY_STATE_ID
    assert payload["features"] == [
        {"name": name, "value": value} for name, value in local.feature_items
    ]
    assert local.feature_schema_sha256 == NBA_RICH_SCHEMA_SHA256
    assert len(local.row_sha256) == 64
    assert replace(local, elo_state_sha256="d" * 64).row_sha256 != local.row_sha256


def test_feature_rows_round_trip_as_canonical_original_only_jsonl(tmp_path: Path) -> None:
    first = build_local_rich_feature_row(
        _bundle(),
        season=2027,
        elo=_elo(),
        action_at=ACTION_AT,
    )
    second = replace(first, question_id="nba-rich-2")
    path = tmp_path / "features.jsonl"

    write_nba_feature_rows_jsonl(path, (first, second))

    assert read_nba_feature_rows_jsonl(path) == (first, second)
    assert path.read_text(encoding="utf-8") == "".join(
        f"{canonical_json(row.canonical_payload())}\n" for row in (first, second)
    )


def test_feature_row_reader_rejects_targets_and_noncanonical_json(tmp_path: Path) -> None:
    row = build_local_rich_feature_row(
        _bundle(),
        season=2027,
        elo=_elo(),
        action_at=ACTION_AT,
    )
    payload = row.canonical_payload()
    payload["label"] = "TEAM"
    path = tmp_path / "features.jsonl"
    path.write_text(f"{canonical_json(payload)}\n", encoding="utf-8")

    with pytest.raises(NbaFeatureRowError, match="invalid NBA feature row"):
        read_nba_feature_rows_jsonl(path)

    path.write_text(f"{json.dumps(row.canonical_payload())}\n", encoding="utf-8")
    with pytest.raises(NbaFeatureRowError, match="canonical JSONL"):
        read_nba_feature_rows_jsonl(path)


def test_feature_row_reader_reconstructs_and_range_checks_values(tmp_path: Path) -> None:
    row = build_local_rich_feature_row(
        _bundle(),
        season=2027,
        elo=_elo(),
        action_at=ACTION_AT,
    )
    payload = row.canonical_payload()
    features = require_list(required_field(payload, "features"), "features")
    first_feature = require_object(features[0], "feature")
    first_feature["value"] = 1_000_000.0
    features[0] = first_feature
    payload["features"] = features
    path = tmp_path / "features.jsonl"
    path.write_text(f"{canonical_json(payload)}\n", encoding="utf-8")

    with pytest.raises(NbaFeatureRowError, match="invalid NBA feature row"):
        read_nba_feature_rows_jsonl(path)


def test_feature_row_jsonl_requires_nonempty_unique_unsuffixed_rows(tmp_path: Path) -> None:
    row = build_local_rich_feature_row(
        _bundle(),
        season=2027,
        elo=_elo(),
        action_at=ACTION_AT,
    )
    path = tmp_path / "features.jsonl"

    with pytest.raises(NbaFeatureRowError, match="must not be empty"):
        write_nba_feature_rows_jsonl(path, ())
    with pytest.raises(NbaFeatureRowError, match="duplicate question ID"):
        write_nba_feature_rows_jsonl(path, (row, row))
    with pytest.raises(NbaFeatureRowError, match="only original rows"):
        write_nba_feature_rows_jsonl(path, (row.side_swap(),))


def test_feature_row_writer_refuses_to_replace_sealed_rows(tmp_path: Path) -> None:
    row = build_local_rich_feature_row(
        _bundle(),
        season=2027,
        elo=_elo(),
        action_at=ACTION_AT,
    )
    path = tmp_path / "features.jsonl"
    write_nba_feature_rows_jsonl(path, (row,))

    with pytest.raises(NbaFeatureRowError, match="cannot be replaced"):
        write_nba_feature_rows_jsonl(path, (row,))


def test_primary_supervised_feature_row_must_be_exactly_t_minus_60() -> None:
    row = build_local_rich_feature_row(
        _bundle(),
        season=2027,
        elo=_elo(),
        action_at=ACTION_AT,
    )

    with pytest.raises(NbaFeatureRowError, match="exactly T-60"):
        replace(row, forecast_cutoff=row.forecast_cutoff - timedelta(seconds=1))


def test_latest_evidence_sets_maximum_input_availability() -> None:
    bundle = _bundle()
    earlier_elo = EVIDENCE_AT - timedelta(minutes=1)

    row = build_local_rich_feature_row(
        bundle,
        season=2027,
        elo=_elo(0.55, available_at=earlier_elo),
        action_at=ACTION_AT,
    )

    assert row.input_available_at == EVIDENCE_AT
    assert row.input_available_at <= row.forecast_cutoff


@pytest.mark.parametrize("probability", [0.0, 1.0, float("nan"), float("inf")])
def test_elo_probability_must_be_finite_and_interior(probability: float) -> None:
    with pytest.raises(NbaFeatureRowError, match="finite and interior"):
        _elo(probability)


def test_season_and_elo_timestamps_fail_closed() -> None:
    with pytest.raises(NbaFeatureRowError, match="season must be a positive integer"):
        build_local_rich_feature_row(
            _bundle(),
            season=0,
            elo=_elo(0.6),
            action_at=ACTION_AT,
        )

    with pytest.raises(NbaFeatureRowError, match="season must be a positive integer"):
        build_local_rich_feature_row(
            _bundle(),
            season=True,
            elo=_elo(0.6),
            action_at=ACTION_AT,
        )

    valid = build_local_rich_feature_row(
        _bundle(),
        season=2027,
        elo=_elo(0.6),
        action_at=ACTION_AT,
    )
    with pytest.raises(NbaFeatureRowError, match="season must be a positive integer"):
        replace(valid, season=cast(int, 2027.0))

    after_cutoff = CUTOFF + timedelta(seconds=1)
    with pytest.raises(NbaFeatureRowError, match="newer than the forecast cutoff"):
        build_local_rich_feature_row(
            _bundle(),
            season=2027,
            elo=_elo(0.6, available_at=after_cutoff),
            action_at=ACTION_AT,
        )

    with pytest.raises(NbaFeatureRowError, match="newer than the protected action"):
        build_local_rich_feature_row(
            _bundle(),
            season=2027,
            elo=_elo(0.6),
            action_at=ELO_AT - timedelta(seconds=1),
        )


def test_elo_state_digest_and_original_question_id_are_required() -> None:
    with pytest.raises(NbaFeatureRowError, match=r"elo\.state_sha256"):
        _elo(state_sha256="not-a-digest")

    bundle = _bundle()
    row = build_local_rich_feature_row(
        bundle,
        season=2027,
        elo=_elo(),
        action_at=ACTION_AT,
    )
    with pytest.raises(NbaFeatureRowError, match="must sum to one"):
        replace(
            row,
            elo_opponent_win_probability=row.elo_opponent_win_probability + 1e-12,
        )

    suffixed_id = f"{bundle.game.question_id}{SIDE_SWAP_SUFFIX}"
    suffixed = replace(
        bundle,
        game=replace(bundle.game, question_id=suffixed_id),
        question=replace(bundle.question, question_id=suffixed_id),
    )
    with pytest.raises(NbaFeatureRowError, match="cannot use the side-swap suffix"):
        build_local_rich_feature_row(
            suffixed,
            season=2027,
            elo=_elo(),
            action_at=ACTION_AT,
        )


def test_tinker_builder_uses_tinker_rights_gate() -> None:
    bundle = _bundle()
    prohibited = replace(bundle.sources[0].rights, tinker_processing="prohibited")
    source = replace(bundle.sources[0], rights=prohibited)
    restricted = replace(bundle, sources=(source,))

    assert build_local_rich_feature_row(
        restricted,
        season=2027,
        elo=_elo(0.6),
        action_at=ACTION_AT,
    )
    with pytest.raises(NbaEvidenceError, match="tinker_processing"):
        build_tinker_rich_feature_row(
            restricted,
            season=2027,
            elo=_elo(0.6),
            action_at=ACTION_AT,
        )


def test_side_swap_is_exact_and_preserves_causality() -> None:
    original = build_local_rich_feature_row(
        _bundle(),
        season=2027,
        elo=_elo(0.1),
        action_at=ACTION_AT,
    )
    zeroed = replace(original.rich_features, rest_days_difference=0.0)
    original = replace(original, rich_features=zeroed)
    swapped = original.side_swap()

    assert swapped.question_id == f"{original.question_id}{SIDE_SWAP_SUFFIX}"
    assert swapped.elo_team_win_probability == 1.0 - original.elo_team_win_probability
    assert swapped.rich_features.vector == tuple(
        0.0 if value == 0.0 else -value for value in original.rich_features.vector
    )
    assert copysign(1.0, swapped.rich_features.rest_days_difference) == 1.0
    assert swapped.season == original.season
    assert swapped.forecast_cutoff == original.forecast_cutoff
    assert swapped.scheduled_tipoff == original.scheduled_tipoff
    assert swapped.elo_available_at == original.elo_available_at
    assert swapped.elo_state_sha256 == original.elo_state_sha256
    assert swapped.input_available_at == original.input_available_at
    assert swapped.evidence_bundle_sha256 == original.evidence_bundle_sha256
    assert swapped.row_sha256 != original.row_sha256
    assert swapped.side_swap() == original
