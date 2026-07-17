"""Tests for sealed, deterministic per-game NBA Elo states."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

from forecastfm.integrity import canonical_json, canonical_sha256
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_elo_state import (
    NbaEloState,
    NbaEloStateError,
    base10_elo_team_probability,
    read_nba_elo_states_jsonl,
    validate_elo_states_against_feature_rows,
    write_nba_elo_states_jsonl,
)
from forecastfm.nba_feature_rows import NbaRichFeatureRow
from forecastfm.nba_rich import NbaRichFeatures

AVAILABLE_AT = datetime(2026, 10, 21, 20, 30, tzinfo=UTC)
CUTOFF = datetime(2026, 10, 21, 22, tzinfo=UTC)
TIPOFF = CUTOFF + timedelta(minutes=60)


def _state(
    question_id: str = "nba-elo-1",
    *,
    team_rating: float = 1_600.0,
) -> NbaEloState:
    return NbaEloState(
        question_id=question_id,
        available_at=AVAILABLE_AT,
        team_rating=team_rating,
        opponent_rating=1_500.0,
        home_advantage=100.0,
        rating_scale=400.0,
        recipe_sha256="a" * 64,
    )


def _feature_row(state: NbaEloState) -> NbaRichFeatureRow:
    probability = state.team_win_probability
    return NbaRichFeatureRow(
        question_id=state.question_id,
        season=2027,
        forecast_cutoff=CUTOFF,
        scheduled_tipoff=TIPOFF,
        elo_team_win_probability=probability,
        elo_opponent_win_probability=1.0 - probability,
        elo_available_at=state.available_at,
        elo_state_sha256=state.state_sha256,
        rich_features=NbaRichFeatures.from_vector((0.0,) * 11),
        evidence_bundle_sha256="b" * 64,
        input_available_at=state.available_at,
    )


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.write_text(f"{canonical_json(payload)}\n", encoding="utf-8")


def test_probability_and_state_digest_are_deterministic_and_input_sensitive() -> None:
    state = _state()
    expected = 1.0 / (1.0 + 10.0**-0.5)

    assert state.team_win_probability == expected
    assert state.team_win_probability == base10_elo_team_probability(
        1_600.0,
        1_500.0,
        100.0,
        400.0,
    )
    assert 0.0 < state.team_win_probability < 1.0
    assert state.state_sha256 == canonical_sha256(state.state_input_payload())
    assert replace(state, team_rating=1_601.0).state_sha256 != state.state_sha256
    assert replace(state, recipe_sha256="c" * 64).state_sha256 != state.state_sha256


def test_jsonl_round_trip_is_canonical_ordered_and_create_only(tmp_path: Path) -> None:
    first = _state()
    second = _state("nba-elo-2", team_rating=1_550.0)
    path = tmp_path / "elo-states.jsonl"

    write_nba_elo_states_jsonl(path, (second, first))

    assert read_nba_elo_states_jsonl(path) == (second, first)
    assert path.read_text(encoding="utf-8") == "".join(
        f"{canonical_json(state.canonical_payload())}\n" for state in (second, first)
    )
    with pytest.raises(NbaEloStateError, match="already exists"):
        write_nba_elo_states_jsonl(path, (first,))


def test_jsonl_requires_nonempty_unique_original_ids(tmp_path: Path) -> None:
    state = _state()
    with pytest.raises(NbaEloStateError, match="must not be empty"):
        write_nba_elo_states_jsonl(tmp_path / "empty.jsonl", ())
    with pytest.raises(NbaEloStateError, match="duplicate question ID"):
        write_nba_elo_states_jsonl(tmp_path / "duplicate.jsonl", (state, state))
    with pytest.raises(NbaEloStateError, match="only original question IDs"):
        _state(f"nba-elo-1{SIDE_SWAP_SUFFIX}")

    empty_path = tmp_path / "read-empty.jsonl"
    empty_path.write_text("", encoding="utf-8")
    with pytest.raises(NbaEloStateError, match="must not be empty"):
        read_nba_elo_states_jsonl(empty_path)


def test_constructor_enforces_exact_types_utc_and_numeric_ranges() -> None:
    with pytest.raises(NbaEloStateError, match="finite float"):
        replace(_state(), team_rating=cast(float, 1_600))
    with pytest.raises(NbaEloStateError, match="team_rating is outside"):
        replace(_state(), team_rating=-1.0)
    with pytest.raises(NbaEloStateError, match="opponent_rating is outside"):
        replace(_state(), opponent_rating=4_001.0)
    with pytest.raises(NbaEloStateError, match="home_advantage is outside"):
        replace(_state(), home_advantage=501.0)
    with pytest.raises(NbaEloStateError, match="rating_scale is outside"):
        replace(_state(), rating_scale=0.0)
    with pytest.raises(NbaEloStateError, match="too extreme"):
        replace(
            _state(),
            team_rating=4_000.0,
            opponent_rating=0.0,
            home_advantage=0.0,
            rating_scale=1.0,
        )

    central = timezone(timedelta(hours=-5))
    with pytest.raises(NbaEloStateError, match="available_at must be in UTC"):
        replace(_state(), available_at=AVAILABLE_AT.astimezone(central))


def test_reader_recomputes_probability_digest_and_canonical_types(tmp_path: Path) -> None:
    state = _state()
    path = tmp_path / "elo-states.jsonl"

    integer_rating = state.canonical_payload()
    integer_rating["team_rating"] = 1_600
    _write_payload(path, integer_rating)
    with pytest.raises(NbaEloStateError, match="invalid NBA Elo state"):
        read_nba_elo_states_jsonl(path)

    changed_probability = state.canonical_payload()
    changed_probability["team_win_probability"] = state.team_win_probability + 0.01
    _write_payload(path, changed_probability)
    with pytest.raises(NbaEloStateError, match="invalid NBA Elo state"):
        read_nba_elo_states_jsonl(path)

    changed_digest = state.canonical_payload()
    changed_digest["state_sha256"] = "d" * 64
    _write_payload(path, changed_digest)
    with pytest.raises(NbaEloStateError, match="invalid NBA Elo state"):
        read_nba_elo_states_jsonl(path)

    target_field = state.canonical_payload()
    target_field["winner"] = "TEAM"
    _write_payload(path, target_field)
    with pytest.raises(NbaEloStateError, match="invalid NBA Elo state"):
        read_nba_elo_states_jsonl(path)

    alternate_utc = state.canonical_payload()
    alternate_utc["available_at"] = "2026-10-21T20:30:00+00:00"
    _write_payload(path, alternate_utc)
    with pytest.raises(NbaEloStateError, match="invalid NBA Elo state"):
        read_nba_elo_states_jsonl(path)

    path.write_text(f"{json.dumps(state.canonical_payload())}\n", encoding="utf-8")
    with pytest.raises(NbaEloStateError, match="canonical JSONL"):
        read_nba_elo_states_jsonl(path)


def test_validator_requires_exact_feature_row_order_and_elo_fields() -> None:
    first = _state()
    second = _state("nba-elo-2", team_rating=1_550.0)
    first_row = _feature_row(first)
    second_row = _feature_row(second)

    validate_elo_states_against_feature_rows(
        (first, second),
        (first_row, second_row),
        action_at=CUTOFF,
    )

    with pytest.raises(NbaEloStateError, match="identical IDs and order"):
        validate_elo_states_against_feature_rows(
            (first, second),
            (second_row, first_row),
            action_at=CUTOFF,
        )
    with pytest.raises(NbaEloStateError, match="availability differs"):
        validate_elo_states_against_feature_rows(
            (first,),
            (replace(first_row, elo_available_at=AVAILABLE_AT - timedelta(minutes=1)),),
            action_at=CUTOFF,
        )
    with pytest.raises(NbaEloStateError, match="digest differs"):
        validate_elo_states_against_feature_rows(
            (first,),
            (replace(first_row, elo_state_sha256="e" * 64),),
            action_at=CUTOFF,
        )

    changed_probability = first.team_win_probability + 0.01
    with pytest.raises(NbaEloStateError, match="team probability differs"):
        validate_elo_states_against_feature_rows(
            (first,),
            (
                replace(
                    first_row,
                    elo_team_win_probability=changed_probability,
                    elo_opponent_win_probability=1.0 - changed_probability,
                ),
            ),
            action_at=CUTOFF,
        )

    with pytest.raises(NbaEloStateError, match="postdate the protected action"):
        validate_elo_states_against_feature_rows(
            (first,),
            (first_row,),
            action_at=AVAILABLE_AT - timedelta(seconds=1),
        )
