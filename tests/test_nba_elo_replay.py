"""Tests for causal, deterministic NBA Elo history replay."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

from forecastfm.integrity import canonical_json, canonical_sha256
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_elo_replay import (
    NbaEloRecipe,
    NbaEloReplayError,
    NbaEloReplayRow,
    NbaGameSite,
    read_nba_elo_replay_rows_jsonl,
    read_nba_elo_replay_rows_jsonl_bytes,
    replay_nba_elo_states,
    validate_nba_elo_replay_states,
    write_nba_elo_replay_rows_jsonl,
)
from forecastfm.nba_resolutions import NbaResolution

BASE_CUTOFF = datetime(2025, 10, 1, 23, tzinfo=UTC)
RECIPE = NbaEloRecipe(
    initial_rating=1_500.0,
    k_factor=20.0,
    rating_scale=400.0,
    home_advantage=100.0,
)


def _row(
    question_id: str,
    day: int,
    team_id: str,
    opponent_id: str,
    *,
    site: NbaGameSite = "neutral",
) -> NbaEloReplayRow:
    cutoff = BASE_CUTOFF + timedelta(days=day)
    return NbaEloReplayRow(
        question_id=question_id,
        source_game_id=f"source-{question_id}",
        season=2025,
        team_id=team_id,
        opponent_id=opponent_id,
        site=site,
        forecast_cutoff=cutoff,
        scheduled_tipoff=cutoff + timedelta(minutes=60),
    )


def _resolution(
    row: NbaEloReplayRow,
    *,
    team_won: bool = True,
    resolved_at: datetime | None = None,
) -> NbaResolution:
    return NbaResolution(
        question_id=row.question_id,
        source_game_id=row.source_game_id,
        team_id=row.team_id,
        opponent_id=row.opponent_id,
        site=row.site,
        team_score=110 if team_won else 100,
        opponent_score=100 if team_won else 110,
        resolved_at=resolved_at or row.scheduled_tipoff + timedelta(hours=3),
        source_id=f"final:{row.source_game_id}",
        snapshot_metadata_sha256="a" * 64,
    )


def test_recipe_digest_is_canonical_and_input_sensitive() -> None:
    assert RECIPE.recipe_sha256 == canonical_sha256(RECIPE.canonical_payload())
    assert replace(RECIPE, k_factor=21.0).recipe_sha256 != RECIPE.recipe_sha256
    assert RECIPE.canonical_payload() == {
        "schema_version": 1,
        "initial_rating": 1_500.0,
        "k_factor": 20.0,
        "rating_scale": 400.0,
        "home_advantage": 100.0,
    }


def test_replay_applies_only_available_results_and_resets_each_season() -> None:
    first = _row("game-1", 0, "A", "B", site="home")
    second = _row("game-2", 1, "A", "C")
    third = _row("game-3", 1, "B", "D", site="away")
    fourth = _row("game-4", 2, "A", "D", site="home")
    next_season = replace(_row("game-5", 100, "A", "B"), season=2026)
    rows = (first, second, third, fourth, next_season)
    resolutions = (
        _resolution(first),
        _resolution(second, team_won=False),
        _resolution(third),
        _resolution(fourth),
        _resolution(next_season),
    )

    states = replay_nba_elo_states(rows, resolutions, RECIPE)
    first_probability = 1.0 / (1.0 + 10.0 ** (-100.0 / 400.0))
    first_change = 20.0 * (1.0 - first_probability)

    assert states[0].team_rating == 1_500.0
    assert states[0].opponent_rating == 1_500.0
    assert states[0].home_advantage == 100.0
    assert states[1].team_rating == 1_500.0 + first_change
    assert states[1].opponent_rating == 1_500.0
    assert states[2].team_rating == 1_500.0 - first_change
    assert states[2].opponent_rating == 1_500.0
    assert states[2].home_advantage == -100.0

    second_change = 20.0 * (0.0 - states[1].team_win_probability)
    third_change = 20.0 * (1.0 - states[2].team_win_probability)
    assert states[3].team_rating == 1_500.0 + first_change + second_change
    assert states[3].opponent_rating == 1_500.0 - third_change
    assert states[4].team_rating == RECIPE.initial_rating
    assert states[4].opponent_rating == RECIPE.initial_rating
    assert all(
        state.available_at == row.forecast_cutoff for state, row in zip(states, rows, strict=True)
    )
    assert all(state.recipe_sha256 == RECIPE.recipe_sha256 for state in states)


def test_result_remains_unavailable_until_its_resolution_timestamp() -> None:
    first = _row("game-1", 0, "A", "B")
    second = _row("game-2", 1, "A", "C")
    third = _row("game-3", 2, "A", "D")
    delayed = _resolution(first, resolved_at=second.forecast_cutoff + timedelta(hours=1))
    resolutions = (
        delayed,
        _resolution(second, resolved_at=third.forecast_cutoff + timedelta(hours=1)),
        _resolution(third),
    )

    states = replay_nba_elo_states((first, second, third), resolutions, RECIPE)

    assert states[1].team_rating == RECIPE.initial_rating
    assert states[2].team_rating > RECIPE.initial_rating


def test_same_cutoff_games_are_forecast_from_one_frozen_rating_state() -> None:
    first = _row("game-1", 0, "A", "B")
    second = _row("game-2", 0, "C", "D")
    later = _row("game-3", 1, "A", "C")
    states = replay_nba_elo_states(
        (first, second, later),
        (_resolution(first), _resolution(second), _resolution(later)),
        RECIPE,
    )

    assert states[0].team_rating == states[1].team_rating == RECIPE.initial_rating
    assert states[0].opponent_rating == states[1].opponent_rating == RECIPE.initial_rating
    assert states[2].team_rating > RECIPE.initial_rating
    assert states[2].opponent_rating > RECIPE.initial_rating


def test_replay_rejects_duplicate_and_nonchronological_schedule_rows() -> None:
    first = _row("game-1", 0, "A", "B")
    same_cutoff = _row("game-2", 0, "A", "C")
    later = _row("game-3", 1, "C", "D")

    with pytest.raises(NbaEloReplayError, match="duplicate question_id"):
        replay_nba_elo_states((first, first), (_resolution(first),), RECIPE)
    with pytest.raises(NbaEloReplayError, match="participate twice"):
        replay_nba_elo_states(
            (first, same_cutoff),
            (_resolution(first), _resolution(same_cutoff)),
            RECIPE,
        )
    with pytest.raises(NbaEloReplayError, match="cutoffs must be monotone"):
        replay_nba_elo_states(
            (later, first),
            (_resolution(later), _resolution(first)),
            RECIPE,
        )
    with pytest.raises(NbaEloReplayError, match="seasons must be monotone"):
        replay_nba_elo_states(
            (replace(first, season=2026), replace(later, season=2025)),
            (_resolution(first), _resolution(later)),
            RECIPE,
        )


def test_replay_requires_one_matching_postgame_resolution_per_row() -> None:
    first = _row("game-1", 0, "A", "B")
    second = _row("game-2", 1, "A", "C")
    extra = _row("extra", 2, "D", "E")

    with pytest.raises(NbaEloReplayError, match=r"missing=\['game-2'\]"):
        replay_nba_elo_states((first, second), (_resolution(first),), RECIPE)
    with pytest.raises(NbaEloReplayError, match=r"extra=\['extra'\]"):
        replay_nba_elo_states(
            (first, second),
            (_resolution(first), _resolution(second), _resolution(extra)),
            RECIPE,
        )
    with pytest.raises(NbaEloReplayError, match="source_game_id differs"):
        replay_nba_elo_states(
            (first,),
            (replace(_resolution(first), source_game_id="wrong"),),
            RECIPE,
        )
    with pytest.raises(NbaEloReplayError, match="orientation differs"):
        replay_nba_elo_states(
            (first,),
            (replace(_resolution(first), team_id="wrong"),),
            RECIPE,
        )
    with pytest.raises(NbaEloReplayError, match="after the scheduled tipoff"):
        replay_nba_elo_states(
            (first,),
            (_resolution(first, resolved_at=first.scheduled_tipoff),),
            RECIPE,
        )


def test_season_reset_waits_for_every_prior_season_result() -> None:
    first = _row("game-1", 0, "A", "B")
    next_season = replace(_row("game-2", 100, "A", "B"), season=2026)
    late_resolution = _resolution(
        first,
        resolved_at=next_season.forecast_cutoff + timedelta(seconds=1),
    )

    with pytest.raises(NbaEloReplayError, match="reset cannot precede"):
        replay_nba_elo_states(
            (first, next_season),
            (late_resolution, _resolution(next_season)),
            RECIPE,
        )


def test_replay_row_constructor_and_recipe_are_strict() -> None:
    row = _row("game-1", 0, "A", "B")
    central = timezone(timedelta(hours=-5))

    with pytest.raises(NbaEloReplayError, match="only original question IDs"):
        replace(row, question_id=f"game-1{SIDE_SWAP_SUFFIX}")
    with pytest.raises(NbaEloReplayError, match="team_id and opponent_id must differ"):
        replace(row, opponent_id="A")
    with pytest.raises(NbaEloReplayError, match="site must be"):
        replace(row, site=cast(NbaGameSite, "arena"))
    with pytest.raises(NbaEloReplayError, match="T-60"):
        replace(row, scheduled_tipoff=row.scheduled_tipoff + timedelta(minutes=1))
    with pytest.raises(NbaEloReplayError, match="must be in UTC"):
        replace(row, forecast_cutoff=row.forecast_cutoff.astimezone(central))
    with pytest.raises(NbaEloReplayError, match="finite float"):
        replace(RECIPE, k_factor=cast(float, 20))
    with pytest.raises(NbaEloReplayError, match="negative zero"):
        replace(RECIPE, home_advantage=-0.0)
    with pytest.raises(NbaEloReplayError, match="supported range"):
        replace(RECIPE, rating_scale=0.0)


def test_replay_row_jsonl_is_canonical_and_create_only(tmp_path: Path) -> None:
    first = _row("game-1", 0, "A", "B", site="home")
    second = _row("game-2", 1, "A", "C")
    path = tmp_path / "elo-replay.jsonl"

    write_nba_elo_replay_rows_jsonl(path, (first, second))

    assert read_nba_elo_replay_rows_jsonl(path) == (first, second)
    assert read_nba_elo_replay_rows_jsonl_bytes(path.read_bytes()) == (first, second)
    assert path.read_text(encoding="utf-8") == "".join(
        f"{canonical_json(row.canonical_payload())}\n" for row in (first, second)
    )
    with pytest.raises(NbaEloReplayError, match="already exists"):
        write_nba_elo_replay_rows_jsonl(path, (first,))


def test_replay_row_reader_rejects_noncanonical_or_changed_records(tmp_path: Path) -> None:
    row = _row("game-1", 0, "A", "B")
    path = tmp_path / "elo-replay.jsonl"
    payload = row.canonical_payload()
    payload["forecast_cutoff"] = row.forecast_cutoff.isoformat()
    path.write_text(f"{canonical_json(payload)}\n", encoding="utf-8")
    with pytest.raises(NbaEloReplayError, match="invalid NBA Elo replay row"):
        read_nba_elo_replay_rows_jsonl(path)

    with pytest.raises(NbaEloReplayError, match="UTF-8"):
        read_nba_elo_replay_rows_jsonl_bytes(b"\xff")

    path.write_text(f"{json.dumps(row.canonical_payload())}\n", encoding="utf-8")
    with pytest.raises(NbaEloReplayError, match="canonical JSONL"):
        read_nba_elo_replay_rows_jsonl(path)

    payload = row.canonical_payload()
    payload["winner"] = "TEAM"
    path.write_text(f"{canonical_json(payload)}\n", encoding="utf-8")
    with pytest.raises(NbaEloReplayError, match="invalid NBA Elo replay row"):
        read_nba_elo_replay_rows_jsonl(path)


def test_validator_requires_byte_and_value_exact_replay_states() -> None:
    first = _row("game-1", 0, "A", "B")
    second = _row("game-2", 1, "A", "C")
    rows = (first, second)
    resolutions = (_resolution(first), _resolution(second))
    states = replay_nba_elo_states(rows, resolutions, RECIPE)

    validate_nba_elo_replay_states(rows, resolutions, RECIPE, states)

    with pytest.raises(NbaEloReplayError, match="count differs"):
        validate_nba_elo_replay_states(rows, resolutions, RECIPE, states[:1])
    with pytest.raises(NbaEloReplayError, match="index 0"):
        validate_nba_elo_replay_states(
            rows,
            resolutions,
            RECIPE,
            (replace(states[0], home_advantage=-0.0), states[1]),
        )
    with pytest.raises(NbaEloReplayError, match="index 0"):
        validate_nba_elo_replay_states(rows, resolutions, RECIPE, tuple(reversed(states)))
