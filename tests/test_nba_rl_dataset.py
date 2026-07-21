"""Tests for the sealed rl-prompt-v1 NBA RL question set."""

import json
from datetime import date
from pathlib import Path

import pytest

from forecastfm.integrity import canonical_json
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_prototype_dataset import PrototypeGameRow
from forecastfm.nba_rl_dataset import (
    ANSWERS_FILENAME,
    MANIFEST_FILENAME,
    PROMPTS_FILENAME,
    RL_SYSTEM_PROMPT,
    NbaRlDatasetError,
    answer_label,
    answer_record,
    build_prompt,
    prompt_record,
    prompt_template_sha256,
    rl_question_id,
    seal_rl_dataset,
    swap_row,
    verify_sealed_dataset,
)

FEATURES = (1.0, 0.0, 3.0, -1.5, 812.25, 2.0, 0.75, 0.4, -3.21, 1.05, 1522.5)

EXPECTED_USER = "\n".join(
    [
        "elo_home_probability: 0.6234",
        "rest_days: +1.000",
        "back_to_back: +0.000",
        "games_last_7: +3.000",
        "road_games_last_7: -1.500",
        "travel_miles: +812.250",
        "travel_time_zones: +2.000",
        "roster_continuity: +0.750",
        "expected_lineup_continuity: +0.400",
        "rolling_team_net_rating: -3.210",
        "rolling_player_value: +1.050",
        "schedule_strength: +1522.500",
        "winner_label:",
    ]
)

EXPECTED_SWAPPED_USER = "\n".join(
    [
        "elo_home_probability: 0.3766",
        "rest_days: -1.000",
        "back_to_back: -0.000",
        "games_last_7: -3.000",
        "road_games_last_7: +1.500",
        "travel_miles: -812.250",
        "travel_time_zones: -2.000",
        "roster_continuity: -0.750",
        "expected_lineup_continuity: -0.400",
        "rolling_team_net_rating: +3.210",
        "rolling_player_value: -1.050",
        "schedule_strength: -1522.500",
        "winner_label:",
    ]
)


def _row(
    game_id: int = 999,
    *,
    season: int = 2025,
    day: date = date(2025, 1, 15),
    home_won: bool = True,
) -> PrototypeGameRow:
    return PrototypeGameRow(
        question_id=f"nba-{game_id}",
        game_id=game_id,
        season=season,
        game_date=day,
        elo_home_probability=0.6234,
        features_standard=FEATURES,
        features_health=None,
        home_won=home_won,
    )


def test_system_prompt_is_frozen() -> None:
    assert RL_SYSTEM_PROMPT == (
        "You are a calibrated NBA forecasting model. Given an Elo prior and pregame "
        "evidence differences (home minus away), estimate the probability that the listed "
        "team wins. Answer with exactly one label: TEAM if the listed team wins, OTHER if "
        "the opponent wins."
    )


def test_build_prompt_byte_exact() -> None:
    system, user = build_prompt(_row(), swapped=False)
    assert system == RL_SYSTEM_PROMPT
    assert user == EXPECTED_USER


def test_side_swap_is_exact_complement() -> None:
    row = _row()
    assert build_prompt(row, swapped=True) == (RL_SYSTEM_PROMPT, EXPECTED_SWAPPED_USER)
    assert swap_row(swap_row(row)) == row
    assert build_prompt(swap_row(row), swapped=False) == build_prompt(row, swapped=True)
    assert rl_question_id(row, swapped=False) == "nba-999-T-60"
    swapped_id = rl_question_id(row, swapped=True)
    assert swapped_id == f"nba-999-T-60{SIDE_SWAP_SUFFIX}"
    assert swapped_id.removesuffix(SIDE_SWAP_SUFFIX) == rl_question_id(row, swapped=False)


def test_answer_labels_track_the_listed_team() -> None:
    assert answer_label(_row(home_won=True), swapped=False) == "TEAM"
    assert answer_label(_row(home_won=True), swapped=True) == "OTHER"
    assert answer_label(_row(home_won=False), swapped=False) == "OTHER"
    assert answer_label(_row(home_won=False), swapped=True) == "TEAM"
    assert answer_label(swap_row(_row()), swapped=False) == answer_label(_row(), swapped=True)


def test_answers_carry_no_prompt_text_and_prompts_carry_no_outcome() -> None:
    row = _row()
    for swapped in (False, True):
        answer = answer_record(row, swapped=swapped)
        assert set(answer) == {"question_id", "winner"}
        answer_text = canonical_json(answer)
        assert RL_SYSTEM_PROMPT not in answer_text
        assert "elo_home_probability" not in answer_text
        record = prompt_record(row, swapped=swapped)
        user = str(record["user"])
        assert user.splitlines()[-1] == "winner_label:"
        for banned in ("TEAM", "OTHER", "won", "score", "home_won"):
            assert banned not in user


def test_seal_reload_reproduces_identical_hashes(tmp_path: Path) -> None:
    rows = [
        _row(2, season=2024, day=date(2024, 3, 1), home_won=False),
        _row(1, season=2025, day=date(2024, 3, 2)),
    ]
    manifest = seal_rl_dataset(rows, tmp_path)
    assert verify_sealed_dataset(tmp_path) == manifest
    assert manifest["prompt_template_sha256"] == prompt_template_sha256()
    assert manifest["total_games"] == 2
    assert manifest["total_prompts"] == 4
    assert manifest["seasons"] == {
        "2024": {"games": 1, "prompts": 2},
        "2025": {"games": 1, "prompts": 2},
    }
    assert manifest["decision_2a"] == "A"
    assert "no health-derived values" in str(manifest["health_disclosure"])

    prompt_lines = (tmp_path / PROMPTS_FILENAME).read_text(encoding="utf-8").splitlines()
    answer_lines = (tmp_path / ANSWERS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(prompt_lines) == len(answer_lines) == 4
    for line in (*prompt_lines, *answer_lines):
        assert canonical_json(json.loads(line)) == line
    prompts = [json.loads(line) for line in prompt_lines]
    assert [record["question_id"] for record in prompts] == [
        "nba-2-T-60",
        f"nba-2-T-60{SIDE_SWAP_SUFFIX}",
        "nba-1-T-60",
        f"nba-1-T-60{SIDE_SWAP_SUFFIX}",
    ]
    answers = [json.loads(line) for line in answer_lines]
    assert [record["winner"] for record in answers] == ["OTHER", "TEAM", "TEAM", "OTHER"]

    with pytest.raises(NbaRlDatasetError, match="refusing to replace sealed artifact"):
        seal_rl_dataset(rows, tmp_path)
    (tmp_path / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    with pytest.raises(NbaRlDatasetError):
        verify_sealed_dataset(tmp_path)
