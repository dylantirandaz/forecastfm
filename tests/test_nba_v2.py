"""Tests for leakage-safe historical NBA v2 features."""

import csv
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from pathlib import Path

import pytest

from forecastfm.nba_data import elo_venue_probability
from forecastfm.nba_v2 import (
    NBA_V2_DATA_LIMITATIONS,
    NBA_V2_FEATURE_NAMES,
    NbaV2DataError,
    NbaV2Example,
    NbaV2Features,
    load_nba_v2_examples,
    side_swap_nba_v2_example,
)
from forecastfm.outcome import OPPONENT_OUTCOME, TEAM_OUTCOME
from forecastfm.prompting import render_case

_COLUMNS: tuple[str, ...] = (
    "game_id",
    "lg_id",
    "_iscopy",
    "year_id",
    "date_game",
    "team_id",
    "pts",
    "elo_i",
    "opp_id",
    "opp_pts",
    "opp_elo_i",
    "game_location",
    "game_result",
    "forecast",
)


@dataclass(frozen=True, slots=True)
class _GameSpec:
    game_id: str
    season: int
    game_date: date
    team: str
    opponent: str
    team_points: int
    opponent_points: int
    team_elo: float = 1_500.0
    opponent_elo: float = 1_500.0
    location: str = "H"


def _selected_copy(game_id: str) -> str:
    return str(sha256(game_id.encode()).digest()[0] % 2)


def _game_id(prefix: str, selected_copy: str = "0") -> str:
    index = 0
    while _selected_copy(f"{prefix}-{index}") != selected_copy:
        index += 1
    return f"{prefix}-{index}"


def _forecast(team_elo: float, opponent_elo: float, location: str) -> float:
    neutral = 1.0 / (1.0 + 10.0 ** ((opponent_elo - team_elo) / 400.0))
    location_name = {"A": "away", "H": "home", "N": "neutral"}[location]
    return elo_venue_probability(neutral, location_name)


def _row(spec: _GameSpec, *, copy: str, swapped: bool) -> dict[str, str]:
    location = {"A": "H", "H": "A", "N": "N"}[spec.location] if swapped else spec.location
    team = spec.opponent if swapped else spec.team
    opponent = spec.team if swapped else spec.opponent
    team_points = spec.opponent_points if swapped else spec.team_points
    opponent_points = spec.team_points if swapped else spec.opponent_points
    team_elo = spec.opponent_elo if swapped else spec.team_elo
    opponent_elo = spec.team_elo if swapped else spec.opponent_elo
    return {
        "game_id": spec.game_id,
        "lg_id": "NBA",
        "_iscopy": copy,
        "year_id": str(spec.season),
        "date_game": spec.game_date.strftime("%m/%d/%Y"),
        "team_id": team,
        "pts": str(team_points),
        "elo_i": str(team_elo),
        "opp_id": opponent,
        "opp_pts": str(opponent_points),
        "opp_elo_i": str(opponent_elo),
        "game_location": location,
        "game_result": "W" if team_points > opponent_points else "L",
        "forecast": str(_forecast(team_elo, opponent_elo, location)),
    }


def _paired_rows(spec: _GameSpec) -> tuple[dict[str, str], dict[str, str]]:
    return _row(spec, copy="0", swapped=False), _row(spec, copy="1", swapped=True)


def _write_games(path: Path, games: tuple[tuple[dict[str, str], dict[str, str]], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_COLUMNS)
        writer.writeheader()
        for pair in games:
            writer.writerows(pair)


def _example_for_date(examples: tuple[NbaV2Example, ...], game_date: date) -> NbaV2Example:
    return next(
        example
        for example in examples
        if example.training_example.case.question.forecast_at.date() == game_date
    )


def test_current_and_future_results_cannot_change_current_features(tmp_path: Path) -> None:
    first_id = _game_id("first")
    current_id = _game_id("current")
    future_id = _game_id("future")
    first = _paired_rows(
        _GameSpec(
            game_id=first_id,
            season=2020,
            game_date=date(2019, 10, 1),
            team="INTERNAL_ALPHA",
            opponent="INTERNAL_BETA",
            team_points=110,
            opponent_points=100,
        )
    )
    current_win = _paired_rows(
        _GameSpec(
            game_id=current_id,
            season=2020,
            game_date=date(2019, 10, 3),
            team="INTERNAL_ALPHA",
            opponent="INTERNAL_GAMMA",
            team_points=120,
            opponent_points=90,
        )
    )
    current_loss = _paired_rows(
        _GameSpec(
            game_id=current_id,
            season=2020,
            game_date=date(2019, 10, 3),
            team="INTERNAL_ALPHA",
            opponent="INTERNAL_GAMMA",
            team_points=80,
            opponent_points=130,
        )
    )
    future_win = _paired_rows(
        _GameSpec(
            game_id=future_id,
            season=2020,
            game_date=date(2019, 10, 5),
            team="INTERNAL_ALPHA",
            opponent="INTERNAL_DELTA",
            team_points=140,
            opponent_points=70,
        )
    )
    future_loss = _paired_rows(
        _GameSpec(
            game_id=future_id,
            season=2020,
            game_date=date(2019, 10, 5),
            team="INTERNAL_ALPHA",
            opponent="INTERNAL_DELTA",
            team_points=70,
            opponent_points=140,
        )
    )
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    _write_games(first_path, (first, current_win, future_win))
    _write_games(second_path, (first, current_loss, future_loss))

    current_a = _example_for_date(load_nba_v2_examples(first_path), date(2019, 10, 3))
    current_b = _example_for_date(load_nba_v2_examples(second_path), date(2019, 10, 3))

    assert current_a.features == current_b.features
    assert current_a.training_example.case == current_b.training_example.case
    assert current_a.training_example.realized_outcome == TEAM_OUTCOME
    assert current_b.training_example.realized_outcome == OPPONENT_OUTCOME
    prompt = render_case(current_a.training_example.case)
    for hidden_value in ("INTERNAL_ALPHA", "INTERNAL_GAMMA", "2019-10-03"):
        assert hidden_value not in prompt


def test_same_date_games_share_the_pre_date_snapshot(tmp_path: Path) -> None:
    first = _paired_rows(
        _GameSpec(
            game_id=_game_id("history"),
            season=2020,
            game_date=date(2019, 10, 1),
            team="ALPHA",
            opponent="BETA",
            team_points=110,
            opponent_points=100,
        )
    )
    same_day_first = _paired_rows(
        _GameSpec(
            game_id=_game_id("same-day-first"),
            season=2020,
            game_date=date(2019, 10, 3),
            team="ALPHA",
            opponent="GAMMA",
            team_points=140,
            opponent_points=80,
        )
    )
    same_day_second = _paired_rows(
        _GameSpec(
            game_id=_game_id("same-day-second"),
            season=2020,
            game_date=date(2019, 10, 3),
            team="ALPHA",
            opponent="DELTA",
            team_points=70,
            opponent_points=130,
        )
    )
    next_day = _paired_rows(
        _GameSpec(
            game_id=_game_id("next-day"),
            season=2020,
            game_date=date(2019, 10, 4),
            team="ALPHA",
            opponent="EPSILON",
            team_points=100,
            opponent_points=90,
        )
    )
    path = tmp_path / "games.csv"
    _write_games(path, (first, same_day_first, same_day_second, next_day))

    examples = load_nba_v2_examples(path)
    same_day = tuple(
        example
        for example in examples
        if example.training_example.case.question.forecast_at.date() == date(2019, 10, 3)
    )
    following = _example_for_date(examples, date(2019, 10, 4))

    assert len(same_day) == 2
    assert same_day[0].features.vector == same_day[1].features.vector
    assert same_day[0].features.trailing_10_history_difference == 1.0
    assert same_day[1].features.trailing_10_history_difference == 1.0
    assert following.features.trailing_10_history_difference == 3.0


def test_history_resets_at_each_season(tmp_path: Path) -> None:
    old_season = _paired_rows(
        _GameSpec(
            game_id=_game_id("old-season"),
            season=2020,
            game_date=date(2020, 6, 1),
            team="ALPHA",
            opponent="BETA",
            team_points=100,
            opponent_points=90,
        )
    )
    new_season = _paired_rows(
        _GameSpec(
            game_id=_game_id("new-season"),
            season=2021,
            game_date=date(2020, 12, 1),
            team="ALPHA",
            opponent="GAMMA",
            team_points=100,
            opponent_points=90,
        )
    )
    path = tmp_path / "games.csv"
    _write_games(path, (old_season, new_season))

    features = load_nba_v2_examples(path)[1].features

    assert features.rest_days_difference == 0.0
    assert features.games_last_7_difference == 0.0
    assert features.trailing_10_history_difference == 0.0


def test_pregame_history_features_use_only_prior_dates(tmp_path: Path) -> None:
    team_history = _paired_rows(
        _GameSpec(
            game_id=_game_id("team-history"),
            season=2020,
            game_date=date(2019, 10, 1),
            team="ALPHA",
            opponent="BETA",
            team_points=110,
            opponent_points=90,
            opponent_elo=1_400.0,
            location="A",
        )
    )
    opponent_history = _paired_rows(
        _GameSpec(
            game_id=_game_id("opponent-history"),
            season=2020,
            game_date=date(2019, 10, 2),
            team="GAMMA",
            opponent="DELTA",
            team_points=90,
            opponent_points=100,
            opponent_elo=1_600.0,
            location="H",
        )
    )
    target = _paired_rows(
        _GameSpec(
            game_id=_game_id("feature-target"),
            season=2020,
            game_date=date(2019, 10, 3),
            team="ALPHA",
            opponent="GAMMA",
            team_points=100,
            opponent_points=90,
        )
    )
    path = tmp_path / "games.csv"
    _write_games(path, (team_history, opponent_history, target))

    features = load_nba_v2_examples(path)[2].features

    assert features.rest_days_difference == 1.0
    assert features.back_to_back_difference == -1.0
    assert features.games_last_7_difference == 0.0
    assert features.road_games_last_7_difference == 1.0
    assert features.trailing_10_win_rate_difference == 1.0
    assert features.trailing_10_margin_difference == 30.0
    assert features.trailing_10_opponent_elo_difference == -200.0
    assert features.trailing_10_history_difference == 0.0


def test_features_and_examples_have_exact_side_swap_involution(tmp_path: Path) -> None:
    first = _paired_rows(
        _GameSpec(
            game_id=_game_id("swap-history"),
            season=2020,
            game_date=date(2019, 10, 1),
            team="ALPHA",
            opponent="BETA",
            team_points=115,
            opponent_points=95,
            location="A",
        )
    )
    second = _paired_rows(
        _GameSpec(
            game_id=_game_id("swap-target"),
            season=2020,
            game_date=date(2019, 10, 3),
            team="ALPHA",
            opponent="GAMMA",
            team_points=105,
            opponent_points=100,
            team_elo=1_620.0,
            opponent_elo=1_480.0,
            location="H",
        )
    )
    path = tmp_path / "games.csv"
    _write_games(path, (first, second))
    original = load_nba_v2_examples(path)[1]

    swapped = side_swap_nba_v2_example(original)

    assert tuple(original.features.as_dict()) == NBA_V2_FEATURE_NAMES
    assert swapped.features.vector == tuple(-value for value in original.features.vector)
    assert swapped.features.venue_adjusted_elo_probabilities == tuple(
        reversed(original.features.venue_adjusted_elo_probabilities)
    )
    assert swapped.training_example.realized_outcome == OPPONENT_OUTCOME
    assert swapped.training_example.case.prior.probabilities == tuple(
        reversed(original.training_example.case.prior.probabilities)
    )
    assert side_swap_nba_v2_example(swapped) == original
    assert original.features.venue_adjusted_elo_probabilities[0] == pytest.approx(
        _forecast(1_620.0, 1_480.0, "H")
    )
    limitations = " ".join(NBA_V2_DATA_LIMITATIONS).lower()
    for missing_input in (
        "tipoff timestamps",
        "arena",
        "injuries",
        "lineups",
        "rosters",
        "player-level",
    ):
        assert missing_input in limitations


def test_feature_log_odds_must_match_the_elo_probabilities() -> None:
    with pytest.raises(NbaV2DataError, match="log-odds"):
        NbaV2Features(
            venue_adjusted_elo_probabilities=(0.6, 0.4),
            venue_adjusted_elo_log_odds=0.0,
            rest_days_difference=0.0,
            back_to_back_difference=0.0,
            games_last_7_difference=0.0,
            road_games_last_7_difference=0.0,
            trailing_10_win_rate_difference=0.0,
            trailing_10_margin_difference=0.0,
            trailing_10_opponent_elo_difference=0.0,
            trailing_10_history_difference=0.0,
        )
