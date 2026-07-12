"""Tests for the pinned FiveThirtyEight NBA data boundary."""

import csv
from hashlib import sha256
from pathlib import Path

import pytest

from forecastfm.nba_data import (
    NbaDataError,
    audit_nba_splits,
    download_nba_elo,
    elo_venue_probability,
    load_nba_splits,
)
from forecastfm.prompting import render_case

_COLUMNS = (
    "gameorder",
    "game_id",
    "lg_id",
    "_iscopy",
    "year_id",
    "date_game",
    "seasongame",
    "is_playoffs",
    "team_id",
    "fran_id",
    "pts",
    "elo_i",
    "elo_n",
    "win_equiv",
    "opp_id",
    "opp_fran",
    "opp_pts",
    "opp_elo_i",
    "opp_elo_n",
    "game_location",
    "game_result",
    "forecast",
    "notes",
)

_ROWS = (
    (
        "1",
        "200906140ORL",
        "NBA",
        "0",
        "2009",
        "6/14/2009",
        "82",
        "1",
        "ORL",
        "Magic",
        "86",
        "1650",
        "1632",
        "55",
        "LAL",
        "Lakers",
        "99",
        "1700",
        "1718",
        "H",
        "L",
        "0.571463",
        "postgame marker",
    ),
    (
        "1",
        "200906140ORL",
        "NBA",
        "1",
        "2009",
        "6/14/2009",
        "82",
        "1",
        "LAL",
        "Lakers",
        "99",
        "1700",
        "1718",
        "58",
        "ORL",
        "Magic",
        "86",
        "1650",
        "1632",
        "A",
        "W",
        "0.428537",
        "postgame marker",
    ),
    (
        "2",
        "200910270CLE",
        "NBA",
        "0",
        "2010",
        "10/27/2009",
        "1",
        "0",
        "CLE",
        "Cavaliers",
        "89",
        "1680",
        "1672",
        "52",
        "BOS",
        "Celtics",
        "85",
        "1660",
        "1668",
        "H",
        "W",
        "0.666139",
        "",
    ),
    (
        "3",
        "201210300MIA",
        "NBA",
        "0",
        "2013",
        "10/30/2012",
        "1",
        "0",
        "MIA",
        "Heat",
        "120",
        "1720",
        "1735",
        "60",
        "BOS",
        "Celtics",
        "107",
        "1685",
        "1670",
        "H",
        "W",
        "0.685060",
        "",
    ),
)


def _write_csv(tmp_path: Path) -> Path:
    path = tmp_path / "nbaallelo.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        csv.writer(file).writerows((_COLUMNS, *_ROWS))
    return path


def _question_id(game_id: str) -> str:
    digest = sha256(f"forecastfm:nba:{game_id}".encode()).hexdigest()
    return f"nba-{digest[:16]}"


def test_load_deduplicates_games_and_splits_chronologically(tmp_path: Path) -> None:
    splits = load_nba_splits(_write_csv(tmp_path))

    assert tuple(example.case.question.question_id for example in splits.train) == (
        _question_id("200906140ORL"),
    )
    assert tuple(example.case.question.question_id for example in splits.validation) == (
        _question_id("200910270CLE"),
    )
    assert tuple(example.case.question.question_id for example in splits.test) == (
        _question_id("201210300MIA"),
    )
    audit_nba_splits(splits)


def test_load_uses_pregame_forecast_and_realized_outcome(tmp_path: Path) -> None:
    example = load_nba_splits(_write_csv(tmp_path)).train[0]

    assert example.target.distribution.outcomes == ("team_wins", "opponent_wins")
    assert example.target.distribution.probabilities == pytest.approx((0.428537, 0.571463))
    assert example.case.prior.probabilities == pytest.approx((0.571463, 0.428537))
    assert example.realized_outcome == "team_wins"
    assert example.target_information_cutoff == example.case.question.forecast_at


def test_prompt_excludes_postgame_fields(tmp_path: Path) -> None:
    example = load_nba_splits(_write_csv(tmp_path)).train[0]

    prompt = render_case(example.case)

    for field in ("pts", "elo_n", "forecast", "game_result", "realized_outcome", "notes"):
        assert f'"{field}"' not in prompt
    assert "postgame marker" not in prompt
    for identifying_value in (
        "200906140ORL",
        "2009-06-14",
        "LAL",
        "Lakers",
        "Magic",
        "ORL",
        example.case.question.question_id,
        example.case.question.resolution_source,
    ):
        assert identifying_value not in prompt


def test_load_rejects_missing_source_columns(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text("game_id\nexample\n", encoding="utf-8")

    with pytest.raises(NbaDataError, match="missing columns"):
        load_nba_splits(path)


def test_download_rejects_an_existing_file_with_the_wrong_hash(tmp_path: Path) -> None:
    path = tmp_path / "nbaallelo.csv"
    path.write_text("not the pinned source\n", encoding="utf-8")

    with pytest.raises(NbaDataError, match="unexpected NBA source SHA-256"):
        download_nba_elo(path)


def test_elo_oracle_is_side_swap_equivariant() -> None:
    home_probability = elo_venue_probability(0.4, "home")
    swapped_away_probability = elo_venue_probability(0.6, "away")

    assert home_probability == pytest.approx(1.0 - swapped_away_probability)
