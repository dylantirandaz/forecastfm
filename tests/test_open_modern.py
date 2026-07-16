"""Tests for one-way preparation of the open-modern NBA source."""

import csv
import json
from datetime import date
from pathlib import Path

import pytest

from forecastfm.integrity import file_sha256
from forecastfm.open_modern import (
    DEVELOPMENT_COLUMNS,
    SOURCE_COLUMNS,
    TEST_INPUT_COLUMNS,
    OpenModernError,
    OpenModernGame,
    load_open_modern_games,
    write_open_modern_artifacts,
)


def _source_rows() -> list[dict[str, str]]:
    return [
        {
            "season": str(season),
            "date": f"{season - 1}-10-27",
            "team1": f"Team {season}",
            "team2": f"Opponent {season}",
            "prob1": "0.6",
            "prob1_outcome": str(season % 2),
            "prob2": "0.4",
            "prob2_outcome": str(1 - season % 2),
        }
        for season in range(2016, 2023)
    ]


def _write_source(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(SOURCE_COLUMNS)
        writer.writerows(tuple(row[column] for column in SOURCE_COLUMNS) for row in rows)


def _load_fixture(path: Path) -> tuple[OpenModernGame, ...]:
    return load_open_modern_games(path, expected_sha256=file_sha256(path))


def test_loads_games_chronologically_with_opaque_unique_ids(tmp_path: Path) -> None:
    source_path = tmp_path / "source.csv"
    _write_source(source_path, list(reversed(_source_rows())))

    games = _load_fixture(source_path)

    assert [game.season for game in games] == list(range(2016, 2023))
    assert all(game.game_id.startswith("nba-538-") for game in games)
    assert len({game.game_id for game in games}) == len(games)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("prob1", "1", "interior probability"),
        ("prob2", "0.5", "sum to one"),
        ("prob1_outcome", "0.0", "zero or one"),
    ],
)
def test_rejects_malformed_probabilities_and_answers(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    source_path = tmp_path / "source.csv"
    rows = _source_rows()
    rows[0][field] = value
    _write_source(source_path, rows)

    with pytest.raises(OpenModernError, match=message):
        _load_fixture(source_path)


def test_rejects_duplicate_unordered_game_identity(tmp_path: Path) -> None:
    source_path = tmp_path / "source.csv"
    rows = _source_rows()
    duplicate = dict(rows[0])
    duplicate["team1"], duplicate["team2"] = duplicate["team2"], duplicate["team1"]
    rows.append(duplicate)
    _write_source(source_path, rows)

    with pytest.raises(OpenModernError, match="identities must be unique"):
        _load_fixture(source_path)


def test_writes_labels_only_to_development_artifact(tmp_path: Path) -> None:
    source_path = tmp_path / "source.csv"
    development_path = tmp_path / "development.csv"
    test_inputs_path = tmp_path / "test_inputs.csv"
    protocol_path = tmp_path / "protocol.json"
    seal_path = tmp_path / "seal.json"
    _write_source(source_path, _source_rows())
    protocol_path.write_text("{}\n", encoding="utf-8")

    result = write_open_modern_artifacts(
        _load_fixture(source_path),
        development_path,
        test_inputs_path,
        protocol_path,
        seal_path,
    )

    with development_path.open(encoding="utf-8", newline="") as file:
        development_reader = csv.DictReader(file)
        assert tuple(development_reader.fieldnames or ()) == DEVELOPMENT_COLUMNS
        assert len(list(development_reader)) == 5
    with test_inputs_path.open(encoding="utf-8", newline="") as file:
        test_reader = csv.DictReader(file)
        assert tuple(test_reader.fieldnames or ()) == TEST_INPUT_COLUMNS
        assert len(list(test_reader)) == 2

    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert result.development_count == 5
    assert result.test_input_count == 2
    assert seal["test_answers"]["materialized"] is False
    assert seal["safety"]["test_labels_written"] is False
    assert "prob1_outcome" not in seal["safety"]["test_input_columns"]
    assert len(seal["test_answers"]["sha256"]) == 64


def test_game_date_may_be_late_in_the_named_season_year() -> None:
    game = OpenModernGame(
        game_id="game",
        season=2020,
        game_date=date(2020, 10, 1),
        team1="Team",
        team2="Opponent",
        prob1=0.6,
        prob2=0.4,
        team1_won=True,
    )

    assert game.season == 2020
