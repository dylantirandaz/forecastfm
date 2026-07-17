"""Tests for one-way preparation of the open-modern NBA source."""

import csv
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples import seal_open_modern_source as seal_script

from forecastfm import open_modern
from forecastfm.integrity import canonical_sha256, file_sha256
from forecastfm.open_modern import (
    DEVELOPMENT_COLUMNS,
    SOURCE_COLUMNS,
    TEST_INPUT_COLUMNS,
    OpenModernError,
    OpenModernSealResult,
    require_open_modern_development,
    seal_open_modern_source,
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


def _seal_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, str]],
) -> tuple[OpenModernSealResult, tuple[Path, ...]]:
    source_path = tmp_path / "source.csv"
    development_path = tmp_path / "development.csv"
    test_inputs_path = tmp_path / "test_inputs.csv"
    protocol_path = tmp_path / "protocol.json"
    exposure_path = tmp_path / "EXPOSURE.md"
    seal_path = tmp_path / "seal.json"
    _write_source(source_path, rows)
    protocol_path.write_text("{}\n", encoding="utf-8")
    exposure_path.write_text("exposure\n", encoding="utf-8")
    dates = tuple(date.fromisoformat(row["date"]) for row in rows)
    monkeypatch.setattr(open_modern, "OPEN_MODERN_SOURCE_SHA256", file_sha256(source_path))
    monkeypatch.setattr(open_modern, "OPEN_MODERN_SOURCE_BYTES", source_path.stat().st_size)
    monkeypatch.setattr(open_modern, "OPEN_MODERN_SOURCE_ROWS", len(rows))
    monkeypatch.setattr(open_modern, "OPEN_MODERN_PROTOCOL_SHA256", file_sha256(protocol_path))
    monkeypatch.setattr(open_modern, "_FIRST_GAME_DATE", min(dates))
    monkeypatch.setattr(open_modern, "_LAST_GAME_DATE", max(dates))
    result = seal_open_modern_source(
        source_path,
        development_path,
        test_inputs_path,
        protocol_path,
        seal_path,
    )
    return result, (
        source_path,
        development_path,
        test_inputs_path,
        protocol_path,
        exposure_path,
        seal_path,
    )


def test_seals_games_chronologically_with_opaque_unique_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, paths = _seal_fixture(tmp_path, monkeypatch, list(reversed(_source_rows())))
    development_path = paths[1]
    with development_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert [int(row["season"]) for row in rows] == list(range(2016, 2021))
    assert all(row["game_id"].startswith("nba-538-") for row in rows)
    assert len({row["game_id"] for row in rows}) == len(rows)


def test_parses_the_same_source_snapshot_it_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "source.csv"
    development_path = tmp_path / "development.csv"
    test_inputs_path = tmp_path / "test_inputs.csv"
    protocol_path = tmp_path / "protocol.json"
    exposure_path = tmp_path / "EXPOSURE.md"
    seal_path = tmp_path / "seal.json"
    original_rows = _source_rows()
    replacement_rows = [dict(row) for row in original_rows]
    replacement_rows[0]["prob1_outcome"] = "1"
    replacement_rows[0]["prob2_outcome"] = "0"
    _write_source(source_path, original_rows)
    protocol_path.write_text("{}\n", encoding="utf-8")
    exposure_path.write_text("exposure\n", encoding="utf-8")

    original_hash = file_sha256(source_path)
    monkeypatch.setattr(open_modern, "OPEN_MODERN_SOURCE_SHA256", original_hash)
    monkeypatch.setattr(open_modern, "OPEN_MODERN_SOURCE_BYTES", source_path.stat().st_size)
    monkeypatch.setattr(open_modern, "OPEN_MODERN_SOURCE_ROWS", len(original_rows))
    monkeypatch.setattr(open_modern, "OPEN_MODERN_PROTOCOL_SHA256", file_sha256(protocol_path))
    monkeypatch.setattr(open_modern, "_FIRST_GAME_DATE", date(2015, 10, 27))
    monkeypatch.setattr(open_modern, "_LAST_GAME_DATE", date(2021, 10, 27))

    real_read_bytes = Path.read_bytes
    source_read_count = 0

    def replace_path_after_snapshot(path: Path) -> bytes:
        nonlocal source_read_count
        snapshot = real_read_bytes(path)
        if path == source_path:
            source_read_count += 1
            if source_read_count == 1:
                _write_source(source_path, replacement_rows)
        return snapshot

    monkeypatch.setattr(Path, "read_bytes", replace_path_after_snapshot)

    seal_open_modern_source(
        source_path,
        development_path,
        test_inputs_path,
        protocol_path,
        seal_path,
    )

    with development_path.open(encoding="utf-8", newline="") as file:
        development_rows = list(csv.DictReader(file))
    assert source_read_count == 1
    assert development_rows[0]["prob1_outcome"] == original_rows[0]["prob1_outcome"]
    assert file_sha256(source_path) != original_hash


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
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    rows = _source_rows()
    rows[0][field] = value

    with pytest.raises(OpenModernError, match=message):
        _seal_fixture(tmp_path, monkeypatch, rows)


def test_rejects_duplicate_unordered_game_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _source_rows()
    duplicate = dict(rows[0])
    duplicate["team1"], duplicate["team2"] = duplicate["team2"], duplicate["team1"]
    rows.append(duplicate)

    with pytest.raises(OpenModernError, match="identities must be unique"):
        _seal_fixture(tmp_path, monkeypatch, rows)


def test_writes_labels_only_to_development_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, paths = _seal_fixture(tmp_path, monkeypatch, _source_rows())
    development_path, test_inputs_path, seal_path = paths[1], paths[2], paths[5]
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
    assert result.development_sha256 == file_sha256(development_path)
    assert result.test_inputs_sha256 == file_sha256(test_inputs_path)
    assert result.seal_sha256 == file_sha256(seal_path)
    assert seal["test_answers"]["answer_artifact_written"] is False
    assert seal["claim"]["literally_unopened"] is False
    assert seal["safety"]["test_labels_written"] is False
    assert "prob1_outcome" not in seal["safety"]["test_input_columns"]
    assert len(seal["test_answers"]["sha256"]) == 64


def test_game_date_may_be_late_in_the_named_season_year(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _source_rows()
    rows[4]["date"] = "2020-10-01"

    result, _ = _seal_fixture(tmp_path, monkeypatch, rows)

    assert result.development_count == 5


def test_exact_writer_refuses_to_replace_a_changed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _source_rows()
    _, paths = _seal_fixture(tmp_path, monkeypatch, rows)
    source_path, development_path, test_inputs_path, protocol_path, _, seal_path = paths
    development_path.write_text("changed\n", encoding="utf-8")

    with pytest.raises(OpenModernError, match="refusing to replace"):
        seal_open_modern_source(
            source_path,
            development_path,
            test_inputs_path,
            protocol_path,
            seal_path,
        )


def test_preflight_rejects_a_late_conflict_before_writing_any_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, paths = _seal_fixture(tmp_path, monkeypatch, _source_rows())
    source_path, development_path, test_inputs_path, protocol_path, _, seal_path = paths
    development_path.unlink()
    test_inputs_path.unlink()
    seal_path.write_text("changed\n", encoding="utf-8")

    with pytest.raises(OpenModernError, match="refusing to replace"):
        seal_open_modern_source(
            source_path,
            development_path,
            test_inputs_path,
            protocol_path,
            seal_path,
        )

    assert not development_path.exists()
    assert not test_inputs_path.exists()
    assert seal_path.read_text(encoding="utf-8") == "changed\n"


def test_atomic_writer_cleans_up_when_exclusive_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, paths = _seal_fixture(tmp_path, monkeypatch, _source_rows())
    source_path, development_path, test_inputs_path, protocol_path, _, seal_path = paths
    development_path.unlink()
    test_inputs_path.unlink()
    seal_path.unlink()

    real_link = open_modern.os.link
    link_count = 0

    def fail_on_second_link(source: Path, destination: Path) -> None:
        nonlocal link_count
        link_count += 1
        if link_count == 2:
            raise OSError("publish failed")
        real_link(source, destination)

    monkeypatch.setattr(open_modern.os, "link", fail_on_second_link)

    with pytest.raises(OSError, match="publish failed"):
        seal_open_modern_source(
            source_path,
            development_path,
            test_inputs_path,
            protocol_path,
            seal_path,
        )

    assert not development_path.exists()
    assert not test_inputs_path.exists()
    assert not seal_path.exists()
    assert not tuple(tmp_path.glob(".*.part"))


def test_public_verifier_rejects_a_changed_development_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development_path = tmp_path / "development.csv"
    seal_path = tmp_path / "seal.json"
    protocol_path = tmp_path / "protocol.json"
    exposure_path = tmp_path / "EXPOSURE.md"
    row = (
        "game-1",
        "2016",
        "2015-10-27",
        "Team",
        "Opponent",
        "0.6",
        "1",
        "0.4",
        "0",
    )
    with development_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(DEVELOPMENT_COLUMNS)
        writer.writerow(row)
    seal_path.write_text("seal\n", encoding="utf-8")
    protocol_path.write_text("protocol\n", encoding="utf-8")
    exposure_path.write_text("exposure\n", encoding="utf-8")
    monkeypatch.setattr(open_modern, "OPEN_MODERN_SOURCE_SEAL_SHA256", file_sha256(seal_path))
    monkeypatch.setattr(open_modern, "OPEN_MODERN_PROTOCOL_SHA256", file_sha256(protocol_path))
    monkeypatch.setattr(open_modern, "OPEN_MODERN_EXPOSURE_SHA256", file_sha256(exposure_path))
    monkeypatch.setattr(
        open_modern,
        "_DEVELOPMENT_CONTRACT",
        SimpleNamespace(
            columns=DEVELOPMENT_COLUMNS,
            sha256=file_sha256(development_path),
            row_count=1,
            ordered_ids_sha256=canonical_sha256(["game-1"]),
        ),
    )
    require_open_modern_development(
        development_path,
        seal_path,
        protocol_path,
        exposure_path,
    )
    development_path.write_text("changed\n", encoding="utf-8")

    with pytest.raises(OpenModernError, match="artifact does not match"):
        require_open_modern_development(
            development_path,
            seal_path,
            protocol_path,
            exposure_path,
        )


def test_cli_removes_temporary_answer_source_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "answers.csv"
    source_path.write_text("temporary answers\n", encoding="utf-8")

    def fail_to_seal(*_paths: Path) -> OpenModernSealResult:
        raise OpenModernError("failed closed")

    monkeypatch.setattr(seal_script, "SOURCE_PATH", source_path)
    monkeypatch.setattr(seal_script, "seal_open_modern_source", fail_to_seal)

    with pytest.raises(OpenModernError, match="failed closed"):
        seal_script.main()

    assert not source_path.exists()
