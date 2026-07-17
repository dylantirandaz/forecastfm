"""Tests for outcome-free open-modern NBA causal features."""

import csv
import inspect
import math
from datetime import date
from hashlib import sha1, sha256
from pathlib import Path

import pytest

import forecastfm.open_modern_features as feature_module
from forecastfm.open_modern import DEVELOPMENT_COLUMNS
from forecastfm.open_modern_features import (
    OPEN_MODERN_CAUSAL_FEATURE_CONTRACT_SHA256,
    OPEN_MODERN_CAUSAL_FEATURE_NAMES,
    RAPTOR_COLUMNS,
    RAPTOR_SOURCE_BYTES,
    RAPTOR_SOURCE_GIT_BLOB,
    RAPTOR_SOURCE_SHA256,
    TEAM_TO_BBREF,
    OpenModernCausalFeatures,
    OpenModernFeatureError,
    OpenModernInputGame,
    build_open_modern_features,
    load_open_modern_feature_inputs,
    load_prior_season_raptor,
)


def _game(
    game_id: str,
    season: int,
    game_date: date,
    teams: tuple[str, str],
    prob1: float,
) -> OpenModernInputGame:
    return OpenModernInputGame(
        game_id=game_id,
        season=season,
        game_date=game_date,
        team1=teams[0],
        team2=teams[1],
        prob1=prob1,
        prob2=1.0 - prob1,
    )


def _raptor_for(games: tuple[OpenModernInputGame, ...]) -> dict[tuple[int, str], float]:
    keys = {
        (game.season - 1, TEAM_TO_BBREF[team])
        for game in games
        for team in (game.team1, game.team2)
    }
    return {key: float(index) / 10.0 for index, key in enumerate(sorted(keys), start=1)}


def _write_development(
    path: Path,
    label_pair: tuple[str, str],
    *,
    prob1: float = 0.6,
) -> None:
    values = {
        "game_id": "game",
        "season": "2020",
        "date": "2019-10-01",
        "team1": "Hawks",
        "team2": "Celtics",
        "prob1": str(prob1),
        "prob2": str(1.0 - prob1),
    }
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(DEVELOPMENT_COLUMNS)
        labels = iter(label_pair)
        row = [
            values[column] if column in values else next(labels) for column in DEVELOPMENT_COLUMNS
        ]
        writer.writerow(row)


def _raptor_row(
    player_id: str,
    season: int,
    season_type: str,
    team: str,
    values: tuple[float, float],
) -> tuple[str, ...]:
    fields = {
        "player_name": player_id,
        "player_id": player_id,
        "season": str(season),
        "season_type": season_type,
        "team": team,
        "poss": str(values[0]),
        "raptor_total": str(values[1]),
    }
    return tuple(fields.get(column, "") for column in RAPTOR_COLUMNS)


def _write_raptor_fixture(path: Path, teams: tuple[str, ...]) -> None:
    rows: list[tuple[str, ...]] = []
    for season in (2019, 2020):
        rows.extend(
            _raptor_row(f"{team}-{season}", season, "RS", team, (100.0, 1.0)) for team in teams
        )
    if "ATL" in teams:
        rows.append(_raptor_row("ATL-extra", 2019, "RS", "ATL", (300.0, 3.0)))
        rows.append(_raptor_row("ATL-playoffs", 2019, "PO", "ATL", (900.0, 99.0)))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(RAPTOR_COLUMNS)
        writer.writerows(rows)


def _pin_raptor_fixture(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = path.read_bytes()
    blob_prefix = f"blob {len(payload)}\0".encode()
    monkeypatch.setattr(feature_module, "RAPTOR_SOURCE_BYTES", len(payload))
    monkeypatch.setattr(feature_module, "RAPTOR_SOURCE_SHA256", sha256(payload).hexdigest())
    monkeypatch.setattr(
        feature_module,
        "RAPTOR_SOURCE_GIT_BLOB",
        sha1(blob_prefix + payload, usedforsecurity=False).hexdigest(),
    )


def test_fixed_mapping_covers_exactly_thirty_source_teams() -> None:
    assert len(TEAM_TO_BBREF) == 30
    assert len(set(TEAM_TO_BBREF.values())) == 30
    assert TEAM_TO_BBREF["76ers"] == "PHI"
    assert TEAM_TO_BBREF["Nets"] == "BRK"
    assert TEAM_TO_BBREF["Trail Blazers"] == "POR"


def test_feature_source_never_names_outcome_columns() -> None:
    source = inspect.getsource(feature_module)

    assert "prob1_outcome" not in source
    assert "prob2_outcome" not in source


def test_poisoned_development_answers_cannot_change_parsed_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_development(first, ("unused", "1"))
    _write_development(second, ("unused", "0"))

    def accept_artifact(*_paths: Path) -> None:
        return None

    monkeypatch.setattr(feature_module, "require_open_modern_development", accept_artifact)
    monkeypatch.setattr(
        feature_module,
        "OPEN_MODERN_DEVELOPMENT_SHA256",
        sha256(first.read_bytes()).hexdigest(),
    )
    first_games = load_open_modern_feature_inputs(
        first,
        seal_path=first,
        protocol_path=first,
        exposure_path=first,
    )
    monkeypatch.setattr(
        feature_module,
        "OPEN_MODERN_DEVELOPMENT_SHA256",
        sha256(second.read_bytes()).hexdigest(),
    )
    second_games = load_open_modern_feature_inputs(
        second,
        seal_path=second,
        protocol_path=second,
        exposure_path=second,
    )

    assert first_games == second_games


def test_feature_loader_parses_the_verified_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "development.csv"
    replacement = tmp_path / "replacement.csv"
    _write_development(path, ("unused", "1"), prob1=0.6)
    _write_development(replacement, ("unused", "1"), prob1=0.7)
    original_payload = path.read_bytes()

    def replace_artifact(*_paths: Path) -> None:
        path.write_bytes(replacement.read_bytes())

    monkeypatch.setattr(feature_module, "require_open_modern_development", replace_artifact)
    monkeypatch.setattr(
        feature_module,
        "OPEN_MODERN_DEVELOPMENT_SHA256",
        sha256(original_payload).hexdigest(),
    )

    games = load_open_modern_feature_inputs(
        path,
        seal_path=path,
        protocol_path=path,
        exposure_path=path,
    )

    assert games[0].prob1 == 0.6
    assert path.read_bytes() == replacement.read_bytes()


def test_same_date_rows_are_built_before_any_history_update() -> None:
    games = (
        _game("same-a", 2020, date(2019, 10, 1), ("Hawks", "Celtics"), 0.6),
        _game("same-b", 2020, date(2019, 10, 1), ("Hawks", "Bulls"), 0.7),
        _game("next", 2020, date(2019, 10, 2), ("Hawks", "Bucks"), 0.55),
    )

    rows = {row.game_id: row for row in build_open_modern_features(games, _raptor_for(games))}

    for game_id in ("same-a", "same-b"):
        features = rows[game_id].features
        assert features.rest_days_difference == 0.0
        assert features.games_last_7_difference == 0.0
        assert features.trailing_10_opponent_probability_difference == 0.0
        assert features.trailing_10_history_difference == 0.0
    next_features = rows["next"].features
    assert next_features.back_to_back_difference == 1.0
    assert next_features.games_last_7_difference == 2.0
    assert next_features.trailing_10_opponent_probability_difference == pytest.approx(-0.15)
    assert next_features.trailing_10_history_difference == 2.0


def test_seven_day_window_is_closed_on_start_and_open_on_game_date() -> None:
    games = (
        _game("start", 2020, date(2019, 10, 1), ("Hawks", "Celtics"), 0.6),
        _game("inside", 2020, date(2019, 10, 7), ("Hawks", "Bulls"), 0.7),
        _game("current", 2020, date(2019, 10, 8), ("Hawks", "Bucks"), 0.55),
    )

    rows = {row.game_id: row for row in build_open_modern_features(games, _raptor_for(games))}
    features = rows["current"].features

    assert features.games_last_7_difference == 2.0
    assert features.rest_days_difference == 0.0
    assert features.back_to_back_difference == 1.0


def test_team_histories_reset_at_each_season_boundary() -> None:
    games = (
        _game("old", 2020, date(2020, 10, 1), ("Hawks", "Bulls"), 0.6),
        _game("new", 2021, date(2020, 12, 22), ("Hawks", "Celtics"), 0.7),
    )

    rows = {row.game_id: row for row in build_open_modern_features(games, _raptor_for(games))}
    features = rows["new"].features

    assert features.rest_days_difference == 0.0
    assert features.back_to_back_difference == 0.0
    assert features.games_last_7_difference == 0.0
    assert features.trailing_10_opponent_probability_difference == 0.0
    assert features.trailing_10_history_difference == 0.0


def test_source_log_odds_and_feature_order_are_exact() -> None:
    game = _game("game", 2020, date(2019, 10, 1), ("Hawks", "Celtics"), 0.8)
    row = build_open_modern_features((game,), _raptor_for((game,)))[0]

    assert row.features.source_log_odds == pytest.approx(math.log(4.0))
    assert tuple(row.features.as_dict()) == OPEN_MODERN_CAUSAL_FEATURE_NAMES


def test_side_swap_exactly_negates_and_normalizes_signed_zero() -> None:
    features = OpenModernCausalFeatures(1.0, 0.0, -1.0, 2.0, -0.25, 3.0, -4.0)

    swapped = features.side_swap()

    assert swapped.vector == (-1.0, 0.0, 1.0, -2.0, 0.25, -3.0, 4.0)
    assert math.copysign(1.0, swapped.rest_days_difference) == 1.0
    assert swapped.side_swap() == features


def test_requires_complete_finite_prior_season_raptor_join() -> None:
    game = _game("game", 2020, date(2019, 10, 1), ("Hawks", "Celtics"), 0.6)
    raptor = {(2019, "ATL"): 1.0}

    with pytest.raises(OpenModernFeatureError, match="join is incomplete"):
        build_open_modern_features((game,), raptor)


def test_pinned_raptor_loader_filters_later_seasons_and_weights_possessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "raptor.csv"
    _write_raptor_fixture(path, tuple(TEAM_TO_BBREF.values()))
    _pin_raptor_fixture(path, monkeypatch)

    raptor = load_prior_season_raptor(path, max_allowed_season=2019)

    assert max(season for season, _ in raptor) == 2019
    assert raptor[(2019, "ATL")] == pytest.approx(2.5)
    assert {(2019, team) for team in TEAM_TO_BBREF.values()}.issubset(raptor)


def test_raptor_loader_parses_the_pinned_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "raptor.csv"
    _write_raptor_fixture(path, tuple(TEAM_TO_BBREF.values()))
    _pin_raptor_fixture(path, monkeypatch)
    original_read_bytes = Path.read_bytes

    def replace_after_read(candidate: Path) -> bytes:
        payload = original_read_bytes(candidate)
        if candidate == path:
            candidate.write_bytes(b"not the pinned RAPTOR source")
        return payload

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)

    raptor = load_prior_season_raptor(path, max_allowed_season=2019)

    assert raptor[(2019, "ATL")] == pytest.approx(2.5)
    assert original_read_bytes(path) == b"not the pinned RAPTOR source"


def test_raptor_pin_includes_commit_blob_size_and_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert RAPTOR_SOURCE_BYTES == 1_922_974
    assert RAPTOR_SOURCE_GIT_BLOB == "7e9f47d175de2a0f86b04bfd175597477c6ae26d"
    assert RAPTOR_SOURCE_SHA256 == (
        "a80bb5d24eb6b9742bb0c68aacf643144e7d39311b3b5aa12199b63d8d7de2aa"
    )
    tampered = tmp_path / "raptor.csv"
    _write_raptor_fixture(tampered, tuple(TEAM_TO_BBREF.values()))
    _pin_raptor_fixture(tampered, monkeypatch)
    tampered.write_bytes(tampered.read_bytes() + b"\n")

    with pytest.raises(OpenModernFeatureError, match="byte size"):
        load_prior_season_raptor(tampered, max_allowed_season=2019)


def test_raptor_loader_requires_full_team_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "raptor.csv"
    teams = tuple(team for team in TEAM_TO_BBREF.values() if team != "WAS")
    _write_raptor_fixture(path, teams)
    _pin_raptor_fixture(path, monkeypatch)

    with pytest.raises(OpenModernFeatureError, match="incomplete team coverage"):
        load_prior_season_raptor(path, max_allowed_season=2019)


def test_feature_contract_is_hash_frozen() -> None:
    assert OPEN_MODERN_CAUSAL_FEATURE_CONTRACT_SHA256 == (
        "7c41ea294d5aa298053753b47a7e6fd5e218515ebe18820121429fa84a1f4ba4"
    )
