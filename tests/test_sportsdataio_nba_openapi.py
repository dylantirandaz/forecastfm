"""Tests for the fixed SportsDataIO NBA OpenAPI request registry."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import cast

import pytest

from forecastfm.nba_raw_capture import NbaRawCaptureRequest
from forecastfm.sportsdataio_nba_openapi import (
    SPORTSDATAIO_NBA_HOST,
    SPORTSDATAIO_NBA_OPENAPI_URL,
    SportsDataIONbaRequestError,
    SportsDataIONbaSeason,
    SportsDataIONbaSeasonSuffix,
    SportsDataIONumberOfGames,
    depth_charts_request,
    games_by_date_final_request,
    games_request,
    injured_players_request,
    player_game_stats_by_date_request,
    require_registered_nba_request,
    starting_lineups_by_date_request,
    team_game_stats_by_season_request,
    transactions_by_date_request,
)


def test_registry_pins_the_public_schema_and_host() -> None:
    assert SPORTSDATAIO_NBA_HOST == "api.sportsdata.io"
    assert SPORTSDATAIO_NBA_OPENAPI_URL == (
        "https://cdn.sportsdata.io/openapi/NBA-openapi-3.1.json"
    )


def test_registry_builds_all_required_openapi_paths() -> None:
    season = SportsDataIONbaSeason(2026)
    game_date = date(2026, 7, 17)

    requests = (
        games_request(season),
        games_by_date_final_request(game_date),
        depth_charts_request(),
        transactions_by_date_request(game_date),
        starting_lineups_by_date_request(game_date),
        injured_players_request(),
        team_game_stats_by_season_request(season, 8, "all"),
        player_game_stats_by_date_request(game_date),
    )

    assert tuple(request.operation for request in requests) == (
        "games",
        "games_by_date_final",
        "depth_charts",
        "transactions_by_date",
        "starting_lineups_by_date",
        "injured_players",
        "team_game_stats_by_season",
        "player_game_stats_by_date",
    )
    assert tuple(request.path for request in requests) == (
        "/v3/nba/scores/json/Games/2026",
        "/v3/nba/scores/json/GamesByDateFinal/2026-JUL-17",
        "/v3/nba/scores/json/DepthCharts",
        "/v3/nba/scores/json/TransactionsByDate/2026-JUL-17",
        "/v3/nba/projections/json/StartingLineupsByDate/2026-JUL-17",
        "/v3/nba/projections/json/InjuredPlayers",
        "/v3/nba/scores/json/TeamGameStatsBySeason/2026/8/all",
        "/v3/nba/stats/json/PlayerGameStatsByDate/2026-JUL-17",
    )
    assert all(request.host == SPORTSDATAIO_NBA_HOST for request in requests)
    assert all(require_registered_nba_request(request) is request for request in requests)


@pytest.mark.parametrize(
    "candidate",
    [
        NbaRawCaptureRequest("games", "evil.example", "/v3/nba/scores/json/Games/2026"),
        NbaRawCaptureRequest("games", SPORTSDATAIO_NBA_HOST, "/v3/nba/scores/json/Games/1945"),
        NbaRawCaptureRequest(
            "games_by_date_final",
            SPORTSDATAIO_NBA_HOST,
            "/v3/nba/scores/json/GamesByDateFinal/2026-JUL-17/extra",
        ),
        NbaRawCaptureRequest(
            "depth_charts",
            SPORTSDATAIO_NBA_HOST,
            "/v3/nba/scores/json/InjuredPlayers",
        ),
    ],
)
def test_registry_rejects_forged_request_identities(candidate: NbaRawCaptureRequest) -> None:
    with pytest.raises(SportsDataIONbaRequestError, match="outside"):
        require_registered_nba_request(candidate)


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [("", "2026"), ("PRE", "2026PRE"), ("POST", "2026POST"), ("STAR", "2026STAR")],
)
def test_season_formats_supported_suffixes(suffix: str, expected: str) -> None:
    season = SportsDataIONbaSeason(2026, cast(SportsDataIONbaSeasonSuffix, suffix))

    assert season.path_value == expected


@pytest.mark.parametrize("year", [True, 1945, 2101])
def test_season_rejects_unsupported_years(year: int) -> None:
    with pytest.raises(SportsDataIONbaRequestError, match="year"):
        SportsDataIONbaSeason(year)


def test_season_rejects_unknown_suffix_at_runtime() -> None:
    with pytest.raises(SportsDataIONbaRequestError, match="suffix"):
        SportsDataIONbaSeason(2026, cast(SportsDataIONbaSeasonSuffix, "REG"))


def test_registry_rejects_datetime_instead_of_calendar_date() -> None:
    value = cast(date, datetime(2026, 7, 17, tzinfo=UTC))

    with pytest.raises(SportsDataIONbaRequestError, match="calendar date"):
        games_by_date_final_request(value)


@pytest.mark.parametrize(("team_id", "number_of_games"), [(0, "all"), (True, "all"), (8, 0)])
def test_team_game_log_rejects_nonpositive_identifiers(
    team_id: int,
    number_of_games: int | str,
) -> None:
    with pytest.raises(SportsDataIONbaRequestError, match="positive integer"):
        team_game_stats_by_season_request(
            SportsDataIONbaSeason(2026),
            team_id,
            cast(SportsDataIONumberOfGames, number_of_games),
        )


def test_team_game_log_formats_a_fixed_positive_count() -> None:
    request = team_game_stats_by_season_request(SportsDataIONbaSeason(2026, "POST"), 8, 25)

    assert request.path == "/v3/nba/scores/json/TeamGameStatsBySeason/2026POST/8/25"
