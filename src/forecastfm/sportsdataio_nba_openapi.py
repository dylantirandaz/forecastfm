"""Typed request identities from SportsDataIO's public NBA OpenAPI schema."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from forecastfm.nba_raw_capture import NbaRawCaptureRequest

SPORTSDATAIO_NBA_HOST = "api.sportsdata.io"
SPORTSDATAIO_NBA_OPENAPI_URL = "https://cdn.sportsdata.io/openapi/NBA-openapi-3.1.json"

type SportsDataIONbaSeasonSuffix = Literal["", "PRE", "POST", "STAR"]
type SportsDataIONumberOfGames = int | Literal["all"]

_MONTHS = (
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
)
_YEAR_PATTERN = r"(?:19(?:4[6-9]|[5-9][0-9])|20[0-9]{2}|2100)"
_SEASON_PATTERN = rf"{_YEAR_PATTERN}(?:PRE|POST|STAR)?"
_DATE_PATTERN = rf"{_YEAR_PATTERN}-(?:{'|'.join(_MONTHS)})-[0-9]{{2}}"
_REGISTERED_PATHS = {
    "games": re.compile(rf"/v3/nba/scores/json/Games/{_SEASON_PATTERN}"),
    "games_by_date_final": re.compile(rf"/v3/nba/scores/json/GamesByDateFinal/{_DATE_PATTERN}"),
    "depth_charts": re.compile(r"/v3/nba/scores/json/DepthCharts"),
    "transactions_by_date": re.compile(rf"/v3/nba/scores/json/TransactionsByDate/{_DATE_PATTERN}"),
    "starting_lineups_by_date": re.compile(
        rf"/v3/nba/projections/json/StartingLineupsByDate/{_DATE_PATTERN}"
    ),
    "injured_players": re.compile(r"/v3/nba/projections/json/InjuredPlayers"),
    "team_game_stats_by_season": re.compile(
        rf"/v3/nba/scores/json/TeamGameStatsBySeason/{_SEASON_PATTERN}/[1-9][0-9]*/"
        r"(?:all|[1-9][0-9]*)"
    ),
    "player_game_stats_by_date": re.compile(
        rf"/v3/nba/stats/json/PlayerGameStatsByDate/{_DATE_PATTERN}"
    ),
}


class SportsDataIONbaRequestError(ValueError):
    """Raised when a typed NBA OpenAPI request value is unsupported."""


@dataclass(frozen=True, slots=True)
class SportsDataIONbaSeason:
    """One season segment accepted by the selected public NBA endpoints."""

    year: int
    suffix: SportsDataIONbaSeasonSuffix = ""

    def __post_init__(self) -> None:
        if type(self.year) is not int or not 1946 <= self.year <= 2100:
            raise SportsDataIONbaRequestError("NBA season year is outside the supported range")
        if self.suffix not in {"", "PRE", "POST", "STAR"}:
            raise SportsDataIONbaRequestError("unsupported NBA season suffix")

    @property
    def path_value(self) -> str:
        """Return the exact OpenAPI season path segment."""
        return f"{self.year}{self.suffix}"


def games_request(season: SportsDataIONbaSeason) -> NbaRawCaptureRequest:
    """Build the published full-season schedule request identity."""
    return _request("games", f"/v3/nba/scores/json/Games/{season.path_value}")


def games_by_date_final_request(game_date: date) -> NbaRawCaptureRequest:
    """Build the published final-score request identity for one date."""
    return _request(
        "games_by_date_final",
        f"/v3/nba/scores/json/GamesByDateFinal/{_date_path(game_date)}",
    )


def depth_charts_request() -> NbaRawCaptureRequest:
    """Build the published current depth-chart request identity."""
    return _request("depth_charts", "/v3/nba/scores/json/DepthCharts")


def transactions_by_date_request(transaction_date: date) -> NbaRawCaptureRequest:
    """Build the published transaction request identity for one date."""
    return _request(
        "transactions_by_date",
        f"/v3/nba/scores/json/TransactionsByDate/{_date_path(transaction_date)}",
    )


def starting_lineups_by_date_request(game_date: date) -> NbaRawCaptureRequest:
    """Build the published projected and confirmed lineup request identity."""
    return _request(
        "starting_lineups_by_date",
        f"/v3/nba/projections/json/StartingLineupsByDate/{_date_path(game_date)}",
    )


def injured_players_request() -> NbaRawCaptureRequest:
    """Build the published current injured-player request identity."""
    return _request("injured_players", "/v3/nba/projections/json/InjuredPlayers")


def team_game_stats_by_season_request(
    season: SportsDataIONbaSeason,
    team_id: int,
    number_of_games: SportsDataIONumberOfGames,
) -> NbaRawCaptureRequest:
    """Build the published team game-log request identity."""
    _require_positive_integer(team_id, "team_id")
    if number_of_games != "all":
        _require_positive_integer(number_of_games, "number_of_games")
    path = (
        f"/v3/nba/scores/json/TeamGameStatsBySeason/{season.path_value}/{team_id}/{number_of_games}"
    )
    return _request("team_game_stats_by_season", path)


def player_game_stats_by_date_request(game_date: date) -> NbaRawCaptureRequest:
    """Build the published player game-stat request identity for one date."""
    return _request(
        "player_game_stats_by_date",
        f"/v3/nba/stats/json/PlayerGameStatsByDate/{_date_path(game_date)}",
    )


def require_registered_nba_request(request: NbaRawCaptureRequest) -> NbaRawCaptureRequest:
    """Require one fixed-host request matching a selected public OpenAPI path."""
    pattern = _REGISTERED_PATHS.get(request.operation)
    if (
        request.host != SPORTSDATAIO_NBA_HOST
        or pattern is None
        or pattern.fullmatch(request.path) is None
    ):
        raise SportsDataIONbaRequestError("request is outside the registered NBA OpenAPI paths")
    return request


def _request(operation: str, path: str) -> NbaRawCaptureRequest:
    return NbaRawCaptureRequest(operation=operation, host=SPORTSDATAIO_NBA_HOST, path=path)


def _date_path(value: date) -> str:
    if type(value) is not date:
        raise SportsDataIONbaRequestError("NBA request date must be a calendar date")
    return f"{value.year:04d}-{_MONTHS[value.month - 1]}-{value.day:02d}"


def _require_positive_integer(value: int, field_name: str) -> None:
    if type(value) is not int or value <= 0:
        raise SportsDataIONbaRequestError(f"{field_name} must be a positive integer")
