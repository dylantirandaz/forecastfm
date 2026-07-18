"""NBA arena geography for travel features.

Arena coordinates and time zones are curated from Wikidata (CC0) and pinned in this file after
manual audit; every entry names its source. The table covers the 2021-22 through 2024-25 window:
the only franchise arena change inside it is the LA Clippers' move from Crypto.com Arena to
Intuit Dome for 2024-25, keyed by tipoff date.

Neutral-site regular-season games in the window (Mexico City and Paris) carry explicit dated
overrides, verified against the schedule join. All other games use the home team's arena.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from math import asin, cos, radians, sin, sqrt
from zoneinfo import ZoneInfo

NBA_ARENAS_SCHEMA_VERSION = 1

_LAC_INTUIT_DOME_FIRST_TIPOFF = date(2024, 10, 23)


class NbaArenaError(ValueError):
    """Raised when an arena lookup or travel computation violates its contract."""


@dataclass(frozen=True, slots=True)
class NbaArena:
    """One venue's identity, coordinates, and IANA time zone."""

    arena_name: str
    latitude: float
    longitude: float
    zone_name: str
    source: str

    def __post_init__(self) -> None:
        if not self.arena_name.strip() or not self.source.strip():
            raise NbaArenaError("arena name and source must not be empty")
        if not -90.0 <= self.latitude <= 90.0 or not -180.0 <= self.longitude <= 180.0:
            raise NbaArenaError("arena coordinates are out of range")
        ZoneInfo(self.zone_name)


_WIKIDATA = "Wikidata (CC0), audited 2026-07-17"

_TEAM_ARENAS: dict[str, NbaArena] = {
    "ATL": NbaArena("State Farm Arena", 33.7573, -84.3963, "America/New_York", _WIKIDATA),
    "BOS": NbaArena("TD Garden", 42.3662, -71.0621, "America/New_York", _WIKIDATA),
    "BKN": NbaArena("Barclays Center", 40.6826, -73.9754, "America/New_York", _WIKIDATA),
    "CHA": NbaArena("Spectrum Center", 35.2251, -80.8392, "America/New_York", _WIKIDATA),
    "CHI": NbaArena("United Center", 41.8807, -87.6742, "America/Chicago", _WIKIDATA),
    "CLE": NbaArena("Rocket Mortgage FieldHouse", 41.4965, -81.6882, "America/New_York", _WIKIDATA),
    "DAL": NbaArena("American Airlines Center", 32.7904, -96.8103, "America/Chicago", _WIKIDATA),
    "DEN": NbaArena("Ball Arena", 39.7487, -105.0077, "America/Denver", _WIKIDATA),
    "DET": NbaArena("Little Caesars Arena", 42.3411, -83.0553, "America/New_York", _WIKIDATA),
    "GSW": NbaArena("Chase Center", 37.7680, -122.3877, "America/Los_Angeles", _WIKIDATA),
    "HOU": NbaArena("Toyota Center", 29.7508, -95.3621, "America/Chicago", _WIKIDATA),
    "IND": NbaArena("Gainbridge Fieldhouse", 39.7640, -86.1555, "America/New_York", _WIKIDATA),
    "LAC": NbaArena("Crypto.com Arena", 34.0430, -118.2673, "America/Los_Angeles", _WIKIDATA),
    "LAL": NbaArena("Crypto.com Arena", 34.0430, -118.2673, "America/Los_Angeles", _WIKIDATA),
    "MEM": NbaArena("FedExForum", 35.1382, -90.0506, "America/Chicago", _WIKIDATA),
    "MIA": NbaArena("Kaseya Center", 25.7814, -80.1870, "America/New_York", _WIKIDATA),
    "MIL": NbaArena("Fiserv Forum", 43.0451, -87.9174, "America/Chicago", _WIKIDATA),
    "MIN": NbaArena("Target Center", 44.9795, -93.2760, "America/Chicago", _WIKIDATA),
    "NOP": NbaArena("Smoothie King Center", 29.9490, -90.0821, "America/Chicago", _WIKIDATA),
    "NYK": NbaArena("Madison Square Garden", 40.7505, -73.9934, "America/New_York", _WIKIDATA),
    "OKC": NbaArena("Paycom Center", 35.4634, -97.5151, "America/Chicago", _WIKIDATA),
    "ORL": NbaArena("Kia Center", 28.5392, -81.3839, "America/New_York", _WIKIDATA),
    "PHI": NbaArena("Wells Fargo Center", 39.9012, -75.1720, "America/New_York", _WIKIDATA),
    "PHX": NbaArena("Footprint Center", 33.4457, -112.0712, "America/Phoenix", _WIKIDATA),
    "POR": NbaArena("Moda Center", 45.5316, -122.6668, "America/Los_Angeles", _WIKIDATA),
    "SAC": NbaArena("Golden 1 Center", 38.5802, -121.4997, "America/Los_Angeles", _WIKIDATA),
    "SAS": NbaArena("Frost Bank Center", 29.4270, -98.4375, "America/Chicago", _WIKIDATA),
    "TOR": NbaArena("Scotiabank Arena", 43.6435, -79.3791, "America/New_York", _WIKIDATA),
    "UTA": NbaArena("Delta Center", 40.7683, -111.9011, "America/Denver", _WIKIDATA),
    "WAS": NbaArena("Capital One Arena", 38.8981, -77.0209, "America/New_York", _WIKIDATA),
}

_LAC_INTUIT_DOME = NbaArena(
    "Intuit Dome",
    33.9451,
    -118.3431,
    "America/Los_Angeles",
    _WIKIDATA,
)

_ARENA_CDMX = NbaArena("Arena CDMX", 19.4969, -99.1753, "America/Mexico_City", _WIKIDATA)
_ACCOR_ARENA = NbaArena("Accor Arena", 48.8385, 2.3785, "Europe/Paris", _WIKIDATA)

_NEUTRAL_SITE_GAMES: dict[tuple[date, str, str], NbaArena] = {
    (date(2022, 12, 17), "MIA", "SAS"): _ARENA_CDMX,
    (date(2023, 11, 9), "ATL", "ORL"): _ARENA_CDMX,
    (date(2024, 1, 11), "BKN", "CLE"): _ACCOR_ARENA,
    (date(2024, 11, 2), "MIA", "WAS"): _ARENA_CDMX,
    (date(2025, 1, 23), "IND", "SAS"): _ACCOR_ARENA,
    (date(2025, 1, 25), "IND", "SAS"): _ACCOR_ARENA,
}

_EARTH_RADIUS_MILES = 3958.7613


def home_arena(team_abbreviation: str, tipoff: datetime) -> NbaArena:
    """Return the home arena for one team at one tipoff, honoring arena moves."""
    if team_abbreviation == "LAC" and tipoff.date() >= _LAC_INTUIT_DOME_FIRST_TIPOFF:
        return _LAC_INTUIT_DOME
    try:
        return _TEAM_ARENAS[team_abbreviation]
    except KeyError as exc:
        raise NbaArenaError(f"unknown team abbreviation: {team_abbreviation}") from exc


def game_arena(
    game_date: date,
    away_abbreviation: str,
    home_abbreviation: str,
    tipoff: datetime,
) -> NbaArena:
    """Return the arena for one game, applying dated neutral-site overrides."""
    override = _NEUTRAL_SITE_GAMES.get((game_date, away_abbreviation, home_abbreviation))
    if override is not None:
        return override
    return home_arena(home_abbreviation, tipoff)


def neutral_site_games() -> tuple[tuple[date, str, str], ...]:
    """Return the declared neutral-site keys for schedule-join verification."""
    return tuple(sorted(_NEUTRAL_SITE_GAMES))


def great_circle_miles(first: NbaArena, second: NbaArena) -> float:
    """Return the haversine distance between two venues in miles."""
    lat1, lon1 = radians(first.latitude), radians(first.longitude)
    lat2, lon2 = radians(second.latitude), radians(second.longitude)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = sin(delta_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2) ** 2
    return 2 * _EARTH_RADIUS_MILES * asin(sqrt(a))


def utc_offset_hours(arena: NbaArena, at: datetime) -> float:
    """Return the venue's UTC offset in hours at one aware moment, including DST."""
    aware = at if at.tzinfo is not None else at.replace(tzinfo=UTC)
    offset = aware.astimezone(ZoneInfo(arena.zone_name)).utcoffset()
    if offset is None:
        raise NbaArenaError(f"cannot resolve UTC offset for {arena.zone_name}")
    return offset.total_seconds() / 3600.0


def travel_time_zone_change(prior: NbaArena, current: NbaArena, at: datetime) -> float:
    """Return the absolute venue UTC-offset change between two consecutive game venues."""
    return abs(utc_offset_hours(current, at) - utc_offset_hours(prior, at))
