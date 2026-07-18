"""Tests for the arena geography module."""

from datetime import UTC, date, datetime

import pytest

from forecastfm.nba_arenas import (
    NbaArenaError,
    game_arena,
    great_circle_miles,
    home_arena,
    neutral_site_games,
    travel_time_zone_change,
    utc_offset_hours,
)


def test_home_arena_covers_all_thirty_teams() -> None:
    tipoff = datetime(2024, 1, 15, 19, 30, tzinfo=UTC)
    abbreviations = [
        "ATL",
        "BOS",
        "BKN",
        "CHA",
        "CHI",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GSW",
        "HOU",
        "IND",
        "LAC",
        "LAL",
        "MEM",
        "MIA",
        "MIL",
        "MIN",
        "NOP",
        "NYK",
        "OKC",
        "ORL",
        "PHI",
        "PHX",
        "POR",
        "SAC",
        "SAS",
        "TOR",
        "UTA",
        "WAS",
    ]
    for abbreviation in abbreviations:
        assert home_arena(abbreviation, tipoff).arena_name
    with pytest.raises(NbaArenaError):
        home_arena("SEA", tipoff)


def test_lac_arena_move_is_keyed_by_tipoff() -> None:
    before = datetime(2024, 4, 10, 19, 30, tzinfo=UTC)
    after = datetime(2024, 10, 25, 19, 30, tzinfo=UTC)
    assert home_arena("LAC", before).arena_name == "Crypto.com Arena"
    assert home_arena("LAC", after).arena_name == "Intuit Dome"
    assert home_arena("LAL", after).arena_name == "Crypto.com Arena"


def test_neutral_site_overrides() -> None:
    tipoff = datetime(2025, 1, 23, 19, 0, tzinfo=UTC)
    arena = game_arena(date(2025, 1, 23), "IND", "SAS", tipoff)
    assert arena.arena_name == "Accor Arena"
    assert arena.zone_name == "Europe/Paris"
    regular = game_arena(date(2025, 1, 26), "IND", "SAS", tipoff)
    assert regular.arena_name == "Frost Bank Center"
    assert len(neutral_site_games()) == 6


def test_great_circle_distance_is_plausible() -> None:
    tipoff = datetime(2024, 1, 15, 19, 30, tzinfo=UTC)
    lax = home_arena("LAL", tipoff)
    nyc = home_arena("NYK", tipoff)
    miles = great_circle_miles(lax, nyc)
    assert 2400.0 < miles < 2500.0
    assert great_circle_miles(nyc, nyc) == pytest.approx(0.0, abs=1e-9)


def test_utc_offset_tracks_dst() -> None:
    tipoff_winter = datetime(2024, 1, 15, 19, 30, tzinfo=UTC)
    tipoff_summer = datetime(2024, 7, 15, 19, 30, tzinfo=UTC)
    nyc = home_arena("NYK", tipoff_winter)
    phx = home_arena("PHX", tipoff_winter)
    assert utc_offset_hours(nyc, tipoff_winter) == -5.0
    assert utc_offset_hours(nyc, tipoff_summer) == -4.0
    assert utc_offset_hours(phx, tipoff_winter) == -7.0
    assert utc_offset_hours(phx, tipoff_summer) == -7.0


def test_travel_time_zone_change() -> None:
    tipoff = datetime(2024, 1, 15, 19, 30, tzinfo=UTC)
    nyc = home_arena("NYK", tipoff)
    lax = home_arena("LAL", tipoff)
    assert travel_time_zone_change(nyc, lax, tipoff) == 3.0
    assert travel_time_zone_change(nyc, nyc, tipoff) == 0.0
