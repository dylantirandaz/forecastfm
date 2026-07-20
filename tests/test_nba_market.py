"""Tests for the dependency-free ESPN market-odds benchmark core."""

import json
from pathlib import Path

import pytest

from forecastfm.nba_market import (
    NbaMarketError,
    american_to_implied_probability,
    load_market_probabilities,
    parse_market_odds,
)

# Trimmed mirror of the retained data/raw/espn/raw/summary-401809238.json pickcenter
# substructure (DraftKings pregame moneyline; ``odds`` is empty in the real payload).
REALISTIC_PAYLOAD = json.dumps(
    {
        "pickcenter": [
            {
                "provider": {"id": "100", "name": "DraftKings", "priority": 1},
                "details": "NY -4.5",
                "overUnder": 241.5,
                "spread": -4.5,
                "overOdds": -112.0,
                "underOdds": -108.0,
                "awayTeamOdds": {
                    "favorite": False,
                    "underdog": True,
                    "moneyLine": 150,
                    "spreadOdds": -118.0,
                    "teamId": "5",
                    "favoriteAtOpen": False,
                },
                "homeTeamOdds": {
                    "favorite": True,
                    "underdog": False,
                    "moneyLine": -180,
                    "spreadOdds": -102.0,
                    "teamId": "18",
                    "favoriteAtOpen": True,
                },
                "moneyline": {
                    "displayName": "Moneyline",
                    "shortDisplayName": "ML",
                    "home": {"close": {"odds": "-180"}, "open": {"odds": "-198"}},
                    "away": {"close": {"odds": "+150"}, "open": {"odds": "+164"}},
                },
            }
        ],
        "odds": [],
    }
).encode("utf-8")


def test_american_negative_moneyline_implied_probability() -> None:
    assert american_to_implied_probability(-150) == pytest.approx(150 / 250)
    assert american_to_implied_probability(-180) == pytest.approx(180 / 280)
    assert american_to_implied_probability(-100) == pytest.approx(0.5)


def test_american_positive_moneyline_implied_probability() -> None:
    assert american_to_implied_probability(150) == pytest.approx(100 / 250)
    assert american_to_implied_probability(230) == pytest.approx(100 / 330)
    assert american_to_implied_probability(100) == pytest.approx(0.5)


def test_american_moneyline_rejects_zero() -> None:
    with pytest.raises(NbaMarketError):
        american_to_implied_probability(0.0)


def test_parse_realistic_payload_devigs_both_sides() -> None:
    odds = parse_market_odds(REALISTIC_PAYLOAD)
    assert odds is not None
    home_raw = 180 / 280
    away_raw = 100 / 250
    total = home_raw + away_raw
    assert odds.home_implied == pytest.approx(home_raw / total)
    assert odds.away_implied == pytest.approx(away_raw / total)
    assert odds.provider == "DraftKings"
    assert odds.details == "NY -4.5"
    assert odds.home_implied + odds.away_implied == pytest.approx(1.0)
    assert home_raw + away_raw > 1.0  # the vig existed before normalization


def test_parse_returns_none_when_no_odds_sections() -> None:
    assert parse_market_odds(b'{"pickcenter": [], "odds": []}') is None
    assert parse_market_odds(b"{}") is None


def test_parse_returns_none_when_market_is_off_the_board() -> None:
    payload = json.dumps(
        {
            "pickcenter": [
                {
                    "provider": {"id": "100", "name": "DraftKings", "priority": 1},
                    "details": "ATL -10.5",
                    "homeTeamOdds": {"favorite": False, "teamId": "27"},
                    "awayTeamOdds": {"favorite": True, "teamId": "1"},
                    "moneyline": {
                        "home": {"close": {"odds": "OFF"}, "open": {"odds": "OFF"}},
                        "away": {"close": {"odds": "OFF"}, "open": {"odds": "OFF"}},
                    },
                }
            ],
            "odds": [],
        }
    ).encode("utf-8")
    assert parse_market_odds(payload) is None


def test_parse_falls_back_to_nested_close_then_open_lines() -> None:
    close_only = json.dumps(
        {
            "pickcenter": [
                {
                    "provider": {"name": "DraftKings"},
                    "moneyline": {
                        "home": {"close": {"odds": "-198"}, "open": {"odds": "-210"}},
                        "away": {"close": {"odds": "+164"}, "open": {"odds": "+175"}},
                    },
                }
            ]
        }
    ).encode("utf-8")
    odds = parse_market_odds(close_only)
    assert odds is not None
    home_raw = 198 / 298
    away_raw = 100 / 264
    total = home_raw + away_raw
    assert odds.home_implied == pytest.approx(home_raw / total)
    open_only = json.dumps(
        {
            "odds": [
                {
                    "provider": {"name": "DraftKings"},
                    "moneyline": {
                        "home": {"open": {"odds": "-210"}},
                        "away": {"open": {"odds": "+175"}},
                    },
                }
            ]
        }
    ).encode("utf-8")
    odds = parse_market_odds(open_only)
    assert odds is not None
    home_raw = 210 / 310
    away_raw = 100 / 275
    total = home_raw + away_raw
    assert odds.home_implied == pytest.approx(home_raw / total)


def test_parse_rejects_malformed_payload() -> None:
    with pytest.raises(NbaMarketError):
        parse_market_odds(b"not json")
    with pytest.raises(NbaMarketError):
        parse_market_odds(b"\xff\xfe")


def test_load_market_probabilities_joins_manifest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "summary-401809238.json").write_bytes(REALISTIC_PAYLOAD)
    (raw_dir / "summary-401809243.json").write_bytes(b'{"pickcenter": [], "odds": []}')
    manifest = {
        "schema_version": 1,
        "games": [
            {"synthetic_game_id": 22500001, "event_id": "401809238"},
            {"synthetic_game_id": 22500002, "event_id": "401809243"},
            {"synthetic_game_id": 22500003, "event_id": "401809999"},
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    probabilities = load_market_probabilities(raw_dir, manifest_path)
    home_raw = 180 / 280
    away_raw = 100 / 250
    assert probabilities == {22500001: pytest.approx(home_raw / (home_raw + away_raw))}
