"""ESPN pregame moneyline odds as a declared market benchmark, never a model feature.

The project's frozen rule is that market odds may only be a DECLARED evaluation
benchmark; they are never joined into model features. Pregame moneylines come from the
retained ESPN summary payloads (``data/raw/espn/raw/summary-<event_id>.json``). In the
retained archive the ``pickcenter`` list carries the sportsbook entries while ``odds``
is always empty, but both shapes are supported because ESPN documents both. "Pregame"
means the last price captured before tipoff: the ``moneyLine`` fields on
``homeTeamOdds``/``awayTeamOdds`` (which mirror the nested ``moneyline.*.close.odds``
strings), falling back to the opener when only it survives. Entries whose market is
``OFF`` carry no moneyline and are skipped. American moneylines convert to implied
probabilities (negative line ``-m``: ``m/(m+100)``; positive line ``+m``:
``100/(m+100)``) and the two sides are de-vigged by normalizing them to sum to one.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import cast

from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_float,
    require_list,
    require_object,
    require_string,
    required_field,
)


class NbaMarketError(ValueError):
    """Raised when a retained ESPN market payload or manifest is malformed."""


@dataclass(frozen=True, slots=True)
class MarketOdds:
    """De-vigged pregame home/away implied probabilities with provenance."""

    home_implied: float
    away_implied: float
    provider: str
    details: str

    def __post_init__(self) -> None:
        for field_name in ("home_implied", "away_implied"):
            value = getattr(self, field_name)
            if not isfinite(value) or not 0.0 < value < 1.0:
                raise NbaMarketError(f"{field_name} must be a probability, got {value!r}")
        if not self.provider.strip():
            raise NbaMarketError("provider must be non-empty")


def american_to_implied_probability(money_line: float) -> float:
    """Convert an American moneyline to its raw (vigged) implied probability."""
    line = require_float(money_line, "money_line")
    if line == 0.0:
        raise NbaMarketError("money_line of 0 is not a valid American line")
    if line < 0.0:
        return -line / (-line + 100.0)
    return 100.0 / (line + 100.0)


def parse_market_odds(payload: bytes) -> MarketOdds | None:
    """Extract pregame moneyline odds from one retained ESPN summary payload.

    Returns ``None`` when the payload carries no usable moneyline (an empty
    ``pickcenter``/``odds``, or entries whose market is off the board).
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NbaMarketError("ESPN summary payload is not UTF-8") from error
    try:
        document = parse_json_object(text)
    except JsonFormatError as error:
        raise NbaMarketError("malformed ESPN summary payload") from error
    for field_name in ("pickcenter", "odds"):
        entries = document.get(field_name)
        if not isinstance(entries, list):
            continue
        for entry in require_list(cast(list[object], entries), field_name):
            odds = _odds_from_entry(entry)
            if odds is not None:
                return odds
    return None


def load_market_probabilities(raw_dir: Path, manifest_path: Path) -> dict[int, float]:
    """Join retained summary odds to synthetic game IDs via the ESPN manifest.

    Returns ``synthetic_game_id -> de-vigged home implied probability`` for every
    manifest game whose retained summary payload carries a usable moneyline; games
    without odds are skipped so callers can count coverage honestly.
    """
    try:
        manifest = parse_json_object(manifest_path.read_text(encoding="utf-8"))
        games = require_list(required_field(manifest, "games"), "games")
    except JsonFormatError as error:
        raise NbaMarketError(f"malformed ESPN manifest: {manifest_path}") from error
    probabilities: dict[int, float] = {}
    for index, game_value in enumerate(games):
        game = require_object(game_value, f"games[{index}]")
        event_id = require_string(required_field(game, "event_id"), "event_id")
        synthetic_game_id = require_float(
            required_field(game, "synthetic_game_id"), "synthetic_game_id"
        )
        if synthetic_game_id != int(synthetic_game_id) or synthetic_game_id <= 0:
            raise NbaMarketError(f"synthetic_game_id must be a positive integer: {game!r}")
        summary_path = raw_dir / f"summary-{event_id}.json"
        if not summary_path.is_file():
            continue
        odds = parse_market_odds(summary_path.read_bytes())
        if odds is not None:
            probabilities[int(synthetic_game_id)] = odds.home_implied
    return probabilities


def _odds_from_entry(entry: object) -> MarketOdds | None:
    if not isinstance(entry, dict):
        return None
    mapping = cast(dict[str, object], entry)
    home_line = _side_money_line(mapping, "homeTeamOdds", "home")
    away_line = _side_money_line(mapping, "awayTeamOdds", "away")
    if home_line is None or away_line is None:
        return None
    home_raw = american_to_implied_probability(home_line)
    away_raw = american_to_implied_probability(away_line)
    total = home_raw + away_raw
    provider = mapping.get("provider")
    provider_name = "unknown"
    if isinstance(provider, dict):
        name = cast(dict[str, object], provider).get("name")
        if isinstance(name, str) and name.strip():
            provider_name = name.strip()
    details = mapping.get("details")
    return MarketOdds(
        home_implied=home_raw / total,
        away_implied=away_raw / total,
        provider=provider_name,
        details=details.strip() if isinstance(details, str) else "",
    )


def _side_money_line(entry: dict[str, object], team_key: str, side: str) -> float | None:
    team_odds = entry.get(team_key)
    if isinstance(team_odds, dict):
        line = cast(dict[str, object], team_odds).get("moneyLine")
        if isinstance(line, int | float) and not isinstance(line, bool) and line != 0:
            return float(line)
    nested = entry.get("moneyline")
    if isinstance(nested, dict):
        side_book = cast(dict[str, object], nested).get(side)
        if isinstance(side_book, dict):
            for phase in ("close", "open"):
                line = _string_money_line(cast(dict[str, object], side_book).get(phase))
                if line is not None:
                    return line
    return None


def _string_money_line(book: object) -> float | None:
    if not isinstance(book, dict):
        return None
    odds = cast(dict[str, object], book).get("odds")
    if not isinstance(odds, str):
        return None
    try:
        line = int(odds.strip().removeprefix("+"))
    except ValueError:
        return None
    return float(line) if line != 0 else None
