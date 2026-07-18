"""Canonical per-game Elo priors for original T-60 NBA feature rows.

This file alone binds and recomputes each probability from supplied pregame ratings.
``nba_elo_replay`` proves the deterministic causal replay; the licensed connector must
still prove complete schedule coverage and raw team/site derivation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import isfinite
from pathlib import Path

from forecastfm.integrity import canonical_json, canonical_sha256
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_string,
    required_field,
)
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_feature_rows import NBA_PRIMARY_STATE_ID, NbaRichFeatureRow

NBA_ELO_STATE_SCHEMA_VERSION = 1

_HASH_CHARACTERS = frozenset("0123456789abcdef")
_RATING_MINIMUM = 0.0
_RATING_MAXIMUM = 4_000.0
_HOME_ADVANTAGE_LIMIT = 500.0
_RATING_SCALE_MINIMUM = 1.0
_RATING_SCALE_MAXIMUM = 2_000.0
_LOG10_ODDS_LIMIT = 12.0
_RECORD_KEYS = {
    "schema_version",
    "state_id",
    "question_id",
    "available_at",
    "team_rating",
    "opponent_rating",
    "home_advantage",
    "rating_scale",
    "recipe_sha256",
    "team_win_probability",
    "state_sha256",
}


class NbaEloStateError(ValueError):
    """Raised when a sealed Elo state violates its deterministic contract."""


@dataclass(frozen=True, slots=True)
class NbaEloState:
    """One original T-60 prior bound to ratings and a frozen Elo recipe."""

    question_id: str
    available_at: datetime
    team_rating: float
    opponent_rating: float
    home_advantage: float
    rating_scale: float
    recipe_sha256: str
    state_id: str = field(default=NBA_PRIMARY_STATE_ID, init=False)

    def __post_init__(self) -> None:
        _require_original_question_id(self.question_id)
        _require_utc(self.available_at, "available_at")
        _validate_elo_numbers(
            self.team_rating,
            self.opponent_rating,
            self.home_advantage,
            self.rating_scale,
        )
        _require_sha256(self.recipe_sha256, "recipe_sha256")

    @property
    def team_win_probability(self) -> float:
        """Recompute the interior team probability with base-10 Elo odds."""
        return _probability_from_validated_values(
            self.team_rating,
            self.opponent_rating,
            self.home_advantage,
            self.rating_scale,
        )

    @property
    def state_sha256(self) -> str:
        """Hash the exact state inputs and schema deterministically."""
        return canonical_sha256(self.state_input_payload())

    def state_input_payload(self) -> dict[str, object]:
        """Return the exact causal inputs covered by ``state_sha256``."""
        return {
            "schema_version": NBA_ELO_STATE_SCHEMA_VERSION,
            "state_id": self.state_id,
            "question_id": self.question_id,
            "available_at": _utc_text(self.available_at),
            "team_rating": self.team_rating,
            "opponent_rating": self.opponent_rating,
            "home_advantage": self.home_advantage,
            "rating_scale": self.rating_scale,
            "recipe_sha256": self.recipe_sha256,
        }

    def canonical_payload(self) -> dict[str, object]:
        """Return the complete canonical JSONL record."""
        return {
            **self.state_input_payload(),
            "team_win_probability": self.team_win_probability,
            "state_sha256": self.state_sha256,
        }


def base10_elo_team_probability(
    team_rating: float,
    opponent_rating: float,
    home_advantage: float,
    rating_scale: float,
) -> float:
    """Return a validated, interior base-10 Elo team probability."""
    _validate_elo_numbers(
        team_rating,
        opponent_rating,
        home_advantage,
        rating_scale,
    )
    return _probability_from_validated_values(
        team_rating,
        opponent_rating,
        home_advantage,
        rating_scale,
    )


def write_nba_elo_states_jsonl(path: Path, states: Iterable[NbaEloState]) -> None:
    """Create immutable canonical JSONL containing nonempty original T-60 states."""
    checked = _require_original_states(tuple(states))
    text = _canonical_jsonl(checked)
    try:
        with path.open("x", encoding="utf-8", newline="") as file:
            file.write(text)
    except FileExistsError as error:
        raise NbaEloStateError("NBA Elo-state JSONL already exists") from error
    except OSError as error:
        raise NbaEloStateError("cannot write NBA Elo-state JSONL") from error


def read_nba_elo_states_jsonl(path: Path) -> tuple[NbaEloState, ...]:
    """Read strict canonical JSONL and recompute every probability and digest."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise NbaEloStateError("cannot read NBA Elo-state JSONL") from error

    states: list[NbaEloState] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            states.append(_state_from_payload(parse_json_object(line)))
        except ValueError as error:
            message = f"invalid NBA Elo state at line {line_number}"
            raise NbaEloStateError(message) from error

    checked = _require_original_states(tuple(states))
    if text != _canonical_jsonl(checked):
        raise NbaEloStateError("NBA Elo states must use canonical JSONL encoding")
    return checked


def validate_elo_states_against_feature_rows(
    states: Sequence[NbaEloState],
    rows: Sequence[NbaRichFeatureRow],
    *,
    action_at: datetime,
) -> None:
    """Require exact state/row order and matching probability, time, and digest."""
    _require_utc(action_at, "action_at")
    checked = _require_original_states(tuple(states))
    state_ids = tuple(state.question_id for state in checked)
    row_ids = tuple(row.question_id for row in rows)
    if state_ids != row_ids:
        raise NbaEloStateError("Elo states and feature rows must have identical IDs and order")

    for state, row in zip(checked, rows, strict=True):
        if state.available_at > action_at:
            raise NbaEloStateError("Elo state cannot postdate the protected action")
        if state.available_at != row.elo_available_at:
            raise NbaEloStateError("Elo availability differs from the feature row")
        if state.state_sha256 != row.elo_state_sha256:
            raise NbaEloStateError("Elo state digest differs from the feature row")
        if state.team_win_probability != row.elo_team_win_probability:
            raise NbaEloStateError("Elo team probability differs from the feature row")
        if 1.0 - state.team_win_probability != row.elo_opponent_win_probability:
            raise NbaEloStateError("Elo opponent probability differs from the feature row")


def _state_from_payload(payload: Mapping[str, object]) -> NbaEloState:
    require_exact_keys(payload, _RECORD_KEYS, "NBA Elo state")
    version = required_field(payload, "schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise JsonFormatError("schema_version must be an integer")
    if version != NBA_ELO_STATE_SCHEMA_VERSION:
        raise JsonFormatError(f"unsupported NBA Elo-state schema version: {version}")
    state_id = _string_field(payload, "state_id")
    if state_id != NBA_PRIMARY_STATE_ID:
        raise JsonFormatError(f"state_id must equal {NBA_PRIMARY_STATE_ID}")

    state = NbaEloState(
        question_id=_string_field(payload, "question_id"),
        available_at=_time_field(payload, "available_at"),
        team_rating=_float_field(payload, "team_rating"),
        opponent_rating=_float_field(payload, "opponent_rating"),
        home_advantage=_float_field(payload, "home_advantage"),
        rating_scale=_float_field(payload, "rating_scale"),
        recipe_sha256=_string_field(payload, "recipe_sha256"),
    )
    stored_probability = _float_field(payload, "team_win_probability")
    if stored_probability != state.team_win_probability:
        raise JsonFormatError("team_win_probability does not match the Elo inputs")
    stored_sha256 = _string_field(payload, "state_sha256")
    if stored_sha256 != state.state_sha256:
        raise JsonFormatError("state_sha256 does not match the Elo inputs")
    return state


def _canonical_jsonl(states: Sequence[NbaEloState]) -> str:
    return "".join(f"{canonical_json(state.canonical_payload())}\n" for state in states)


def _require_original_states(states: tuple[NbaEloState, ...]) -> tuple[NbaEloState, ...]:
    if not states:
        raise NbaEloStateError("NBA Elo-state JSONL must not be empty")
    question_ids = tuple(state.question_id for state in states)
    if len(set(question_ids)) != len(question_ids):
        raise NbaEloStateError("NBA Elo-state JSONL contains a duplicate question ID")
    return states


def _require_original_question_id(question_id: object) -> None:
    if not isinstance(question_id, str) or not question_id.strip():
        raise NbaEloStateError("question_id must be a nonempty string")
    if question_id.endswith(SIDE_SWAP_SUFFIX):
        raise NbaEloStateError("NBA Elo states may contain only original question IDs")


def _validate_elo_numbers(
    team_rating: object,
    opponent_rating: object,
    home_advantage: object,
    rating_scale: object,
) -> None:
    team = _require_float(team_rating, "team_rating")
    opponent = _require_float(opponent_rating, "opponent_rating")
    home = _require_float(home_advantage, "home_advantage")
    scale = _require_float(rating_scale, "rating_scale")
    if not _RATING_MINIMUM <= team <= _RATING_MAXIMUM:
        raise NbaEloStateError("team_rating is outside the supported range")
    if not _RATING_MINIMUM <= opponent <= _RATING_MAXIMUM:
        raise NbaEloStateError("opponent_rating is outside the supported range")
    if not -_HOME_ADVANTAGE_LIMIT <= home <= _HOME_ADVANTAGE_LIMIT:
        raise NbaEloStateError("home_advantage is outside the supported range")
    if not _RATING_SCALE_MINIMUM <= scale <= _RATING_SCALE_MAXIMUM:
        raise NbaEloStateError("rating_scale is outside the supported range")
    log10_odds = (team + home - opponent) / scale
    if abs(log10_odds) > _LOG10_ODDS_LIMIT:
        raise NbaEloStateError("Elo inputs are too extreme for an interior probability")


def _probability_from_validated_values(
    team_rating: float,
    opponent_rating: float,
    home_advantage: float,
    rating_scale: float,
) -> float:
    log10_odds = (team_rating + home_advantage - opponent_rating) / rating_scale
    probability = 1.0 / (1.0 + 10.0 ** (-log10_odds))
    if not 0.0 < probability < 1.0:
        raise NbaEloStateError("Elo probability must be interior")
    return probability


def _require_float(value: object, field_name: str) -> float:
    if not isinstance(value, float) or not isfinite(value):
        raise NbaEloStateError(f"{field_name} must be a finite float")
    return value


def _require_sha256(value: object, field_name: str) -> None:
    if not isinstance(value, str):
        raise NbaEloStateError(f"{field_name} must be a string")
    if len(value) != 64 or any(character not in _HASH_CHARACTERS for character in value):
        raise NbaEloStateError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_utc(value: object, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise NbaEloStateError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise NbaEloStateError(f"{field_name} must be in UTC")


def _utc_text(value: datetime) -> str:
    _require_utc(value, "datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _string_field(payload: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(payload, field_name), field_name)


def _float_field(payload: Mapping[str, object], field_name: str) -> float:
    value = required_field(payload, field_name)
    if not isinstance(value, float) or not isfinite(value):
        raise JsonFormatError(f"{field_name} must be a finite JSON float")
    return value


def _time_field(payload: Mapping[str, object], field_name: str) -> datetime:
    text = _string_field(payload, field_name)
    if not text.endswith("Z"):
        raise JsonFormatError(f"{field_name} must use canonical UTC Z notation")
    try:
        parsed = datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise JsonFormatError(f"{field_name} must be an ISO 8601 datetime") from error
    _require_utc(parsed, field_name)
    if text != _utc_text(parsed):
        raise JsonFormatError(f"{field_name} must use canonical UTC notation")
    return parsed.astimezone(UTC)
