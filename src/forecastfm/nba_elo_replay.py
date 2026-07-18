"""Deterministic, causal NBA Elo replay from a sealed schedule and final scores.

The replay proves how supplied schedule rows and resolutions produce Elo states. It
does not prove that the rows cover the full NBA schedule or correctly encode each
team and site. Those facts still depend on the licensed connector and a separately
sealed schedule artifact.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import copysign, isfinite
from pathlib import Path
from typing import Literal

from forecastfm.integrity import canonical_json, canonical_sha256
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_string,
    required_field,
)
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_elo_state import NbaEloState, NbaEloStateError
from forecastfm.nba_feature_rows import NBA_PRIMARY_STATE_ID
from forecastfm.nba_resolutions import NbaResolution

NBA_ELO_RECIPE_SCHEMA_VERSION = 1
NBA_ELO_REPLAY_ROW_SCHEMA_VERSION = 1

type NbaGameSite = Literal["home", "away", "neutral"]

_ALLOWED_SITES = frozenset({"home", "away", "neutral"})
_PRIMARY_STATE_OFFSET = timedelta(minutes=60)
_RATING_MINIMUM = 0.0
_RATING_MAXIMUM = 4_000.0
_K_FACTOR_MAXIMUM = 400.0
_RATING_SCALE_MINIMUM = 1.0
_RATING_SCALE_MAXIMUM = 2_000.0
_HOME_ADVANTAGE_MAXIMUM = 500.0
_ROW_KEYS = {
    "schema_version",
    "question_id",
    "source_game_id",
    "season",
    "team_id",
    "opponent_id",
    "site",
    "forecast_cutoff",
    "scheduled_tipoff",
}


class NbaEloReplayError(ValueError):
    """Raised when an NBA Elo replay input or output violates its contract."""


@dataclass(frozen=True, slots=True)
class NbaEloRecipe:
    """Frozen numerical choices for one base-10 Elo replay."""

    initial_rating: float
    k_factor: float
    rating_scale: float
    home_advantage: float

    def __post_init__(self) -> None:
        _require_float_range(
            self.initial_rating,
            "initial_rating",
            minimum=_RATING_MINIMUM,
            maximum=_RATING_MAXIMUM,
        )
        _require_float_range(
            self.k_factor,
            "k_factor",
            minimum=0.0,
            maximum=_K_FACTOR_MAXIMUM,
            minimum_is_open=True,
        )
        _require_float_range(
            self.rating_scale,
            "rating_scale",
            minimum=_RATING_SCALE_MINIMUM,
            maximum=_RATING_SCALE_MAXIMUM,
        )
        _require_float_range(
            self.home_advantage,
            "home_advantage",
            minimum=0.0,
            maximum=_HOME_ADVANTAGE_MAXIMUM,
        )

    def canonical_payload(self) -> dict[str, object]:
        """Return the exact recipe inputs covered by ``recipe_sha256``."""
        return {
            "schema_version": NBA_ELO_RECIPE_SCHEMA_VERSION,
            "initial_rating": self.initial_rating,
            "k_factor": self.k_factor,
            "rating_scale": self.rating_scale,
            "home_advantage": self.home_advantage,
        }

    @property
    def recipe_sha256(self) -> str:
        """Return the canonical digest of the complete Elo recipe."""
        return canonical_sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class NbaEloReplayRow:
    """One original T-60 game needed to replay a pregame Elo state."""

    question_id: str
    source_game_id: str
    season: int
    team_id: str
    opponent_id: str
    site: NbaGameSite
    forecast_cutoff: datetime
    scheduled_tipoff: datetime

    def __post_init__(self) -> None:
        _require_id(self.question_id, "question_id")
        if self.question_id.endswith(SIDE_SWAP_SUFFIX):
            raise NbaEloReplayError("replay rows may contain only original question IDs")
        _require_id(self.source_game_id, "source_game_id")
        _require_season(self.season)
        _require_id(self.team_id, "team_id")
        _require_id(self.opponent_id, "opponent_id")
        if self.team_id == self.opponent_id:
            raise NbaEloReplayError("team_id and opponent_id must differ")
        if self.site not in _ALLOWED_SITES:
            raise NbaEloReplayError("site must be home, away, or neutral")
        _require_utc(self.forecast_cutoff, "forecast_cutoff")
        _require_utc(self.scheduled_tipoff, "scheduled_tipoff")
        if self.scheduled_tipoff - self.forecast_cutoff != _PRIMARY_STATE_OFFSET:
            raise NbaEloReplayError(f"replay rows must use the {NBA_PRIMARY_STATE_ID} cutoff")

    def canonical_payload(self) -> dict[str, object]:
        """Return the complete canonical replay-row record."""
        return {
            "schema_version": NBA_ELO_REPLAY_ROW_SCHEMA_VERSION,
            "question_id": self.question_id,
            "source_game_id": self.source_game_id,
            "season": self.season,
            "team_id": self.team_id,
            "opponent_id": self.opponent_id,
            "site": self.site,
            "forecast_cutoff": _utc_text(self.forecast_cutoff),
            "scheduled_tipoff": _utc_text(self.scheduled_tipoff),
        }


@dataclass(frozen=True, slots=True)
class _PendingResult:
    row: NbaEloReplayRow
    resolution: NbaResolution
    expected_team_score: float


def replay_nba_elo_states(
    rows: Sequence[NbaEloReplayRow],
    resolutions: Sequence[NbaResolution],
    recipe: NbaEloRecipe,
) -> tuple[NbaEloState, ...]:
    """Replay exact original T-60 Elo states without using future results."""
    checked_rows = _require_replay_rows(tuple(rows))
    resolution_by_question = _align_resolutions(checked_rows, tuple(resolutions))
    _require_clean_season_boundaries(checked_rows, resolution_by_question)

    states: list[NbaEloState] = []
    start = 0
    while start < len(checked_rows):
        season = checked_rows[start].season
        end = start + 1
        while end < len(checked_rows) and checked_rows[end].season == season:
            end += 1
        states.extend(
            _replay_one_season(
                checked_rows[start:end],
                resolution_by_question,
                recipe,
            )
        )
        start = end
    return tuple(states)


def validate_nba_elo_replay_states(
    rows: Sequence[NbaEloReplayRow],
    resolutions: Sequence[NbaResolution],
    recipe: NbaEloRecipe,
    states: Sequence[NbaEloState],
) -> None:
    """Require supplied states to equal a fresh replay in value and canonical bytes."""
    expected = replay_nba_elo_states(rows, resolutions, recipe)
    supplied = tuple(states)
    if len(supplied) != len(expected):
        raise NbaEloReplayError("supplied Elo-state count differs from the replay")
    for index, (actual, wanted) in enumerate(zip(supplied, expected, strict=True)):
        actual_bytes = canonical_json(actual.canonical_payload()).encode("utf-8")
        wanted_bytes = canonical_json(wanted.canonical_payload()).encode("utf-8")
        if actual_bytes != wanted_bytes:
            raise NbaEloReplayError(f"supplied Elo state differs from replay at index {index}")


def write_nba_elo_replay_rows_jsonl(
    path: Path,
    rows: Iterable[NbaEloReplayRow],
) -> None:
    """Create immutable canonical JSONL containing one exact ordered replay input."""
    checked = _require_replay_rows(tuple(rows))
    text = _canonical_jsonl(checked)
    try:
        with path.open("x", encoding="utf-8", newline="") as file:
            file.write(text)
    except FileExistsError as error:
        raise NbaEloReplayError("NBA Elo replay-row JSONL already exists") from error
    except OSError as error:
        raise NbaEloReplayError("cannot write NBA Elo replay-row JSONL") from error


def read_nba_elo_replay_rows_jsonl(path: Path) -> tuple[NbaEloReplayRow, ...]:
    """Read strict canonical replay rows while preserving chronological order."""
    try:
        value = path.read_bytes()
    except OSError as error:
        raise NbaEloReplayError("cannot read NBA Elo replay-row JSONL") from error
    return read_nba_elo_replay_rows_jsonl_bytes(value)


def read_nba_elo_replay_rows_jsonl_bytes(value: bytes) -> tuple[NbaEloReplayRow, ...]:
    """Parse strict canonical replay rows from one captured immutable byte buffer."""
    try:
        text = value.decode("utf-8")
    except UnicodeError as error:
        raise NbaEloReplayError("NBA Elo replay rows must use UTF-8") from error

    rows: list[NbaEloReplayRow] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            rows.append(_row_from_payload(parse_json_object(line)))
        except ValueError as error:
            message = f"invalid NBA Elo replay row at line {line_number}"
            raise NbaEloReplayError(message) from error

    checked = _require_replay_rows(tuple(rows))
    if text != _canonical_jsonl(checked):
        raise NbaEloReplayError("NBA Elo replay rows must use canonical JSONL encoding")
    return checked


def _replay_one_season(
    rows: tuple[NbaEloReplayRow, ...],
    resolutions: Mapping[str, NbaResolution],
    recipe: NbaEloRecipe,
) -> tuple[NbaEloState, ...]:
    ratings: dict[str, float] = {}
    pending: list[_PendingResult] = []
    states: list[NbaEloState] = []
    start = 0
    while start < len(rows):
        cutoff = rows[start].forecast_cutoff
        _apply_available_results(pending, cutoff, ratings, recipe)
        end = start + 1
        while end < len(rows) and rows[end].forecast_cutoff == cutoff:
            end += 1

        batch = rows[start:end]
        # Freeze the whole cutoff batch before any of its results can update Elo.
        batch_states = tuple(_state_for_row(row, ratings, recipe) for row in batch)
        states.extend(batch_states)
        pending.extend(
            _PendingResult(row, resolutions[row.question_id], state.team_win_probability)
            for row, state in zip(batch, batch_states, strict=True)
        )
        start = end
    return tuple(states)


def _apply_available_results(
    pending: list[_PendingResult],
    cutoff: datetime,
    ratings: dict[str, float],
    recipe: NbaEloRecipe,
) -> None:
    available = sorted(
        (item for item in pending if item.resolution.resolved_at <= cutoff),
        key=lambda item: (item.resolution.resolved_at, item.row.question_id),
    )
    for item in available:
        actual_team_score = 1.0 if item.resolution.team_won else 0.0
        change = recipe.k_factor * (actual_team_score - item.expected_team_score)
        ratings[item.row.team_id] = _normalize_zero(ratings[item.row.team_id] + change)
        ratings[item.row.opponent_id] = _normalize_zero(ratings[item.row.opponent_id] - change)
    pending[:] = [item for item in pending if item.resolution.resolved_at > cutoff]


def _state_for_row(
    row: NbaEloReplayRow,
    ratings: dict[str, float],
    recipe: NbaEloRecipe,
) -> NbaEloState:
    team_rating = ratings.setdefault(row.team_id, recipe.initial_rating)
    opponent_rating = ratings.setdefault(row.opponent_id, recipe.initial_rating)
    try:
        return NbaEloState(
            question_id=row.question_id,
            available_at=row.forecast_cutoff,
            team_rating=team_rating,
            opponent_rating=opponent_rating,
            home_advantage=_effective_home_advantage(row.site, recipe.home_advantage),
            rating_scale=recipe.rating_scale,
            recipe_sha256=recipe.recipe_sha256,
        )
    except NbaEloStateError as error:
        raise NbaEloReplayError("replayed Elo state is outside its supported range") from error


def _effective_home_advantage(site: NbaGameSite, value: float) -> float:
    if site == "home":
        return value
    if site == "away" and value != 0.0:
        return -value
    return 0.0


def _normalize_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _require_replay_rows(
    rows: tuple[NbaEloReplayRow, ...],
) -> tuple[NbaEloReplayRow, ...]:
    if not rows:
        raise NbaEloReplayError("NBA Elo replay rows must not be empty")
    _require_unique_values((row.question_id for row in rows), "question_id")
    _require_unique_values((row.source_game_id for row in rows), "source_game_id")

    previous = rows[0]
    _require_unique_teams_at_cutoff(rows)
    for row in rows[1:]:
        if row.season < previous.season:
            raise NbaEloReplayError("replay seasons must be monotone")
        if row.forecast_cutoff < previous.forecast_cutoff:
            raise NbaEloReplayError("replay forecast cutoffs must be monotone")
        if row.season > previous.season and row.forecast_cutoff == previous.forecast_cutoff:
            raise NbaEloReplayError("a new season must begin at a later cutoff")
        previous = row
    return rows


def _require_unique_teams_at_cutoff(rows: Sequence[NbaEloReplayRow]) -> None:
    active: set[tuple[datetime, str]] = set()
    for row in rows:
        for team_id in (row.team_id, row.opponent_id):
            key = (row.forecast_cutoff, team_id)
            if key in active:
                raise NbaEloReplayError("a team cannot participate twice at one cutoff")
            active.add(key)


def _align_resolutions(
    rows: tuple[NbaEloReplayRow, ...],
    resolutions: tuple[NbaResolution, ...],
) -> dict[str, NbaResolution]:
    _require_unique_values((item.question_id for item in resolutions), "resolution question_id")
    _require_unique_values(
        (item.source_game_id for item in resolutions), "resolution source_game_id"
    )
    by_question = {item.question_id: item for item in resolutions}
    row_ids = {row.question_id for row in rows}
    missing = sorted(row_ids - by_question.keys())
    extra = sorted(by_question.keys() - row_ids)
    if missing or extra:
        raise NbaEloReplayError(
            f"resolutions differ from replay rows; missing={missing}, extra={extra}"
        )
    for row in rows:
        resolution = by_question[row.question_id]
        if resolution.source_game_id != row.source_game_id:
            raise NbaEloReplayError("resolution source_game_id differs from its replay row")
        if (
            resolution.team_id,
            resolution.opponent_id,
            resolution.site,
        ) != (row.team_id, row.opponent_id, row.site):
            raise NbaEloReplayError("resolution orientation differs from its replay row")
        if resolution.resolved_at <= row.scheduled_tipoff:
            raise NbaEloReplayError("result availability must be after the scheduled tipoff")
    return by_question


def _require_clean_season_boundaries(
    rows: tuple[NbaEloReplayRow, ...],
    resolutions: Mapping[str, NbaResolution],
) -> None:
    prior_season = rows[0].season
    prior_resolution_times: list[datetime] = []
    for row in rows:
        if row.season != prior_season:
            if max(prior_resolution_times) > row.forecast_cutoff:
                raise NbaEloReplayError("a season reset cannot precede prior result availability")
            prior_season = row.season
            prior_resolution_times = []
        prior_resolution_times.append(resolutions[row.question_id].resolved_at)


def _row_from_payload(payload: Mapping[str, object]) -> NbaEloReplayRow:
    require_exact_keys(payload, _ROW_KEYS, "NBA Elo replay row")
    version = _integer_field(payload, "schema_version")
    if version != NBA_ELO_REPLAY_ROW_SCHEMA_VERSION:
        raise JsonFormatError(f"unsupported NBA Elo replay-row schema version: {version}")
    return NbaEloReplayRow(
        question_id=_string_field(payload, "question_id"),
        source_game_id=_string_field(payload, "source_game_id"),
        season=_integer_field(payload, "season"),
        team_id=_string_field(payload, "team_id"),
        opponent_id=_string_field(payload, "opponent_id"),
        site=_site_field(payload),
        forecast_cutoff=_time_field(payload, "forecast_cutoff"),
        scheduled_tipoff=_time_field(payload, "scheduled_tipoff"),
    )


def _canonical_jsonl(rows: Sequence[NbaEloReplayRow]) -> str:
    return "".join(f"{canonical_json(row.canonical_payload())}\n" for row in rows)


def _require_unique_values(values: Iterable[str], field_name: str) -> None:
    items = tuple(values)
    if len(set(items)) != len(items):
        raise NbaEloReplayError(f"NBA Elo replay contains a duplicate {field_name}")


def _require_id(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip() or not value.isprintable():
        raise NbaEloReplayError(f"{field_name} must be a clean nonempty string")


def _require_season(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NbaEloReplayError("season must be a positive integer")


def _require_float_range(
    value: object,
    field_name: str,
    *,
    minimum: float,
    maximum: float,
    minimum_is_open: bool = False,
) -> None:
    if not isinstance(value, float) or not isfinite(value):
        raise NbaEloReplayError(f"{field_name} must be a finite float")
    if value == 0.0 and copysign(1.0, value) < 0.0:
        raise NbaEloReplayError(f"{field_name} cannot use negative zero")
    minimum_failed = value <= minimum if minimum_is_open else value < minimum
    if minimum_failed or value > maximum:
        raise NbaEloReplayError(f"{field_name} is outside the supported range")


def _require_utc(value: object, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise NbaEloReplayError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise NbaEloReplayError(f"{field_name} must be in UTC")


def _utc_text(value: datetime) -> str:
    _require_utc(value, "datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _string_field(payload: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(payload, field_name), field_name)


def _integer_field(payload: Mapping[str, object], field_name: str) -> int:
    value = required_field(payload, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise JsonFormatError(f"{field_name} must be an integer")
    return value


def _site_field(payload: Mapping[str, object]) -> NbaGameSite:
    value = _string_field(payload, "site")
    if value == "home":
        return "home"
    if value == "away":
        return "away"
    if value == "neutral":
        return "neutral"
    raise JsonFormatError("site must be home, away, or neutral")


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
