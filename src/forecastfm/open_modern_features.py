"""Outcome-free, point-in-time features for the open-modern NBA cohort."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha1, sha256
from itertools import groupby
from math import isclose, isfinite, log
from pathlib import Path
from types import MappingProxyType

from forecastfm.integrity import canonical_sha256
from forecastfm.open_modern import (
    DEVELOPMENT_COLUMNS,
    OPEN_MODERN_DEVELOPMENT_SHA256,
    OPEN_MODERN_TEST_INPUTS_SHA256,
    TEST_INPUT_COLUMNS,
    OpenModernError,
    require_open_modern_development,
    require_open_modern_test_inputs,
)

RAPTOR_SOURCE_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/"
    "4c1ff5e3aef1816ae04af63218015066e186c147/nba-raptor/modern_RAPTOR_by_team.csv"
)
RAPTOR_SOURCE_COMMIT = "4c1ff5e3aef1816ae04af63218015066e186c147"
RAPTOR_SOURCE_GIT_BLOB = "7e9f47d175de2a0f86b04bfd175597477c6ae26d"
RAPTOR_SOURCE_SHA256 = "a80bb5d24eb6b9742bb0c68aacf643144e7d39311b3b5aa12199b63d8d7de2aa"
RAPTOR_SOURCE_BYTES = 1_922_974

RAPTOR_COLUMNS = (
    "player_name",
    "player_id",
    "season",
    "season_type",
    "team",
    "poss",
    "mp",
    "raptor_box_offense",
    "raptor_box_defense",
    "raptor_box_total",
    "raptor_onoff_offense",
    "raptor_onoff_defense",
    "raptor_onoff_total",
    "raptor_offense",
    "raptor_defense",
    "raptor_total",
    "war_total",
    "war_reg_season",
    "war_playoffs",
    "predator_offense",
    "predator_defense",
    "predator_total",
    "pace_impact",
)

TEAM_TO_BBREF: Mapping[str, str] = MappingProxyType(
    {
        "76ers": "PHI",
        "Bucks": "MIL",
        "Bulls": "CHI",
        "Cavaliers": "CLE",
        "Celtics": "BOS",
        "Clippers": "LAC",
        "Grizzlies": "MEM",
        "Hawks": "ATL",
        "Heat": "MIA",
        "Hornets": "CHA",
        "Jazz": "UTA",
        "Kings": "SAC",
        "Knicks": "NYK",
        "Lakers": "LAL",
        "Magic": "ORL",
        "Mavericks": "DAL",
        "Nets": "BRK",
        "Nuggets": "DEN",
        "Pacers": "IND",
        "Pelicans": "NOP",
        "Pistons": "DET",
        "Raptors": "TOR",
        "Rockets": "HOU",
        "Spurs": "SAS",
        "Suns": "PHO",
        "Thunder": "OKC",
        "Timberwolves": "MIN",
        "Trail Blazers": "POR",
        "Warriors": "GSW",
        "Wizards": "WAS",
    }
)

OPEN_MODERN_CAUSAL_FEATURE_NAMES = (
    "source_log_odds",
    "rest_days_difference",
    "back_to_back_difference",
    "games_last_7_difference",
    "trailing_10_opponent_probability_difference",
    "trailing_10_history_difference",
    "prior_season_raptor_total_difference",
)
OPEN_MODERN_CAUSAL_FEATURE_CONTRACT_SHA256 = canonical_sha256(
    {
        "schema_version": 1,
        "orientation": "team1 minus team2",
        "feature_order": list(OPEN_MODERN_CAUSAL_FEATURE_NAMES),
        "formulas": {
            "source_log_odds": "ln(prob1 / prob2)",
            "rest_days_difference": (
                "max((D - last_date).days - 1, 0), default 0, team1 minus team2"
            ),
            "back_to_back_difference": (
                "indicator((D - last_date).days == 1), default 0, team1 minus team2"
            ),
            "games_last_7_difference": (
                "count prior input games dated in [D-7, D), team1 minus team2"
            ),
            "trailing_10_opponent_probability_difference": (
                "mean opponent source probability over last 10 input games, default 0.5, "
                "team1 minus team2"
            ),
            "trailing_10_history_difference": (
                "min(strictly prior input game count, 10), team1 minus team2"
            ),
            "prior_season_raptor_total_difference": (
                "completed game-season-minus-one RS sum(poss * raptor_total) / sum(poss), "
                "team1 minus team2"
            ),
        },
        "history": {
            "source": "input probabilities and scheduled dates only",
            "same_date": "build every row before any history update",
            "season_boundary": "reset every team history",
        },
        "raptor": {
            "commit": RAPTOR_SOURCE_COMMIT,
            "git_blob_sha": RAPTOR_SOURCE_GIT_BLOB,
            "sha256": RAPTOR_SOURCE_SHA256,
            "byte_size": RAPTOR_SOURCE_BYTES,
            "season_type": "RS",
            "join_coverage": "required for both teams in every game",
        },
        "side_swap": "exact feature negation with signed zero normalized to positive zero",
        "realized_outcomes": "forbidden",
    }
)


class OpenModernFeatureError(OpenModernError):
    """Raised when a causal input or lagged feature source is invalid."""


@dataclass(frozen=True, slots=True)
class OpenModernInputGame:
    """One game containing only fields available before its result."""

    game_id: str
    season: int
    game_date: date
    team1: str
    team2: str
    prob1: float
    prob2: float

    def __post_init__(self) -> None:
        _validate_input_game(self)


@dataclass(frozen=True, slots=True)
class OpenModernCausalFeatures:
    """Team-one-minus-team-two values built without realized outcomes."""

    source_log_odds: float
    rest_days_difference: float
    back_to_back_difference: float
    games_last_7_difference: float
    trailing_10_opponent_probability_difference: float
    trailing_10_history_difference: float
    prior_season_raptor_total_difference: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in self.vector):
            raise OpenModernFeatureError("open-modern causal features must be finite")

    @property
    def vector(self) -> tuple[float, ...]:
        """Return values in stable model order."""
        return (
            self.source_log_odds,
            self.rest_days_difference,
            self.back_to_back_difference,
            self.games_last_7_difference,
            self.trailing_10_opponent_probability_difference,
            self.trailing_10_history_difference,
            self.prior_season_raptor_total_difference,
        )

    def as_dict(self) -> dict[str, float]:
        """Return a readable mapping in stable model order."""
        return dict(zip(OPEN_MODERN_CAUSAL_FEATURE_NAMES, self.vector, strict=True))

    def side_swap(self) -> OpenModernCausalFeatures:
        """Swap the listed sides by exactly negating every feature."""
        return _features_from_vector(tuple(_negate(value) for value in self.vector))


@dataclass(frozen=True, slots=True)
class OpenModernFeatureRow:
    """One game identifier paired with its point-in-time feature vector."""

    game_id: str
    season: int
    game_date: date
    team1: str
    team2: str
    features: OpenModernCausalFeatures


@dataclass(frozen=True, slots=True)
class _PriorGame:
    game_date: date
    opponent_probability: float


def load_open_modern_feature_inputs(
    path: Path,
    *,
    seal_path: Path,
    protocol_path: Path,
    exposure_path: Path,
) -> tuple[OpenModernInputGame, ...]:
    """Load only pregame fields from a verified development or test-input CSV."""
    payload = path.read_bytes()
    header = _require_verified_input(path, payload, seal_path, protocol_path, exposure_path)
    return _read_input_fields(payload, header)


def load_prior_season_raptor(
    path: Path,
    *,
    max_allowed_season: int,
) -> dict[tuple[int, str], float]:
    """Load pinned regular-season RAPTOR through an explicit completed season."""
    payload = path.read_bytes()
    _require_pinned_raptor(payload)
    totals = _read_raptor_totals(payload, max_allowed_season)
    result = {
        key: weighted_sum / possessions
        for key, (weighted_sum, possessions) in totals.items()
        if possessions > 0.0
    }
    _require_full_raptor_seasons(result)
    return result


def build_open_modern_features(
    games: Sequence[OpenModernInputGame],
    prior_season_raptor: Mapping[tuple[int, str], float],
) -> tuple[OpenModernFeatureRow, ...]:
    """Build causal features, batching every date before updating histories."""
    ordered = tuple(sorted(games, key=_input_sort_key))
    _require_unique_input_games(ordered)
    _require_raptor_coverage(ordered, prior_season_raptor)

    rows: list[OpenModernFeatureRow] = []
    for _, season_games in groupby(ordered, key=lambda game: game.season):
        rows.extend(_build_season_features(tuple(season_games), prior_season_raptor))
    return tuple(rows)


def _build_season_features(
    games: Sequence[OpenModernInputGame],
    prior_season_raptor: Mapping[tuple[int, str], float],
) -> list[OpenModernFeatureRow]:
    histories: dict[str, list[_PriorGame]] = {}
    rows: list[OpenModernFeatureRow] = []
    for _, date_games in groupby(games, key=lambda game: game.game_date):
        batch = tuple(date_games)
        rows.extend(_feature_row(game, histories, prior_season_raptor) for game in batch)
        for game in batch:
            _update_histories(game, histories)
    return rows


def _feature_row(
    game: OpenModernInputGame,
    histories: Mapping[str, Sequence[_PriorGame]],
    raptor: Mapping[tuple[int, str], float],
) -> OpenModernFeatureRow:
    team_history = histories.get(game.team1, ())
    opponent_history = histories.get(game.team2, ())
    features = OpenModernCausalFeatures(
        source_log_odds=log(game.prob1 / game.prob2),
        rest_days_difference=_difference(
            _rest_days(team_history, game.game_date),
            _rest_days(opponent_history, game.game_date),
        ),
        back_to_back_difference=_difference(
            _back_to_back(team_history, game.game_date),
            _back_to_back(opponent_history, game.game_date),
        ),
        games_last_7_difference=_difference(
            _games_last_7(team_history, game.game_date),
            _games_last_7(opponent_history, game.game_date),
        ),
        trailing_10_opponent_probability_difference=_difference(
            _trailing_opponent_probability(team_history),
            _trailing_opponent_probability(opponent_history),
        ),
        trailing_10_history_difference=_difference(
            min(len(team_history), 10),
            min(len(opponent_history), 10),
        ),
        prior_season_raptor_total_difference=_raptor_difference(game, raptor),
    )
    return OpenModernFeatureRow(
        game_id=game.game_id,
        season=game.season,
        game_date=game.game_date,
        team1=game.team1,
        team2=game.team2,
        features=features,
    )


def _update_histories(
    game: OpenModernInputGame,
    histories: dict[str, list[_PriorGame]],
) -> None:
    histories.setdefault(game.team1, []).append(_PriorGame(game.game_date, game.prob2))
    histories.setdefault(game.team2, []).append(_PriorGame(game.game_date, game.prob1))


def _rest_days(history: Sequence[_PriorGame], game_date: date) -> float:
    if not history:
        return 0.0
    return float(max((game_date - history[-1].game_date).days - 1, 0))


def _back_to_back(history: Sequence[_PriorGame], game_date: date) -> float:
    if not history:
        return 0.0
    return float((game_date - history[-1].game_date).days == 1)


def _games_last_7(history: Sequence[_PriorGame], game_date: date) -> float:
    start = game_date - timedelta(days=7)
    return float(sum(start <= prior.game_date < game_date for prior in history))


def _trailing_opponent_probability(history: Sequence[_PriorGame]) -> float:
    recent = history[-10:]
    if not recent:
        return 0.5
    return sum(game.opponent_probability for game in recent) / len(recent)


def _raptor_difference(
    game: OpenModernInputGame,
    raptor: Mapping[tuple[int, str], float],
) -> float:
    prior_season = game.season - 1
    team_value = raptor[(prior_season, TEAM_TO_BBREF[game.team1])]
    opponent_value = raptor[(prior_season, TEAM_TO_BBREF[game.team2])]
    return _difference(team_value, opponent_value)


def _difference(team_value: float | int, opponent_value: float | int) -> float:
    value = float(team_value) - float(opponent_value)
    return 0.0 if value == 0.0 else value


def _negate(value: float) -> float:
    return 0.0 if value == 0.0 else -value


def _features_from_vector(values: tuple[float, ...]) -> OpenModernCausalFeatures:
    return OpenModernCausalFeatures(*values)


def _read_header(payload: bytes) -> tuple[str, ...]:
    with io.StringIO(_decode_csv(payload), newline="") as file:
        header = next(csv.reader(file), None)
    if header is None:
        raise OpenModernFeatureError("open-modern input CSV is empty")
    return tuple(header)


def _require_verified_input(
    path: Path,
    payload: bytes,
    seal_path: Path,
    protocol_path: Path,
    exposure_path: Path,
) -> tuple[str, ...]:
    # Verify the in-memory snapshot before asking the shared verifier to bind its controls.
    # Parsing then uses this same snapshot and never reopens the artifact path.
    header = _read_header(payload)
    if header == DEVELOPMENT_COLUMNS:
        if sha256(payload).hexdigest() != OPEN_MODERN_DEVELOPMENT_SHA256:
            raise OpenModernFeatureError("open-modern development snapshot does not match")
        require_open_modern_development(path, seal_path, protocol_path, exposure_path)
        return header
    if header == TEST_INPUT_COLUMNS:
        if sha256(payload).hexdigest() != OPEN_MODERN_TEST_INPUTS_SHA256:
            raise OpenModernFeatureError("open-modern test-input snapshot does not match")
        require_open_modern_test_inputs(path, seal_path, protocol_path, exposure_path)
        return header
    raise OpenModernFeatureError("open-modern input columns do not match a safe artifact")


def _read_input_fields(
    payload: bytes,
    header: tuple[str, ...],
) -> tuple[OpenModernInputGame, ...]:
    indexes = tuple(header.index(name) for name in TEST_INPUT_COLUMNS)
    games: list[OpenModernInputGame] = []
    with io.StringIO(_decode_csv(payload), newline="") as file:
        reader = csv.reader(file)
        next(reader)
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise OpenModernFeatureError(f"line {line_number}: column count does not match")
            values = tuple(row[index] for index in indexes)
            games.append(_parse_input_fields(values, line_number))
    ordered = tuple(sorted(games, key=_input_sort_key))
    _require_unique_input_games(ordered)
    return ordered


def _parse_input_fields(values: tuple[str, ...], line_number: int) -> OpenModernInputGame:
    game_id, season_text, date_text, team1, team2, prob1_text, prob2_text = values
    try:
        season = int(season_text)
        game_date = date.fromisoformat(date_text)
        prob1 = float(prob1_text)
        prob2 = float(prob2_text)
    except ValueError as error:
        raise OpenModernFeatureError(f"line {line_number}: input value is invalid") from error
    if str(season) != season_text:
        raise OpenModernFeatureError(f"line {line_number}: season is not canonical")
    return OpenModernInputGame(game_id, season, game_date, team1, team2, prob1, prob2)


def _validate_input_game(game: OpenModernInputGame) -> None:
    if not game.game_id or game.game_id != game.game_id.strip():
        raise OpenModernFeatureError("game ID must be present and trimmed")
    if game.game_date.year not in {game.season - 1, game.season}:
        raise OpenModernFeatureError("game date and season disagree")
    if game.team1 not in TEAM_TO_BBREF or game.team2 not in TEAM_TO_BBREF:
        raise OpenModernFeatureError("every team must have a fixed Basketball Reference code")
    if game.team1 == game.team2:
        raise OpenModernFeatureError("teams must differ")
    if not all(isfinite(value) and 0.0 < value < 1.0 for value in (game.prob1, game.prob2)):
        raise OpenModernFeatureError("pregame probabilities must be finite and interior")
    if not isclose(game.prob1 + game.prob2, 1.0, abs_tol=1e-9):
        raise OpenModernFeatureError("pregame probabilities must sum to one")


def _input_sort_key(game: OpenModernInputGame) -> tuple[int, date, str, str, str]:
    return (game.season, game.game_date, game.team1, game.team2, game.game_id)


def _require_unique_input_games(games: Sequence[OpenModernInputGame]) -> None:
    ids = {game.game_id for game in games}
    identities = {
        (game.season, game.game_date, *sorted((game.team1, game.team2))) for game in games
    }
    if len(ids) != len(games) or len(identities) != len(games):
        raise OpenModernFeatureError("open-modern input game identities must be unique")


def _require_raptor_coverage(
    games: Sequence[OpenModernInputGame],
    raptor: Mapping[tuple[int, str], float],
) -> None:
    required = {
        (game.season - 1, TEAM_TO_BBREF[team])
        for game in games
        for team in (game.team1, game.team2)
    }
    missing = sorted(required.difference(raptor))
    if missing:
        raise OpenModernFeatureError(f"prior-season RAPTOR join is incomplete: {missing[0]}")
    if not all(isfinite(raptor[key]) for key in required):
        raise OpenModernFeatureError("prior-season RAPTOR values must be finite")


def _require_pinned_raptor(payload: bytes) -> None:
    if len(payload) != RAPTOR_SOURCE_BYTES:
        raise OpenModernFeatureError("RAPTOR source byte size does not match")
    if sha256(payload).hexdigest() != RAPTOR_SOURCE_SHA256:
        raise OpenModernFeatureError("RAPTOR source SHA-256 does not match")
    prefix = f"blob {len(payload)}\0".encode()
    git_blob = sha1(prefix + payload, usedforsecurity=False).hexdigest()
    if git_blob != RAPTOR_SOURCE_GIT_BLOB:
        raise OpenModernFeatureError("RAPTOR source Git blob does not match")


def _read_raptor_totals(
    payload: bytes,
    max_allowed_season: int,
) -> dict[tuple[int, str], tuple[float, float]]:
    totals: dict[tuple[int, str], tuple[float, float]] = {}
    with io.StringIO(_decode_csv(payload), newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != RAPTOR_COLUMNS:
            raise OpenModernFeatureError("RAPTOR source columns do not match")
        for line_number, row in enumerate(reader, start=2):
            _add_raptor_row(totals, row, line_number, max_allowed_season)
    if not totals:
        raise OpenModernFeatureError("RAPTOR source has no allowed regular-season rows")
    return totals


def _decode_csv(payload: bytes) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OpenModernFeatureError("CSV source is not valid UTF-8") from error


def _add_raptor_row(
    totals: dict[tuple[int, str], tuple[float, float]],
    row: Mapping[str, str | None],
    line_number: int,
    max_allowed_season: int,
) -> None:
    season = _raptor_integer(row.get("season"), "season", line_number)
    if season > max_allowed_season or row.get("season_type") != "RS":
        return
    team = row.get("team")
    if team is None:
        raise OpenModernFeatureError(f"line {line_number}: RAPTOR team is missing")
    if team not in TEAM_TO_BBREF.values():
        raise OpenModernFeatureError(f"line {line_number}: RAPTOR team is unknown")
    possessions = _raptor_float(row.get("poss"), "possessions", line_number)
    value = _raptor_float(row.get("raptor_total"), "RAPTOR total", line_number)
    if possessions < 0.0:
        raise OpenModernFeatureError(f"line {line_number}: possessions must be nonnegative")
    key = (season, team)
    weighted_sum, possession_sum = totals.get(key, (0.0, 0.0))
    totals[key] = (weighted_sum + possessions * value, possession_sum + possessions)


def _raptor_integer(value: str | None, field: str, line_number: int) -> int:
    try:
        result = int(value or "")
    except ValueError as error:
        raise OpenModernFeatureError(f"line {line_number}: RAPTOR {field} is invalid") from error
    if str(result) != value:
        raise OpenModernFeatureError(f"line {line_number}: RAPTOR {field} is not canonical")
    return result


def _raptor_float(value: str | None, field: str, line_number: int) -> float:
    try:
        result = float(value or "")
    except ValueError as error:
        raise OpenModernFeatureError(f"line {line_number}: RAPTOR {field} is invalid") from error
    if not isfinite(result):
        raise OpenModernFeatureError(f"line {line_number}: RAPTOR {field} must be finite")
    return result


def _require_full_raptor_seasons(raptor: Mapping[tuple[int, str], float]) -> None:
    teams_by_season: dict[int, set[str]] = {}
    for season, team in raptor:
        teams_by_season.setdefault(season, set()).add(team)
    expected = set(TEAM_TO_BBREF.values())
    incomplete = [season for season, teams in teams_by_season.items() if teams != expected]
    if incomplete:
        raise OpenModernFeatureError(
            f"RAPTOR season has incomplete team coverage: {min(incomplete)}"
        )
