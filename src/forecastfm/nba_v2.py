"""Leakage-safe historical NBA features derived from the pinned Elo source.

Team identities are used only inside the chronological state machine. Public
records contain an opaque question ID, an Elo prior, and oriented numeric
features. Every game on one date is featurized before any result from that date
updates history.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from itertools import groupby
from math import isclose, isfinite, log
from pathlib import Path
from typing import Literal

from forecastfm.models import (
    Distribution,
    EvidenceCard,
    ForecastCase,
    ForecastPrediction,
    ForecastQuestion,
    TrainingExample,
)
from forecastfm.nba_data import (
    DATE_AMBIGUOUS_GAME_IDS,
    ELO_TARGET_TOLERANCE,
    SIDE_SWAP_SUFFIX,
    SOURCE_URL,
    elo_venue_probability,
)
from forecastfm.outcome import NBA_OUTCOMES, OPPONENT_OUTCOME, TEAM_OUTCOME

NBA_V2_FEATURE_NAMES = (
    "venue_adjusted_elo_log_odds",
    "rest_days_difference",
    "back_to_back_difference",
    "games_last_7_difference",
    "road_games_last_7_difference",
    "trailing_10_win_rate_difference",
    "trailing_10_margin_difference",
    "trailing_10_opponent_elo_difference",
    "trailing_10_history_difference",
)

NBA_V2_DATA_LIMITATIONS = (
    "Game times have date granularity, not true publication or tipoff timestamps.",
    "Road-game counts are a travel proxy; the source has no arena, mileage, or time zone.",
    "The source has no injuries, expected lineups, rosters, or player-level data.",
    "Rolling metrics are limited to prior game results, margins, and pregame Elo ratings.",
)

_REQUIRED_COLUMNS = frozenset(
    {
        "_iscopy",
        "date_game",
        "elo_i",
        "forecast",
        "game_id",
        "game_location",
        "game_result",
        "lg_id",
        "opp_elo_i",
        "opp_id",
        "opp_pts",
        "pts",
        "team_id",
        "year_id",
    }
)
_DEFAULT_ELO = 1_500.0
_DEFAULT_WIN_RATE = 0.5
_RECENT_DAYS = 7
_TRAILING_GAMES = 10

type _Location = Literal["away", "home", "neutral"]


class NbaV2DataError(ValueError):
    """Raised when source data violates the historical v2 contract."""


@dataclass(frozen=True, slots=True)
class NbaV2Features:
    """One anonymous, oriented, exactly side-swappable feature vector."""

    venue_adjusted_elo_probabilities: tuple[float, float]
    venue_adjusted_elo_log_odds: float
    rest_days_difference: float
    back_to_back_difference: float
    games_last_7_difference: float
    road_games_last_7_difference: float
    trailing_10_win_rate_difference: float
    trailing_10_margin_difference: float
    trailing_10_opponent_elo_difference: float
    trailing_10_history_difference: float

    def __post_init__(self) -> None:
        probabilities = self.venue_adjusted_elo_probabilities
        if not all(isfinite(value) and 0.0 < value < 1.0 for value in probabilities):
            raise NbaV2DataError("venue-adjusted Elo probabilities must be finite and interior")
        if not isclose(sum(probabilities), 1.0, abs_tol=1e-12):
            raise NbaV2DataError("venue-adjusted Elo probabilities must sum to one")
        expected_log_odds = log(probabilities[0] / probabilities[1])
        if not isclose(self.venue_adjusted_elo_log_odds, expected_log_odds, abs_tol=1e-12):
            raise NbaV2DataError("venue-adjusted Elo log-odds do not match probabilities")
        if not all(isfinite(value) for value in self.vector):
            raise NbaV2DataError("NBA v2 features must be finite")

    @property
    def vector(self) -> tuple[float, ...]:
        """Return feature values in the stable ``NBA_V2_FEATURE_NAMES`` order."""
        return (
            self.venue_adjusted_elo_log_odds,
            self.rest_days_difference,
            self.back_to_back_difference,
            self.games_last_7_difference,
            self.road_games_last_7_difference,
            self.trailing_10_win_rate_difference,
            self.trailing_10_margin_difference,
            self.trailing_10_opponent_elo_difference,
            self.trailing_10_history_difference,
        )

    def as_dict(self) -> dict[str, float]:
        """Return a readable feature mapping with stable insertion order."""
        return dict(zip(NBA_V2_FEATURE_NAMES, self.vector, strict=True))

    def side_swap(self) -> NbaV2Features:
        """Exchange listed team and opponent while preserving exact involution."""
        team_probability, opponent_probability = self.venue_adjusted_elo_probabilities
        return NbaV2Features(
            venue_adjusted_elo_probabilities=(opponent_probability, team_probability),
            venue_adjusted_elo_log_odds=_negate(self.venue_adjusted_elo_log_odds),
            rest_days_difference=_negate(self.rest_days_difference),
            back_to_back_difference=_negate(self.back_to_back_difference),
            games_last_7_difference=_negate(self.games_last_7_difference),
            road_games_last_7_difference=_negate(self.road_games_last_7_difference),
            trailing_10_win_rate_difference=_negate(self.trailing_10_win_rate_difference),
            trailing_10_margin_difference=_negate(self.trailing_10_margin_difference),
            trailing_10_opponent_elo_difference=_negate(self.trailing_10_opponent_elo_difference),
            trailing_10_history_difference=_negate(self.trailing_10_history_difference),
        )


@dataclass(frozen=True, slots=True)
class NbaV2Example:
    """An anonymous ForecastFM example paired with its numeric feature vector."""

    training_example: TrainingExample
    features: NbaV2Features
    season: int

    def __post_init__(self) -> None:
        if self.season <= 0:
            raise NbaV2DataError("NBA v2 season must be positive")
        case = self.training_example.case
        if case.question.outcomes != NBA_OUTCOMES:
            raise NbaV2DataError("NBA v2 examples require canonical binary outcomes")
        if case.prior.probabilities != self.features.venue_adjusted_elo_probabilities:
            raise NbaV2DataError("NBA v2 Elo prior does not match its feature record")


@dataclass(frozen=True, slots=True)
class _Game:
    game_id: str
    season: int
    game_date: date
    team_id: str
    opponent_id: str
    team_elo: float
    opponent_elo: float
    location: _Location
    team_points: int
    opponent_points: int
    team_won: bool


@dataclass(frozen=True, slots=True)
class _PastGame:
    game_date: date
    won: bool
    margin: int
    opponent_elo: float
    road: bool


@dataclass(frozen=True, slots=True)
class _HistorySummary:
    rest_days: int
    back_to_back: int
    games_last_7: int
    road_games_last_7: int
    trailing_win_rate: float
    trailing_margin: float
    trailing_opponent_elo: float
    trailing_history: int


def load_nba_v2_examples(path: Path) -> tuple[NbaV2Example, ...]:
    """Load selected NBA games and derive features using only earlier dates."""
    games = _load_selected_games(path)
    examples: list[NbaV2Example] = []
    for _, season_values in groupby(games, key=lambda game: game.season):
        histories: dict[str, list[_PastGame]] = {}
        season_games = tuple(season_values)
        for _, date_values in groupby(season_games, key=lambda game: game.game_date):
            date_games = tuple(date_values)
            examples.extend(_build_example(game, histories) for game in date_games)
            for game in date_games:
                _update_histories(game, histories)
    return tuple(examples)


def side_swap_nba_v2_example(example: NbaV2Example) -> NbaV2Example:
    """Exchange every oriented input and label in an NBA v2 example."""
    training = example.training_example
    realized_outcome = training.realized_outcome
    if realized_outcome == TEAM_OUTCOME:
        swapped_outcome = OPPONENT_OUTCOME
    elif realized_outcome == OPPONENT_OUTCOME:
        swapped_outcome = TEAM_OUTCOME
    else:
        raise NbaV2DataError("NBA v2 side swap requires a realized winner")

    features = example.features.side_swap()
    question = replace(
        training.case.question,
        question_id=_side_swap_question_id(training.case.question.question_id),
    )
    case = replace(
        training.case,
        question=question,
        prior=Distribution(
            outcomes=NBA_OUTCOMES,
            probabilities=features.venue_adjusted_elo_probabilities,
        ),
        evidence=(_feature_card(features, question.forecast_at),),
    )
    target = ForecastPrediction(
        distribution=Distribution(
            outcomes=NBA_OUTCOMES,
            probabilities=tuple(reversed(training.target.distribution.probabilities)),
        )
    )
    return NbaV2Example(
        training_example=replace(
            training,
            case=case,
            target=target,
            realized_outcome=swapped_outcome,
        ),
        features=features,
        season=example.season,
    )


def _load_selected_games(path: Path) -> tuple[_Game, ...]:
    games: list[_Game] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise NbaV2DataError(f"NBA source is missing columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            if _required(row, "lg_id", line_number) != "NBA":
                continue
            game_id = _required(row, "game_id", line_number)
            if game_id in DATE_AMBIGUOUS_GAME_IDS:
                continue
            if _required(row, "_iscopy", line_number) != _selected_copy(game_id):
                continue
            if game_id in seen_ids:
                raise NbaV2DataError(f"selected game appears more than once: {game_id}")
            seen_ids.add(game_id)
            games.append(_parse_game(row, line_number))

    return tuple(sorted(games, key=lambda game: (game.season, game.game_date, game.game_id)))


def _parse_game(row: Mapping[str, str | None], line_number: int) -> _Game:
    game_id = _required(row, "game_id", line_number)
    team_id = _required(row, "team_id", line_number)
    opponent_id = _required(row, "opp_id", line_number)
    if team_id == opponent_id:
        raise NbaV2DataError(f"game lists one team twice: {game_id}")

    team_points = _integer(row, "pts", line_number)
    opponent_points = _integer(row, "opp_pts", line_number)
    if team_points == opponent_points:
        raise NbaV2DataError(f"NBA game cannot end tied: {game_id}")
    team_won = _result(row, line_number)
    if team_won != (team_points > opponent_points):
        raise NbaV2DataError(f"result and score disagree: {game_id}")

    team_elo = _number(row, "elo_i", line_number)
    opponent_elo = _number(row, "opp_elo_i", line_number)
    location = _location(row, line_number)
    neutral_probability = 1.0 / (1.0 + 10.0 ** ((opponent_elo - team_elo) / 400.0))
    elo_probability = elo_venue_probability(neutral_probability, location)
    source_forecast = _number(row, "forecast", line_number)
    if abs(source_forecast - elo_probability) > ELO_TARGET_TOLERANCE:
        raise NbaV2DataError(f"forecast differs from the Elo oracle on source line {line_number}")

    return _Game(
        game_id=game_id,
        season=_integer(row, "year_id", line_number),
        game_date=_game_date(row, line_number),
        team_id=team_id,
        opponent_id=opponent_id,
        team_elo=team_elo,
        opponent_elo=opponent_elo,
        location=location,
        team_points=team_points,
        opponent_points=opponent_points,
        team_won=team_won,
    )


def _build_example(
    game: _Game,
    histories: Mapping[str, list[_PastGame]],
) -> NbaV2Example:
    team = _summarize_history(histories.get(game.team_id, ()), game.game_date)
    opponent = _summarize_history(histories.get(game.opponent_id, ()), game.game_date)
    neutral_probability = 1.0 / (1.0 + 10.0 ** ((game.opponent_elo - game.team_elo) / 400.0))
    team_probability = elo_venue_probability(neutral_probability, game.location)
    probabilities = (team_probability, 1.0 - team_probability)
    features = NbaV2Features(
        venue_adjusted_elo_probabilities=probabilities,
        venue_adjusted_elo_log_odds=log(probabilities[0] / probabilities[1]),
        rest_days_difference=float(team.rest_days - opponent.rest_days),
        back_to_back_difference=float(team.back_to_back - opponent.back_to_back),
        games_last_7_difference=float(team.games_last_7 - opponent.games_last_7),
        road_games_last_7_difference=float(team.road_games_last_7 - opponent.road_games_last_7),
        trailing_10_win_rate_difference=team.trailing_win_rate - opponent.trailing_win_rate,
        trailing_10_margin_difference=team.trailing_margin - opponent.trailing_margin,
        trailing_10_opponent_elo_difference=team.trailing_opponent_elo
        - opponent.trailing_opponent_elo,
        trailing_10_history_difference=float(team.trailing_history - opponent.trailing_history),
    )
    forecast_at = datetime.combine(game.game_date, datetime.min.time(), tzinfo=UTC)
    outcomes = NBA_OUTCOMES
    baseline = Distribution(outcomes=outcomes, probabilities=probabilities)
    case = ForecastCase(
        question=ForecastQuestion(
            question_id=_anonymous_question_id(game.game_id),
            text="Will the listed team defeat its opponent in this NBA game?",
            resolution_rule="Resolve to the team with the higher final score.",
            resolution_source=SOURCE_URL,
            outcomes=outcomes,
            forecast_at=forecast_at,
            resolves_at=forecast_at + timedelta(days=2),
        ),
        prior=baseline,
        prior_source="Venue-adjusted probability from pinned FiveThirtyEight pregame Elo",
        prior_as_of=forecast_at,
        evidence=(_feature_card(features, forecast_at),),
    )
    realized_outcome = TEAM_OUTCOME if game.team_won else OPPONENT_OUTCOME
    training = TrainingExample(
        case=case,
        target=ForecastPrediction(distribution=baseline),
        target_information_cutoff=forecast_at,
        target_method="Venue-adjusted FiveThirtyEight pregame Elo baseline metadata",
        realized_outcome=realized_outcome,
    )
    return NbaV2Example(training_example=training, features=features, season=game.season)


def _summarize_history(history: Sequence[_PastGame], game_date: date) -> _HistorySummary:
    if history:
        elapsed_days = (game_date - history[-1].game_date).days
        if elapsed_days <= 0:
            raise NbaV2DataError("historical features require strictly earlier game dates")
        rest_days = elapsed_days - 1
        back_to_back = int(elapsed_days == 1)
    else:
        rest_days = 0
        back_to_back = 0

    recent = tuple(
        game for game in history if 0 < (game_date - game.game_date).days <= _RECENT_DAYS
    )
    trailing = tuple(history[-_TRAILING_GAMES:])
    if trailing:
        trailing_win_rate = sum(game.won for game in trailing) / len(trailing)
        trailing_margin = sum(game.margin for game in trailing) / len(trailing)
        trailing_opponent_elo = sum(game.opponent_elo for game in trailing) / len(trailing)
    else:
        trailing_win_rate = _DEFAULT_WIN_RATE
        trailing_margin = 0.0
        trailing_opponent_elo = _DEFAULT_ELO

    return _HistorySummary(
        rest_days=rest_days,
        back_to_back=back_to_back,
        games_last_7=len(recent),
        road_games_last_7=sum(game.road for game in recent),
        trailing_win_rate=trailing_win_rate,
        trailing_margin=trailing_margin,
        trailing_opponent_elo=trailing_opponent_elo,
        trailing_history=len(trailing),
    )


def _update_histories(game: _Game, histories: dict[str, list[_PastGame]]) -> None:
    team_history = histories.setdefault(game.team_id, [])
    opponent_history = histories.setdefault(game.opponent_id, [])
    margin = game.team_points - game.opponent_points
    team_history.append(
        _PastGame(
            game_date=game.game_date,
            won=game.team_won,
            margin=margin,
            opponent_elo=game.opponent_elo,
            road=game.location == "away",
        )
    )
    opponent_history.append(
        _PastGame(
            game_date=game.game_date,
            won=not game.team_won,
            margin=-margin,
            opponent_elo=game.team_elo,
            road=game.location == "home",
        )
    )


def _feature_card(features: NbaV2Features, available_at: datetime) -> EvidenceCard:
    values = json.dumps(features.as_dict(), separators=(",", ":"))
    return EvidenceCard(
        text=f"Pregame numeric features: {values}",
        source=SOURCE_URL,
        available_at=available_at,
    )


def _required(row: Mapping[str, str | None], field: str, line_number: int) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise NbaV2DataError(f"missing {field} on source line {line_number}")
    return value.strip()


def _integer(row: Mapping[str, str | None], field: str, line_number: int) -> int:
    value = _required(row, field, line_number)
    try:
        return int(value)
    except ValueError as error:
        raise NbaV2DataError(f"invalid integer {field} on source line {line_number}") from error


def _number(row: Mapping[str, str | None], field: str, line_number: int) -> float:
    value = _required(row, field, line_number)
    try:
        number = float(value)
    except ValueError as error:
        raise NbaV2DataError(f"invalid number {field} on source line {line_number}") from error
    if not isfinite(number):
        raise NbaV2DataError(f"non-finite {field} on source line {line_number}")
    return number


def _game_date(row: Mapping[str, str | None], line_number: int) -> date:
    value = _required(row, "date_game", line_number)
    try:
        return datetime.strptime(value, "%m/%d/%Y").replace(tzinfo=UTC).date()
    except ValueError as error:
        raise NbaV2DataError(f"invalid date_game on source line {line_number}") from error


def _location(row: Mapping[str, str | None], line_number: int) -> _Location:
    value = _required(row, "game_location", line_number)
    locations: dict[str, _Location] = {"A": "away", "H": "home", "N": "neutral"}
    try:
        return locations[value]
    except KeyError as error:
        raise NbaV2DataError(f"invalid game_location on source line {line_number}") from error


def _result(row: Mapping[str, str | None], line_number: int) -> bool:
    value = _required(row, "game_result", line_number)
    if value == "W":
        return True
    if value == "L":
        return False
    raise NbaV2DataError(f"invalid game_result on source line {line_number}")


def _selected_copy(game_id: str) -> str:
    return str(sha256(game_id.encode()).digest()[0] % 2)


def _anonymous_question_id(game_id: str) -> str:
    digest = sha256(f"forecastfm:nba-v2:{game_id}".encode()).hexdigest()
    return f"nba-v2-{digest[:16]}"


def _side_swap_question_id(question_id: str) -> str:
    if question_id.endswith(SIDE_SWAP_SUFFIX):
        return question_id.removesuffix(SIDE_SWAP_SUFFIX)
    return f"{question_id}{SIDE_SWAP_SUFFIX}"


def _negate(value: float) -> float:
    if value == 0.0:
        return 0.0
    return -value
