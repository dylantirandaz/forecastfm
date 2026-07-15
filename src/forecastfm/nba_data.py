"""Load real, leakage-filtered NBA forecasts from FiveThirtyEight."""

import csv
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlretrieve

from forecastfm.models import (
    Distribution,
    EvidenceCard,
    ForecastCase,
    ForecastPrediction,
    ForecastQuestion,
    TrainingExample,
)
from forecastfm.prompting import render_case

SOURCE_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/"
    "6d880e939ad3d11d94c137c911681b3cf718fd74/nba-elo/nbaallelo.csv"
)
SOURCE_PAGE = "https://github.com/fivethirtyeight/data/tree/master/nba-elo"
SOURCE_SHA256 = "d46ed3540ee8d9eca31b3e94cc8c777e0be5156173d814ebf65b8195e8d616bc"
LICENSE_URL = "https://github.com/fivethirtyeight/data/blob/master/LICENSE"

TRAIN_LAST_SEASON = 2009
VALIDATION_LAST_SEASON = 2012
SOURCE_NBA_GAME_COUNT = 59_008
ELO_HOME_ADVANTAGE = 100.0
ELO_TARGET_TOLERANCE = 1e-6
_VENUE_ELO_ADJUSTMENT = {
    "away": -ELO_HOME_ADVANTAGE,
    "home": ELO_HOME_ADVANTAGE,
    "neutral": 0.0,
}
DATE_AMBIGUOUS_GAME_IDS = frozenset(
    {
        "195403081MLH",
        "197002081CHI",
        "197903231PHI",
        "198304131SAS",
    }
)
SIDE_SWAP_SUFFIX = "-side-swap"

_REQUIRED_COLUMNS = {
    "_iscopy",
    "date_game",
    "elo_i",
    "forecast",
    "game_id",
    "game_location",
    "game_result",
    "lg_id",
    "opp_elo_i",
    "year_id",
}


class NbaDataError(ValueError):
    """Raised when the pinned NBA source violates its expected schema."""


@dataclass(frozen=True, slots=True)
class NbaSplits:
    """Chronological, game-level NBA training, validation, and test splits."""

    train: tuple[TrainingExample, ...]
    validation: tuple[TrainingExample, ...]
    test: tuple[TrainingExample, ...]
    source_game_count: int
    duplicate_prompt_count: int


def download_nba_elo(path: Path) -> None:
    """Download the pinned source once and verify its exact SHA-256 hash."""
    if path.is_file():
        _require_expected_hash(path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path: Path | None = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as file:
            partial_path = Path(file.name)
        urlretrieve(SOURCE_URL, partial_path)
        _require_expected_hash(partial_path)
        partial_path.replace(path)
    finally:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_nba_splits(path: Path) -> NbaSplits:
    """Load unique NBA games and split them by season without random leakage."""
    train: list[TrainingExample] = []
    validation: list[TrainingExample] = []
    test: list[TrainingExample] = []
    source_game_count = 0

    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise NbaDataError(f"NBA source is missing columns: {', '.join(missing)}")

        for line_number, row in enumerate(reader, start=2):
            if _required(row, "lg_id", line_number) != "NBA":
                continue
            game_id = _required(row, "game_id", line_number)
            if game_id in DATE_AMBIGUOUS_GAME_IDS:
                continue
            if _required(row, "_iscopy", line_number) != _selected_copy(game_id):
                continue

            source_game_count += 1
            season = _integer(row, "year_id", line_number)
            example = _training_example(row, line_number)
            if season <= TRAIN_LAST_SEASON:
                train.append(example)
            elif season <= VALIDATION_LAST_SEASON:
                validation.append(example)
            else:
                test.append(example)

    seen_prompts: set[str] = set()
    unique_train = _unique_prompts(train, seen_prompts)
    unique_validation = _unique_prompts(validation, seen_prompts)
    unique_test = _unique_prompts(test, seen_prompts)
    unique_count = len(unique_train) + len(unique_validation) + len(unique_test)
    return NbaSplits(
        train=unique_train,
        validation=unique_validation,
        test=unique_test,
        source_game_count=source_game_count,
        duplicate_prompt_count=source_game_count - unique_count,
    )


def audit_nba_splits(splits: NbaSplits) -> dict[str, int]:
    """Reject identity leakage, one-hot targets, and duplicate split membership."""
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    venue_counts = {"away": 0, "home": 0, "neutral": 0}
    unique_count = len(splits.train) + len(splits.validation) + len(splits.test)
    if unique_count + splits.duplicate_prompt_count != splits.source_game_count:
        raise NbaDataError("NBA split accounting is inconsistent")
    for examples in (splits.train, splits.validation, splits.test):
        for example in examples:
            venue = _audit_example(example, seen_ids, seen_prompts)
            venue_counts[venue] += 1

    non_neutral_count = venue_counts["away"] + venue_counts["home"]
    if splits.source_game_count >= 100:
        minority_share = min(venue_counts["away"], venue_counts["home"]) / non_neutral_count
        if minority_share < 0.45:
            raise NbaDataError("selected NBA perspective is not balanced")
    return venue_counts


def _audit_example(
    example: TrainingExample,
    seen_ids: set[str],
    seen_prompts: set[str],
) -> str:
    question = example.case.question
    if question.question_id in seen_ids:
        raise NbaDataError(f"game appears in multiple rows or splits: {question.question_id}")
    seen_ids.add(question.question_id)

    prompt = render_case(example.case)
    prompt_hash = sha256(prompt.encode()).hexdigest()
    if prompt_hash in seen_prompts:
        raise NbaDataError(f"model prompt appears more than once: {question.question_id}")
    seen_prompts.add(prompt_hash)
    forbidden_values = (
        question.question_id,
        question.forecast_at.date().isoformat(),
        question.resolution_source,
        "game_result",
        "realized_outcome",
        "target_method",
    )
    if any(value in prompt for value in forbidden_values):
        raise NbaDataError(f"model prompt leaks metadata for {question.question_id}")
    if not all(0.0 < value < 1.0 for value in example.target.distribution.probabilities):
        raise NbaDataError(f"target must remain probabilistic: {question.question_id}")
    if example.realized_outcome is None:
        raise NbaDataError(f"missing evaluation outcome: {question.question_id}")

    evidence_text = example.case.evidence[0].text
    for venue in ("away", "home", "neutral"):
        if evidence_text.endswith(f"{venue}."):
            return venue
    raise NbaDataError(f"invalid model-facing venue: {question.question_id}")


def _training_example(row: Mapping[str, str | None], line_number: int) -> TrainingExample:
    game_id = _required(row, "game_id", line_number)
    forecast_at = _game_date(row, line_number)
    forecast = _number(row, "forecast", line_number)
    location = _location(row, line_number)
    team_elo = _number(row, "elo_i", line_number)
    opponent_elo = _number(row, "opp_elo_i", line_number)
    neutral_probability = _neutral_elo_probability(team_elo, opponent_elo)
    oracle_probability = elo_venue_probability(neutral_probability, location)
    if abs(forecast - oracle_probability) > ELO_TARGET_TOLERANCE:
        raise NbaDataError(f"forecast differs from the Elo oracle on source line {line_number}")

    forecast = round(forecast, 7)
    outcomes = ("team_wins", "opponent_wins")

    case = ForecastCase(
        question=ForecastQuestion(
            question_id=_anonymous_question_id(game_id),
            text="Will the listed team defeat its opponent in this NBA game?",
            resolution_rule="Resolve to the team with the higher final score.",
            resolution_source=SOURCE_URL,
            outcomes=outcomes,
            forecast_at=forecast_at,
            resolves_at=forecast_at + timedelta(days=2),
        ),
        prior=Distribution(
            outcomes=outcomes,
            probabilities=_binary_probabilities(neutral_probability),
        ),
        prior_source="Neutral-court probability from FiveThirtyEight pregame Elo ratings",
        prior_as_of=forecast_at,
        evidence=(
            EvidenceCard(
                text=f"Venue for the listed team: {location}.",
                source=SOURCE_URL,
                available_at=forecast_at,
            ),
        ),
    )
    return TrainingExample(
        case=case,
        target=ForecastPrediction(
            distribution=Distribution(
                outcomes=outcomes,
                probabilities=(forecast, round(1.0 - forecast, 7)),
            )
        ),
        target_information_cutoff=forecast_at,
        target_method="FiveThirtyEight retrospective pregame NBA Elo forecast",
        realized_outcome=_realized_outcome(row, line_number),
    )


def _required(row: Mapping[str, str | None], field: str, line_number: int) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise NbaDataError(f"missing {field} on source line {line_number}")
    return value.strip()


def _neutral_elo_probability(team_elo: float, opponent_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_elo - team_elo) / 400.0))


def elo_venue_probability(neutral_probability: float, location: str) -> float:
    """Apply FiveThirtyEight's fixed venue adjustment to neutral Elo odds."""
    if not 0.0 < neutral_probability < 1.0:
        raise NbaDataError("neutral probability must be strictly between zero and one")
    if location not in _VENUE_ELO_ADJUSTMENT:
        raise NbaDataError(f"unknown venue: {location}")
    prior_odds = neutral_probability / (1.0 - neutral_probability)
    adjusted_odds = prior_odds * 10.0 ** (_VENUE_ELO_ADJUSTMENT[location] / 400.0)
    return adjusted_odds / (1.0 + adjusted_odds)


def side_swap_nba_case(case: ForecastCase) -> ForecastCase:
    """Exchange the listed team and opponent in an anonymous NBA case."""
    if case.question.outcomes != ("team_wins", "opponent_wins"):
        raise NbaDataError("NBA side swap requires canonical binary outcomes")
    if len(case.evidence) != 1:
        raise NbaDataError("NBA side swap requires exactly one venue card")

    card = case.evidence[0]
    location = _venue_from_text(card.text)
    swapped_location = {"away": "home", "home": "away", "neutral": "neutral"}[location]
    swapped_question = replace(
        case.question,
        question_id=_side_swap_question_id(case.question.question_id),
    )
    swapped_prior = Distribution(
        outcomes=case.prior.outcomes,
        probabilities=tuple(reversed(case.prior.probabilities)),
    )
    swapped_card = replace(card, text=f"Venue for the listed team: {swapped_location}.")
    return replace(
        case,
        question=swapped_question,
        prior=swapped_prior,
        evidence=(swapped_card,),
    )


def side_swap_nba_example(example: TrainingExample) -> TrainingExample:
    """Exchange sides, teacher probabilities, and winner in an NBA example."""
    if example.realized_outcome == "team_wins":
        swapped_outcome = "opponent_wins"
    elif example.realized_outcome == "opponent_wins":
        swapped_outcome = "team_wins"
    else:
        raise NbaDataError("NBA side-swap training requires a realized winner")

    swapped_target = ForecastPrediction(
        distribution=Distribution(
            outcomes=example.target.distribution.outcomes,
            probabilities=tuple(reversed(example.target.distribution.probabilities)),
        )
    )
    return replace(
        example,
        case=side_swap_nba_case(example.case),
        target=swapped_target,
        realized_outcome=swapped_outcome,
    )


def _binary_probabilities(team_probability: float) -> tuple[float, float]:
    rounded_probability = round(team_probability, 7)
    return rounded_probability, round(1.0 - rounded_probability, 7)


def _venue_from_text(text: str) -> str:
    prefix = "Venue for the listed team: "
    if not text.startswith(prefix) or not text.endswith("."):
        raise NbaDataError("NBA side swap found an unexpected venue card")
    location = text.removeprefix(prefix).removesuffix(".")
    if location not in _VENUE_ELO_ADJUSTMENT:
        raise NbaDataError("NBA side swap found an unknown venue")
    return location


def _side_swap_question_id(question_id: str) -> str:
    if question_id.endswith(SIDE_SWAP_SUFFIX):
        return question_id.removesuffix(SIDE_SWAP_SUFFIX)
    return f"{question_id}{SIDE_SWAP_SUFFIX}"


def _selected_copy(game_id: str) -> str:
    return str(sha256(game_id.encode()).digest()[0] % 2)


def _anonymous_question_id(game_id: str) -> str:
    digest = sha256(f"forecastfm:nba:{game_id}".encode()).hexdigest()
    return f"nba-{digest[:16]}"


def _unique_prompts(
    examples: list[TrainingExample],
    seen_prompts: set[str],
) -> tuple[TrainingExample, ...]:
    unique: list[TrainingExample] = []
    for example in examples:
        prompt_hash = sha256(render_case(example.case).encode()).hexdigest()
        if prompt_hash in seen_prompts:
            continue
        seen_prompts.add(prompt_hash)
        unique.append(example)
    return tuple(unique)


def _integer(row: Mapping[str, str | None], field: str, line_number: int) -> int:
    value = _required(row, field, line_number)
    try:
        return int(value)
    except ValueError as error:
        raise NbaDataError(f"invalid integer {field} on source line {line_number}") from error


def _number(row: Mapping[str, str | None], field: str, line_number: int) -> float:
    value = _required(row, field, line_number)
    try:
        number = float(value)
    except ValueError as error:
        raise NbaDataError(f"invalid number {field} on source line {line_number}") from error
    if not isfinite(number):
        raise NbaDataError(f"non-finite {field} on source line {line_number}")
    return number


def _game_date(row: Mapping[str, str | None], line_number: int) -> datetime:
    value = _required(row, "date_game", line_number)
    try:
        return datetime.strptime(value, "%m/%d/%Y").replace(tzinfo=UTC)
    except ValueError as error:
        raise NbaDataError(f"invalid date_game on source line {line_number}") from error


def _location(row: Mapping[str, str | None], line_number: int) -> str:
    value = _required(row, "game_location", line_number)
    names = {"A": "away", "H": "home", "N": "neutral"}
    try:
        return names[value]
    except KeyError as error:
        raise NbaDataError(f"invalid game_location on source line {line_number}") from error


def _realized_outcome(
    row: Mapping[str, str | None],
    line_number: int,
) -> str:
    result = _required(row, "game_result", line_number)
    if result == "W":
        return "team_wins"
    if result == "L":
        return "opponent_wins"
    raise NbaDataError(f"invalid game_result on source line {line_number}")


def _require_expected_hash(path: Path) -> None:
    actual = file_sha256(path)
    if actual != SOURCE_SHA256:
        raise NbaDataError(f"unexpected NBA source SHA-256: {actual}")
