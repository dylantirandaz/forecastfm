"""Outcome-classification prompts and probability conversion."""

from math import exp, isfinite
from typing import Protocol

from forecastfm.models import Distribution, ForecastCase, ForecastPrediction, TrainingExample
from forecastfm.prompting import ChatMessage, render_case

OUTCOME_INPUT_SCHEMA_VERSION = 2

TEAM_OUTCOME = "team_wins"
OPPONENT_OUTCOME = "opponent_wins"
NBA_OUTCOMES = (TEAM_OUTCOME, OPPONENT_OUTCOME)

TEAM_LABEL = "TEAM"
OPPONENT_LABEL = "OTHER"
OUTCOME_LABELS = (TEAM_LABEL, OPPONENT_LABEL)

OUTCOME_SYSTEM_PROMPT = """You are ForecastFM, a calibrated probabilistic forecaster.
Use only the prior and evidence supplied to you.
Forecast which side will win.
Return exactly TEAM if the listed team will win.
Return exactly OTHER if the opponent will win.
Do not return JSON, punctuation, or an explanation."""


class OutcomeForecastError(ValueError):
    """Raised when an outcome-classification contract is violated."""


class TokenCodec(Protocol):
    """Tokenizer operations needed to verify fixed classifier labels."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        """Encode text into token IDs."""
        ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        """Decode token IDs into text."""
        ...


def build_outcome_messages(case: ForecastCase) -> tuple[ChatMessage, ...]:
    """Build a target-free forecasting prompt for the outcome classifier."""
    _require_nba_outcomes(case.question.outcomes)
    return (
        ChatMessage(role="system", content=OUTCOME_SYSTEM_PROMPT),
        ChatMessage(role="user", content=render_case(case)),
    )


def label_for_example(example: TrainingExample) -> str:
    """Map the realized winner to its fixed classifier label."""
    _require_nba_outcomes(example.case.question.outcomes)
    if example.realized_outcome == TEAM_OUTCOME:
        return TEAM_LABEL
    if example.realized_outcome == OPPONENT_OUTCOME:
        return OPPONENT_LABEL
    raise OutcomeForecastError("outcome training requires a realized NBA winner")


def require_label(label: str) -> str:
    """Return a supported classifier label or fail closed."""
    if label not in OUTCOME_LABELS:
        raise OutcomeForecastError(f"unsupported outcome label: {label}")
    return label


def require_label_token_ids(tokenizer: TokenCodec) -> tuple[int, int]:
    """Verify that both classifier labels are distinct, exact single tokens."""
    team_token = _require_single_token(tokenizer, TEAM_LABEL)
    opponent_token = _require_single_token(tokenizer, OPPONENT_LABEL)
    if team_token == opponent_token:
        raise OutcomeForecastError("outcome labels must use different token IDs")
    return team_token, opponent_token


def team_probability_from_logprobs(
    team_logprob: float,
    opponent_logprob: float,
) -> float:
    """Normalize two label log-probabilities into a stable team-win probability."""
    if not isfinite(team_logprob) or not isfinite(opponent_logprob):
        raise OutcomeForecastError("label log-probabilities must be finite")

    difference = team_logprob - opponent_logprob
    if difference >= 0.0:
        return 1.0 / (1.0 + exp(-difference))
    ratio = exp(difference)
    return ratio / (1.0 + ratio)


def prediction_from_logprobs(
    team_logprob: float,
    opponent_logprob: float,
) -> ForecastPrediction:
    """Create the canonical NBA forecast from the two label scores."""
    team_probability = team_probability_from_logprobs(team_logprob, opponent_logprob)
    return ForecastPrediction(
        distribution=Distribution(
            outcomes=NBA_OUTCOMES,
            probabilities=(team_probability, 1.0 - team_probability),
        )
    )


def symmetric_team_probability(
    original_team_probability: float,
    swapped_team_probability: float,
) -> float:
    """Average an original forecast with the complement of its side swap."""
    for probability in (original_team_probability, swapped_team_probability):
        if not isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise OutcomeForecastError("side-swap probabilities must be between zero and one")
    return (original_team_probability + 1.0 - swapped_team_probability) / 2.0


def _require_single_token(tokenizer: TokenCodec, label: str) -> int:
    token_ids = tokenizer.encode(label, add_special_tokens=False)
    if len(token_ids) != 1:
        raise OutcomeForecastError(f"outcome label is not one token: {label}")
    if tokenizer.decode(token_ids, skip_special_tokens=False) != label:
        raise OutcomeForecastError(f"outcome label does not round-trip exactly: {label}")
    return token_ids[0]


def _require_nba_outcomes(outcomes: tuple[str, ...]) -> None:
    if outcomes != NBA_OUTCOMES:
        raise OutcomeForecastError("outcome classifier requires canonical NBA outcomes")
