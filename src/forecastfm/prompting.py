"""Prompt rendering and strict model-output parsing."""

import json
from typing import Literal, TypedDict

from forecastfm.json_utils import (
    parse_json_object,
    require_exact_keys,
    require_float,
    require_object,
    required_field,
)
from forecastfm.models import (
    Distribution,
    ForecastCase,
    ForecastPrediction,
    TrainingExample,
)

MODEL_INPUT_SCHEMA_VERSION = 2

SYSTEM_PROMPT = """You are ForecastFM, a calibrated probabilistic forecaster.
Use only the prior and evidence supplied to you.
Return valid JSON with exactly one key: probabilities.
The probabilities must use every outcome exactly once and sum to one."""


class ChatMessage(TypedDict):
    """One role-tagged chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


def render_case(case: ForecastCase) -> str:
    """Render only information the model needs to make its forecast."""
    question = case.question
    value = {
        "question": question.text,
        "resolution_rule": question.resolution_rule,
        "outcomes": list(question.outcomes),
        "prior": case.prior.as_dict(),
        "evidence": [card.text for card in case.evidence],
    }
    return json.dumps(value, indent=2, sort_keys=True)


def render_prediction(prediction: ForecastPrediction) -> str:
    """Render a prediction in the exact model response schema."""
    value = {"probabilities": prediction.distribution.as_dict()}
    return json.dumps(value, sort_keys=True)


def build_forecast_messages(case: ForecastCase) -> tuple[ChatMessage, ...]:
    """Create a target-free conversation for evaluation or inference."""
    return (
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=render_case(case)),
    )


def build_sft_messages(example: TrainingExample) -> tuple[ChatMessage, ...]:
    """Add the supervised answer only at the training boundary."""
    return (
        *build_forecast_messages(example.case),
        ChatMessage(role="assistant", content=render_prediction(example.target)),
    )


def parse_prediction(text: str, outcomes: tuple[str, ...]) -> ForecastPrediction:
    """Parse a model response without accepting missing or invented outcomes."""
    record = parse_json_object(text)
    require_exact_keys(record, {"probabilities"}, "prediction")
    probability_record = require_object(
        required_field(record, "probabilities"),
        "probabilities",
    )
    require_exact_keys(probability_record, set(outcomes), "probabilities")
    probabilities = tuple(
        require_float(required_field(probability_record, outcome), f"probabilities.{outcome}")
        for outcome in outcomes
    )
    return ForecastPrediction(
        distribution=Distribution(outcomes=outcomes, probabilities=probabilities)
    )
