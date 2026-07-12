"""Tests for chat rendering and response parsing."""

import json

import pytest

from forecastfm.json_utils import JsonFormatError
from forecastfm.prompting import (
    build_forecast_messages,
    build_sft_messages,
    parse_prediction,
    render_case,
    render_prediction,
)
from tests.helpers import make_training_example


def test_prediction_round_trip() -> None:
    example = make_training_example()
    rendered = render_prediction(example.target)

    parsed = parse_prediction(rendered, example.case.question.outcomes)

    assert parsed == example.target


def test_prediction_schema_contains_only_probabilities() -> None:
    example = make_training_example()

    rendered = render_prediction(example.target)

    assert json.loads(rendered) == {"probabilities": example.target.distribution.as_dict()}


def test_prediction_rejects_invented_outcome() -> None:
    text = '{"probabilities":{"yes":0.5,"maybe":0.5}}'

    with pytest.raises(JsonFormatError, match="keys differ"):
        parse_prediction(text, ("yes", "no"))


@pytest.mark.parametrize(
    "text",
    [
        '{"probabilities":{"yes":NaN,"no":0.5}}',
        '{"probabilities":{"yes":0.5,"yes":0.5}}',
        f'{{"probabilities":{{"yes":{10**400},"no":0.5}}}}',
    ],
)
def test_prediction_rejects_nonstandard_json(text: str) -> None:
    with pytest.raises(JsonFormatError):
        parse_prediction(text, ("yes", "no"))


def test_prediction_rejects_extra_key() -> None:
    text = '{"probabilities":{"yes":0.5,"no":0.5},"summary":"uncertain"}'

    with pytest.raises(JsonFormatError, match="keys differ"):
        parse_prediction(text, ("yes", "no"))


def test_sft_messages_have_stable_roles() -> None:
    example = make_training_example()

    messages = build_sft_messages(example)

    assert [message["role"] for message in messages] == ["system", "user", "assistant"]


def test_forecast_messages_have_no_training_target() -> None:
    example = make_training_example()

    messages = build_forecast_messages(example.case)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert render_prediction(example.target) not in {message["content"] for message in messages}


def test_model_case_contains_only_predictive_fields() -> None:
    example = make_training_example()

    record = json.loads(render_case(example.case))

    assert set(record) == {"evidence", "outcomes", "prior", "question", "resolution_rule"}
    assert example.case.question.question_id not in str(record)
    assert example.case.question.resolution_source not in str(record)
    assert example.case.question.forecast_at.isoformat() not in str(record)
