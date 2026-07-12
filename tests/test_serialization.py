"""Tests for strict JSONL round trips."""

from pathlib import Path

import pytest

from forecastfm.json_utils import JsonFormatError
from forecastfm.serialization import (
    SCHEMA_VERSION,
    read_jsonl,
    training_example_from_dict,
    training_example_to_dict,
    write_jsonl,
)
from tests.helpers import make_training_example


def test_jsonl_round_trip(tmp_path: Path) -> None:
    examples = (make_training_example(),)
    path = tmp_path / "examples.jsonl"

    write_jsonl(examples, path)

    assert read_jsonl(path) == examples


def test_serialized_schema_contains_no_summary() -> None:
    example = make_training_example()

    record = training_example_to_dict(example)

    assert record["schema_version"] == SCHEMA_VERSION == 3
    assert "summary" not in record


def test_jsonl_error_includes_line_number(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(JsonFormatError, match="line 1"):
        read_jsonl(path)


def test_boolean_schema_version_is_rejected() -> None:
    with pytest.raises(JsonFormatError, match="must be an integer"):
        training_example_from_dict(
            {
                "schema_version": True,
                "question_id": "unused",
                "question": "unused",
                "resolution_rule": "unused",
                "resolution_source": "unused",
                "outcomes": [],
                "forecast_at": "unused",
                "resolves_at": "unused",
                "prior": [],
                "prior_source": "unused",
                "prior_as_of": "unused",
                "evidence": [],
                "target": [],
                "target_information_cutoff": "unused",
                "target_method": "unused",
                "realized_outcome": None,
            }
        )
