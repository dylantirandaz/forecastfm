"""Strict JSONL serialization for forecast training examples."""

import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_float,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.models import (
    Distribution,
    EvidenceCard,
    ForecastCase,
    ForecastPrediction,
    ForecastQuestion,
    TrainingExample,
)

SCHEMA_VERSION = 3
RECORD_KEYS = {
    "schema_version",
    "question_id",
    "question",
    "resolution_rule",
    "resolution_source",
    "outcomes",
    "forecast_at",
    "resolves_at",
    "prior",
    "prior_source",
    "prior_as_of",
    "evidence",
    "target",
    "target_information_cutoff",
    "target_method",
    "realized_outcome",
}


def training_example_to_dict(example: TrainingExample) -> dict[str, object]:
    """Convert a training example into a human-readable JSON object."""
    question = example.case.question
    return {
        "schema_version": SCHEMA_VERSION,
        "question_id": question.question_id,
        "question": question.text,
        "resolution_rule": question.resolution_rule,
        "resolution_source": question.resolution_source,
        "outcomes": list(question.outcomes),
        "forecast_at": question.forecast_at.isoformat(),
        "resolves_at": question.resolves_at.isoformat(),
        "prior": list(example.case.prior.probabilities),
        "prior_source": example.case.prior_source,
        "prior_as_of": example.case.prior_as_of.isoformat(),
        "evidence": [
            {
                "text": card.text,
                "source": card.source,
                "available_at": card.available_at.isoformat(),
            }
            for card in example.case.evidence
        ],
        "target": list(example.target.distribution.probabilities),
        "target_information_cutoff": example.target_information_cutoff.isoformat(),
        "target_method": example.target_method,
        "realized_outcome": example.realized_outcome,
    }


def training_example_from_dict(record: Mapping[str, object]) -> TrainingExample:
    """Validate a decoded JSON object and create a training example."""
    require_exact_keys(record, RECORD_KEYS, "training example")
    schema_version = required_field(record, "schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise JsonFormatError("schema_version must be an integer")
    if schema_version != SCHEMA_VERSION:
        raise JsonFormatError(f"unsupported schema version: {schema_version}")

    outcomes = _string_tuple(required_field(record, "outcomes"), "outcomes")
    prior = Distribution(
        outcomes=outcomes,
        probabilities=_float_tuple(required_field(record, "prior"), "prior"),
    )
    target = Distribution(
        outcomes=outcomes,
        probabilities=_float_tuple(required_field(record, "target"), "target"),
    )
    question = ForecastQuestion(
        question_id=require_string(required_field(record, "question_id"), "question_id"),
        text=require_string(required_field(record, "question"), "question"),
        resolution_rule=require_string(
            required_field(record, "resolution_rule"),
            "resolution_rule",
        ),
        resolution_source=require_string(
            required_field(record, "resolution_source"),
            "resolution_source",
        ),
        outcomes=outcomes,
        forecast_at=_datetime(required_field(record, "forecast_at"), "forecast_at"),
        resolves_at=_datetime(required_field(record, "resolves_at"), "resolves_at"),
    )
    evidence = _evidence_tuple(required_field(record, "evidence"))
    realized_value = required_field(record, "realized_outcome")
    realized_outcome = (
        None if realized_value is None else require_string(realized_value, "realized_outcome")
    )
    return TrainingExample(
        case=ForecastCase(
            question=question,
            prior=prior,
            prior_source=require_string(
                required_field(record, "prior_source"),
                "prior_source",
            ),
            prior_as_of=_datetime(required_field(record, "prior_as_of"), "prior_as_of"),
            evidence=evidence,
        ),
        target=ForecastPrediction(distribution=target),
        target_information_cutoff=_datetime(
            required_field(record, "target_information_cutoff"),
            "target_information_cutoff",
        ),
        target_method=require_string(
            required_field(record, "target_method"),
            "target_method",
        ),
        realized_outcome=realized_outcome,
    )


def write_jsonl(examples: Iterable[TrainingExample], path: Path) -> None:
    """Write one validated training example per JSONL line."""
    with path.open("w", encoding="utf-8") as file:
        for example in examples:
            json.dump(training_example_to_dict(example), file, sort_keys=True)
            file.write("\n")


def read_jsonl(path: Path) -> tuple[TrainingExample, ...]:
    """Read and validate all training examples in a JSONL file."""
    examples: list[TrainingExample] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = parse_json_object(line)
                examples.append(training_example_from_dict(record))
            except (JsonFormatError, ValueError) as error:
                raise JsonFormatError(f"invalid JSONL record on line {line_number}") from error
    return tuple(examples)


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    values = require_list(value, field_name)
    return tuple(
        require_string(item, f"{field_name}[{index}]") for index, item in enumerate(values)
    )


def _float_tuple(value: object, field_name: str) -> tuple[float, ...]:
    values = require_list(value, field_name)
    return tuple(require_float(item, f"{field_name}[{index}]") for index, item in enumerate(values))


def _datetime(value: object, field_name: str) -> datetime:
    text = require_string(value, field_name)
    try:
        return datetime.fromisoformat(text)
    except ValueError as error:
        raise JsonFormatError(f"{field_name} must be an ISO 8601 datetime") from error


def _evidence_tuple(value: object) -> tuple[EvidenceCard, ...]:
    values = require_list(value, "evidence")
    cards: list[EvidenceCard] = []
    keys = {"text", "source", "available_at"}
    for index, item in enumerate(values):
        field_name = f"evidence[{index}]"
        record = require_object(item, field_name)
        require_exact_keys(record, keys, field_name)
        cards.append(
            EvidenceCard(
                text=require_string(required_field(record, "text"), f"{field_name}.text"),
                source=require_string(
                    required_field(record, "source"),
                    f"{field_name}.source",
                ),
                available_at=_datetime(
                    required_field(record, "available_at"),
                    f"{field_name}.available_at",
                ),
            )
        )
    return tuple(cards)
