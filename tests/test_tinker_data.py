"""Tests for the screened Tinker data boundary."""

from dataclasses import replace
from pathlib import Path

import pytest

from forecastfm.models import EvidenceCard, ForecastCase
from forecastfm.tinker_data import (
    build_forecast_record,
    build_sft_record,
    write_forecast_jsonl,
    write_sft_jsonl,
)
from forecastfm.tinker_screening import TinkerScreeningError
from tests.helpers import make_training_example


def test_sft_record_contains_three_messages() -> None:
    example = make_training_example()

    record = build_sft_record(example)

    assert len(record["messages"]) == 3


def test_forecast_record_contains_no_assistant_target() -> None:
    case = make_training_example().case

    record = build_forecast_record(case)

    assert record["question_id"] == case.question.question_id
    assert [message["role"] for message in record["messages"]] == ["system", "user"]


def test_sft_export_rejects_health_language() -> None:
    example = make_training_example()
    unsafe_card = EvidenceCard(
        text="A player has an injury.",
        source="test",
        available_at=example.case.question.forecast_at,
    )
    unsafe_case = ForecastCase(
        question=example.case.question,
        prior=example.case.prior,
        prior_source=example.case.prior_source,
        prior_as_of=example.case.prior_as_of,
        evidence=(unsafe_card,),
    )

    with pytest.raises(TinkerScreeningError, match="injury"):
        build_sft_record(replace(example, case=unsafe_case))


def test_forecast_export_rejects_health_language() -> None:
    example = make_training_example()
    unsafe_card = EvidenceCard(
        text="A player has an injury.",
        source="test",
        available_at=example.case.question.forecast_at,
    )
    unsafe_case = replace(example.case, evidence=(unsafe_card,))

    with pytest.raises(TinkerScreeningError, match="injury"):
        build_forecast_record(unsafe_case)


def test_sft_export_checks_the_target_method() -> None:
    example = make_training_example()

    with pytest.raises(TinkerScreeningError, match="medical"):
        build_sft_record(replace(example, target_method="A medical report."))


def test_write_sft_jsonl(tmp_path: Path) -> None:
    examples = (make_training_example(), make_training_example())
    path = tmp_path / "sft.jsonl"

    write_sft_jsonl(examples, path)

    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_write_forecast_jsonl(tmp_path: Path) -> None:
    case = make_training_example().case
    path = tmp_path / "forecasts.jsonl"

    write_forecast_jsonl((case,), path)

    text = path.read_text(encoding="utf-8")
    assert '"assistant"' not in text
    assert '"target"' not in text
    assert '"realized_outcome"' not in text


def test_write_sft_jsonl_does_not_leave_partial_destination(tmp_path: Path) -> None:
    safe_example = make_training_example()
    unsafe_example = replace(
        make_training_example(),
        target_method="A medical detail entered the target.",
    )
    path = tmp_path / "sft.jsonl"
    path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(TinkerScreeningError):
        write_sft_jsonl((safe_example, unsafe_example), path)

    assert path.read_text(encoding="utf-8") == "existing\n"
    assert tuple(tmp_path.iterdir()) == (path,)
