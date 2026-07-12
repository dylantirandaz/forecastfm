"""Vendor-neutral SFT data export for a later Tinker training call."""

import json
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypedDict

from forecastfm.models import ForecastCase, TrainingExample
from forecastfm.prompting import ChatMessage, build_forecast_messages, build_sft_messages
from forecastfm.tinker_screening import require_health_screen_passes


class SftRecord(TypedDict):
    """A chat record accepted by common supervised-training loaders."""

    messages: list[ChatMessage]


class ForecastRecord(TypedDict):
    """A target-free model input keyed by an opaque internal identity."""

    question_id: str
    messages: list[ChatMessage]


type JsonlRecord = SftRecord | ForecastRecord


def build_sft_record(example: TrainingExample) -> SftRecord:
    """Create a supervised record after a conservative health-term screen."""
    require_health_screen_passes(example)
    return SftRecord(messages=list(build_sft_messages(example)))


def build_forecast_record(case: ForecastCase) -> ForecastRecord:
    """Create a record that cannot expose a target or realized outcome."""
    return ForecastRecord(
        question_id=case.question.question_id,
        messages=list(build_forecast_messages(case)),
    )


def write_sft_jsonl(examples: Iterable[TrainingExample], path: Path) -> None:
    """Atomically write screened chat records without importing the Tinker SDK."""
    records = (build_sft_record(example) for example in examples)
    _write_jsonl(records, path)


def write_forecast_jsonl(cases: Iterable[ForecastCase], path: Path) -> None:
    """Atomically write target-free records for evaluation or inference."""
    records = (build_forecast_record(case) for case in cases)
    _write_jsonl(records, path)


def _write_jsonl(records: Iterable[JsonlRecord], path: Path) -> None:
    partial_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".part",
            delete=False,
        ) as file:
            partial_path = Path(file.name)
            for record in records:
                json.dump(record, file, sort_keys=True)
                file.write("\n")
        partial_path.replace(path)
    except Exception:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)
        raise
