"""Vendor-neutral SFT data export for a later Tinker training call."""

import json
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal, TypedDict, cast

from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.models import ForecastCase, TrainingExample
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.outcome import (
    OUTCOME_SYSTEM_PROMPT,
    build_outcome_messages,
    label_for_example,
    require_label,
)
from forecastfm.prompting import ChatMessage, build_forecast_messages, build_sft_messages
from forecastfm.tinker_screening import (
    require_case_health_screen_passes,
    require_health_screen_passes,
)


class SftRecord(TypedDict):
    """A chat record accepted by common supervised-training loaders."""

    messages: list[ChatMessage]


class ForecastRecord(TypedDict):
    """A target-free model input keyed by an opaque internal identity."""

    question_id: str
    messages: list[ChatMessage]


class OutcomeTrainingRecord(TypedDict):
    """A target-free prompt paired with a fixed realized-winner label."""

    question_id: str
    messages: list[ChatMessage]
    label: str


type JsonlRecord = SftRecord | ForecastRecord | OutcomeTrainingRecord


def build_sft_record(example: TrainingExample) -> SftRecord:
    """Create a supervised record after a conservative health-term screen."""
    require_health_screen_passes(example)
    return SftRecord(messages=list(build_sft_messages(example)))


def build_forecast_record(case: ForecastCase) -> ForecastRecord:
    """Create a record that cannot expose a target or realized outcome."""
    require_case_health_screen_passes(case)
    return ForecastRecord(
        question_id=case.question.question_id,
        messages=list(build_forecast_messages(case)),
    )


def build_outcome_training_record(example: TrainingExample) -> OutcomeTrainingRecord:
    """Create an outcome record whose label comes only from the realized winner."""
    require_health_screen_passes(example)
    return OutcomeTrainingRecord(
        question_id=example.case.question.question_id,
        messages=list(build_outcome_messages(example.case)),
        label=label_for_example(example),
    )


def build_outcome_forecast_record(case: ForecastCase) -> ForecastRecord:
    """Create a target-free input for fixed-label outcome inference."""
    require_case_health_screen_passes(case)
    return ForecastRecord(
        question_id=case.question.question_id,
        messages=list(build_outcome_messages(case)),
    )


def write_sft_jsonl(examples: Iterable[TrainingExample], path: Path) -> None:
    """Atomically write screened chat records without importing the Tinker SDK."""
    records = (build_sft_record(example) for example in examples)
    _write_jsonl(records, path)


def write_forecast_jsonl(cases: Iterable[ForecastCase], path: Path) -> None:
    """Atomically write target-free records for evaluation or inference."""
    records = (build_forecast_record(case) for case in cases)
    _write_jsonl(records, path)


def write_outcome_training_jsonl(
    examples: Iterable[TrainingExample],
    path: Path,
) -> None:
    """Atomically write screened realized-winner classification records."""
    records = (build_outcome_training_record(example) for example in examples)
    _write_jsonl(records, path)


def write_outcome_forecast_jsonl(cases: Iterable[ForecastCase], path: Path) -> None:
    """Atomically write target-free fixed-label inference records."""
    records = (build_outcome_forecast_record(case) for case in cases)
    _write_jsonl(records, path)


def read_outcome_training_jsonl(path: Path) -> tuple[OutcomeTrainingRecord, ...]:
    """Read and strictly validate fixed-label outcome training records."""
    records: list[OutcomeTrainingRecord] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(_parse_outcome_training_record(line))
            except (JsonFormatError, ValueError) as error:
                raise JsonFormatError(
                    f"invalid outcome training record on line {line_number}"
                ) from error
    return tuple(records)


def read_outcome_forecast_jsonl(path: Path) -> tuple[ForecastRecord, ...]:
    """Read target-free fixed-label inference records."""
    records: list[ForecastRecord] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(_parse_outcome_forecast_record(line))
            except (JsonFormatError, ValueError) as error:
                raise JsonFormatError(
                    f"invalid outcome forecast record on line {line_number}"
                ) from error
    return tuple(records)


def pair_outcome_forecast_records(
    records: tuple[ForecastRecord, ...],
) -> tuple[tuple[ForecastRecord, ForecastRecord], ...]:
    """Group adjacent original/swap prompts without opening answer records."""
    if len(records) % 2 != 0:
        raise JsonFormatError("outcome forecast records contain an incomplete side-swap pair")
    pairs: list[tuple[ForecastRecord, ForecastRecord]] = []
    seen_ids: set[str] = set()
    for index in range(0, len(records), 2):
        original = records[index]
        swapped = records[index + 1]
        pair_ids = {original["question_id"], swapped["question_id"]}
        if seen_ids & pair_ids:
            raise JsonFormatError("outcome forecast records contain duplicate IDs")
        seen_ids.update(pair_ids)
        if swapped["question_id"] != f"{original['question_id']}{SIDE_SWAP_SUFFIX}":
            raise JsonFormatError("outcome forecast records are not adjacent side-swap pairs")
        pairs.append((original, swapped))
    return tuple(pairs)


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


def _parse_outcome_training_record(text: str) -> OutcomeTrainingRecord:
    record = parse_json_object(text)
    require_exact_keys(record, {"question_id", "messages", "label"}, "outcome record")
    question_id = require_string(required_field(record, "question_id"), "question_id")
    label = require_label(require_string(required_field(record, "label"), "label"))
    values = require_list(required_field(record, "messages"), "messages")
    messages = [_parse_chat_message(value, index) for index, value in enumerate(values)]
    if [message["role"] for message in messages] != ["system", "user"]:
        raise JsonFormatError("outcome training messages must be target-free system and user turns")
    if messages[0]["content"] != OUTCOME_SYSTEM_PROMPT:
        raise JsonFormatError("outcome training record uses an unexpected system prompt")
    return OutcomeTrainingRecord(question_id=question_id, messages=messages, label=label)


def _parse_outcome_forecast_record(text: str) -> ForecastRecord:
    record = parse_json_object(text)
    require_exact_keys(record, {"question_id", "messages"}, "outcome forecast record")
    question_id = require_string(required_field(record, "question_id"), "question_id")
    values = require_list(required_field(record, "messages"), "messages")
    messages = [_parse_chat_message(value, index) for index, value in enumerate(values)]
    if [message["role"] for message in messages] != ["system", "user"]:
        raise JsonFormatError("outcome forecast messages must be target-free system and user turns")
    if messages[0]["content"] != OUTCOME_SYSTEM_PROMPT:
        raise JsonFormatError("outcome forecast record uses an unexpected system prompt")
    return ForecastRecord(question_id=question_id, messages=messages)


def _parse_chat_message(value: object, index: int) -> ChatMessage:
    field_name = f"messages[{index}]"
    record = require_object(value, field_name)
    require_exact_keys(record, {"role", "content"}, field_name)
    role = require_string(required_field(record, "role"), f"{field_name}.role")
    if role not in {"system", "user", "assistant"}:
        raise JsonFormatError(f"unsupported message role: {role}")
    return ChatMessage(
        role=cast(Literal["system", "user", "assistant"], role),
        content=require_string(required_field(record, "content"), f"{field_name}.content"),
    )
