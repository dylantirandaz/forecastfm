"""Strict helpers for converting untyped JSON into typed Python values."""

import json
from collections.abc import Mapping
from math import isfinite
from typing import Never, cast


class JsonFormatError(ValueError):
    """Raised when decoded JSON does not match an expected shape."""


def _reject_nonfinite_constant(value: str) -> Never:
    raise JsonFormatError(f"non-finite JSON number: {value}")


def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise JsonFormatError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_object(text: str) -> dict[str, object]:
    """Decode a JSON object while containing `Any` at this boundary."""
    try:
        value = cast(
            object,
            json.loads(
                text,
                parse_constant=_reject_nonfinite_constant,
                object_pairs_hook=_object_from_pairs,
            ),
        )
    except json.JSONDecodeError as error:
        raise JsonFormatError("invalid JSON") from error
    return require_object(value, "root")


def require_object(value: object, field_name: str) -> dict[str, object]:
    """Validate and return a string-keyed JSON object."""
    if not isinstance(value, Mapping):
        raise JsonFormatError(f"{field_name} must be an object")
    mapping = cast(Mapping[object, object], value)
    result: dict[str, object] = {}
    for key, item in mapping.items():
        if not isinstance(key, str):
            raise JsonFormatError(f"{field_name} keys must be strings")
        result[key] = item
    return result


def require_list(value: object, field_name: str) -> list[object]:
    """Validate and return a JSON list."""
    if not isinstance(value, list):
        raise JsonFormatError(f"{field_name} must be a list")
    return cast(list[object], value)


def require_string(value: object, field_name: str) -> str:
    """Validate and return a JSON string."""
    if not isinstance(value, str):
        raise JsonFormatError(f"{field_name} must be a string")
    return value


def require_float(value: object, field_name: str) -> float:
    """Validate and return a JSON number as a float."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise JsonFormatError(f"{field_name} must be a number")
    try:
        result = float(value)
    except OverflowError as error:
        raise JsonFormatError(f"{field_name} must be a finite number") from error
    if not isfinite(result):
        raise JsonFormatError(f"{field_name} must be a finite number")
    return result


def required_field(mapping: Mapping[str, object], field_name: str) -> object:
    """Return a required field with a clear error when it is absent."""
    try:
        return mapping[field_name]
    except KeyError as error:
        raise JsonFormatError(f"missing field: {field_name}") from error


def require_exact_keys(mapping: Mapping[str, object], keys: set[str], field_name: str) -> None:
    """Reject missing and unexpected object keys."""
    actual_keys = set(mapping)
    if actual_keys != keys:
        missing = sorted(keys - actual_keys)
        extra = sorted(actual_keys - keys)
        raise JsonFormatError(f"{field_name} keys differ; missing={missing}, extra={extra}")
