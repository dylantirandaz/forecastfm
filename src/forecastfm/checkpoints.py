"""Strict readers for completed Tinker checkpoint logs."""

from pathlib import Path

from forecastfm.json_utils import parse_json_object, require_string, required_field


def read_final_checkpoint(path: Path) -> dict[str, object]:
    """Read one final state/sampler pair from the end of a checkpoint log."""
    try:
        lines = tuple(line for line in path.read_text(encoding="utf-8").splitlines() if line)
    except FileNotFoundError as error:
        raise RuntimeError("Tinker checkpoint log is missing; finish training first") from error
    if not lines:
        raise RuntimeError("Tinker checkpoint log is empty")

    records = tuple(parse_json_object(line) for line in lines)
    names = tuple(require_string(required_field(record, "name"), "name") for record in records)
    if names.count("final") != 1:
        raise RuntimeError("Tinker checkpoint log must contain exactly one final record")
    if names[-1] != "final":
        raise RuntimeError("Tinker final checkpoint must be the last record")

    final = records[-1]
    state_path = require_string(required_field(final, "state_path"), "state_path")
    sampler_path = require_string(required_field(final, "sampler_path"), "sampler_path")
    state_suffix = "/weights/final"
    sampler_suffix = "/sampler_weights/final"
    if not state_path.startswith("tinker://") or not state_path.endswith(state_suffix):
        raise RuntimeError("Tinker final state path has an unexpected format")
    if not sampler_path.startswith("tinker://") or not sampler_path.endswith(sampler_suffix):
        raise RuntimeError("Tinker final sampler path has an unexpected format")
    if state_path.removesuffix(state_suffix) != sampler_path.removesuffix(sampler_suffix):
        raise RuntimeError("Tinker final state and sampler must belong to the same run")
    return final
