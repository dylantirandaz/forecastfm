"""Sealed rl-prompt-v1 question set for the NBA RL run (protocol prerequisite 5).

The prompt template is frozen byte-for-byte as version ``rl-prompt-v1``:

SYSTEM (exactly one line):
    You are a calibrated NBA forecasting model. Given an Elo prior and pregame evidence
    differences (home minus away), estimate the probability that the listed team wins.
    Answer with exactly one label: TEAM if the listed team wins, OTHER if the opponent wins.

USER (one line per field, in this exact order with this exact formatting):
    elo_home_probability: {p:.4f}
    {name}: {value:+.3f}     repeated for the 11 names in NBA_RICH_FEATURE_NAMES order
    winner_label:

``winner_label:`` is the answer position for rollouts; prompts never contain the realized
winner. Decision 2a is A (default, strict): prompts contain the MOV Elo prior and the 11
standard features only — no health-derived value appears anywhere in this dataset.

Each game contributes two orientations bound by the exact side-swap contract: the swapped
prompt negates every feature and complements the Elo probability (1 - p), so the two
orientations are exactly recoverable from each other. Question IDs are
``nba-{game_id}-T-60`` for the original and gain SIDE_SWAP_SUFFIX when swapped. Answers
live in a separate file so prompt text and outcomes never mix.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from forecastfm.integrity import canonical_json, canonical_sha256, file_sha256
from forecastfm.json_utils import JsonFormatError, require_object, require_string, required_field
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_prototype_dataset import PrototypeGameRow
from forecastfm.nba_rich import NBA_RICH_FEATURE_NAMES
from forecastfm.outcome import OPPONENT_LABEL, TEAM_LABEL

RL_PROMPT_TEMPLATE_VERSION = "rl-prompt-v1"
RL_DATASET_SCHEMA_VERSION = 1
RL_ELO_FIELD = "elo_home_probability"
RL_ANSWER_FIELD = "winner_label"
ORIGINAL_ORIENTATION = "original"
SWAPPED_ORIENTATION = "side-swap"
PROMPTS_FILENAME = "prompts.jsonl"
ANSWERS_FILENAME = "answers.jsonl"
MANIFEST_FILENAME = "manifest.json"

RL_SYSTEM_PROMPT = (
    "You are a calibrated NBA forecasting model. Given an Elo prior and pregame evidence "
    "differences (home minus away), estimate the probability that the listed team wins. "
    "Answer with exactly one label: TEAM if the listed team wins, OTHER if the opponent wins."
)

RL_USER_TEMPLATE = "\n".join(
    [
        f"{RL_ELO_FIELD}: {{probability:.4f}}",
        *(f"{name}: {{value:+.3f}}" for name in NBA_RICH_FEATURE_NAMES),
        f"{RL_ANSWER_FIELD}:",
    ]
)

_CHRONOLOGICAL_SPLIT_NOTE = (
    "rows are sealed in chronological (game_date, game_id) order; train/evaluation splits "
    "are chronological by season label and exact prompt overlap across splits is rejected"
)
_HEALTH_DISCLOSURE = (
    "decision 2a is A: no health-derived values (no unavailable_rotation_minutes, no "
    "unavailable_rotation_value, no projected_rotation_value) appear anywhere in the prompts"
)


class NbaRlDatasetError(RuntimeError):
    """Raised when the RL question set cannot be built, sealed, or verified."""


def rl_question_id(row: PrototypeGameRow, *, swapped: bool) -> str:
    """Return the RL question ID for one orientation of one game."""
    base = f"nba-{row.game_id}-T-60"
    return f"{base}{SIDE_SWAP_SUFFIX}" if swapped else base


def swap_row(row: PrototypeGameRow) -> PrototypeGameRow:
    """Return the away-perspective view of one row; an exact involution."""
    return replace(
        row,
        elo_home_probability=1.0 - row.elo_home_probability,
        features_standard=tuple(-value for value in row.features_standard),
        home_won=not row.home_won,
    )


def build_prompt(row: PrototypeGameRow, *, swapped: bool) -> tuple[str, str]:
    """Render the frozen rl-prompt-v1 (system, user) pair for one orientation."""
    if len(row.features_standard) != len(NBA_RICH_FEATURE_NAMES):
        raise NbaRlDatasetError("RL prompts require exactly the 11 standard features")
    probability = row.elo_home_probability
    features = row.features_standard
    if swapped:
        probability = 1.0 - probability
        features = tuple(-value for value in features)
    if not 0.0 <= probability <= 1.0:
        raise NbaRlDatasetError("elo_home_probability must lie in [0, 1]")
    lines = [f"{RL_ELO_FIELD}: {probability:.4f}"]
    lines.extend(
        f"{name}: {value:+.3f}"
        for name, value in zip(NBA_RICH_FEATURE_NAMES, features, strict=True)
    )
    lines.append(f"{RL_ANSWER_FIELD}:")
    return RL_SYSTEM_PROMPT, "\n".join(lines)


def answer_label(row: PrototypeGameRow, *, swapped: bool) -> str:
    """Return TEAM when the orientation's listed team won; otherwise OTHER."""
    listed_won = not row.home_won if swapped else row.home_won
    return TEAM_LABEL if listed_won else OPPONENT_LABEL


def prompt_record(row: PrototypeGameRow, *, swapped: bool) -> dict[str, object]:
    """Return the canonical prompts.jsonl record for one orientation."""
    system, user = build_prompt(row, swapped=swapped)
    return {
        "question_id": rl_question_id(row, swapped=swapped),
        "system": system,
        "user": user,
        "orientation": SWAPPED_ORIENTATION if swapped else ORIGINAL_ORIENTATION,
        "season": row.season,
        "game_date": row.game_date.isoformat(),
    }


def answer_record(row: PrototypeGameRow, *, swapped: bool) -> dict[str, object]:
    """Return the canonical answers.jsonl record for one orientation."""
    return {
        "question_id": rl_question_id(row, swapped=swapped),
        "winner": answer_label(row, swapped=swapped),
    }


def prompt_template_sha256() -> str:
    """Hash the frozen rl-prompt-v1 template (system text plus user skeleton)."""
    return canonical_sha256(
        {
            "template_version": RL_PROMPT_TEMPLATE_VERSION,
            "system": RL_SYSTEM_PROMPT,
            "user": RL_USER_TEMPLATE,
        }
    )


def seal_rl_dataset(
    rows: list[PrototypeGameRow],
    output_dir: Path,
) -> dict[str, object]:
    """Write prompts, answers, and manifest create-only; return the manifest payload."""
    ordered = sorted(rows, key=lambda row: (row.game_date, row.game_id))
    if len({row.game_id for row in ordered}) != len(ordered):
        raise NbaRlDatasetError("RL rows contain a duplicate game_id")
    prompt_lines: list[str] = []
    answer_lines: list[str] = []
    for row in ordered:
        for swapped in (False, True):
            prompt_lines.append(canonical_json(prompt_record(row, swapped=swapped)))
            answer_lines.append(canonical_json(answer_record(row, swapped=swapped)))
    prompts_path = output_dir / PROMPTS_FILENAME
    answers_path = output_dir / ANSWERS_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_create_only(prompts_path, "".join(f"{line}\n" for line in prompt_lines))
    _write_create_only(answers_path, "".join(f"{line}\n" for line in answer_lines))
    manifest = _manifest(ordered, prompts_path, answers_path)
    _write_create_only(output_dir / MANIFEST_FILENAME, f"{canonical_json(manifest)}\n")
    return manifest


def verify_sealed_dataset(output_dir: Path) -> dict[str, object]:
    """Reload a sealed dataset and confirm every recorded hash reproduces."""
    manifest_path = output_dir / MANIFEST_FILENAME
    try:
        manifest = require_object(json.loads(manifest_path.read_text(encoding="utf-8")), "root")
        template_hash = require_string(
            required_field(manifest, "prompt_template_sha256"), "prompt_template_sha256"
        )
    except OSError as error:
        raise NbaRlDatasetError("cannot read the RL dataset manifest") from error
    except JsonFormatError as error:
        raise NbaRlDatasetError("RL dataset manifest is malformed") from error
    if template_hash != prompt_template_sha256():
        raise NbaRlDatasetError("sealed prompt template hash differs from rl-prompt-v1")
    files = require_object(required_field(manifest, "files"), "files")
    for name in (PROMPTS_FILENAME, ANSWERS_FILENAME):
        entry = require_object(required_field(files, name), name)
        recorded = require_string(required_field(entry, "sha256"), f"{name}.sha256")
        try:
            actual = file_sha256(output_dir / name)
        except OSError as error:
            raise NbaRlDatasetError(f"cannot read sealed {name}") from error
        if actual != recorded:
            raise NbaRlDatasetError(f"sealed {name} hash differs from the manifest")
    return manifest


def _manifest(
    rows: list[PrototypeGameRow],
    prompts_path: Path,
    answers_path: Path,
) -> dict[str, object]:
    seasons: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = seasons.setdefault(str(row.season), {"games": 0, "prompts": 0})
        entry["games"] += 1
        entry["prompts"] += 2
    return {
        "schema_version": RL_DATASET_SCHEMA_VERSION,
        "prompt_template_version": RL_PROMPT_TEMPLATE_VERSION,
        "prompt_template_sha256": prompt_template_sha256(),
        "candidate_labels": [TEAM_LABEL, OPPONENT_LABEL],
        "elo_field": RL_ELO_FIELD,
        "answer_field": RL_ANSWER_FIELD,
        "orientations_per_game": [ORIGINAL_ORIENTATION, SWAPPED_ORIENTATION],
        "total_games": len(rows),
        "total_prompts": 2 * len(rows),
        "seasons": seasons,
        "files": {
            PROMPTS_FILENAME: {"sha256": file_sha256(prompts_path)},
            ANSWERS_FILENAME: {"sha256": file_sha256(answers_path)},
        },
        "chronological_split_note": _CHRONOLOGICAL_SPLIT_NOTE,
        "decision_2a": "A",
        "health_disclosure": _HEALTH_DISCLOSURE,
        "side_swap_contract": (
            "swapped prompts negate all 11 features and complement elo_home_probability; "
            "labels flip so the listed team's result is reported"
        ),
        "created_by": "examples/build_rl_dataset.py",
    }


def _write_create_only(path: Path, text: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as file:
            file.write(text)
    except FileExistsError as error:
        raise NbaRlDatasetError(f"refusing to replace sealed artifact: {path}") from error
    except OSError as error:
        raise NbaRlDatasetError(f"cannot write sealed artifact: {path}") from error
