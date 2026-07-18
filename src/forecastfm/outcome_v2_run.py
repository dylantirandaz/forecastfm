"""Immutable local commitment for one prepared outcome-v2 paid run."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path

from forecastfm.integrity import (
    bytes_sha256,
    canonical_json,
    canonical_sha256,
    file_sha256,
    text_sha256,
)
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.outcome import OUTCOME_INPUT_SCHEMA_VERSION
from forecastfm.outcome_v2_config import (
    OUTCOME_RENDERER_NAME,
    outcome_v2_elo_recipe,
    outcome_v2_evaluation_policy,
    outcome_v2_inference_settings,
    outcome_v2_sft_settings,
    outcome_v2_training_settings,
)
from forecastfm.outcome_v2_preflight import OutcomeV2Preflight, PreparedOutcomeV2Run
from forecastfm.outcome_v2_prompt import OUTCOME_V2_SYSTEM_PROMPT
from forecastfm.run_config import TORCH_VERSION, model_settings, package_versions

OUTCOME_V2_RUN_LOCK_SCHEMA_VERSION = 2

OUTCOME_V2_LOCKED_CODE_FILES = (
    (
        "publication_workflow",
        Path(".github/workflows/outcome-v2-publication-timestamp.yml"),
    ),
    ("entrypoint", Path("examples/train_tinker_outcome_v2_sft.py")),
    ("fit_rich_baseline", Path("examples/fit_nba_rich_baseline.py")),
    ("predict_rich_baseline", Path("examples/predict_nba_rich_baseline.py")),
    ("paid_runtime", Path("examples/tinker_outcome_v2_runtime.py")),
    (
        "paid_inference_runtime",
        Path("examples/tinker_outcome_v2_inference_runtime.py"),
    ),
    ("shared_outcome_dataset", Path("examples/train_tinker_outcome_sft.py")),
    ("outcome_v2_run", Path("src/forecastfm/outcome_v2_run.py")),
    ("outcome_v2_aggregation", Path("src/forecastfm/outcome_v2_aggregation.py")),
    ("outcome_v2_experiment", Path("src/forecastfm/outcome_v2_experiment.py")),
    ("outcome_v2_coverage", Path("src/forecastfm/outcome_v2_coverage.py")),
    ("outcome_v2_inference", Path("src/forecastfm/outcome_v2_inference.py")),
    ("outcome_v2_preflight", Path("src/forecastfm/outcome_v2_preflight.py")),
    ("outcome_v2_prompt", Path("src/forecastfm/outcome_v2_prompt.py")),
    ("outcome_v2_rolling", Path("src/forecastfm/outcome_v2_rolling.py")),
    (
        "outcome_v2_rolling_score",
        Path("src/forecastfm/outcome_v2_rolling_score.py"),
    ),
    (
        "outcome_v2_rolling_gate",
        Path("src/forecastfm/outcome_v2_rolling_gate.py"),
    ),
    ("outcome_v2_sft_gate", Path("src/forecastfm/outcome_v2_sft_gate.py")),
    ("outcome_v2_config", Path("src/forecastfm/outcome_v2_config.py")),
    ("outcome", Path("src/forecastfm/outcome.py")),
    ("tinker_data", Path("src/forecastfm/tinker_data.py")),
    ("tinker_screening", Path("src/forecastfm/tinker_screening.py")),
    ("run_config", Path("src/forecastfm/run_config.py")),
    ("integrity", Path("src/forecastfm/integrity.py")),
    (
        "github_actions_receipt",
        Path("src/forecastfm/github_actions_receipt.py"),
    ),
    ("json_utils", Path("src/forecastfm/json_utils.py")),
    ("local_config", Path("src/forecastfm/local_config.py")),
    ("ledger", Path("src/forecastfm/ledger.py")),
    ("models", Path("src/forecastfm/models.py")),
    ("prompting", Path("src/forecastfm/prompting.py")),
    ("elo_residual", Path("src/forecastfm/elo_residual.py")),
    ("nba_data", Path("src/forecastfm/nba_data.py")),
    ("nba_elo_replay", Path("src/forecastfm/nba_elo_replay.py")),
    ("nba_elo_state", Path("src/forecastfm/nba_elo_state.py")),
    ("nba_evaluation_gate", Path("src/forecastfm/nba_evaluation_gate.py")),
    ("nba_evidence", Path("src/forecastfm/nba_evidence.py")),
    ("nba_evidence_io", Path("src/forecastfm/nba_evidence_io.py")),
    ("nba_feature_rows", Path("src/forecastfm/nba_feature_rows.py")),
    ("nba_resolutions", Path("src/forecastfm/nba_resolutions.py")),
    ("nba_rich", Path("src/forecastfm/nba_rich.py")),
    ("nba_rich_baseline", Path("src/forecastfm/nba_rich_baseline.py")),
    ("nba_rights_lock", Path("src/forecastfm/nba_rights_lock.py")),
    ("nba_snapshot_pack", Path("src/forecastfm/nba_snapshot_pack.py")),
    ("outcome_v2_metrics", Path("src/forecastfm/outcome_v2_metrics.py")),
    ("publication", Path("src/forecastfm/publication.py")),
)

_KIND = "forecastfm_outcome_v2_paid_run"
_STATUS = "committed_before_remote_client"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40,64}")
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_LOCK_KEYS = {
    "schema_version",
    "kind",
    "status",
    "action_at",
    "created_at",
    "code_revision",
    "code_sha256",
    "preflight",
    "training_bytes_sha256",
    "sft_settings",
    "training_settings",
    "inference_settings",
    "elo_recipe",
    "evaluation_policy",
    "model",
    "model_reference_sha256",
    "prompt",
    "packages",
    "uv_lock_sha256",
}
_HASHED_CONFIG_KEYS = {"config", "sha256"}
_PROMPT_KEYS = {
    "outcome_input_schema_version",
    "system_prompt",
    "system_prompt_sha256",
}
_PREFLIGHT_FIELD_NAMES = tuple(item.name for item in fields(OutcomeV2Preflight))

type JsonObject = dict[str, object]


class OutcomeV2RunError(ValueError):
    """Raised when an outcome-v2 paid-run commitment is invalid or stale."""


@dataclass(frozen=True, slots=True)
class OutcomeV2RunLock:
    """One strict canonical JSON commitment retained as immutable bytes."""

    canonical_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_immutable_bytes(self.canonical_bytes)
        _record_from_bytes(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        """Return the digest of the exact bytes written to disk."""
        return bytes_sha256(self.canonical_bytes)

    def to_record(self) -> JsonObject:
        """Return a newly decoded copy of the strict lock record."""
        return _record_from_bytes(self.canonical_bytes)


def build_outcome_v2_run_lock(
    project_root: Path,
    prepared: PreparedOutcomeV2Run,
    code_revision: str,
) -> OutcomeV2RunLock:
    """Build the exact local commitment for one already prepared paid run."""
    _require_revision(code_revision)
    preflight = _preflight_payload(prepared.proof)
    training_sha256 = bytes_sha256(prepared.training_jsonl)
    if training_sha256 != prepared.proof.training_sha256:
        raise OutcomeV2RunError("prepared training bytes differ from the preflight proof")

    model = model_settings()
    model["renderer"] = OUTCOME_RENDERER_NAME
    elo_recipe = outcome_v2_elo_recipe()
    evaluation_policy = outcome_v2_evaluation_policy()
    action_at = _utc_text(prepared.proof.action_at, "action_at")
    record: JsonObject = {
        "schema_version": OUTCOME_V2_RUN_LOCK_SCHEMA_VERSION,
        "kind": _KIND,
        "status": _STATUS,
        "action_at": action_at,
        "created_at": action_at,
        "code_revision": code_revision,
        "code_sha256": _code_hashes(project_root),
        "preflight": preflight,
        "training_bytes_sha256": training_sha256,
        "sft_settings": outcome_v2_sft_settings(),
        "training_settings": outcome_v2_training_settings(),
        "inference_settings": outcome_v2_inference_settings(),
        "elo_recipe": {
            "config": elo_recipe.canonical_payload(),
            "sha256": elo_recipe.recipe_sha256,
        },
        "evaluation_policy": {
            "config": evaluation_policy.canonical_payload(),
            "sha256": evaluation_policy.policy_sha256,
        },
        "model": model,
        "model_reference_sha256": canonical_sha256(model),
        "prompt": {
            "outcome_input_schema_version": OUTCOME_INPUT_SCHEMA_VERSION,
            "system_prompt": OUTCOME_V2_SYSTEM_PROMPT,
            "system_prompt_sha256": text_sha256(OUTCOME_V2_SYSTEM_PROMPT),
        },
        "packages": {**package_versions(), "torch": TORCH_VERSION},
        "uv_lock_sha256": _file_hash(project_root / "uv.lock", "uv.lock"),
    }
    return OutcomeV2RunLock(canonical_json(record).encode("utf-8"))


def write_outcome_v2_run_lock(path: Path, lock: OutcomeV2RunLock) -> str:
    """Create the canonical lock exactly once and durably flush its bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as file:
            file.write(lock.canonical_bytes)
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError:
        raise
    except OSError as error:
        raise OutcomeV2RunError("cannot write the outcome-v2 run lock") from error
    return lock.sha256


def verify_outcome_v2_run_lock(
    project_root: Path,
    path: Path,
    prepared: PreparedOutcomeV2Run,
    code_revision: str,
) -> OutcomeV2RunLock:
    """Verify exact lock bytes against current code, config, and prepared data."""
    _require_revision(code_revision)
    try:
        actual = OutcomeV2RunLock(path.read_bytes())
    except OSError as error:
        raise OutcomeV2RunError("cannot read the outcome-v2 run lock") from error
    expected = build_outcome_v2_run_lock(project_root, prepared, code_revision)
    if actual.canonical_bytes != expected.canonical_bytes:
        raise OutcomeV2RunError(
            "outcome-v2 run lock differs from current code, config, or prepared data"
        )
    return actual


def require_outcome_v2_run_static_contract(
    project_root: Path,
    lock: OutcomeV2RunLock,
) -> None:
    """Revalidate every current code/config contract that does not need source data."""
    record = lock.to_record()
    expected_model = model_settings()
    expected_model["renderer"] = OUTCOME_RENDERER_NAME
    elo_recipe = outcome_v2_elo_recipe()
    evaluation_policy = outcome_v2_evaluation_policy()
    expected: tuple[tuple[str, object], ...] = (
        ("code_sha256", _code_hashes(project_root)),
        ("sft_settings", outcome_v2_sft_settings()),
        ("training_settings", outcome_v2_training_settings()),
        ("inference_settings", outcome_v2_inference_settings()),
        (
            "elo_recipe",
            {
                "config": elo_recipe.canonical_payload(),
                "sha256": elo_recipe.recipe_sha256,
            },
        ),
        (
            "evaluation_policy",
            {
                "config": evaluation_policy.canonical_payload(),
                "sha256": evaluation_policy.policy_sha256,
            },
        ),
        ("model", expected_model),
        ("model_reference_sha256", canonical_sha256(expected_model)),
        (
            "prompt",
            {
                "outcome_input_schema_version": OUTCOME_INPUT_SCHEMA_VERSION,
                "system_prompt": OUTCOME_V2_SYSTEM_PROMPT,
                "system_prompt_sha256": text_sha256(OUTCOME_V2_SYSTEM_PROMPT),
            },
        ),
        ("packages", {**package_versions(), "torch": TORCH_VERSION}),
        ("uv_lock_sha256", _file_hash(project_root / "uv.lock", "uv.lock")),
    )
    for field_name, expected_value in expected:
        if required_field(record, field_name) != expected_value:
            raise OutcomeV2RunError(
                f"outcome-v2 run lock differs from the current {field_name} contract"
            )


def _preflight_payload(proof: OutcomeV2Preflight) -> JsonObject:
    payload = proof.canonical_payload()
    if tuple(payload) != _PREFLIGHT_FIELD_NAMES:
        raise OutcomeV2RunError("outcome-v2 preflight fields changed; update the run lock")
    _validate_preflight_payload(payload)
    return payload


def _code_hashes(project_root: Path) -> JsonObject:
    return {
        logical_name: _file_hash(project_root / relative_path, f"code file {logical_name}")
        for logical_name, relative_path in OUTCOME_V2_LOCKED_CODE_FILES
    }


def _file_hash(path: Path, field_name: str) -> str:
    try:
        return file_sha256(path)
    except OSError as error:
        raise OutcomeV2RunError(f"cannot hash {field_name}") from error


def _require_immutable_bytes(value: object) -> None:
    if not isinstance(value, bytes):
        raise OutcomeV2RunError("outcome-v2 run lock requires immutable bytes")


def _record_from_bytes(value: bytes) -> JsonObject:
    try:
        text = value.decode("utf-8")
        record = parse_json_object(text)
    except (JsonFormatError, UnicodeError) as error:
        raise OutcomeV2RunError("outcome-v2 run lock must be one UTF-8 JSON object") from error
    if text != canonical_json(record):
        raise OutcomeV2RunError("outcome-v2 run lock must use canonical JSON bytes")
    _validate_record(record)
    return record


def _validate_record(record: Mapping[str, object]) -> None:
    try:
        _validate_lock_identity(record)
        _validate_lock_bindings(record)
    except JsonFormatError as error:
        raise OutcomeV2RunError("invalid outcome-v2 run lock structure") from error


def _validate_lock_identity(record: Mapping[str, object]) -> None:
    require_exact_keys(record, _LOCK_KEYS, "outcome-v2 run lock")
    if _integer(record, "schema_version") != OUTCOME_V2_RUN_LOCK_SCHEMA_VERSION:
        raise OutcomeV2RunError("unsupported outcome-v2 run lock schema")
    if _string(record, "kind") != _KIND:
        raise OutcomeV2RunError("unexpected outcome-v2 run lock kind")
    if _string(record, "status") != _STATUS:
        raise OutcomeV2RunError("outcome-v2 run lock is not committed before remote client")
    action_at = _parse_utc(_string(record, "action_at"), "action_at")
    created_at = _parse_utc(_string(record, "created_at"), "created_at")
    if action_at != created_at:
        raise OutcomeV2RunError("action_at and created_at must be identical")
    _require_revision(_string(record, "code_revision"))


def _validate_lock_bindings(record: Mapping[str, object]) -> None:
    _validate_code_hashes(_object(record, "code_sha256"))
    preflight = _object(record, "preflight")
    _validate_preflight_payload(preflight)
    if _string(preflight, "action_at") != _string(record, "action_at"):
        raise OutcomeV2RunError("preflight action_at must equal the lock action_at")
    training_sha256 = _string(record, "training_bytes_sha256")
    _require_hash(training_sha256, "training_bytes_sha256")
    if training_sha256 != _string(preflight, "training_sha256"):
        raise OutcomeV2RunError("training bytes hash differs from the preflight record")
    _validate_nonempty_objects(record)
    _validate_batch_size_bindings(record, preflight)
    _validate_hashed_config(_object(record, "elo_recipe"), "elo_recipe")
    _validate_hashed_config(_object(record, "evaluation_policy"), "evaluation_policy")
    _validate_model(record)
    _validate_prompt(_object(record, "prompt"))
    _require_hash(_string(record, "uv_lock_sha256"), "uv_lock_sha256")


def _validate_nonempty_objects(record: Mapping[str, object]) -> None:
    for field_name in (
        "sft_settings",
        "training_settings",
        "inference_settings",
        "model",
        "packages",
    ):
        if not _object(record, field_name):
            raise OutcomeV2RunError(f"{field_name} must not be empty")


def _validate_batch_size_bindings(
    record: Mapping[str, object], preflight: Mapping[str, object]
) -> None:
    batch_sizes = {
        _positive_integer(preflight, "batch_size"),
        _positive_integer(_object(record, "sft_settings"), "batch_size"),
        _positive_integer(_object(record, "training_settings"), "batch_size"),
    }
    if len(batch_sizes) != 1:
        raise OutcomeV2RunError("preflight and training batch sizes must match")


def _validate_model(record: Mapping[str, object]) -> None:
    model = _object(record, "model")
    model_hash = _string(record, "model_reference_sha256")
    _require_hash(model_hash, "model_reference_sha256")
    if canonical_sha256(model) != model_hash:
        raise OutcomeV2RunError("model_reference_sha256 does not match the model")


def _validate_code_hashes(record: Mapping[str, object]) -> None:
    expected_names = {name for name, _ in OUTCOME_V2_LOCKED_CODE_FILES}
    require_exact_keys(record, expected_names, "code_sha256")
    for name in expected_names:
        _require_hash(_string(record, name), f"code_sha256.{name}")


def _validate_preflight_payload(record: Mapping[str, object]) -> None:
    require_exact_keys(record, set(_PREFLIGHT_FIELD_NAMES), "preflight")
    for field_name in _PREFLIGHT_FIELD_NAMES:
        value = required_field(record, field_name)
        if field_name.endswith("_sha256"):
            digest = require_string(value, field_name)
            _require_hash(digest, f"preflight.{field_name}")
    _parse_utc(_string(record, "action_at"), "preflight.action_at")
    if _string(record, "action_time_source") != "internal_paid_preparation":
        raise OutcomeV2RunError("preflight action time must be derived during paid-run preparation")
    _validate_evaluation_seasons(record)
    _validate_preflight_counts(record)


def _validate_evaluation_seasons(record: Mapping[str, object]) -> None:
    seasons = require_list(
        required_field(record, "untouched_evaluation_seasons"),
        "preflight.untouched_evaluation_seasons",
    )
    if len(seasons) < 2:
        raise OutcomeV2RunError("preflight must bind at least two evaluation seasons")
    validated_seasons: list[int] = []
    for season in seasons:
        if isinstance(season, bool) or not isinstance(season, int):
            raise OutcomeV2RunError("preflight evaluation seasons must be integers")
        validated_seasons.append(season)
    if len(validated_seasons) != len(set(validated_seasons)):
        raise OutcomeV2RunError("preflight evaluation seasons must be unique")


def _validate_preflight_counts(record: Mapping[str, object]) -> None:
    row_count = _positive_integer(record, "row_count")
    pair_count = _positive_integer(record, "pair_count")
    _positive_integer(record, "batch_size")
    if row_count != pair_count * 2:
        raise OutcomeV2RunError("preflight row_count must contain exact side-swap pairs")


def _validate_hashed_config(record: Mapping[str, object], field_name: str) -> None:
    require_exact_keys(record, _HASHED_CONFIG_KEYS, field_name)
    config = _object(record, "config")
    expected_hash = _string(record, "sha256")
    _require_hash(expected_hash, f"{field_name}.sha256")
    if canonical_sha256(config) != expected_hash:
        raise OutcomeV2RunError(f"{field_name}.sha256 does not match its config")


def _validate_prompt(record: Mapping[str, object]) -> None:
    require_exact_keys(record, _PROMPT_KEYS, "prompt")
    _integer(record, "outcome_input_schema_version")
    prompt = _string(record, "system_prompt")
    prompt_hash = _string(record, "system_prompt_sha256")
    _require_hash(prompt_hash, "prompt.system_prompt_sha256")
    if text_sha256(prompt) != prompt_hash:
        raise OutcomeV2RunError("system_prompt_sha256 does not match the prompt")


def _object(record: Mapping[str, object], field_name: str) -> JsonObject:
    return require_object(required_field(record, field_name), field_name)


def _string(record: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = required_field(record, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutcomeV2RunError(f"{field_name} must be an integer")
    return value


def _positive_integer(record: Mapping[str, object], field_name: str) -> int:
    value = _integer(record, field_name)
    if value <= 0:
        raise OutcomeV2RunError(f"{field_name} must be positive")
    return value


def _require_revision(value: str) -> None:
    if _REVISION_PATTERN.fullmatch(value) is None:
        raise OutcomeV2RunError("code_revision must be a lowercase 40-64 character Git ID")


def _require_hash(value: str, field_name: str) -> None:
    if _HASH_PATTERN.fullmatch(value) is None:
        raise OutcomeV2RunError(f"{field_name} must be a lowercase SHA-256 digest")


def _parse_utc(value: str, field_name: str) -> datetime:
    if not value.endswith("Z"):
        raise OutcomeV2RunError(f"{field_name} must use canonical UTC Z notation")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise OutcomeV2RunError(f"{field_name} must be an ISO 8601 datetime") from error
    if _utc_text(parsed, field_name) != value:
        raise OutcomeV2RunError(f"{field_name} must use canonical UTC notation")
    return parsed.astimezone(UTC)


def _utc_text(value: datetime, field_name: str) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise OutcomeV2RunError(f"{field_name} must be in UTC")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
