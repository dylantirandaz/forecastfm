"""Exact, externally timed schedule coverage for rolling outcome-v2 forecasts."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from forecastfm.github_actions_receipt import (
    GitHubActionsReceipt,
    GitHubActionsReceiptRequest,
    read_github_actions_receipt,
    verify_github_actions_receipt,
)
from forecastfm.integrity import bytes_sha256, canonical_json, canonical_sha256
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.nba_elo_replay import (
    NbaEloReplayError,
    NbaEloReplayRow,
    read_nba_elo_replay_rows_jsonl_bytes,
)
from forecastfm.nba_provider_conformance import (
    NBA_PROVIDER_CONFORMANCE_PROOF_SCOPE,
    NBA_PROVIDER_CONFORMANCE_SCHEMA_VERSION,
    NbaProviderConformanceReport,
)
from forecastfm.outcome_v2_rolling import (
    OutcomeV2ProspectivePlanArtifacts,
    outcome_v2_receipt_policy,
    verify_outcome_v2_prospective_plan,
)

OUTCOME_V2_SCHEDULE_COVERAGE_SCHEMA_VERSION = 1

_KIND = "forecastfm_outcome_v2_claimed_inventory_schedule_coverage"
_STATUS = "structurally_bound_to_claimed_conformance_report"
_PROVIDER_AUTHENTICITY = "required_separately"
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_SEAL_KEYS = {
    "commitment_deadline",
    "coverage_policy_sha256",
    "created_at",
    "game_count",
    "kind",
    "plan_sha256",
    "provider_authenticity",
    "provider_conformance_report",
    "provider_conformance_report_sha256",
    "provider_proof_scope",
    "question_ids_sha256",
    "replay_rows_sha256",
    "schema_version",
    "season",
    "source_game_ids_sha256",
    "status",
}
_REPORT_KEYS = {
    "cohort_sha256",
    "connector_id",
    "connector_sha256",
    "cutoff_selection_sha256",
    "inventory_sha256",
    "proof_scope",
    "replay_rows_sha256",
    "revision_count",
    "schedule_derivation_sha256",
    "schedule_game_count",
    "schedule_season_types",
    "schema_version",
    "snapshot_pack_sha256",
    "status",
}

type JsonObject = dict[str, object]


class OutcomeV2CoverageError(ValueError):
    """Raised when schedule coverage is incomplete, late, or inconsistent."""


@dataclass(frozen=True, slots=True)
class OutcomeV2ScheduleCoverageArtifacts:
    """A verified plan, one canonical season schedule, and its conformance report."""

    plan: OutcomeV2ProspectivePlanArtifacts
    plan_path: Path
    replay_rows_path: Path
    provider_conformance_report: NbaProviderConformanceReport


@dataclass(frozen=True, slots=True)
class OutcomeV2ScheduleCoverageConfig:
    """Local seal timing fixed before the external publication receipt."""

    created_at: datetime
    commitment_deadline: datetime

    def __post_init__(self) -> None:
        _require_utc(self.created_at, "created_at")
        _require_utc(self.commitment_deadline, "commitment_deadline")
        if self.created_at >= self.commitment_deadline:
            raise OutcomeV2CoverageError("coverage seal must predate its commitment deadline")


@dataclass(frozen=True, slots=True)
class OutcomeV2ScheduleCoverageSeal:
    """Canonical commitment to one claimed provider-inventory season."""

    canonical_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _seal_record(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        """Return the digest of the exact canonical seal bytes."""
        return bytes_sha256(self.canonical_bytes)

    def to_record(self) -> JsonObject:
        """Return a fresh strict record decoded from the seal."""
        return _seal_record(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class OutcomeV2ScheduleCoverageReceiptArtifacts:
    """One local coverage seal and its live GitHub publication receipt."""

    coverage: OutcomeV2ScheduleCoverageArtifacts
    seal_path: Path
    seal_repository_path: str
    receipt_path: Path

    def __post_init__(self) -> None:
        _require_repository_path(self.seal_repository_path)


@dataclass(frozen=True, slots=True)
class VerifiedOutcomeV2ScheduleCoverage:
    """One structurally bound season schedule with a live commitment time."""

    seal: OutcomeV2ScheduleCoverageSeal
    rows: tuple[NbaEloReplayRow, ...]
    receipt: GitHubActionsReceipt
    externally_committed_at: datetime


def build_outcome_v2_schedule_coverage_seal(
    artifacts: OutcomeV2ScheduleCoverageArtifacts,
    config: OutcomeV2ScheduleCoverageConfig,
) -> OutcomeV2ScheduleCoverageSeal:
    """Bind one complete reviewed season schedule before its first feature input."""
    plan = verify_outcome_v2_prospective_plan(artifacts.plan, artifacts.plan_path)
    plan_record = plan.to_record()
    replay_bytes = _read_bytes(artifacts.replay_rows_path, "schedule replay rows")
    rows = _parse_replay_rows(replay_bytes)
    seasons = {row.season for row in rows}
    if len(seasons) != 1:
        raise OutcomeV2CoverageError("one coverage seal must contain exactly one season")
    season = next(iter(seasons))
    if season not in _integer_tuple(plan_record, "seasons"):
        raise OutcomeV2CoverageError("coverage season is outside the prospective plan")
    plan_created_at = _parse_utc(_string(plan_record, "created_at"), "plan.created_at")
    earliest_cutoff = min(row.forecast_cutoff for row in rows)
    if not plan_created_at <= config.created_at < config.commitment_deadline <= earliest_cutoff:
        raise OutcomeV2CoverageError("coverage timing is outside the prospective season window")

    report = artifacts.provider_conformance_report.canonical_payload()
    _require_report(report)
    if _positive_integer(report, "schedule_game_count") != len(rows):
        raise OutcomeV2CoverageError("provider report game count differs from schedule rows")
    expected_replay_sha256 = canonical_sha256([row.canonical_payload() for row in rows])
    if _string(report, "replay_rows_sha256") != expected_replay_sha256:
        raise OutcomeV2CoverageError("provider report differs from the exact schedule rows")
    record: JsonObject = {
        "schema_version": OUTCOME_V2_SCHEDULE_COVERAGE_SCHEMA_VERSION,
        "kind": _KIND,
        "status": _STATUS,
        "created_at": _utc_text(config.created_at, "created_at"),
        "commitment_deadline": _utc_text(
            config.commitment_deadline,
            "commitment_deadline",
        ),
        "plan_sha256": plan.sha256,
        "coverage_policy_sha256": _string(plan_record, "coverage_policy_sha256"),
        "season": season,
        "replay_rows_sha256": bytes_sha256(replay_bytes),
        "question_ids_sha256": canonical_sha256([row.question_id for row in rows]),
        "source_game_ids_sha256": canonical_sha256([row.source_game_id for row in rows]),
        "game_count": len(rows),
        "provider_conformance_report": report,
        "provider_conformance_report_sha256": canonical_sha256(report),
        "provider_proof_scope": NBA_PROVIDER_CONFORMANCE_PROOF_SCOPE,
        "provider_authenticity": _PROVIDER_AUTHENTICITY,
    }
    seal = OutcomeV2ScheduleCoverageSeal(canonical_json(record).encode("utf-8"))
    if _read_bytes(artifacts.plan_path, "prospective plan") != plan.canonical_bytes:
        raise OutcomeV2CoverageError("prospective plan changed while sealing coverage")
    if _read_bytes(artifacts.replay_rows_path, "schedule replay rows") != replay_bytes:
        raise OutcomeV2CoverageError("schedule replay rows changed while sealing coverage")
    return seal


def write_outcome_v2_schedule_coverage_seal(
    path: Path,
    seal: OutcomeV2ScheduleCoverageSeal,
) -> str:
    """Create and durably flush one schedule coverage seal without replacement."""
    _write_once(path, seal.canonical_bytes, "schedule coverage seal")
    return seal.sha256


def read_outcome_v2_schedule_coverage_seal(path: Path) -> OutcomeV2ScheduleCoverageSeal:
    """Read one strict canonical schedule coverage seal."""
    return OutcomeV2ScheduleCoverageSeal(_read_bytes(path, "schedule coverage seal"))


def verify_outcome_v2_schedule_coverage_seal(
    artifacts: OutcomeV2ScheduleCoverageArtifacts,
    path: Path,
) -> OutcomeV2ScheduleCoverageSeal:
    """Rebuild a schedule coverage seal and compare its exact bytes."""
    actual = read_outcome_v2_schedule_coverage_seal(path)
    record = actual.to_record()
    expected = build_outcome_v2_schedule_coverage_seal(
        artifacts,
        OutcomeV2ScheduleCoverageConfig(
            created_at=_parse_utc(_string(record, "created_at"), "created_at"),
            commitment_deadline=_parse_utc(
                _string(record, "commitment_deadline"),
                "commitment_deadline",
            ),
        ),
    )
    if actual.canonical_bytes != expected.canonical_bytes:
        raise OutcomeV2CoverageError("schedule coverage seal differs from current artifacts")
    return actual


def verify_outcome_v2_schedule_coverage_receipt(
    artifacts: OutcomeV2ScheduleCoverageReceiptArtifacts,
    token: str | None = None,
) -> VerifiedOutcomeV2ScheduleCoverage:
    """Re-fetch GitHub and require the exact season schedule before its deadline."""
    seal = verify_outcome_v2_schedule_coverage_seal(
        artifacts.coverage,
        artifacts.seal_path,
    )
    plan = verify_outcome_v2_prospective_plan(
        artifacts.coverage.plan,
        artifacts.coverage.plan_path,
    )
    record = seal.to_record()
    if _string(record, "plan_sha256") != plan.sha256:
        raise OutcomeV2CoverageError("coverage seal differs from the verified plan")
    receipt = read_github_actions_receipt(artifacts.receipt_path)
    request = GitHubActionsReceiptRequest(
        run_id=_positive_integer(_object(receipt.to_record(), "run"), "id"),
        artifact_path=artifacts.seal_repository_path,
        artifact_bytes=seal.canonical_bytes,
        not_before=_parse_utc(_string(record, "created_at"), "created_at"),
        deadline=_parse_utc(
            _string(record, "commitment_deadline"),
            "commitment_deadline",
        ),
    )
    verify_github_actions_receipt(outcome_v2_receipt_policy(plan), request, receipt, token)
    committed_at = _parse_utc(
        _string(_object(receipt.to_record(), "run"), "created_at"),
        "receipt.run.created_at",
    )
    rows = _verified_receipted_rows(artifacts.coverage.replay_rows_path, record)
    return VerifiedOutcomeV2ScheduleCoverage(seal, rows, receipt, committed_at)


def _verified_receipted_rows(
    path: Path,
    seal: Mapping[str, object],
) -> tuple[NbaEloReplayRow, ...]:
    value = _read_bytes(path, "schedule replay rows")
    if bytes_sha256(value) != _string(seal, "replay_rows_sha256"):
        raise OutcomeV2CoverageError("schedule rows changed after coverage verification")
    rows = _parse_replay_rows(value)
    if len(rows) != _positive_integer(seal, "game_count"):
        raise OutcomeV2CoverageError("receipted schedule row count is invalid")
    if {row.season for row in rows} != {_positive_integer(seal, "season")}:
        raise OutcomeV2CoverageError("receipted schedule season is invalid")
    if canonical_sha256([row.question_id for row in rows]) != _string(
        seal,
        "question_ids_sha256",
    ):
        raise OutcomeV2CoverageError("receipted schedule question IDs are invalid")
    if canonical_sha256([row.source_game_id for row in rows]) != _string(
        seal,
        "source_game_ids_sha256",
    ):
        raise OutcomeV2CoverageError("receipted schedule source IDs are invalid")
    return rows


def _seal_record(value: bytes) -> JsonObject:
    try:
        text = value.decode("utf-8")
        record = parse_json_object(text)
        require_exact_keys(record, _SEAL_KEYS, "schedule coverage seal")
        if text != canonical_json(record):
            raise OutcomeV2CoverageError("schedule coverage seal must use canonical JSON bytes")
        if _integer(record, "schema_version") != OUTCOME_V2_SCHEDULE_COVERAGE_SCHEMA_VERSION:
            raise OutcomeV2CoverageError("unsupported schedule coverage schema")
        _require_value(record, "kind", _KIND)
        _require_value(record, "status", _STATUS)
        created_at = _parse_utc(_string(record, "created_at"), "created_at")
        deadline = _parse_utc(
            _string(record, "commitment_deadline"),
            "commitment_deadline",
        )
        if created_at >= deadline:
            raise OutcomeV2CoverageError("coverage seal must predate its deadline")
        for field_name in (
            "plan_sha256",
            "coverage_policy_sha256",
            "replay_rows_sha256",
            "question_ids_sha256",
            "source_game_ids_sha256",
            "provider_conformance_report_sha256",
        ):
            _require_hash(_string(record, field_name), field_name)
        _positive_integer(record, "season")
        game_count = _positive_integer(record, "game_count")
        _require_value(record, "provider_proof_scope", NBA_PROVIDER_CONFORMANCE_PROOF_SCOPE)
        _require_value(record, "provider_authenticity", _PROVIDER_AUTHENTICITY)
        report = _object(record, "provider_conformance_report")
        _require_report(report)
        if canonical_sha256(report) != _string(
            record,
            "provider_conformance_report_sha256",
        ):
            raise OutcomeV2CoverageError("provider conformance report hash is invalid")
        if _positive_integer(report, "schedule_game_count") != game_count:
            raise OutcomeV2CoverageError("provider report game count differs from coverage seal")
    except (JsonFormatError, UnicodeError) as error:
        raise OutcomeV2CoverageError("invalid schedule coverage seal") from error
    return record


def _require_report(report: Mapping[str, object]) -> None:
    require_exact_keys(report, _REPORT_KEYS, "provider conformance report")
    if _integer(report, "schema_version") != NBA_PROVIDER_CONFORMANCE_SCHEMA_VERSION:
        raise OutcomeV2CoverageError("unsupported provider conformance report schema")
    _require_value(report, "status", "passed")
    _require_value(report, "proof_scope", NBA_PROVIDER_CONFORMANCE_PROOF_SCOPE)
    for field_name in (
        "inventory_sha256",
        "connector_sha256",
        "snapshot_pack_sha256",
        "cutoff_selection_sha256",
        "schedule_derivation_sha256",
        "replay_rows_sha256",
        "cohort_sha256",
    ):
        _require_hash(_string(report, field_name), field_name)
    _require_text(_string(report, "connector_id"), "connector_id")
    if _string_tuple(report, "schedule_season_types") != ("regular",):
        raise OutcomeV2CoverageError("provider report differs from the regular-season policy")
    _positive_integer(report, "revision_count")
    _positive_integer(report, "schedule_game_count")


def _parse_replay_rows(value: bytes) -> tuple[NbaEloReplayRow, ...]:
    try:
        return read_nba_elo_replay_rows_jsonl_bytes(value)
    except NbaEloReplayError as error:
        raise OutcomeV2CoverageError("cannot load canonical schedule replay rows") from error


def _write_once(path: Path, value: bytes, description: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError:
        raise
    except OSError as error:
        raise OutcomeV2CoverageError(f"cannot write {description}") from error


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise OutcomeV2CoverageError(f"cannot read {description}") from error


def _require_repository_path(value: str) -> None:
    prefix = "prospective/outcome_v2/rolling/"
    if not value.startswith(prefix) or value.endswith("/") or ".." in value.split("/"):
        raise OutcomeV2CoverageError(
            "coverage receipt path must be a file under prospective/outcome_v2/rolling"
        )


def _require_hash(value: str, field_name: str) -> None:
    if _HASH_PATTERN.fullmatch(value) is None:
        raise OutcomeV2CoverageError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_text(value: str, field_name: str) -> None:
    if not value.strip() or value != value.strip():
        raise OutcomeV2CoverageError(f"{field_name} must be a nonempty trimmed string")


def _require_value(record: Mapping[str, object], field_name: str, expected: object) -> None:
    if required_field(record, field_name) != expected:
        raise OutcomeV2CoverageError(f"unexpected {field_name}")


def _object(record: Mapping[str, object], field_name: str) -> JsonObject:
    return require_object(required_field(record, field_name), field_name)


def _string(record: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = required_field(record, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutcomeV2CoverageError(f"{field_name} must be an integer")
    return value


def _positive_integer(record: Mapping[str, object], field_name: str) -> int:
    value = _integer(record, field_name)
    if value <= 0:
        raise OutcomeV2CoverageError(f"{field_name} must be positive")
    return value


def _integer_tuple(record: Mapping[str, object], field_name: str) -> tuple[int, ...]:
    value = require_list(required_field(record, field_name), field_name)
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise OutcomeV2CoverageError(f"{field_name} must contain integers")
        result.append(item)
    return tuple(result)


def _string_tuple(record: Mapping[str, object], field_name: str) -> tuple[str, ...]:
    values = require_list(required_field(record, field_name), field_name)
    return tuple(
        require_string(value, f"{field_name}[{index}]") for index, value in enumerate(values)
    )


def _utc_text(value: datetime, field_name: str) -> str:
    _require_utc(value, field_name)
    return value.isoformat().replace("+00:00", "Z")


def _parse_utc(value: str, field_name: str) -> datetime:
    if not value.endswith("Z"):
        raise OutcomeV2CoverageError(f"{field_name} must use canonical UTC Z notation")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        raise OutcomeV2CoverageError(f"{field_name} must be an ISO 8601 datetime") from error
    _require_utc(parsed, field_name)
    if _utc_text(parsed, field_name) != value:
        raise OutcomeV2CoverageError(f"{field_name} must use canonical UTC notation")
    return parsed


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise OutcomeV2CoverageError(f"{field_name} must be a UTC datetime")
    if value.astimezone(UTC) != value:
        raise OutcomeV2CoverageError(f"{field_name} must be a UTC datetime")
