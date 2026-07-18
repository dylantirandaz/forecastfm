"""Structural rolling gate reports built from a verified outcome-v2 scoring seal."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from forecastfm.integrity import bytes_sha256, canonical_json, text_sha256
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_object,
    require_string,
    required_field,
)
from forecastfm.nba_evaluation_gate import (
    NbaEvaluationGateArtifacts,
    NbaEvaluationGateError,
    NbaEvaluationGatePolicy,
    NbaEvaluationGateReport,
    verify_untouched_nba_evaluation_gate,
)
from forecastfm.outcome_v2_rolling_score import (
    OutcomeV2RollingScoreError,
    OutcomeV2RollingScoringArtifacts,
    OutcomeV2RollingScoringFiles,
    OutcomeV2RollingScoringSeal,
    verify_outcome_v2_rolling_scoring_seal,
)

OUTCOME_V2_ROLLING_GATE_SCHEMA_VERSION = 1

_GATE_KIND = "forecastfm_outcome_v2_claimed_rolling_gate"
_GATE_STATUS = "structural_claim_only"
_REQUIRED_SEPARATELY = "required_separately"
_AUTHORIZATION_DENIED = "denied"
_HASH_CHARACTERS = frozenset("0123456789abcdef")
_GATE_REPORT_KEYS = {
    "schema_version",
    "kind",
    "status",
    "scoring_seal_sha256",
    "aggregate_seal_sha256",
    "snapshot_pack_sha256",
    "resolutions_sha256",
    "resolution_availability_sha256",
    "evaluation_policy_sha256",
    "calibration_sha256",
    "generic_report_sha256",
    "generic_report",
    "provider_resolution_authenticity",
    "remote_execution_attestation",
    "prospective_win_authorization",
    "rl_authorization",
}

type JsonObject = dict[str, object]


@dataclass(frozen=True, slots=True)
class OutcomeV2RollingGateReport:
    """Self-contained structural report binding the generic gate to rolling provenance."""

    canonical_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _gate_report_record(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        """Return the digest of the exact wrapper-report bytes."""
        return bytes_sha256(self.canonical_bytes)

    def to_record(self) -> JsonObject:
        """Return a fresh strict wrapper-report record."""
        return _gate_report_record(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class OutcomeV2RollingGateArtifacts:
    """All artifacts required by the only valid rolling evaluation entrypoint."""

    scoring: OutcomeV2RollingScoringArtifacts
    scoring_files: OutcomeV2RollingScoringFiles
    supplied_report_path: Path | None = None
    supplied_rolling_report_path: Path | None = None


def verify_outcome_v2_rolling_gate(
    artifacts: OutcomeV2RollingGateArtifacts,
    policy: NbaEvaluationGatePolicy,
    token: str | None = None,
) -> OutcomeV2RollingGateReport:
    """Authorize the rolling lift gate only after production provider review exists."""
    _require_reviewed_resolution_proof()
    return verify_outcome_v2_claimed_rolling_gate(artifacts, policy, token)


def verify_outcome_v2_claimed_rolling_gate(
    artifacts: OutcomeV2RollingGateArtifacts,
    policy: NbaEvaluationGatePolicy,
    token: str | None = None,
) -> OutcomeV2RollingGateReport:
    """Recompute the structural gate without authenticating opaque provider finals."""
    try:
        seal = verify_outcome_v2_rolling_scoring_seal(
            artifacts.scoring,
            artifacts.scoring_files,
            token,
        )
        seal_record = seal.to_record()
        if policy.policy_sha256 != _string(seal_record, "evaluation_policy_sha256"):
            raise OutcomeV2RollingScoreError(
                "rolling gate policy differs from the externally committed plan"
            )
        report = verify_untouched_nba_evaluation_gate(
            NbaEvaluationGateArtifacts(
                cohort_path=artifacts.scoring_files.cohort_path,
                answers_path=artifacts.scoring_files.answers_path,
                forecasts_path=artifacts.scoring.aggregate_files.forecasts_path,
                calibration_path=artifacts.scoring.calibration_path,
                supplied_report_path=artifacts.supplied_report_path,
            ),
            policy=policy,
        )
    except NbaEvaluationGateError as error:
        raise OutcomeV2RollingScoreError("rolling evaluation gate failed") from error
    _require_gate_report_binding(seal, report)
    wrapper = _build_rolling_gate_report(seal, report)
    if artifacts.supplied_rolling_report_path is not None:
        _require_file_bytes(
            artifacts.supplied_rolling_report_path,
            wrapper.canonical_bytes,
            "claimed rolling gate report",
        )
    return wrapper


def write_outcome_v2_claimed_rolling_gate_report(
    path: Path,
    report: OutcomeV2RollingGateReport,
) -> str:
    """Create the structural wrapper report without replacing a prior claim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as file:
            file.write(report.canonical_bytes)
    except FileExistsError as error:
        raise OutcomeV2RollingScoreError("claimed rolling gate report already exists") from error
    except OSError as error:
        raise OutcomeV2RollingScoreError("cannot write claimed rolling gate report") from error
    return report.sha256


def _require_reviewed_resolution_proof() -> None:
    """Fail closed until a reviewed provider final-score parser is installed."""
    raise OutcomeV2RollingScoreError(
        "outcome-v2 has no reviewed production final-score parser and authenticity verifier"
    )


def _require_gate_report_binding(
    seal: OutcomeV2RollingScoringSeal,
    report: NbaEvaluationGateReport,
) -> None:
    seal_record = seal.to_record()
    payload = report.payload
    try:
        report_artifacts = require_object(required_field(payload, "artifacts"), "artifacts")
        evaluation = require_object(required_field(payload, "evaluation"), "evaluation")
    except JsonFormatError as error:
        raise OutcomeV2RollingScoreError("rolling gate report has invalid provenance") from error
    expected_hashes = {
        "cohort_sha256": _string(seal_record, "cohort_sha256"),
        "answers_sha256": _string(seal_record, "answers_sha256"),
        "forecasts_sha256": _string(seal_record, "forecasts_sha256"),
        "calibration_sha256": _string(seal_record, "calibration_sha256"),
    }
    for field_name, expected in expected_hashes.items():
        if _string(report_artifacts, field_name) != expected:
            raise OutcomeV2RollingScoreError(
                f"rolling gate {field_name} differs from its scoring seal"
            )
    if required_field(evaluation, "question_ids_sha256") != required_field(
        seal_record,
        "question_ids_sha256",
    ):
        raise OutcomeV2RollingScoreError("rolling gate question IDs differ from its scoring seal")
    if required_field(evaluation, "seasons") != required_field(seal_record, "seasons"):
        raise OutcomeV2RollingScoreError("rolling gate seasons differ from its scoring seal")


def _build_rolling_gate_report(
    seal: OutcomeV2RollingScoringSeal,
    generic_report: NbaEvaluationGateReport,
) -> OutcomeV2RollingGateReport:
    seal_record = seal.to_record()
    record: JsonObject = {
        "schema_version": OUTCOME_V2_ROLLING_GATE_SCHEMA_VERSION,
        "kind": _GATE_KIND,
        "status": _GATE_STATUS,
        "scoring_seal_sha256": seal.sha256,
        "aggregate_seal_sha256": _string(seal_record, "aggregate_seal_sha256"),
        "snapshot_pack_sha256": _string(seal_record, "snapshot_pack_sha256"),
        "resolutions_sha256": _string(seal_record, "resolutions_sha256"),
        "resolution_availability_sha256": _string(
            seal_record,
            "resolution_availability_sha256",
        ),
        "evaluation_policy_sha256": _string(
            seal_record,
            "evaluation_policy_sha256",
        ),
        "calibration_sha256": _string(seal_record, "calibration_sha256"),
        "generic_report_sha256": generic_report.sha256,
        "generic_report": generic_report.payload,
        "provider_resolution_authenticity": _REQUIRED_SEPARATELY,
        "remote_execution_attestation": _REQUIRED_SEPARATELY,
        "prospective_win_authorization": _AUTHORIZATION_DENIED,
        "rl_authorization": _AUTHORIZATION_DENIED,
    }
    return OutcomeV2RollingGateReport(canonical_json(record).encode("utf-8"))


def _gate_report_record(value: bytes) -> JsonObject:
    try:
        text = value.decode("utf-8")
        record = parse_json_object(text)
        require_exact_keys(record, _GATE_REPORT_KEYS, "claimed rolling gate report")
        if text != canonical_json(record):
            raise OutcomeV2RollingScoreError(
                "claimed rolling gate report must use canonical JSON bytes"
            )
        _validate_gate_report_record(record)
    except (JsonFormatError, UnicodeError) as error:
        raise OutcomeV2RollingScoreError("invalid claimed rolling gate report") from error
    return record


def _validate_gate_report_record(record: Mapping[str, object]) -> None:
    if _integer(record, "schema_version") != OUTCOME_V2_ROLLING_GATE_SCHEMA_VERSION:
        raise OutcomeV2RollingScoreError("unsupported claimed rolling gate schema")
    _require_value(record, "kind", _GATE_KIND)
    _require_value(record, "status", _GATE_STATUS)
    for field_name in (
        "scoring_seal_sha256",
        "aggregate_seal_sha256",
        "snapshot_pack_sha256",
        "resolutions_sha256",
        "resolution_availability_sha256",
        "evaluation_policy_sha256",
        "calibration_sha256",
        "generic_report_sha256",
    ):
        _require_hash(_string(record, field_name), field_name)
    generic = require_object(required_field(record, "generic_report"), "generic_report")
    if text_sha256(canonical_json(generic)) != _string(record, "generic_report_sha256"):
        raise OutcomeV2RollingScoreError("generic report hash is invalid")
    if required_field(generic, "status") != "passed":
        raise OutcomeV2RollingScoreError("generic report is not a passing evaluation")
    _require_value(record, "provider_resolution_authenticity", _REQUIRED_SEPARATELY)
    _require_value(record, "remote_execution_attestation", _REQUIRED_SEPARATELY)
    _require_value(record, "prospective_win_authorization", _AUTHORIZATION_DENIED)
    _require_value(record, "rl_authorization", _AUTHORIZATION_DENIED)


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise OutcomeV2RollingScoreError(f"cannot read {description}") from error


def _require_file_bytes(path: Path, expected: bytes, description: str) -> None:
    if _read_bytes(path, description) != expected:
        raise OutcomeV2RollingScoreError(f"{description} differs from verified scoring inputs")


def _string(record: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = required_field(record, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutcomeV2RollingScoreError(f"{field_name} must be an integer")
    return value


def _require_hash(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in _HASH_CHARACTERS for character in value):
        raise OutcomeV2RollingScoreError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_value(record: Mapping[str, object], field_name: str, expected: str) -> None:
    if _string(record, field_name) != expected:
        raise OutcomeV2RollingScoreError(f"rolling scoring seal has invalid {field_name}")
