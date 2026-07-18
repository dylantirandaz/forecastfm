"""Live GitHub Actions receipts for externally timed immutable artifacts."""

from __future__ import annotations

import base64
import binascii
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from http.client import HTTPException, HTTPSConnection
from math import isfinite
from pathlib import Path
from typing import Protocol, final

from forecastfm.integrity import bytes_sha256, canonical_json, canonical_sha256
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_object,
    require_string,
    required_field,
)

GITHUB_API_VERSION = "2026-03-10"
GITHUB_API_HOST = "api.github.com"
GITHUB_RECEIPT_SCHEMA_VERSION = 1
MAX_GITHUB_FILE_BYTES = 1024 * 1024

_KIND = "forecastfm_github_actions_receipt"
_STATUS = "live_github_record_verified"
_AUTHORITY = "github_actions_rest_api"
_CLAIM = "exact_artifact_bytes_existed_in_qualifying_commit_by_run_created_at"
_USER_AGENT = "forecastfm-receipt-verifier/0.1"
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_REPOSITORY_PATTERN = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_BRANCH_PATTERN = re.compile(r"[A-Za-z0-9._/-]+")
_PATH_PATTERN = re.compile(r"[A-Za-z0-9._/-]+")
_RECEIPT_KEYS = {
    "api_version",
    "artifact",
    "authority",
    "claim",
    "deadline",
    "kind",
    "not_before",
    "policy",
    "policy_sha256",
    "run",
    "schema_version",
    "status",
    "workflow",
}
_POLICY_KEYS = {
    "branch",
    "event",
    "repository",
    "workflow_path",
    "workflow_sha256",
    "workflow_id",
}
_RUN_KEYS = {
    "created_at",
    "head_sha",
    "id",
    "path",
    "run_attempt",
    "url",
    "workflow_id",
}
_FILE_KEYS = {"git_blob_sha", "path", "sha256"}

type JsonObject = dict[str, object]


class GitHubActionsReceiptError(ValueError):
    """Raised when GitHub cannot prove an artifact existed inside its causal window."""


class GitHubJsonApi(Protocol):
    """Minimal read-only GitHub API surface used by the receipt verifier."""

    @property
    def api_version(self) -> str:
        """Return the REST API version sent with every request."""
        ...

    def get_json(self, path: str) -> JsonObject:
        """Return one JSON object from an absolute GitHub API path."""
        ...


@final
class GitHubRestClient:
    """Small fail-closed GitHub REST client with no automatic retries."""

    __slots__ = ("_timeout_seconds", "_token")

    def __init__(self, token: str | None = None, timeout_seconds: float = 30.0) -> None:
        """Configure optional authentication and one bounded request timeout."""
        if token is not None and (
            not token
            or not token.isascii()
            or any(character.isspace() or ord(character) < 33 for character in token)
        ):
            raise GitHubActionsReceiptError(
                "GitHub token must use visible ASCII without whitespace"
            )
        if not isfinite(timeout_seconds) or not 0.0 < timeout_seconds <= 60.0:
            raise GitHubActionsReceiptError("GitHub timeout must be between zero and 60 seconds")
        self._token = token
        self._timeout_seconds = timeout_seconds

    @property
    def api_version(self) -> str:
        """Return the pinned GitHub REST API version."""
        return GITHUB_API_VERSION

    def get_json(self, path: str) -> JsonObject:
        """Fetch one bounded JSON object, failing closed on any transport error."""
        _require_api_path(path)
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": _USER_AGENT,
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        }
        if self._token is not None:
            headers["Authorization"] = f"Bearer {self._token}"
        connection = HTTPSConnection(GITHUB_API_HOST, timeout=self._timeout_seconds)
        try:
            connection.request("GET", path, headers=headers)
            response = connection.getresponse()
            if response.status != 200:
                raise GitHubActionsReceiptError("GitHub verification did not return HTTP 200")
            value = response.read(_MAX_RESPONSE_BYTES + 1)
        except GitHubActionsReceiptError:
            raise
        except (HTTPException, OSError, TimeoutError) as error:
            raise GitHubActionsReceiptError("GitHub verification is unavailable") from error
        except ValueError:
            raise GitHubActionsReceiptError("GitHub verification is unavailable") from None
        finally:
            connection.close()
        if len(value) > _MAX_RESPONSE_BYTES:
            raise GitHubActionsReceiptError("GitHub response exceeds the verifier limit")
        try:
            return parse_json_object(value.decode("utf-8"))
        except (JsonFormatError, UnicodeError) as error:
            raise GitHubActionsReceiptError("GitHub returned invalid JSON") from error


@dataclass(frozen=True, slots=True)
class GitHubActionsReceiptPolicy:
    """Frozen GitHub repository, workflow, and event identity."""

    repository: str
    branch: str
    workflow_path: str
    workflow_sha256: str
    workflow_id: int
    event: str = "push"

    def __post_init__(self) -> None:
        _require_repository(self.repository)
        _require_branch(self.branch)
        _require_repo_path(self.workflow_path, "workflow_path")
        if not self.workflow_path.startswith(".github/workflows/"):
            raise GitHubActionsReceiptError("workflow_path must be under .github/workflows")
        _require_hash(self.workflow_sha256, "workflow_sha256")
        _require_positive_workflow_id(self.workflow_id)
        if self.event != "push":
            raise GitHubActionsReceiptError("GitHub receipt event must equal push")

    @property
    def run_path(self) -> str:
        """Return GitHub's exact workflow-run path for this branch."""
        return f"{self.workflow_path}@{self.branch}"

    def canonical_payload(self) -> JsonObject:
        """Return the exact policy payload bound into each receipt."""
        return {
            "repository": self.repository,
            "branch": self.branch,
            "event": self.event,
            "workflow_path": self.workflow_path,
            "workflow_sha256": self.workflow_sha256,
            "workflow_id": self.workflow_id,
        }

    @property
    def policy_sha256(self) -> str:
        """Return the deterministic receipt-policy digest."""
        return canonical_sha256(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class GitHubActionsReceiptRequest:
    """Expected artifact bytes and the only acceptable external-time window."""

    run_id: int
    artifact_path: str
    artifact_bytes: bytes = field(repr=False)
    not_before: datetime
    deadline: datetime

    def __post_init__(self) -> None:
        _require_positive_run_id(self.run_id)
        _require_repo_path(self.artifact_path, "artifact_path")
        _require_file_bytes(self.artifact_bytes, "artifact")
        _require_utc(self.not_before, "not_before")
        _require_utc(self.deadline, "deadline")
        if self.not_before >= self.deadline:
            raise GitHubActionsReceiptError("not_before must precede deadline")


@dataclass(frozen=True, slots=True)
class GitHubActionsReceipt:
    """Canonical local record whose claims must be rechecked against GitHub live."""

    canonical_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require_receipt_bytes(self.canonical_bytes)
        _receipt_record(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        """Return the digest of the exact canonical receipt bytes."""
        return bytes_sha256(self.canonical_bytes)

    def to_record(self) -> JsonObject:
        """Return a newly decoded copy of the strict receipt record."""
        return _receipt_record(self.canonical_bytes)


def github_client_from_environment() -> GitHubRestClient:
    """Build a client using optional ``GITHUB_TOKEN`` without exposing its value."""
    return GitHubRestClient(os.environ.get("GITHUB_TOKEN"))


def build_github_actions_receipt(
    policy: GitHubActionsReceiptPolicy,
    request: GitHubActionsReceiptRequest,
    token: str | None = None,
) -> GitHubActionsReceipt:
    """Verify GitHub live and build one canonical external-time receipt."""
    return _build_github_actions_receipt(GitHubRestClient(token), policy, request)


def _build_github_actions_receipt(
    api: GitHubJsonApi,
    policy: GitHubActionsReceiptPolicy,
    request: GitHubActionsReceiptRequest,
) -> GitHubActionsReceipt:
    """Pure protocol-backed implementation retained for deterministic unit tests."""
    if api.api_version != GITHUB_API_VERSION:
        raise GitHubActionsReceiptError("GitHub API version differs from the frozen policy")
    if request.artifact_path == policy.workflow_path:
        raise GitHubActionsReceiptError("artifact_path must differ from workflow_path")

    run = api.get_json(_run_endpoint(policy, request.run_id))
    verified_run = _verify_run(run, policy, request)
    head_sha = _string(verified_run, "head_sha")
    workflow = _fetch_file(api, policy, policy.workflow_path, head_sha)
    artifact = _fetch_file(api, policy, request.artifact_path, head_sha)
    if workflow.value_sha256 != policy.workflow_sha256:
        raise GitHubActionsReceiptError("remote workflow bytes differ from the frozen policy")
    if artifact.value != request.artifact_bytes:
        raise GitHubActionsReceiptError("remote artifact bytes differ from the expected artifact")

    record: JsonObject = {
        "schema_version": GITHUB_RECEIPT_SCHEMA_VERSION,
        "kind": _KIND,
        "status": _STATUS,
        "authority": _AUTHORITY,
        "claim": _CLAIM,
        "api_version": GITHUB_API_VERSION,
        "policy": policy.canonical_payload(),
        "policy_sha256": policy.policy_sha256,
        "run": verified_run,
        "workflow": workflow.receipt_payload(),
        "artifact": artifact.receipt_payload(),
        "not_before": _utc_text(request.not_before, "not_before"),
        "deadline": _utc_text(request.deadline, "deadline"),
    }
    return GitHubActionsReceipt(canonical_json(record).encode("utf-8"))


def verify_github_actions_receipt(
    policy: GitHubActionsReceiptPolicy,
    request: GitHubActionsReceiptRequest,
    receipt: GitHubActionsReceipt,
    token: str | None = None,
) -> None:
    """Re-fetch GitHub and require an existing receipt to match exactly."""
    _verify_github_actions_receipt(GitHubRestClient(token), policy, request, receipt)


def _verify_github_actions_receipt(
    api: GitHubJsonApi,
    policy: GitHubActionsReceiptPolicy,
    request: GitHubActionsReceiptRequest,
    receipt: GitHubActionsReceipt,
) -> None:
    """Protocol-backed receipt comparison retained for deterministic unit tests."""
    expected = _build_github_actions_receipt(api, policy, request)
    if receipt.canonical_bytes != expected.canonical_bytes:
        raise GitHubActionsReceiptError("saved receipt differs from live GitHub evidence")


def write_github_actions_receipt(path: Path, receipt: GitHubActionsReceipt) -> str:
    """Create and durably flush one canonical receipt without replacement."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(receipt.canonical_bytes)
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError:
        raise
    except OSError as error:
        raise GitHubActionsReceiptError("cannot write GitHub Actions receipt") from error
    return receipt.sha256


def read_github_actions_receipt(path: Path) -> GitHubActionsReceipt:
    """Read one strict canonical GitHub Actions receipt."""
    try:
        return GitHubActionsReceipt(path.read_bytes())
    except OSError as error:
        raise GitHubActionsReceiptError("cannot read GitHub Actions receipt") from error


@dataclass(frozen=True, slots=True)
class _RemoteFile:
    path: str
    git_blob_sha: str
    value: bytes = field(repr=False)

    @property
    def value_sha256(self) -> str:
        return bytes_sha256(self.value)

    def receipt_payload(self) -> JsonObject:
        return {
            "path": self.path,
            "git_blob_sha": self.git_blob_sha,
            "sha256": self.value_sha256,
        }


def _verify_run(
    run: Mapping[str, object],
    policy: GitHubActionsReceiptPolicy,
    request: GitHubActionsReceiptRequest,
) -> JsonObject:
    run_id = _positive_integer(run, "id")
    attempt = _positive_integer(run, "run_attempt")
    workflow_id = _positive_integer(run, "workflow_id")
    head_sha = _string(run, "head_sha")
    _require_git_sha(head_sha, "head_sha")
    if run_id != request.run_id or attempt != 1:
        raise GitHubActionsReceiptError("GitHub run identity or attempt is invalid")
    if workflow_id != policy.workflow_id:
        raise GitHubActionsReceiptError("GitHub run workflow_id differs from policy")
    _require_run_value(run, "head_branch", policy.branch)
    _require_run_value(run, "event", policy.event)
    _require_run_value(run, "path", policy.run_path)
    _require_run_value(run, "status", "completed")
    _require_run_value(run, "conclusion", "success")
    _require_nested_value(run, "repository", "full_name", policy.repository)
    _require_nested_value(run, "head_repository", "full_name", policy.repository)
    _require_nested_value(run, "head_commit", "id", head_sha)

    created_at = _parse_utc(_string(run, "created_at"), "run.created_at")
    observed_updated_at = _parse_utc(_string(run, "updated_at"), "run.updated_at")
    if created_at < request.not_before:
        raise GitHubActionsReceiptError("GitHub run predates the artifact's causal window")
    if observed_updated_at < created_at:
        raise GitHubActionsReceiptError("GitHub run update predates its creation")
    if created_at >= request.deadline:
        raise GitHubActionsReceiptError("GitHub run was not created before the deadline")

    expected_url = f"https://github.com/{policy.repository}/actions/runs/{run_id}"
    _require_run_value(run, "html_url", expected_url)
    return {
        "id": run_id,
        "run_attempt": attempt,
        "workflow_id": workflow_id,
        "path": policy.run_path,
        "head_sha": head_sha,
        "created_at": _utc_text(created_at, "run.created_at"),
        "url": expected_url,
    }


def _fetch_file(
    api: GitHubJsonApi,
    policy: GitHubActionsReceiptPolicy,
    path: str,
    head_sha: str,
) -> _RemoteFile:
    response = api.get_json(_content_endpoint(policy, path, head_sha))
    if _string(response, "type") != "file":
        raise GitHubActionsReceiptError("GitHub content is not a file")
    if _string(response, "path") != path:
        raise GitHubActionsReceiptError("GitHub content path differs from the requested path")
    if _string(response, "encoding") != "base64":
        raise GitHubActionsReceiptError("GitHub content must use base64 encoding")
    size = _nonnegative_integer(response, "size")
    if size > MAX_GITHUB_FILE_BYTES:
        raise GitHubActionsReceiptError("GitHub content exceeds the one-megabyte limit")
    encoded = _string(response, "content")
    try:
        value = base64.b64decode(encoded.replace("\n", ""), validate=True)
    except (ValueError, binascii.Error) as error:
        raise GitHubActionsReceiptError("GitHub content has invalid base64 bytes") from error
    if len(value) != size:
        raise GitHubActionsReceiptError("GitHub content size differs from decoded bytes")
    _require_file_bytes(value, "GitHub content")
    git_blob_sha = _string(response, "sha")
    _require_git_sha(git_blob_sha, "git_blob_sha")
    return _RemoteFile(path, git_blob_sha, value)


def _receipt_record(value: bytes) -> JsonObject:
    try:
        text = value.decode("utf-8")
        record = parse_json_object(text)
    except (JsonFormatError, UnicodeError) as error:
        raise GitHubActionsReceiptError("receipt must be one UTF-8 JSON object") from error
    if text != canonical_json(record):
        raise GitHubActionsReceiptError("receipt must use canonical JSON bytes")
    _validate_receipt_record(record)
    return record


def _validate_receipt_record(record: Mapping[str, object]) -> None:
    try:
        require_exact_keys(record, _RECEIPT_KEYS, "receipt")
        if _integer(record, "schema_version") != GITHUB_RECEIPT_SCHEMA_VERSION:
            raise GitHubActionsReceiptError("unsupported receipt schema")
        _require_run_value(record, "kind", _KIND)
        _require_run_value(record, "status", _STATUS)
        _require_run_value(record, "authority", _AUTHORITY)
        _require_run_value(record, "claim", _CLAIM)
        _require_run_value(record, "api_version", GITHUB_API_VERSION)
        policy = _object(record, "policy")
        require_exact_keys(policy, _POLICY_KEYS, "policy")
        parsed_policy = GitHubActionsReceiptPolicy(
            repository=_string(policy, "repository"),
            branch=_string(policy, "branch"),
            workflow_path=_string(policy, "workflow_path"),
            workflow_sha256=_string(policy, "workflow_sha256"),
            workflow_id=_positive_integer(policy, "workflow_id"),
            event=_string(policy, "event"),
        )
        if parsed_policy.policy_sha256 != _string(record, "policy_sha256"):
            raise GitHubActionsReceiptError("receipt policy hash is invalid")
        _validate_receipt_run(_object(record, "run"), parsed_policy)
        workflow = _object(record, "workflow")
        _validate_receipt_file(workflow, parsed_policy.workflow_path)
        if _string(workflow, "sha256") != parsed_policy.workflow_sha256:
            raise GitHubActionsReceiptError("receipt workflow hash differs from policy")
        artifact = _object(record, "artifact")
        _validate_receipt_file(artifact, _string(artifact, "path"))
        if _string(artifact, "path") == parsed_policy.workflow_path:
            raise GitHubActionsReceiptError("receipt artifact must differ from the workflow")
        not_before = _parse_utc(_string(record, "not_before"), "not_before")
        deadline = _parse_utc(_string(record, "deadline"), "deadline")
        run = _object(record, "run")
        created_at = _parse_utc(_string(run, "created_at"), "run.created_at")
        if not not_before <= created_at < deadline:
            raise GitHubActionsReceiptError("receipt timestamps violate the causal window")
    except JsonFormatError as error:
        raise GitHubActionsReceiptError("invalid receipt structure") from error


def _validate_receipt_run(
    run: Mapping[str, object],
    policy: GitHubActionsReceiptPolicy,
) -> None:
    require_exact_keys(run, _RUN_KEYS, "run")
    _positive_integer(run, "id")
    if _positive_integer(run, "run_attempt") != 1:
        raise GitHubActionsReceiptError("receipt must bind the first run attempt")
    if _positive_integer(run, "workflow_id") != policy.workflow_id:
        raise GitHubActionsReceiptError("receipt workflow_id differs from policy")
    _require_run_value(run, "path", policy.run_path)
    _require_git_sha(_string(run, "head_sha"), "run.head_sha")
    _parse_utc(_string(run, "created_at"), "run.created_at")
    expected_url = f"https://github.com/{policy.repository}/actions/runs/{_integer(run, 'id')}"
    _require_run_value(run, "url", expected_url)


def _validate_receipt_file(file: Mapping[str, object], expected_path: str) -> None:
    require_exact_keys(file, _FILE_KEYS, "receipt file")
    _require_repo_path(_string(file, "path"), "receipt file path")
    if _string(file, "path") != expected_path:
        raise GitHubActionsReceiptError("receipt file path is invalid")
    _require_git_sha(_string(file, "git_blob_sha"), "receipt file git_blob_sha")
    _require_hash(_string(file, "sha256"), "receipt file sha256")


def _run_endpoint(policy: GitHubActionsReceiptPolicy, run_id: int) -> str:
    return f"/repos/{policy.repository}/actions/runs/{run_id}"


def _content_endpoint(policy: GitHubActionsReceiptPolicy, path: str, head_sha: str) -> str:
    return f"/repos/{policy.repository}/contents/{path}?ref={head_sha}"


def _require_api_path(path: str) -> None:
    if (
        not path.startswith("/")
        or path.startswith("//")
        or "://" in path
        or any(character.isspace() or ord(character) < 32 for character in path)
    ):
        raise GitHubActionsReceiptError("invalid GitHub API path")


def _require_repository(value: str) -> None:
    if _REPOSITORY_PATTERN.fullmatch(value) is None:
        raise GitHubActionsReceiptError("repository must use owner/name form")


def _require_branch(value: str) -> None:
    if (
        _BRANCH_PATTERN.fullmatch(value) is None
        or value.startswith("/")
        or value.endswith("/")
        or ".." in value
    ):
        raise GitHubActionsReceiptError("branch is invalid")


def _require_repo_path(value: str, field_name: str) -> None:
    parts = value.split("/")
    if (
        _PATH_PATTERN.fullmatch(value) is None
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise GitHubActionsReceiptError(f"{field_name} must be a safe repository path")


def _require_hash(value: str, field_name: str) -> None:
    if _HASH_PATTERN.fullmatch(value) is None:
        raise GitHubActionsReceiptError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_git_sha(value: str, field_name: str) -> None:
    if _GIT_SHA_PATTERN.fullmatch(value) is None:
        raise GitHubActionsReceiptError(f"{field_name} must be a lowercase 40-character Git SHA")


def _require_file_bytes(value: object, description: str) -> None:
    if not isinstance(value, bytes) or not value:
        raise GitHubActionsReceiptError(f"{description} must be nonempty immutable bytes")
    if len(value) > MAX_GITHUB_FILE_BYTES:
        raise GitHubActionsReceiptError(f"{description} exceeds the one-megabyte limit")


def _require_positive_run_id(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GitHubActionsReceiptError("run_id must be a positive integer")


def _require_positive_workflow_id(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GitHubActionsReceiptError("workflow_id must be a positive integer")


def _require_receipt_bytes(value: object) -> None:
    if not isinstance(value, bytes):
        raise GitHubActionsReceiptError("receipt requires immutable bytes")


def _require_run_value(record: Mapping[str, object], field_name: str, expected: str) -> None:
    if _string(record, field_name) != expected:
        raise GitHubActionsReceiptError(f"GitHub run {field_name} differs from policy")


def _require_nested_value(
    record: Mapping[str, object],
    object_name: str,
    field_name: str,
    expected: str,
) -> None:
    value = _object(record, object_name)
    if _string(value, field_name) != expected:
        raise GitHubActionsReceiptError(f"GitHub run {object_name}.{field_name} is invalid")


def _parse_utc(value: str, field_name: str) -> datetime:
    if not value.endswith("Z"):
        raise GitHubActionsReceiptError(f"{field_name} must use canonical UTC Z notation")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise GitHubActionsReceiptError(f"{field_name} must be an ISO 8601 datetime") from error
    if _utc_text(parsed, field_name) != value:
        raise GitHubActionsReceiptError(f"{field_name} must use canonical UTC notation")
    return parsed.astimezone(UTC)


def _require_utc(value: object, field_name: str) -> None:
    _utc_text(value, field_name)


def _utc_text(value: object, field_name: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise GitHubActionsReceiptError(f"{field_name} must be a UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise GitHubActionsReceiptError(f"{field_name} must be a UTC datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _object(record: Mapping[str, object], field_name: str) -> JsonObject:
    return require_object(required_field(record, field_name), field_name)


def _string(record: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = required_field(record, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise GitHubActionsReceiptError(f"{field_name} must be an integer")
    return value


def _positive_integer(record: Mapping[str, object], field_name: str) -> int:
    value = _integer(record, field_name)
    if value <= 0:
        raise GitHubActionsReceiptError(f"{field_name} must be positive")
    return value


def _nonnegative_integer(record: Mapping[str, object], field_name: str) -> int:
    value = _integer(record, field_name)
    if value < 0:
        raise GitHubActionsReceiptError(f"{field_name} must be non-negative")
    return value
