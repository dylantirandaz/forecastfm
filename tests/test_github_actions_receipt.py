"""Tests for live, externally timed GitHub Actions artifact receipts."""

import base64
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import forecastfm.github_actions_receipt as receipt_module
from forecastfm.github_actions_receipt import (
    GITHUB_API_VERSION,
    GitHubActionsReceipt,
    GitHubActionsReceiptError,
    GitHubActionsReceiptPolicy,
    GitHubActionsReceiptRequest,
    build_github_actions_receipt,
    read_github_actions_receipt,
    verify_github_actions_receipt,
    write_github_actions_receipt,
)
from forecastfm.integrity import bytes_sha256, canonical_json
from forecastfm.json_utils import require_object, required_field

REPOSITORY = "dylantirandaz/forecastfm"
BRANCH = "main"
WORKFLOW_PATH = ".github/workflows/outcome-v2-publication-timestamp.yml"
ARTIFACT_PATH = "prospective/outcome_v2/rolling/batch-1/generation-lock.json"
WORKFLOW_BYTES = b"name: timestamp\n"
ARTIFACT_BYTES = b'{"kind":"generation-lock"}'
HEAD_SHA = "a" * 40
RUN_ID = 123456
WORKFLOW_ID = 654321
NOT_BEFORE = datetime(2026, 10, 20, 17, tzinfo=UTC)
CREATED_AT = NOT_BEFORE + timedelta(minutes=1)
COMPLETED_AT = CREATED_AT + timedelta(seconds=20)
DEADLINE = NOT_BEFORE + timedelta(minutes=10)

type JsonObject = dict[str, object]


class FakeGitHubApi:
    """Deterministic read-only API used to test exact endpoint binding."""

    def __init__(self, responses: dict[str, JsonObject]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    @property
    def api_version(self) -> str:
        return GITHUB_API_VERSION

    def get_json(self, path: str) -> JsonObject:
        self.calls.append(path)
        try:
            return self.responses[path]
        except KeyError as error:
            raise AssertionError(f"unexpected API path: {path}") from error


class _FakeHttpResponse:
    def __init__(self, value: bytes, status: int = 200) -> None:
        self.value = value
        self.status = status

    def read(self, amount: int) -> bytes:
        return self.value[:amount]


class _FakeHttpsConnection:
    def __init__(self, api: FakeGitHubApi, status: int = 200) -> None:
        self.api = api
        self.status = status
        self.path: str | None = None
        self.headers: dict[str, str] = {}
        self.closed = False

    def request(
        self,
        method: str,
        path: str,
        _body: object = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        assert method == "GET"
        self.path = path
        self.headers = headers or {}

    def getresponse(self) -> _FakeHttpResponse:
        if self.path is None:
            raise AssertionError("request must precede getresponse")
        value = canonical_json(self.api.get_json(self.path)).encode("utf-8")
        return _FakeHttpResponse(value, self.status)

    def close(self) -> None:
        self.closed = True


def _install_api(
    monkeypatch: pytest.MonkeyPatch,
    api: FakeGitHubApi,
    *,
    status: int = 200,
) -> list[_FakeHttpsConnection]:
    connections: list[_FakeHttpsConnection] = []

    def connect(host: str, timeout: float) -> _FakeHttpsConnection:
        assert host == "api.github.com"
        assert timeout == 30.0
        connection = _FakeHttpsConnection(api, status)
        connections.append(connection)
        return connection

    monkeypatch.setattr(receipt_module, "HTTPSConnection", connect)
    return connections


def _utc_text(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _policy(workflow_bytes: bytes = WORKFLOW_BYTES) -> GitHubActionsReceiptPolicy:
    return GitHubActionsReceiptPolicy(
        repository=REPOSITORY,
        branch=BRANCH,
        workflow_path=WORKFLOW_PATH,
        workflow_sha256=bytes_sha256(workflow_bytes),
        workflow_id=WORKFLOW_ID,
    )


def _request(
    artifact_bytes: bytes = ARTIFACT_BYTES,
    *,
    not_before: datetime = NOT_BEFORE,
    deadline: datetime = DEADLINE,
) -> GitHubActionsReceiptRequest:
    return GitHubActionsReceiptRequest(
        run_id=RUN_ID,
        artifact_path=ARTIFACT_PATH,
        artifact_bytes=artifact_bytes,
        not_before=not_before,
        deadline=deadline,
    )


def _run_record(**changes: object) -> JsonObject:
    record: JsonObject = {
        "id": RUN_ID,
        "run_attempt": 1,
        "workflow_id": WORKFLOW_ID,
        "head_sha": HEAD_SHA,
        "head_branch": BRANCH,
        "event": "push",
        "path": f"{WORKFLOW_PATH}@{BRANCH}",
        "status": "completed",
        "conclusion": "success",
        "created_at": _utc_text(CREATED_AT),
        "updated_at": _utc_text(COMPLETED_AT),
        "html_url": f"https://github.com/{REPOSITORY}/actions/runs/{RUN_ID}",
        "repository": {"full_name": REPOSITORY},
        "head_repository": {"full_name": REPOSITORY},
        "head_commit": {"id": HEAD_SHA},
    }
    record.update(changes)
    return record


def _content(path: str, value: bytes, git_blob_sha: str) -> JsonObject:
    encoded = base64.b64encode(value).decode("ascii")
    return {
        "type": "file",
        "path": path,
        "encoding": "base64",
        "content": encoded,
        "size": len(value),
        "sha": git_blob_sha,
    }


def _api(
    *,
    run: JsonObject | None = None,
    workflow_bytes: bytes = WORKFLOW_BYTES,
    artifact_bytes: bytes = ARTIFACT_BYTES,
) -> FakeGitHubApi:
    policy = _policy()
    run_path = f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}"
    workflow_path = f"/repos/{REPOSITORY}/contents/{WORKFLOW_PATH}?ref={HEAD_SHA}"
    artifact_path = f"/repos/{REPOSITORY}/contents/{ARTIFACT_PATH}?ref={HEAD_SHA}"
    return FakeGitHubApi(
        {
            run_path: run or _run_record(),
            workflow_path: _content(policy.workflow_path, workflow_bytes, "b" * 40),
            artifact_path: _content(ARTIFACT_PATH, artifact_bytes, "c" * 40),
        }
    )


def _object(record: JsonObject, field_name: str) -> JsonObject:
    return require_object(required_field(record, field_name), field_name)


def test_receipt_binds_live_run_and_exact_commit_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    connections = _install_api(monkeypatch, api)
    policy = _policy()
    request = _request()

    receipt = build_github_actions_receipt(policy, request)
    record = receipt.to_record()
    run = _object(record, "run")
    artifact = _object(record, "artifact")
    workflow = _object(record, "workflow")

    assert record["authority"] == "github_actions_rest_api"
    assert record["claim"] == (
        "exact_artifact_bytes_existed_in_qualifying_commit_by_run_created_at"
    )
    assert record["policy_sha256"] == policy.policy_sha256
    assert run["head_sha"] == HEAD_SHA
    assert run["run_attempt"] == 1
    assert run["created_at"] == _utc_text(CREATED_AT)
    assert artifact["path"] == ARTIFACT_PATH
    assert artifact["sha256"] == bytes_sha256(ARTIFACT_BYTES)
    assert workflow["path"] == WORKFLOW_PATH
    assert workflow["sha256"] == bytes_sha256(WORKFLOW_BYTES)
    assert api.calls == [
        f"/repos/{REPOSITORY}/actions/runs/{RUN_ID}",
        f"/repos/{REPOSITORY}/contents/{WORKFLOW_PATH}?ref={HEAD_SHA}",
        f"/repos/{REPOSITORY}/contents/{ARTIFACT_PATH}?ref={HEAD_SHA}",
    ]

    path = tmp_path / "receipt.json"
    assert write_github_actions_receipt(path, receipt) == receipt.sha256
    assert read_github_actions_receipt(path) == receipt
    with pytest.raises(FileExistsError):
        write_github_actions_receipt(path, receipt)

    api = _api()
    _install_api(monkeypatch, api)
    verify_github_actions_receipt(policy, request, receipt)
    assert all(connection.closed for connection in connections)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"run_attempt": 2}, "identity or attempt"),
        ({"status": "in_progress"}, "status differs"),
        ({"conclusion": "failure"}, "conclusion differs"),
        ({"event": "workflow_dispatch"}, "event differs"),
        ({"head_branch": "forecast"}, "head_branch differs"),
        ({"path": WORKFLOW_PATH}, "path differs"),
        ({"workflow_id": WORKFLOW_ID + 1}, "workflow_id differs"),
        ({"repository": {"full_name": "attacker/fork"}}, "repository.full_name"),
        ({"head_commit": {"id": "d" * 40}}, "head_commit.id"),
    ],
)
def test_receipt_rejects_wrong_run_identity(
    changes: JsonObject,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _run_record(**changes)
    _install_api(monkeypatch, _api(run=run))

    with pytest.raises(GitHubActionsReceiptError, match=message):
        build_github_actions_receipt(_policy(), _request())


def test_receipt_requires_run_creation_inside_causal_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    too_early = _run_record(created_at=_utc_text(NOT_BEFORE - timedelta(seconds=1)))
    too_late = _run_record(created_at=_utc_text(DEADLINE), updated_at=_utc_text(DEADLINE))

    _install_api(monkeypatch, _api(run=too_early))
    with pytest.raises(GitHubActionsReceiptError, match="predates"):
        build_github_actions_receipt(_policy(), _request())
    _install_api(monkeypatch, _api(run=too_late))
    with pytest.raises(GitHubActionsReceiptError, match="created before the deadline"):
        build_github_actions_receipt(_policy(), _request())


def test_receipt_requires_exact_remote_artifact_and_frozen_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_api(monkeypatch, _api(artifact_bytes=b"different artifact"))
    with pytest.raises(GitHubActionsReceiptError, match="artifact bytes"):
        build_github_actions_receipt(_policy(), _request())
    _install_api(monkeypatch, _api(workflow_bytes=b"different workflow"))
    with pytest.raises(GitHubActionsReceiptError, match="workflow bytes"):
        build_github_actions_receipt(_policy(), _request())


def test_saved_receipt_is_not_trusted_without_live_reverification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    _install_api(monkeypatch, api)
    receipt = build_github_actions_receipt(_policy(), _request())
    record = receipt.to_record()
    artifact = _object(record, "artifact")
    artifact["sha256"] = "0" * 64
    record["artifact"] = artifact
    forged = GitHubActionsReceipt(canonical_json(record).encode("utf-8"))

    _install_api(monkeypatch, _api())
    with pytest.raises(GitHubActionsReceiptError, match="live GitHub"):
        verify_github_actions_receipt(_policy(), _request(), forged)


def test_receipt_structure_rejects_policy_file_contradictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_api(monkeypatch, _api())
    receipt = build_github_actions_receipt(_policy(), _request())
    wrong_workflow = receipt.to_record()
    workflow = _object(wrong_workflow, "workflow")
    workflow["sha256"] = "0" * 64
    wrong_workflow["workflow"] = workflow
    with pytest.raises(GitHubActionsReceiptError, match="workflow hash"):
        GitHubActionsReceipt(canonical_json(wrong_workflow).encode("utf-8"))

    wrong_artifact = receipt.to_record()
    artifact = _object(wrong_artifact, "artifact")
    artifact["path"] = WORKFLOW_PATH
    wrong_artifact["artifact"] = artifact
    with pytest.raises(GitHubActionsReceiptError, match="differ from the workflow"):
        GitHubActionsReceipt(canonical_json(wrong_artifact).encode("utf-8"))


def test_updated_at_is_not_misrepresented_as_a_commit_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_api(monkeypatch, _api())
    receipt = build_github_actions_receipt(_policy(), _request())
    changed = _run_record(updated_at=_utc_text(COMPLETED_AT + timedelta(days=1)))
    _install_api(monkeypatch, _api(run=changed))

    verify_github_actions_receipt(_policy(), _request(), receipt)

    assert "updated" not in canonical_json(receipt.to_record())


def test_transport_rejects_redirect_without_following_or_leaking_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connections = _install_api(monkeypatch, _api(), status=302)

    with pytest.raises(GitHubActionsReceiptError, match="HTTP 200"):
        build_github_actions_receipt(_policy(), _request(), token="secret-token")

    assert len(connections) == 1
    assert connections[0].headers["Authorization"] == "Bearer secret-token"
    assert connections[0].closed


def test_receipt_policy_and_request_reject_ambiguous_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(GitHubActionsReceiptError, match="safe repository path"):
        replace(_policy(), workflow_path=".github/workflows/../unsafe.yml")
    with pytest.raises(GitHubActionsReceiptError, match="must equal push"):
        replace(_policy(), event="workflow_dispatch")
    with pytest.raises(GitHubActionsReceiptError, match="visible ASCII"):
        receipt_module.GitHubRestClient("secret\nleak")
    with pytest.raises(GitHubActionsReceiptError, match="not_before must precede"):
        _request(deadline=NOT_BEFORE)
    _install_api(monkeypatch, _api())
    with pytest.raises(GitHubActionsReceiptError, match="artifact_path must differ"):
        build_github_actions_receipt(
            _policy(),
            replace(_request(), artifact_path=WORKFLOW_PATH),
        )


def test_receipt_requires_canonical_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_api(monkeypatch, _api())
    receipt = build_github_actions_receipt(_policy(), _request())

    with pytest.raises(GitHubActionsReceiptError, match="canonical"):
        GitHubActionsReceipt(receipt.canonical_bytes + b"\n")
