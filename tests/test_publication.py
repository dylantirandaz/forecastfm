"""Tests for shared immutable-publication gates."""

from collections.abc import Callable
from pathlib import Path

import pytest

from forecastfm import publication
from forecastfm.publication import PublicationError

REVISION = "a" * 40
REMOTE_URL = "https://github.com/example/forecastfm.git"


def _published_git_output(
    calls: list[tuple[str, ...]],
) -> Callable[..., str]:
    def run(_root: Path, *arguments: str) -> str:
        calls.append(arguments)
        values: dict[tuple[str, ...], str] = {
            ("status", "--porcelain", "--untracked-files=all"): "",
            ("rev-parse", "HEAD"): REVISION,
            ("remote", "get-url", "origin"): REMOTE_URL,
            ("ls-remote", "--exit-code", "origin", "refs/heads/main"): (
                f"{REVISION}\trefs/heads/main"
            ),
        }
        return values.get(arguments, "")

    return run


def test_require_published_head_returns_authoritative_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(publication, "git_output", _published_git_output(calls))

    proof = publication.require_published_head(tmp_path, REMOTE_URL)

    assert proof.commit == REVISION
    assert proof.remote_ref == "refs/heads/main"
    assert calls[-1][0] == "ls-remote"


def test_require_published_head_rejects_a_dirty_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dirty(_root: Path, *_arguments: str) -> str:
        return " M changed.py"

    monkeypatch.setattr(publication, "git_output", dirty)

    with pytest.raises(PublicationError, match="clean"):
        publication.require_published_head(tmp_path, REMOTE_URL)


def test_require_published_head_can_allow_runtime_output_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(publication, "git_output", _published_git_output(calls))

    proof = publication.require_published_head(
        tmp_path,
        REMOTE_URL,
        require_clean=False,
    )

    assert proof.commit == REVISION
    assert not any(call[0] == "status" for call in calls)


def test_require_protocol_unchanged_rejects_changed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def changed(_root: Path, *arguments: str) -> str:
        return "src/forecastfm/outcome.py" if arguments[0] == "diff" else ""

    monkeypatch.setattr(publication, "git_output", changed)

    with pytest.raises(PublicationError, match="changed"):
        publication.require_protocol_unchanged(
            tmp_path,
            REVISION,
            REVISION,
            (tmp_path / "src/forecastfm/outcome.py",),
        )
