"""Small Git publication gates for immutable evaluation protocols."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

DEFAULT_REMOTE: Final = "origin"
DEFAULT_REMOTE_REF: Final = "refs/heads/main"


class PublicationError(RuntimeError):
    """Raised when local evidence is not identical to the published revision."""


@dataclass(frozen=True, slots=True)
class PublicationProof:
    """The authoritative repository state observed at one protocol gate."""

    commit: str
    remote: str
    remote_url: str
    remote_ref: str


def require_published_head(
    project_root: Path,
    expected_remote_url: str,
    remote: str = DEFAULT_REMOTE,
    remote_ref: str = DEFAULT_REMOTE_REF,
    *,
    require_clean: bool = True,
) -> PublicationProof:
    """Require HEAD at the authoritative remote, optionally with a clean tree."""
    if require_clean and git_output(
        project_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    ):
        raise PublicationError("working tree must be clean")
    head = git_output(project_root, "rev-parse", "HEAD")
    _require_revision(head)
    remote_url = git_output(project_root, "remote", "get-url", remote)
    if remote_url != expected_remote_url:
        raise PublicationError("remote URL differs from the frozen repository")
    fields = git_output(
        project_root,
        "ls-remote",
        "--exit-code",
        remote,
        remote_ref,
    ).split()
    if fields != [head, remote_ref]:
        raise PublicationError("published remote revision differs from HEAD")
    return PublicationProof(head, remote, remote_url, remote_ref)


def require_paths_at_head(project_root: Path, head: str, paths: tuple[Path, ...]) -> None:
    """Require each path to be tracked and byte-identical to HEAD."""
    _require_revision(head)
    for path in paths:
        relative = str(path.resolve().relative_to(project_root.resolve()))
        tracked = git_output(project_root, "ls-files", "--error-unmatch", "--", relative)
        if tracked != relative:
            raise PublicationError(f"required path is not tracked: {relative}")
        head_object = git_output(project_root, "rev-parse", f"{head}:{relative}")
        working_object = git_output(project_root, "hash-object", "--", relative)
        if head_object != working_object:
            raise PublicationError(f"working path differs from HEAD: {relative}")


def require_protocol_unchanged(
    project_root: Path,
    protocol_revision: str,
    head: str,
    paths: tuple[Path, ...],
) -> None:
    """Require frozen protocol files to remain unchanged at a descendant HEAD."""
    _require_revision(protocol_revision)
    _require_revision(head)
    git_output(project_root, "merge-base", "--is-ancestor", protocol_revision, head)
    relative_paths = tuple(
        str(path.resolve().relative_to(project_root.resolve())) for path in paths
    )
    changed = git_output(
        project_root,
        "diff",
        "--name-only",
        f"{protocol_revision}..{head}",
        "--",
        *relative_paths,
    )
    if changed:
        raise PublicationError("evaluation protocol changed after it was frozen")


def git_output(project_root: Path, *arguments: str) -> str:
    """Run one read-only Git command and return stripped stdout."""
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise PublicationError("Git publication verification failed") from error
    return result.stdout.strip()


def _require_revision(value: str) -> None:
    if len(value) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise PublicationError("Git revision is invalid")
