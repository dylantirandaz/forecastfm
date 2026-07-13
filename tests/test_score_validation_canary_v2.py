"""Fail-closed publication-gate tests for canary-v2 scoring."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from examples import score_validation_canary_v2

from forecastfm.canary import CanaryManifest, CanaryValidationError, SealedGenerations

REVISION = "a" * 40
OBJECT_ID = "f" * 40


def _published_git_output(
    calls: list[tuple[str, ...]],
) -> Callable[..., str]:
    def git_output(*arguments: str) -> str:
        calls.append(arguments)
        fixed_results: dict[tuple[str, ...], str] = {
            ("status", "--porcelain", "--untracked-files=all"): "",
            ("rev-parse", "HEAD"): REVISION,
            ("remote", "get-url", score_validation_canary_v2.REMOTE_NAME): (
                score_validation_canary_v2.EXPECTED_REMOTE_URL
            ),
        }
        if arguments in fixed_results:
            result = fixed_results[arguments]
        elif arguments[:3] == ("ls-files", "--error-unmatch", "--"):
            result = arguments[3]
        elif (arguments[0] == "rev-parse" and ":" in arguments[1]) or arguments[:2] == (
            "hash-object",
            "--",
        ):
            result = OBJECT_ID
        elif arguments[:2] in {
            ("merge-base", "--is-ancestor"),
            ("diff", "--name-only"),
        }:
            result = ""
        elif arguments == (
            "ls-remote",
            "--exit-code",
            score_validation_canary_v2.REMOTE_NAME,
            score_validation_canary_v2.REMOTE_REF,
        ):
            result = f"{REVISION}\t{score_validation_canary_v2.REMOTE_REF}"
        else:
            raise AssertionError(f"unexpected Git call: {arguments}")
        return result

    return git_output


def test_publication_gate_requires_exact_artifacts_and_authoritative_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        score_validation_canary_v2,
        "_git_output",
        _published_git_output(calls),
    )

    proof = score_validation_canary_v2.require_published_raw_outputs(REVISION)

    assert proof.commit == REVISION
    assert proof.remote_url == score_validation_canary_v2.EXPECTED_REMOTE_URL
    assert proof.remote_ref == "refs/heads/main"
    tracked = [call[3] for call in calls if call[:3] == ("ls-files", "--error-unmatch", "--")]
    assert tracked == list(score_validation_canary_v2.PUBLISHED_ARTIFACTS)
    assert calls[-1][0] == "ls-remote"


@pytest.mark.parametrize(
    ("dirty_status", "remote_revision"),
    [("?? raw.jsonl", REVISION), ("", "b" * 40)],
)
def test_publication_gate_fails_closed_for_dirty_or_unpublished_outputs(
    dirty_status: str,
    remote_revision: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    accepted = _published_git_output(calls)

    def git_output(*arguments: str) -> str:
        if arguments == ("status", "--porcelain", "--untracked-files=all"):
            return dirty_status
        if arguments[0] == "ls-remote":
            return f"{remote_revision}\t{score_validation_canary_v2.REMOTE_REF}"
        return accepted(*arguments)

    monkeypatch.setattr(score_validation_canary_v2, "_git_output", git_output)

    with pytest.raises(CanaryValidationError):
        score_validation_canary_v2.require_published_raw_outputs(REVISION)


def test_publication_gate_rejects_protocol_code_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    accepted = _published_git_output(calls)

    def git_output(*arguments: str) -> str:
        if arguments[:2] == ("diff", "--name-only"):
            return "src/forecastfm/canary.py"
        return accepted(*arguments)

    monkeypatch.setattr(score_validation_canary_v2, "_git_output", git_output)

    with pytest.raises(CanaryValidationError, match="protocol code changed"):
        score_validation_canary_v2.require_published_raw_outputs(REVISION)
    assert not any(call[0] == "ls-remote" for call in calls)


def test_publication_gate_rejects_local_bytes_that_differ_from_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    accepted = _published_git_output(calls)

    def git_output(*arguments: str) -> str:
        if arguments[:2] == ("hash-object", "--"):
            return "e" * 40
        return accepted(*arguments)

    monkeypatch.setattr(score_validation_canary_v2, "_git_output", git_output)

    with pytest.raises(CanaryValidationError, match="differs from HEAD"):
        score_validation_canary_v2.require_published_raw_outputs(REVISION)
    assert not any(call[0] == "ls-remote" for call in calls)


@dataclass(frozen=True)
class _Manifest:
    protocol_code_revision: str


@dataclass(frozen=True)
class _Generations:
    manifest: _Manifest


def test_main_checks_publication_before_any_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generations = cast(
        SealedGenerations,
        _Generations(manifest=_Manifest(protocol_code_revision=REVISION)),
    )
    scoring_called = False

    def reject_publication(_revision: str) -> score_validation_canary_v2.PublicationProof:
        raise CanaryValidationError("not published")

    def scoring_trap(*_arguments: object) -> object:
        nonlocal scoring_called
        scoring_called = True
        raise AssertionError("scoring must not run")

    def load_generations(*_arguments: Path) -> SealedGenerations:
        return generations

    def load_manifest(*_arguments: Path) -> tuple[CanaryManifest, tuple[()]]:
        return generations.manifest, ()

    monkeypatch.setattr(score_validation_canary_v2, "SCORES_PATH", tmp_path / "scores.json")
    monkeypatch.setattr(score_validation_canary_v2, "load_canary", load_manifest)
    monkeypatch.setattr(
        score_validation_canary_v2,
        "load_sealed_generations",
        load_generations,
    )
    monkeypatch.setattr(
        score_validation_canary_v2,
        "require_published_raw_outputs",
        reject_publication,
    )
    monkeypatch.setattr(score_validation_canary_v2, "score_primary", scoring_trap)
    monkeypatch.setattr(score_validation_canary_v2, "score_historical", scoring_trap)

    with pytest.raises(CanaryValidationError, match="not published"):
        score_validation_canary_v2.main()
    assert not scoring_called
