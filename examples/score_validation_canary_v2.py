"""Score canary v2 only after its raw outputs are committed and published."""

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from examples.build_validation_canary_v2 import EXPECTED_REMOTE_URL

from forecastfm.canary import (
    CanaryValidationError,
    load_canary,
    load_sealed_generations,
    score_primary,
)
from forecastfm.canary_history import score_historical
from forecastfm.integrity import file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANARY_DIRECTORY = PROJECT_ROOT / "evaluation" / "validation_canary_v2"
MANIFEST_PATH = CANARY_DIRECTORY / "manifest.json"
PROMPTS_PATH = CANARY_DIRECTORY / "prompts.jsonl"
RAW_DIRECTORY = CANARY_DIRECTORY / "raw"
ATTEMPT_PATH = RAW_DIRECTORY / "attempt.json"
BASE_PATH = RAW_DIRECTORY / "base.jsonl"
ADAPTER_PATH = RAW_DIRECTORY / "adapter.jsonl"
SEAL_PATH = RAW_DIRECTORY / "manifest.json"
ANSWERS_PATH = PROJECT_ROOT / "data" / "processed" / "nba_elo_validation_answers.jsonl"
SCORES_PATH = CANARY_DIRECTORY / "scores.json"
REMOTE_NAME = "origin"
REMOTE_REF = "refs/heads/main"

PUBLISHED_ARTIFACTS = (
    "evaluation/validation_canary_v2/manifest.json",
    "evaluation/validation_canary_v2/prompts.jsonl",
    "evaluation/validation_canary_v2/raw/attempt.json",
    "evaluation/validation_canary_v2/raw/base.jsonl",
    "evaluation/validation_canary_v2/raw/adapter.jsonl",
    "evaluation/validation_canary_v2/raw/manifest.json",
)
PROTOCOL_PATHS = (
    "src/forecastfm",
    "examples/build_validation_canary_v2.py",
    "examples/run_tinker_canary.py",
    "examples/run_tinker_canary_v2.py",
    "examples/score_validation_canary_v2.py",
)


@dataclass(frozen=True, slots=True)
class PublicationProof:
    """Authoritative remote revision checked before historical answer access."""

    commit: str
    remote: str
    remote_url: str
    remote_ref: str


def main() -> None:
    """Verify sealed published outputs before opening historical answers."""
    if SCORES_PATH.exists():
        raise RuntimeError(f"refusing to replace frozen scores: {SCORES_PATH}")
    manifest, _prompts = load_canary(MANIFEST_PATH, PROMPTS_PATH)
    publication = require_published_raw_outputs(manifest.protocol_code_revision)
    generations = load_sealed_generations(
        SEAL_PATH,
        MANIFEST_PATH,
        PROMPTS_PATH,
        BASE_PATH,
        ADAPTER_PATH,
    )
    primary = score_primary(generations)
    historical = score_historical(generations, ANSWERS_PATH)
    report = {
        "schema_version": 1,
        "kind": "forecastfm_validation_canary_v2_scores",
        "warning": (
            "Historical scores are contamination-prone diagnostics, not prospective evidence."
        ),
        "publication": asdict(publication),
        "commitments": {
            "generation_seal_sha256": file_sha256(SEAL_PATH),
            "canary_manifest_sha256": file_sha256(MANIFEST_PATH),
        },
        "execution": {
            "logical_sdk_call_count": len(generations.base) + len(generations.adapter),
            "transport_request_count_observable": False,
            "transport_retry_note": generations.manifest.decoding["transport_retry_note"],
        },
        "primary_answer_free": asdict(primary),
        "secondary_historical": asdict(historical),
    }
    try:
        with SCORES_PATH.open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")
    except FileExistsError as error:
        raise RuntimeError(f"refusing to replace frozen scores: {SCORES_PATH}") from error
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


def require_published_raw_outputs(protocol_revision: str) -> PublicationProof:
    """Prove exact raw artifacts are at the authoritative remote head."""
    if not _is_revision(protocol_revision):
        raise CanaryValidationError("canary protocol revision is invalid")
    if _git_output("status", "--porcelain", "--untracked-files=all"):
        raise CanaryValidationError("working tree must be clean before historical scoring")
    head = _git_output("rev-parse", "HEAD")
    if not _is_revision(head):
        raise CanaryValidationError("Git HEAD is not a valid revision")
    _require_artifacts_at_head(head)
    _git_output("merge-base", "--is-ancestor", protocol_revision, head)
    changed_protocol = _git_output(
        "diff",
        "--name-only",
        f"{protocol_revision}..{head}",
        "--",
        *PROTOCOL_PATHS,
    )
    if changed_protocol:
        raise CanaryValidationError("canary protocol code changed after cohort freeze")
    remote_revision = _authoritative_remote_revision()
    if remote_revision != head:
        raise CanaryValidationError("raw-output HEAD is not published at origin/main")
    return PublicationProof(
        commit=head,
        remote=REMOTE_NAME,
        remote_url=EXPECTED_REMOTE_URL,
        remote_ref=REMOTE_REF,
    )


def _require_artifacts_at_head(head: str) -> None:
    for artifact in PUBLISHED_ARTIFACTS:
        tracked = _git_output("ls-files", "--error-unmatch", "--", artifact)
        if tracked != artifact:
            raise CanaryValidationError(f"required raw artifact is not tracked: {artifact}")
        head_object = _git_output("rev-parse", f"{head}:{artifact}")
        working_object = _git_output("hash-object", "--", artifact)
        if head_object != working_object:
            raise CanaryValidationError(f"local artifact differs from HEAD: {artifact}")


def _authoritative_remote_revision() -> str:
    if _git_output("remote", "get-url", REMOTE_NAME) != EXPECTED_REMOTE_URL:
        raise CanaryValidationError("origin does not point to the frozen forecastfm repository")
    output = _git_output("ls-remote", "--exit-code", REMOTE_NAME, REMOTE_REF)
    lines = output.splitlines()
    if len(lines) != 1:
        raise CanaryValidationError("origin/main returned an unexpected publication record")
    fields = lines[0].split()
    if len(fields) != 2 or fields[1] != REMOTE_REF or not _is_revision(fields[0]):
        raise CanaryValidationError("origin/main returned an invalid publication record")
    return fields[0]


def _git_output(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise CanaryValidationError("Git publication verification failed") from error
    return result.stdout.strip()


def _is_revision(value: str) -> bool:
    return len(value) in {40, 64} and all(character in "0123456789abcdef" for character in value)


if __name__ == "__main__":
    main()
