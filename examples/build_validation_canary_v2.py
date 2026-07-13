"""Freeze the disjoint no-thinking validation canary without opening answers."""

import subprocess
from pathlib import Path

from forecastfm.canary import (
    CANARY_SIZE,
    V2_PROTOCOL,
    CanaryManifest,
    CanaryModels,
    CanarySource,
    build_canary_artifacts,
)
from forecastfm.integrity import canonical_sha256, file_sha256
from forecastfm.json_utils import (
    parse_json_object,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.run_config import BASE_MODEL, decoding_settings
from forecastfm.run_lock import verify_experiment_lock, verify_training_lock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.json"
SOURCE_PROMPTS_PATH = PROJECT_ROOT / "data" / "processed" / "nba_elo_validation_prompts.jsonl"
TRAINING_LOCK_PATH = PROJECT_ROOT / "prospective" / "training_lock.json"
EXPERIMENT_PATH = PROJECT_ROOT / "prospective" / "experiment.json"
V1_MANIFEST_PATH = PROJECT_ROOT / "evaluation" / "validation_canary" / "manifest.json"
SMOKE_DIRECTORY = PROJECT_ROOT / "evaluation" / "format_smoke" / "v1"
SMOKE_RESULT_PATH = SMOKE_DIRECTORY / "result.json"
SMOKE_SEAL_PATH = SMOKE_DIRECTORY / "raw" / "manifest.json"
OUTPUT_DIRECTORY = PROJECT_ROOT / "evaluation" / "validation_canary_v2"
PROMPTS_PATH = OUTPUT_DIRECTORY / "prompts.jsonl"
MANIFEST_PATH = OUTPUT_DIRECTORY / "manifest.json"
REMOTE_NAME = "origin"
REMOTE_REF = "refs/heads/main"
EXPECTED_REMOTE_URL = "https://github.com/dylantirandaz/forecastfm.git"

EXPECTED_DATA_MANIFEST_SHA256 = "7cdc6959011c0f7526e626c63b55cef33553805c195c30fe2eac86b52d61c025"
EXPECTED_PROMPTS_SHA256 = "a1c018c09107039101ee9426331f4bbd80bc704ffe23c2d72c628995c245b3cb"
EXPECTED_ANSWERS_SHA256 = "2f91173d30ed835d02663761dcf83c23347a6f9cdb2c0305ac719b852b1c460f"
EXPECTED_V1_MANIFEST_SHA256 = "bb5396100c6867f4cca2ddbfffc600aef7109baeb8660055b3bc366a02d1c1f1"
EXPECTED_V1_QUESTION_IDS_SHA256 = "99e2efda839eca77547f0ee91140463a3ac6cb580fcd24e6a9e2dc90619c1dba"
EXPECTED_SMOKE_RESULT_SHA256 = "ae64bb8e7b3e5243ad1d771c894606197e5aacc427ae8eb14c6c4249ab45b80e"
EXPECTED_SMOKE_SEAL_SHA256 = "6f341c8f84f782a8b130efc3b7520b519d685b200417083c3c63dbbb3f9e4155"
EXPECTED_V2_QUESTION_IDS_SHA256 = "acefb83bb24ea238195a3c9b8499a1d61e17bfbd31ac4a9c16ce3636ad6d04f1"
EXPECTED_V2_PROMPTS_SHA256 = "3815edd18b060b393aa08092845b1012d5051259b3786843b8ccb30933fd0f7d"
EXPECTED_VALIDATION_COUNT = 3_679
TRANSPORT_RETRY_NOTE = (
    "Tinker 0.22.7 may retransmit one logical request with the same session and sequence ID."
)


def main() -> None:
    """Verify published protocol inputs, then exclusively freeze canary v2."""
    revision = require_published_protocol()
    training_lock = verify_training_lock(PROJECT_ROOT, TRAINING_LOCK_PATH)
    experiment = verify_experiment_lock(TRAINING_LOCK_PATH, EXPERIMENT_PATH)
    _verify_dataset_manifest()
    retired_ids = verify_retired_v1()
    verify_format_smoke()
    selected_ids = verify_v2_selection(retired_ids)

    commitments = protocol_commitments()
    source = CanarySource(
        validation_prompts_path=SOURCE_PROMPTS_PATH,
        validation_prompts_sha256=EXPECTED_PROMPTS_SHA256,
        validation_answers_sha256=EXPECTED_ANSWERS_SHA256,
        dataset_manifest_sha256=EXPECTED_DATA_MANIFEST_SHA256,
        expected_question_ids_sha256=EXPECTED_V2_QUESTION_IDS_SHA256,
    )
    models = CanaryModels(
        training_lock_sha256=file_sha256(TRAINING_LOCK_PATH),
        experiment_sha256=file_sha256(EXPERIMENT_PATH),
        base_model=BASE_MODEL,
        adapter_sampler_path=require_string(
            required_field(experiment, "adapter_sampler_path"),
            "adapter_sampler_path",
        ),
        decoding=v2_decoding_settings(),
        protocol_code_revision=revision,
        protocol_commitments=commitments,
    )
    manifest = build_canary_artifacts(
        source,
        models,
        PROMPTS_PATH,
        MANIFEST_PATH,
        V2_PROTOCOL,
    )
    verify_v2_manifest(manifest, selected_ids)
    if manifest.protocol_commitments != commitments:
        raise RuntimeError("v2 manifest does not bind its verified prerequisites")
    if manifest.training_lock_sha256 != file_sha256(TRAINING_LOCK_PATH):
        raise RuntimeError("v2 canary does not bind the verified training lock")
    if required_field(training_lock, "status") != "awaiting_trained_sampler":
        raise RuntimeError("training lock has an unexpected status")
    print(f"Frozen 64 disjoint games and 128 prompts at {PROMPTS_PATH}.")
    print(f"Manifest: {MANIFEST_PATH}")


def require_published_protocol() -> str:
    """Require clean protocol code at the authoritative published revision."""
    if git_output("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked files must be clean before freezing canary v2")
    revision = git_output("rev-parse", "HEAD")
    if revision != git_output("rev-parse", "origin/main"):
        raise RuntimeError("canary v2 protocol must be published to origin/main")
    if git_output("remote", "get-url", REMOTE_NAME) != EXPECTED_REMOTE_URL:
        raise RuntimeError("origin does not point to the frozen forecastfm repository")
    remote = git_output("ls-remote", "--exit-code", REMOTE_NAME, REMOTE_REF).split()
    if remote != [revision, REMOTE_REF]:
        raise RuntimeError("authoritative origin/main differs from the v2 protocol revision")
    return revision


def verify_retired_v1() -> frozenset[str]:
    """Return the exact retired v1 IDs after checking their public commitments."""
    if file_sha256(V1_MANIFEST_PATH) != EXPECTED_V1_MANIFEST_SHA256:
        raise RuntimeError("retired v1 manifest differs from its commitment")
    manifest = parse_json_object(V1_MANIFEST_PATH.read_text(encoding="utf-8"))
    selection = require_object(required_field(manifest, "selection"), "selection")
    values = require_list(required_field(selection, "ordered_question_ids"), "question_ids")
    question_ids = tuple(
        require_string(value, f"question_ids[{index}]") for index, value in enumerate(values)
    )
    committed_hash = require_string(
        required_field(selection, "ordered_question_ids_sha256"),
        "ordered_question_ids_sha256",
    )
    if len(question_ids) != CANARY_SIZE or len(set(question_ids)) != CANARY_SIZE:
        raise RuntimeError("retired v1 cohort must contain exactly 64 unique IDs")
    if committed_hash != EXPECTED_V1_QUESTION_IDS_SHA256:
        raise RuntimeError("retired v1 ID commitment differs from the expected digest")
    if canonical_sha256(list(question_ids)) != EXPECTED_V1_QUESTION_IDS_SHA256:
        raise RuntimeError("retired v1 IDs differ from their commitment")
    return frozenset(question_ids)


def verify_format_smoke() -> None:
    """Require the exact passing no-thinking smoke result and raw-output seal."""
    if file_sha256(SMOKE_RESULT_PATH) != EXPECTED_SMOKE_RESULT_SHA256:
        raise RuntimeError("format-smoke result differs from its commitment")
    if file_sha256(SMOKE_SEAL_PATH) != EXPECTED_SMOKE_SEAL_SHA256:
        raise RuntimeError("format-smoke seal differs from its commitment")
    result = parse_json_object(SMOKE_RESULT_PATH.read_text(encoding="utf-8"))
    if required_field(result, "passed") is not True:
        raise RuntimeError("format smoke did not pass both model arms")
    seal_hash = require_string(required_field(result, "seal_sha256"), "seal_sha256")
    if seal_hash != EXPECTED_SMOKE_SEAL_SHA256:
        raise RuntimeError("format-smoke result does not bind the expected seal")


def verify_v2_selection(retired_ids: frozenset[str]) -> tuple[str, ...]:
    """Derive and verify the answer-blind v2 ID cohort from prompt records only."""
    if file_sha256(SOURCE_PROMPTS_PATH) != EXPECTED_PROMPTS_SHA256:
        raise RuntimeError("validation prompts differ from their frozen commitment")
    question_ids: list[str] = []
    for index, line in enumerate(SOURCE_PROMPTS_PATH.read_text(encoding="utf-8").splitlines()):
        record = parse_json_object(line)
        question_ids.append(
            require_string(required_field(record, "question_id"), f"question_id[{index}]")
        )
    if len(question_ids) != EXPECTED_VALIDATION_COUNT or len(set(question_ids)) != len(
        question_ids
    ):
        raise RuntimeError("validation prompt IDs are incomplete or duplicated")
    start = V2_PROTOCOL.selection_start
    selected = tuple(sorted(question_ids)[start : start + CANARY_SIZE])
    if len(selected) != CANARY_SIZE or set(selected) & retired_ids:
        raise RuntimeError("v2 selection is incomplete or overlaps the retired cohort")
    if canonical_sha256(list(selected)) != EXPECTED_V2_QUESTION_IDS_SHA256:
        raise RuntimeError("v2 question IDs differ from their frozen commitment")
    return selected


def verify_v2_manifest(manifest: CanaryManifest, selected_ids: tuple[str, ...]) -> None:
    """Require the exact answer-blind source, cohort, and prompt commitments."""
    expected = (
        EXPECTED_PROMPTS_SHA256,
        EXPECTED_ANSWERS_SHA256,
        EXPECTED_DATA_MANIFEST_SHA256,
        selected_ids,
        EXPECTED_V2_QUESTION_IDS_SHA256,
        EXPECTED_V2_PROMPTS_SHA256,
    )
    actual = (
        manifest.source_prompt_sha256,
        manifest.source_answer_sha256,
        manifest.dataset_manifest_sha256,
        manifest.question_ids,
        manifest.question_ids_sha256,
        manifest.prompts_sha256,
    )
    if actual != expected:
        raise RuntimeError("v2 manifest differs from its frozen answer-blind commitments")


def protocol_commitments() -> dict[str, str]:
    """Return the exact prerequisite hashes embedded in the v2 manifest."""
    return {
        "retired_v1_manifest_sha256": EXPECTED_V1_MANIFEST_SHA256,
        "retired_v1_question_ids_sha256": EXPECTED_V1_QUESTION_IDS_SHA256,
        "format_smoke_result_sha256": EXPECTED_SMOKE_RESULT_SHA256,
        "format_smoke_seal_sha256": EXPECTED_SMOKE_SEAL_SHA256,
    }


def v2_decoding_settings() -> dict[str, object]:
    """Bind no-thinking inference and the SDK transport-retry limitation."""
    return {
        **decoding_settings(),
        "renderer": V2_PROTOCOL.renderer_name,
        "transport_retry_note": TRANSPORT_RETRY_NOTE,
    }


def _verify_dataset_manifest() -> None:
    if file_sha256(DATA_MANIFEST_PATH) != EXPECTED_DATA_MANIFEST_SHA256:
        raise RuntimeError("dataset manifest differs from its frozen commitment")
    manifest = parse_json_object(DATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    outputs = require_object(required_field(manifest, "outputs"), "outputs")
    expected = {
        SOURCE_PROMPTS_PATH.name: EXPECTED_PROMPTS_SHA256,
        "nba_elo_validation_answers.jsonl": EXPECTED_ANSWERS_SHA256,
    }
    actual = {name: require_string(required_field(outputs, name), name) for name in expected}
    if actual != expected:
        raise RuntimeError("dataset manifest differs from the validation commitments")


def git_output(*arguments: str) -> str:
    """Run a read-only Git query in the project repository."""
    result = subprocess.run(
        ("git", *arguments),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    main()
