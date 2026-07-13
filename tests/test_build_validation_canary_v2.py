"""Answer-free checks for the disjoint canary-v2 builder."""

import pytest
from examples import build_validation_canary_v2

from forecastfm.integrity import canonical_sha256

REVISION = "a" * 40


def test_v2_selection_is_exact_and_disjoint_from_retired_v1() -> None:
    retired_ids = build_validation_canary_v2.verify_retired_v1()
    selected = build_validation_canary_v2.verify_v2_selection(retired_ids)

    assert len(selected) == 64
    assert not set(selected) & retired_ids
    assert selected == tuple(sorted(selected))
    assert (
        canonical_sha256(list(selected))
        == build_validation_canary_v2.EXPECTED_V2_QUESTION_IDS_SHA256
    )
    assert selected[0] == "nba-04affad47fabcab1"
    assert selected[-1] == "nba-08c23006b9fd18e9"


def test_v2_prerequisite_commitments_are_exact() -> None:
    build_validation_canary_v2.verify_format_smoke()

    assert build_validation_canary_v2.protocol_commitments() == {
        "retired_v1_manifest_sha256": (build_validation_canary_v2.EXPECTED_V1_MANIFEST_SHA256),
        "retired_v1_question_ids_sha256": (
            build_validation_canary_v2.EXPECTED_V1_QUESTION_IDS_SHA256
        ),
        "format_smoke_result_sha256": (build_validation_canary_v2.EXPECTED_SMOKE_RESULT_SHA256),
        "format_smoke_seal_sha256": build_validation_canary_v2.EXPECTED_SMOKE_SEAL_SHA256,
    }


def test_v2_decoding_uses_the_passing_no_thinking_renderer() -> None:
    decoding = build_validation_canary_v2.v2_decoding_settings()

    assert decoding["renderer"] == "qwen3_5_disable_thinking"
    assert decoding["transport_retry_note"] == build_validation_canary_v2.TRANSPORT_RETRY_NOTE


def test_builder_requires_the_authoritative_remote_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def git_output(*arguments: str) -> str:
        calls.append(arguments)
        if arguments == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        if arguments in {("rev-parse", "HEAD"), ("rev-parse", "origin/main")}:
            return REVISION
        if arguments == ("remote", "get-url", build_validation_canary_v2.REMOTE_NAME):
            return build_validation_canary_v2.EXPECTED_REMOTE_URL
        if arguments == (
            "ls-remote",
            "--exit-code",
            build_validation_canary_v2.REMOTE_NAME,
            build_validation_canary_v2.REMOTE_REF,
        ):
            return f"{REVISION}\t{build_validation_canary_v2.REMOTE_REF}"
        raise AssertionError(f"unexpected Git call: {arguments}")

    monkeypatch.setattr(build_validation_canary_v2, "git_output", git_output)

    assert build_validation_canary_v2.require_published_protocol() == REVISION
    assert calls[-1][0] == "ls-remote"
