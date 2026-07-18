"""Tests for the immutable outcome-v2 paid-run commitment."""

from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from forecastfm import outcome_v2_run
from forecastfm.integrity import bytes_sha256, canonical_json, canonical_sha256, file_sha256
from forecastfm.json_utils import require_object, required_field
from forecastfm.outcome import OUTCOME_INPUT_SCHEMA_VERSION, OUTCOME_SYSTEM_PROMPT
from forecastfm.outcome_v2_config import (
    OUTCOME_RENDERER_NAME,
    outcome_v2_elo_recipe,
    outcome_v2_evaluation_policy,
    outcome_v2_inference_settings,
    outcome_v2_sft_settings,
    outcome_v2_training_settings,
)
from forecastfm.outcome_v2_preflight import (
    OutcomeV2Preflight,
    OutcomeV2PreflightError,
    PreparedOutcomeV2Run,
)
from forecastfm.outcome_v2_prompt import OUTCOME_V2_SYSTEM_PROMPT
from forecastfm.outcome_v2_run import (
    OUTCOME_V2_LOCKED_CODE_FILES,
    OutcomeV2RunError,
    OutcomeV2RunLock,
    build_outcome_v2_run_lock,
    require_outcome_v2_run_static_contract,
    verify_outcome_v2_run_lock,
    write_outcome_v2_run_lock,
)
from forecastfm.run_config import TORCH_VERSION, model_settings, package_versions

REVISION = "a" * 40
ACTION_AT = datetime(2026, 7, 17, 12, 34, 56, tzinfo=UTC)
TRAINING_BYTES = b'{"label":"TEAM","question_id":"game-1"}\n'


def _proof(
    training_bytes: bytes = TRAINING_BYTES,
    *,
    row_count: int = 12,
    pair_count: int = 6,
    batch_size: int = 14,
) -> OutcomeV2Preflight:
    return OutcomeV2Preflight(
        manifest_sha256="1" * 64,
        action_at=ACTION_AT,
        action_time_source="internal_paid_preparation",
        untouched_evaluation_seasons=(2024, 2025),
        training_sha256=bytes_sha256(training_bytes),
        feature_rows_sha256="2" * 64,
        snapshot_pack_sha256="3" * 64,
        evidence_bundles_sha256="4" * 64,
        elo_states_sha256="5" * 64,
        elo_replay_sha256="6" * 64,
        seasons_sha256="7" * 64,
        resolutions_sha256="8" * 64,
        rights_lock_sha256="9" * 64,
        evaluation_feature_rows_sha256="a" * 64,
        evaluation_elo_replay_sha256="b" * 64,
        evaluation_elo_states_sha256="c" * 64,
        evaluation_resolutions_sha256="d" * 64,
        calibration_sha256="1" * 64,
        rich_baseline_model_sha256="e" * 64,
        rich_baseline_forecast_lock_sha256="f" * 64,
        evaluation_report_sha256="0" * 64,
        row_count=row_count,
        pair_count=pair_count,
        batch_size=batch_size,
    )


def _prepared(proof: OutcomeV2Preflight | None = None) -> PreparedOutcomeV2Run:
    return PreparedOutcomeV2Run(proof or _proof(), TRAINING_BYTES)


def _make_project(path: Path) -> None:
    for logical_name, relative_path in OUTCOME_V2_LOCKED_CODE_FILES:
        full_path = path / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(f"fixture for {logical_name}\n", encoding="utf-8")
    (path / "uv.lock").write_text("fixture lock\n", encoding="utf-8")


def _object(record: dict[str, object], field_name: str) -> dict[str, object]:
    return require_object(required_field(record, field_name), field_name)


def _assert_elo_offset_settings(record: dict[str, object]) -> None:
    training = _object(record, "training_settings")
    assert training["sdk_method"] == "forward_backward_custom_async"
    assert training["final_logit"] == "logit(elo_team_probability)+logp(TEAM)-logp(OTHER)"
    assert training["full_vocabulary_cross_entropy"] is False
    inference = _object(record, "inference_settings")
    assert inference["api_method"] == "compute_logprobs_async"
    assert inference["generated_text_used"] is False
    assert inference["sdk_internal_unused_generated_tokens_per_call"] == 1
    assert "text_generation" not in inference


def test_lock_is_deterministic_and_binds_every_frozen_input(tmp_path: Path) -> None:
    _make_project(tmp_path)
    prepared = _prepared()

    first = build_outcome_v2_run_lock(tmp_path, prepared, REVISION)
    second = build_outcome_v2_run_lock(tmp_path, prepared, REVISION)
    record = first.to_record()

    assert first.canonical_bytes == second.canonical_bytes
    assert record["status"] == "committed_before_remote_client"
    assert record["action_at"] == ACTION_AT.isoformat().replace("+00:00", "Z")
    assert record["created_at"] == record["action_at"]
    assert record["code_revision"] == REVISION
    assert record["training_bytes_sha256"] == bytes_sha256(TRAINING_BYTES)

    preflight = _object(record, "preflight")
    assert set(preflight) == {item.name for item in fields(OutcomeV2Preflight)}
    assert preflight == prepared.proof.canonical_payload()
    assert preflight["action_time_source"] == "internal_paid_preparation"
    assert preflight["training_sha256"] == prepared.proof.training_sha256

    assert _object(record, "sft_settings") == outcome_v2_sft_settings()
    assert _object(record, "training_settings") == outcome_v2_training_settings()
    assert _object(record, "inference_settings") == outcome_v2_inference_settings()
    _assert_elo_offset_settings(record)

    elo = _object(record, "elo_recipe")
    assert elo["config"] == outcome_v2_elo_recipe().canonical_payload()
    assert elo["sha256"] == outcome_v2_elo_recipe().recipe_sha256
    evaluation = _object(record, "evaluation_policy")
    assert evaluation["config"] == outcome_v2_evaluation_policy().canonical_payload()
    assert evaluation["sha256"] == outcome_v2_evaluation_policy().policy_sha256

    expected_model = model_settings()
    expected_model["renderer"] = OUTCOME_RENDERER_NAME
    assert _object(record, "model") == expected_model
    assert record["model_reference_sha256"] == canonical_sha256(expected_model)
    assert _object(record, "packages") == {**package_versions(), "torch": TORCH_VERSION}

    prompt = _object(record, "prompt")
    assert prompt["outcome_input_schema_version"] == OUTCOME_INPUT_SCHEMA_VERSION
    assert prompt["system_prompt"] == OUTCOME_V2_SYSTEM_PROMPT
    assert prompt["system_prompt_sha256"] == bytes_sha256(OUTCOME_V2_SYSTEM_PROMPT.encode("utf-8"))
    assert record["uv_lock_sha256"] == file_sha256(tmp_path / "uv.lock")

    code_hashes = _object(record, "code_sha256")
    assert set(code_hashes) == {name for name, _ in OUTCOME_V2_LOCKED_CODE_FILES}
    for logical_name, relative_path in OUTCOME_V2_LOCKED_CODE_FILES:
        assert code_hashes[logical_name] == file_sha256(tmp_path / relative_path)


def test_static_contract_rejects_a_self_consistent_old_prompt(tmp_path: Path) -> None:
    _make_project(tmp_path)
    record = build_outcome_v2_run_lock(tmp_path, _prepared(), REVISION).to_record()
    record["prompt"] = {
        "outcome_input_schema_version": OUTCOME_INPUT_SCHEMA_VERSION,
        "system_prompt": OUTCOME_SYSTEM_PROMPT,
        "system_prompt_sha256": bytes_sha256(OUTCOME_SYSTEM_PROMPT.encode("utf-8")),
    }
    forged = OutcomeV2RunLock(canonical_json(record).encode("utf-8"))

    with pytest.raises(OutcomeV2RunError, match="current prompt contract"):
        require_outcome_v2_run_static_contract(tmp_path, forged)


def test_static_contract_rejects_a_self_consistent_weaker_policy(tmp_path: Path) -> None:
    _make_project(tmp_path)
    record = build_outcome_v2_run_lock(tmp_path, _prepared(), REVISION).to_record()
    policy = _object(record, "evaluation_policy")
    config = _object(policy, "config")
    config["minimum_games_per_season"] = 1
    policy["config"] = config
    policy["sha256"] = canonical_sha256(config)
    record["evaluation_policy"] = policy
    forged = OutcomeV2RunLock(canonical_json(record).encode("utf-8"))

    with pytest.raises(OutcomeV2RunError, match="current evaluation_policy contract"):
        require_outcome_v2_run_static_contract(tmp_path, forged)


def test_prepared_run_rejects_wrong_bytes_and_caller_time() -> None:
    proof = _proof()

    with pytest.raises(OutcomeV2PreflightError, match="training bytes differ"):
        PreparedOutcomeV2Run(proof, b"different bytes")

    caller_time = replace(proof, action_time_source="caller_supplied_offline_check")
    with pytest.raises(OutcomeV2PreflightError, match="internally derived action time"):
        PreparedOutcomeV2Run(caller_time, TRAINING_BYTES)

    with pytest.raises(OutcomeV2PreflightError, match="batch size differs"):
        _prepared(_proof(batch_size=13))


def test_lock_changes_with_preflight_revision_and_signed_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_project(tmp_path)
    prepared = _prepared()
    baseline = build_outcome_v2_run_lock(tmp_path, prepared, REVISION)

    changed_proof = replace(prepared.proof, feature_rows_sha256="e" * 64)
    changed_preflight = build_outcome_v2_run_lock(
        tmp_path,
        _prepared(changed_proof),
        REVISION,
    )
    changed_rich_model = build_outcome_v2_run_lock(
        tmp_path,
        _prepared(replace(prepared.proof, rich_baseline_model_sha256="0" * 64)),
        REVISION,
    )
    changed_revision = build_outcome_v2_run_lock(tmp_path, prepared, "f" * 40)

    negative_zero_settings = outcome_v2_training_settings()
    negative_zero_settings["learning_rate"] = -0.0
    monkeypatch.setattr(
        outcome_v2_run,
        "outcome_v2_training_settings",
        lambda: negative_zero_settings,
    )
    negative_zero = build_outcome_v2_run_lock(tmp_path, prepared, REVISION)

    assert changed_preflight.canonical_bytes != baseline.canonical_bytes
    assert changed_rich_model.canonical_bytes != baseline.canonical_bytes
    assert changed_revision.canonical_bytes != baseline.canonical_bytes
    assert negative_zero.canonical_bytes != baseline.canonical_bytes


def test_final_partial_batch_is_allowed(tmp_path: Path) -> None:
    _make_project(tmp_path)

    lock = build_outcome_v2_run_lock(
        tmp_path,
        _prepared(_proof(row_count=12, pair_count=6)),
        REVISION,
    )

    assert lock.to_record()["status"] == "committed_before_remote_client"


def test_writer_is_create_only_and_verifier_checks_exact_bytes(tmp_path: Path) -> None:
    _make_project(tmp_path)
    prepared = _prepared()
    lock = build_outcome_v2_run_lock(tmp_path, prepared, REVISION)
    path = tmp_path / "locks/outcome_v2.json"

    digest = write_outcome_v2_run_lock(path, lock)

    assert path.read_bytes() == lock.canonical_bytes
    assert digest == lock.sha256
    assert verify_outcome_v2_run_lock(tmp_path, path, prepared, REVISION) == lock
    with pytest.raises(FileExistsError):
        write_outcome_v2_run_lock(path, lock)


def test_verifier_rejects_changed_code_config_and_preflight(tmp_path: Path) -> None:
    _make_project(tmp_path)
    prepared = _prepared()
    lock = build_outcome_v2_run_lock(tmp_path, prepared, REVISION)
    path = tmp_path / "outcome_v2.json"
    write_outcome_v2_run_lock(path, lock)

    config_path = tmp_path / "src/forecastfm/outcome_v2_config.py"
    original_config = config_path.read_text(encoding="utf-8")
    config_path.write_text("changed config\n", encoding="utf-8")
    with pytest.raises(OutcomeV2RunError, match="differs"):
        verify_outcome_v2_run_lock(tmp_path, path, prepared, REVISION)

    config_path.write_text(original_config, encoding="utf-8")
    changed_proof = replace(prepared.proof, evaluation_report_sha256="e" * 64)
    with pytest.raises(OutcomeV2RunError, match="differs"):
        verify_outcome_v2_run_lock(
            tmp_path,
            path,
            _prepared(changed_proof),
            REVISION,
        )


def test_verifier_rejects_changed_runtime_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_project(tmp_path)
    prepared = _prepared()
    path = tmp_path / "outcome_v2.json"
    write_outcome_v2_run_lock(
        path,
        build_outcome_v2_run_lock(tmp_path, prepared, REVISION),
    )
    changed_settings = outcome_v2_training_settings()
    changed_settings["max_steps"] = 129
    monkeypatch.setattr(
        outcome_v2_run,
        "outcome_v2_training_settings",
        lambda: changed_settings,
    )

    with pytest.raises(OutcomeV2RunError, match="differs"):
        verify_outcome_v2_run_lock(tmp_path, path, prepared, REVISION)


def test_malformed_and_stale_locks_are_rejected(tmp_path: Path) -> None:
    _make_project(tmp_path)
    prepared = _prepared()
    lock = build_outcome_v2_run_lock(tmp_path, prepared, REVISION)

    with pytest.raises(OutcomeV2RunError, match="canonical JSON"):
        OutcomeV2RunLock(lock.canonical_bytes + b"\n")

    with pytest.raises(OutcomeV2RunError, match="immutable bytes"):
        OutcomeV2RunLock(cast(bytes, bytearray(lock.canonical_bytes)))

    extra_field = lock.to_record()
    extra_field["unexpected"] = True
    with pytest.raises(OutcomeV2RunError, match="structure"):
        OutcomeV2RunLock(canonical_json(extra_field).encode("utf-8"))

    wrong_source = lock.to_record()
    preflight = _object(wrong_source, "preflight")
    preflight["action_time_source"] = "caller_supplied_offline_check"
    wrong_source["preflight"] = preflight
    with pytest.raises(OutcomeV2RunError, match="derived during paid-run preparation"):
        OutcomeV2RunLock(canonical_json(wrong_source).encode("utf-8"))

    wrong_time = lock.to_record()
    preflight = _object(wrong_time, "preflight")
    preflight["action_at"] = "2026-07-17T12:34:57Z"
    wrong_time["preflight"] = preflight
    with pytest.raises(OutcomeV2RunError, match="must equal"):
        OutcomeV2RunLock(canonical_json(wrong_time).encode("utf-8"))

    wrong_batch_size = lock.to_record()
    preflight = _object(wrong_batch_size, "preflight")
    preflight["batch_size"] = 13
    wrong_batch_size["preflight"] = preflight
    with pytest.raises(OutcomeV2RunError, match="batch sizes must match"):
        OutcomeV2RunLock(canonical_json(wrong_batch_size).encode("utf-8"))


@pytest.mark.parametrize("revision", ["a" * 39, "A" * 40, "g" * 40, "a" * 65])
def test_revision_must_be_a_lowercase_git_id(tmp_path: Path, revision: str) -> None:
    _make_project(tmp_path)

    with pytest.raises(OutcomeV2RunError, match="lowercase 40-64"):
        build_outcome_v2_run_lock(tmp_path, _prepared(), revision)


def test_lock_contains_no_secret_local_path_or_agreement_content(tmp_path: Path) -> None:
    _make_project(tmp_path)
    secret = "super-secret-tinker-key"

    lock = build_outcome_v2_run_lock(tmp_path, _prepared(), REVISION)
    text = lock.canonical_bytes.decode("utf-8")
    record = lock.to_record()

    assert secret not in text
    assert "api_key" not in text.lower()
    assert str(tmp_path) not in text
    assert "agreement" not in text.lower()
    assert "training_path" not in text
    assert "import tinker" not in Path(outcome_v2_run.__file__).read_text(encoding="utf-8")
    assert set(_object(record, "code_sha256")) == {name for name, _ in OUTCOME_V2_LOCKED_CODE_FILES}


def test_every_locked_code_file_exists_and_missing_inputs_fail(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    locked_paths = dict(OUTCOME_V2_LOCKED_CODE_FILES)
    assert locked_paths["publication_workflow"] == Path(
        ".github/workflows/outcome-v2-publication-timestamp.yml"
    )
    assert locked_paths["github_actions_receipt"] == Path(
        "src/forecastfm/github_actions_receipt.py"
    )
    assert locked_paths["outcome_v2_aggregation"] == Path(
        "src/forecastfm/outcome_v2_aggregation.py"
    )
    assert locked_paths["outcome_v2_coverage"] == Path("src/forecastfm/outcome_v2_coverage.py")
    assert locked_paths["outcome_v2_rolling"] == Path("src/forecastfm/outcome_v2_rolling.py")
    assert locked_paths["outcome_v2_rolling_score"] == Path(
        "src/forecastfm/outcome_v2_rolling_score.py"
    )
    assert locked_paths["outcome_v2_rolling_gate"] == Path(
        "src/forecastfm/outcome_v2_rolling_gate.py"
    )
    for _, relative_path in OUTCOME_V2_LOCKED_CODE_FILES:
        assert (project_root / relative_path).is_file()

    with pytest.raises(OutcomeV2RunError, match="cannot hash code file"):
        build_outcome_v2_run_lock(tmp_path, _prepared(), REVISION)

    _make_project(tmp_path)
    (tmp_path / "uv.lock").unlink()
    with pytest.raises(OutcomeV2RunError, match=r"cannot hash uv\.lock"):
        build_outcome_v2_run_lock(tmp_path, _prepared(), REVISION)
