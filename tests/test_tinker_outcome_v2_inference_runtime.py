"""Offline tests for the single-application-attempt outcome-v2 inference runner."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from math import log
from pathlib import Path
from typing import cast

import pytest
import tinker
from examples import tinker_outcome_v2_inference_runtime as runtime
from tinker.lib.retry_handler import RetryConfig  # pyright: ignore[reportMissingTypeStubs]

from forecastfm.integrity import canonical_json, canonical_sha256
from forecastfm.nba_evaluation_gate import read_nba_evaluation_forecasts_jsonl
from forecastfm.nba_feature_rows import NbaRichFeatureRow
from forecastfm.nba_rich import NbaRichFeatures
from forecastfm.outcome import OPPONENT_LABEL, TEAM_LABEL
from forecastfm.outcome_v2_config import outcome_v2_inference_settings
from forecastfm.outcome_v2_inference import (
    OutcomeV2GenerationArtifacts,
    OutcomeV2GenerationLock,
    binary_forecasts_from_inference_records,
    outcome_v2_prompt_pairs_jsonl_bytes,
    rendered_prompt_token_ids_sha256,
)

TEAM_TOKEN_ID = 10
OPPONENT_TOKEN_ID = 20


class FakeLogprobClient:
    """Return fixed label scores while recording every logical call."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    async def compute_logprobs_async(
        self,
        prompt: tinker.ModelInput,
    ) -> list[float | None]:
        tokens = tuple(prompt.to_ints())
        self.calls.append(tokens)
        score = log(0.4) if tokens[-1] == TEAM_TOKEN_ID else log(0.2)
        return [None] * (prompt.length - 1) + [score]


class DrainedFailureClient:
    """Fail TEAM calls while allowing every sibling candidate call to finish."""

    def __init__(self) -> None:
        self.calls = 0
        self.finished = 0

    async def compute_logprobs_async(
        self,
        prompt: tinker.ModelInput,
    ) -> list[float | None]:
        self.calls += 1
        try:
            if prompt.to_ints()[-1] == TEAM_TOKEN_ID:
                raise RuntimeError("secret provider detail")
            await asyncio.sleep(0.01)
            return [None] * (prompt.length - 1) + [log(0.2)]
        finally:
            self.finished += 1


class MalformedClient:
    """Return an invalid candidate response without raising remotely."""

    def __init__(self) -> None:
        self.calls = 0

    async def compute_logprobs_async(
        self,
        prompt: tinker.ModelInput,
    ) -> list[float | None]:
        self.calls += 1
        if prompt.to_ints()[-1] == TEAM_TOKEN_ID:
            return []
        return [None] * (prompt.length - 1) + [log(0.2)]


def _row(question_id: str, season: int) -> NbaRichFeatureRow:
    cutoff = datetime(season, 10, 21, 22, tzinfo=UTC)
    return NbaRichFeatureRow(
        question_id=question_id,
        source_game_id=f"source-{question_id}",
        team_id=f"team-{question_id}",
        opponent_id=f"opponent-{question_id}",
        site="neutral",
        season=season,
        forecast_cutoff=cutoff,
        scheduled_tipoff=cutoff + timedelta(hours=1),
        elo_team_win_probability=0.6,
        elo_opponent_win_probability=0.4,
        elo_available_at=cutoff - timedelta(hours=1),
        elo_state_sha256="a" * 64,
        rich_features=NbaRichFeatures.from_vector(
            (1.0, -1.0, 2.0, -2.0, 300.0, -1.0, 0.2, -0.2, 4.0, -4.0, 25.0)
        ),
        evidence_bundle_sha256="b" * 64,
        input_available_at=cutoff - timedelta(minutes=30),
    )


def _prepared() -> runtime.PreparedOutcomeV2Inference:
    rows = (_row("post-sft-2026", 2026), _row("post-sft-2027", 2027))
    games = tuple(_rendered_game(row, 100 + index * 10) for index, row in enumerate(rows))
    return runtime.PreparedOutcomeV2Inference(
        generation_lock=_generation_lock(rows),
        sampler_path="tinker://run/sampler/final",
        games=games,
    )


def _generation_lock(rows: tuple[NbaRichFeatureRow, ...]) -> OutcomeV2GenerationLock:
    settings = outcome_v2_inference_settings()
    question_ids = [row.question_id for row in rows]
    record: dict[str, object] = {
        "schema_version": 1,
        "kind": "forecastfm_outcome_v2_generation_lock",
        "status": "committed_before_remote_calls",
        "candidate_role": "forecastfm_outcome_v2_sft_adapter",
        "created_at": "2026-07-17T18:00:00Z",
        "outcome_v2_run_lock_sha256": "1" * 64,
        "outcome_v2_experiment_lock_sha256": "2" * 64,
        "sampler_path": "tinker://run/sampler/final",
        "feature_rows_sha256": "3" * 64,
        "feature_row_sha256s": [row.row_sha256 for row in rows],
        "prompt_pairs_sha256": "4" * 64,
        "question_ids": question_ids,
        "question_ids_sha256": canonical_sha256(question_ids),
        "tabular_seasons": [2024, 2025],
        "evaluation_seasons": [2026, 2027],
        "season_relation": "strictly_later_than_run_lock_tabular_seasons",
        "game_count": 2,
        "orientation_count": 4,
        "renderer_name": "qwen3_5_disable_thinking",
        "label_token_ids": {TEAM_LABEL: TEAM_TOKEN_ID, OPPONENT_LABEL: OPPONENT_TOKEN_ID},
        "inference_settings": settings,
        "inference_settings_sha256": canonical_sha256(settings),
        "call_policy": {
            "logical_calls_per_game": 4,
            "expected_logical_calls": 8,
            "application_attempts_per_game": 1,
            "application_retries": 0,
            "sdk_retry_logic_enabled": False,
            "sdk_internal_retransmission_window_seconds": 300,
            "generated_text_used": False,
            "sdk_internal_unused_generated_tokens_per_call": 1,
            "transport_retry_note": settings["transport_retry_note"],
        },
    }
    return OutcomeV2GenerationLock(canonical_json(record).encode())


def _rendered_game(row: NbaRichFeatureRow, token: int) -> runtime.RenderedOutcomeV2Game:
    original_tokens = (token, token + 1)
    swapped_tokens = (token + 2, token + 3)
    original = tinker.ModelInput.from_ints(list(original_tokens))
    swapped = tinker.ModelInput.from_ints(list(swapped_tokens))
    return runtime.RenderedOutcomeV2Game(
        row=row,
        original_prompt_token_ids_sha256=rendered_prompt_token_ids_sha256(original_tokens),
        swapped_prompt_token_ids_sha256=rendered_prompt_token_ids_sha256(swapped_tokens),
        candidate_inputs=(
            original.append_int(TEAM_TOKEN_ID),
            original.append_int(OPPONENT_TOKEN_ID),
            swapped.append_int(TEAM_TOKEN_ID),
            swapped.append_int(OPPONENT_TOKEN_ID),
        ),
    )


def _paths(tmp_path: Path) -> runtime.OutcomeV2InferenceRuntimePaths:
    artifacts = OutcomeV2GenerationArtifacts(
        project_root=tmp_path,
        run_lock_path=tmp_path / "run.json",
        experiment_lock_path=tmp_path / "experiment.json",
        feature_rows_path=tmp_path / "features.jsonl",
    )
    return runtime.OutcomeV2InferenceRuntimePaths(
        artifacts=artifacts,
        generation_lock_path=tmp_path / "generation.json",
        prompts_path=tmp_path / "prompts.jsonl",
        journal_path=tmp_path / "journal.jsonl",
        inference_records_path=tmp_path / "records.jsonl",
        forecasts_path=tmp_path / "forecasts.jsonl",
    )


def _install_prepared(
    monkeypatch: pytest.MonkeyPatch,
    prepared: runtime.PreparedOutcomeV2Inference,
) -> None:
    def prepare(
        _paths_value: runtime.OutcomeV2InferenceRuntimePaths,
    ) -> runtime.PreparedOutcomeV2Inference:
        return prepared

    monkeypatch.setattr(runtime, "prepare_outcome_v2_inference", prepare)


def _run(
    paths: runtime.OutcomeV2InferenceRuntimePaths,
    verifier: runtime.PrecommitVerifier,
    factory: runtime.SamplingClientFactory,
) -> tuple[runtime.InferenceRecord, ...]:
    return asyncio.run(
        runtime.run_outcome_v2_inference(
            paths,
            verifier,
            create_sampling_client=factory,
        )
    )


def test_success_uses_four_calls_per_game_and_verifies_before_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    paths = _paths(tmp_path)
    _install_prepared(monkeypatch, prepared)
    events: list[str] = []
    client = FakeLogprobClient()

    def verify(lock: OutcomeV2GenerationLock, path: Path) -> None:
        assert lock == prepared.generation_lock
        assert path == paths.generation_lock_path
        events.append("verified")

    async def create(
        sampler_path: str,
        generation_lock_sha256: str,
        retry_config: RetryConfig,
    ) -> runtime.CandidateLogprobClient:
        assert events == ["verified"]
        assert sampler_path == prepared.sampler_path
        assert generation_lock_sha256 == prepared.generation_lock.sha256
        assert retry_config.enable_retry_logic is False
        events.append("client")
        return client

    records = _run(paths, verify, create)

    assert events == ["verified", "client"]
    assert len(client.calls) == 8
    assert all(record.status == "completed" for record in records)
    assert paths.inference_records_path.is_file()
    rows = tuple(game.row for game in prepared.games)
    assert paths.prompts_path.read_bytes() == outcome_v2_prompt_pairs_jsonl_bytes(rows)
    assert read_nba_evaluation_forecasts_jsonl(paths.forecasts_path) == (
        binary_forecasts_from_inference_records(records, prepared.generation_lock)
    )
    assert len(paths.journal_path.read_text().splitlines()) == 4


def test_default_client_binds_provider_metadata_to_generation_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    paths = _paths(tmp_path)
    _install_prepared(monkeypatch, prepared)
    client = FakeLogprobClient()

    class FakeService:
        async def create_sampling_client_async(
            self,
            model_path: str | None = None,
            base_model: str | None = None,
            retry_config: RetryConfig | None = None,
        ) -> tinker.SamplingClient:
            assert model_path == prepared.sampler_path
            assert base_model is None
            assert retry_config is not None
            assert retry_config.enable_retry_logic is False
            return cast(tinker.SamplingClient, client)

    def service_client(
        user_metadata: dict[str, str] | None = None,
    ) -> FakeService:
        assert user_metadata == {
            "outcome_v2_generation_lock_sha256": prepared.generation_lock.sha256
        }
        return FakeService()

    monkeypatch.setattr(runtime.tinker, "ServiceClient", service_client)
    records = asyncio.run(
        runtime.run_outcome_v2_inference(
            paths,
            lambda _lock, _path: None,
        )
    )

    assert len(client.calls) == 8
    assert all(record.status == "completed" for record in records)


@pytest.mark.parametrize(
    ("client_factory", "expected_reason"),
    [
        (DrainedFailureClient, "candidate_call_exception:RuntimeError"),
        (MalformedClient, "candidate_output_invalid"),
    ],
)
def test_remote_and_malformed_failures_are_drained_and_do_not_stop_later_games(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client_factory: Callable[[], DrainedFailureClient | MalformedClient],
    expected_reason: str,
) -> None:
    prepared = _prepared()
    paths = _paths(tmp_path)
    _install_prepared(monkeypatch, prepared)
    client = client_factory()

    async def create(
        _sampler_path: str,
        _generation_lock_sha256: str,
        _retry_config: RetryConfig,
    ) -> runtime.CandidateLogprobClient:
        return cast(runtime.CandidateLogprobClient, client)

    records = _run(paths, lambda _lock, _path: None, create)

    assert client.calls == 8
    if isinstance(client, DrainedFailureClient):
        assert client.finished == 8
    assert [record.failure_reason for record in records] == [expected_reason, expected_reason]
    assert "secret provider detail" not in paths.journal_path.read_text()


def test_started_only_restart_becomes_terminal_failure_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    paths = _paths(tmp_path)
    _install_prepared(monkeypatch, prepared)
    first = prepared.games[0]
    paths.journal_path.write_bytes(
        f"{canonical_json(_started_payload(prepared, first))}\n".encode() + b'{"torn_terminal":'
    )
    client = FakeLogprobClient()

    async def create(
        _sampler_path: str,
        _generation_lock_sha256: str,
        _retry_config: RetryConfig,
    ) -> runtime.CandidateLogprobClient:
        return client

    records = _run(paths, lambda _lock, _path: None, create)

    assert records[0].failure_reason == "interrupted_after_start"
    assert records[1].status == "completed"
    assert len(client.calls) == 4
    assert b"torn_terminal" not in paths.journal_path.read_bytes()

    async def unexpected_create(
        _sampler_path: str,
        _generation_lock_sha256: str,
        _retry_config: RetryConfig,
    ) -> runtime.CandidateLogprobClient:
        raise AssertionError("terminal games must not create a new client")

    assert _run(paths, lambda _lock, _path: None, unexpected_create) == records
    assert len(client.calls) == 4


def _started_payload(
    prepared: runtime.PreparedOutcomeV2Inference,
    game: runtime.RenderedOutcomeV2Game,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "started",
        "generation_lock_sha256": prepared.generation_lock.sha256,
        "sequence": 0,
        "question_id": game.row.question_id,
        "feature_row_sha256": game.row.row_sha256,
        "original_prompt_token_ids_sha256": game.original_prompt_token_ids_sha256,
        "swapped_prompt_token_ids_sha256": game.swapped_prompt_token_ids_sha256,
        "record": None,
    }


def test_rejected_precommit_never_constructs_a_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    paths = _paths(tmp_path)
    _install_prepared(monkeypatch, prepared)

    def reject(_lock: OutcomeV2GenerationLock, _path: Path) -> None:
        raise RuntimeError("missing external precommit receipt")

    async def unexpected_create(
        _sampler_path: str,
        _generation_lock_sha256: str,
        _retry_config: RetryConfig,
    ) -> runtime.CandidateLogprobClient:
        raise AssertionError("client was constructed before precommit verification")

    with pytest.raises(RuntimeError, match="missing external precommit receipt"):
        _run(paths, reject, unexpected_create)
    assert not paths.journal_path.exists()


def test_runtime_has_no_answer_resolution_or_cohort_imports() -> None:
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    import_lines = "\n".join(
        line for line in source.splitlines() if line.startswith(("from ", "import "))
    )
    assert "answer" not in import_lines.lower()
    assert "resolution" not in import_lines.lower()
    assert "cohort" not in import_lines.lower()
