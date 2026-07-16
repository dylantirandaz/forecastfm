"""Offline tests for the answer-blind Tinker outcome runner."""

import asyncio
import fcntl
from dataclasses import replace
from datetime import UTC, datetime
from math import log
from pathlib import Path
from typing import cast

import pytest
import tinker
from examples import run_tinker_outcome_development as runner
from tinker.lib.retry_handler import RetryConfig  # pyright: ignore[reportMissingTypeStubs]
from tinker_cookbook import renderers

from forecastfm.integrity import canonical_sha256
from forecastfm.nba_data import side_swap_nba_example
from forecastfm.outcome_evaluation import ModelRole, OutcomeEvaluationManifest
from forecastfm.tinker_data import (
    ForecastRecord,
    build_outcome_forecast_record,
)
from tests.helpers import make_nba_training_example


class FakeRenderer:
    """Render each target-free orientation to one fixed prompt."""

    def build_generation_prompt(
        self,
        messages: list[renderers.Message],
        role: str = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        assert len(messages) == 2
        assert role == "assistant"
        assert prefill is None
        return tinker.ModelInput.from_ints([1, 2, 3])


class FakeLogprobClient:
    """Return fixed candidate scores or one terminal provider error."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[int, ...]] = []

    async def compute_logprobs_async(
        self,
        prompt: tinker.ModelInput,
    ) -> list[float | None]:
        tokens = tuple(prompt.to_ints())
        self.calls.append(tokens)
        if self.fail:
            raise RuntimeError("provider failed")
        values = {10: log(0.4), 20: log(0.1)}
        return [None] * (len(tokens) - 1) + [values[tokens[-1]]]


class DelayedFailureClient:
    """Fail some calls immediately while proving all sibling calls finish."""

    def __init__(self) -> None:
        self.active = 0
        self.finished = 0

    async def compute_logprobs_async(
        self,
        prompt: tinker.ModelInput,
    ) -> list[float | None]:
        self.active += 1
        try:
            if prompt.to_ints()[-1] == 10:
                raise RuntimeError("provider failed")
            await asyncio.sleep(0.01)
            return [None] * (prompt.length - 1) + [log(0.1)]
        finally:
            self.active -= 1
            self.finished += 1


class BadResponseClient:
    """Return the wrong response length for a stable local failure code."""

    async def compute_logprobs_async(
        self,
        prompt: tinker.ModelInput,
    ) -> list[float | None]:
        assert prompt.length > 0
        return []


class FakeService:
    """Capture base-versus-adapter client construction."""

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, str | None, bool]] = []

    async def create_sampling_client_async(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> tinker.SamplingClient:
        assert retry_config is not None
        self.calls.append((model_path, base_model, retry_config.enable_retry_logic))
        return cast(tinker.SamplingClient, FakeLogprobClient())


def _manifest() -> OutcomeEvaluationManifest:
    question_ids = ("nba-eval-0",)
    return OutcomeEvaluationManifest(
        created_at=datetime(2026, 7, 16, 3, tzinfo=UTC).isoformat(),
        protocol_revision="a" * 40,
        source_manifest_sha256="b" * 64,
        source_prompts_sha256="c" * 64,
        source_answers_sha256="d" * 64,
        frozen_prompts_sha256="c" * 64,
        training_lock_sha256="e" * 64,
        experiment_sha256="f" * 64,
        base_model="Qwen/Qwen3.5-4B",
        adapter_sampler_path="tinker://run/sampler_weights/final",
        renderer_name="qwen3_5_disable_thinking",
        team_token_id=10,
        opponent_token_id=20,
        game_count=1,
        orientation_count=2,
        logical_calls_per_game_per_arm=4,
        expected_total_logical_calls=8,
        max_active_arms=1,
        application_retries=0,
        transport_retry_note="same logical request may be retransmitted",
        question_ids=question_ids,
        question_ids_sha256=canonical_sha256(list(question_ids)),
        scoring_policy={"primary": "mean_log_loss"},
    )


def _prompt_pair() -> tuple[ForecastRecord, ForecastRecord]:
    example = make_nba_training_example()
    original = replace(
        example,
        case=replace(
            example.case,
            question=replace(example.case.question, question_id="nba-eval-0"),
        ),
    )
    swapped = side_swap_nba_example(original)
    return (
        build_outcome_forecast_record(original.case),
        build_outcome_forecast_record(swapped.case),
    )


def _inputs() -> runner.RunInputs:
    return runner.RunInputs(
        manifest=_manifest(),
        prompt_pairs=(_prompt_pair(),),
        renderer=cast(renderers.Renderer, FakeRenderer()),
        label_token_ids=(10, 20),
    )


def test_client_factory_uses_base_and_adapter_with_retries_disabled() -> None:
    service = FakeService()

    manifest = _manifest()
    asyncio.run(
        runner.create_clients(
            cast(tinker.ServiceClient, service),
            manifest.base_model,
            manifest.adapter_sampler_path,
        )
    )

    assert service.calls == [
        (None, "Qwen/Qwen3.5-4B", False),
        ("tinker://run/sampler_weights/final", None, False),
    ]


def test_score_arm_makes_four_candidate_calls_and_retains_raw_scores() -> None:
    client = FakeLogprobClient()
    original, swapped = _prompt_pair()

    record = asyncio.run(runner.score_arm(0, "base", client, _inputs(), (original, swapped)))

    assert record.status == "completed"
    assert record.symmetric_team_probability == 0.5
    assert len(client.calls) == 4
    assert record.original is not None
    assert record.original_prompt_tokens == (1, 2, 3)


def test_provider_error_is_one_terminal_arm_failure() -> None:
    client = FakeLogprobClient(fail=True)
    original, swapped = _prompt_pair()

    record = asyncio.run(runner.score_arm(0, "adapter", client, _inputs(), (original, swapped)))

    assert record.status == "failed"
    assert record.error == "candidate_call_exception:RuntimeError"
    assert len(client.calls) == 4


def test_provider_error_waits_for_every_sibling_call() -> None:
    client = DelayedFailureClient()
    original, swapped = _prompt_pair()

    record = asyncio.run(runner.score_arm(0, "adapter", client, _inputs(), (original, swapped)))

    assert record.status == "failed"
    assert client.finished == 4
    assert client.active == 0


def test_malformed_candidate_response_uses_actionable_failure_code() -> None:
    original, swapped = _prompt_pair()

    record = asyncio.run(
        runner.score_arm(0, "base", BadResponseClient(), _inputs(), (original, swapped))
    )

    assert record.error == "candidate_call_exception:unexpected_logprob_count"


def test_pending_loop_stops_after_first_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeLogprobClient(fail=True)
    journal_path = tmp_path / "journal.jsonl"
    attempt_path = tmp_path / "attempt.json"
    attempt_path.write_text("attempt\n", encoding="utf-8")
    monkeypatch.setattr(runner, "JOURNAL_PATH", journal_path)
    monkeypatch.setattr(runner, "ATTEMPT_PATH", attempt_path)
    clients: dict[ModelRole, tinker.SamplingClient] = {
        "base": cast(tinker.SamplingClient, client),
        "adapter": cast(tinker.SamplingClient, FakeLogprobClient()),
    }

    with pytest.raises(runner.OutcomeEvaluationError, match="stopped after terminal"):
        asyncio.run(
            runner.run_pending(
                _inputs(),
                clients,
                {},
                ((0, "base"), (0, "adapter")),
            )
        )

    assert len(client.calls) == 4
    assert journal_path.read_text(encoding="utf-8").count('"kind": "started"') == 1


def test_started_only_journal_unit_becomes_terminal_without_a_new_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "journal.jsonl"
    attempt_path = tmp_path / "attempt.json"
    attempt_path.write_text("attempt\n", encoding="utf-8")
    monkeypatch.setattr(runner, "JOURNAL_PATH", journal_path)
    monkeypatch.setattr(runner, "ATTEMPT_PATH", attempt_path)
    runner.append_started(0, "base", "nba-eval-0")
    completed, started = runner.read_journal(_manifest())

    runner.terminalize_interrupted(_inputs(), completed, started)

    recovered, _started = runner.read_journal(_manifest())
    assert recovered[(0, "base")].error == "indeterminate_interrupted"


def test_journal_rejects_overlapping_active_arms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "journal.jsonl"
    attempt_path = tmp_path / "attempt.json"
    attempt_path.write_text("attempt\n", encoding="utf-8")
    monkeypatch.setattr(runner, "JOURNAL_PATH", journal_path)
    monkeypatch.setattr(runner, "ATTEMPT_PATH", attempt_path)
    runner.append_started(0, "base", "nba-eval-0")
    runner.append_started(0, "adapter", "nba-eval-0")

    with pytest.raises(runner.OutcomeEvaluationError, match="overlapping active arms"):
        runner.read_journal(_manifest())


def test_partial_journal_tail_recovers_as_an_interrupted_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "journal.jsonl"
    attempt_path = tmp_path / "attempt.json"
    attempt_path.write_text("attempt\n", encoding="utf-8")
    monkeypatch.setattr(runner, "JOURNAL_PATH", journal_path)
    monkeypatch.setattr(runner, "ATTEMPT_PATH", attempt_path)
    runner.append_started(0, "base", "nba-eval-0")
    with journal_path.open("ab") as file:
        file.write(b'{"schema_version":1,"kind":"completed"')

    completed, started = runner.read_journal(_manifest(), recover_partial=True)
    runner.terminalize_interrupted(_inputs(), completed, started)
    recovered, _started = runner.read_journal(_manifest())

    assert recovered[(0, "base")].error == "indeterminate_interrupted"
    assert "recovered_partial_tail" in journal_path.read_text(encoding="utf-8")


def test_empty_crash_created_journal_recovers_without_skipping_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "journal.jsonl"
    attempt_path = tmp_path / "attempt.json"
    journal_path.touch()
    attempt_path.write_text("attempt\n", encoding="utf-8")
    monkeypatch.setattr(runner, "JOURNAL_PATH", journal_path)
    monkeypatch.setattr(runner, "ATTEMPT_PATH", attempt_path)

    completed, started = runner.read_journal(_manifest(), recover_partial=True)

    assert not completed
    assert not started
    assert "recovered_partial_tail" in journal_path.read_text(encoding="utf-8")


def test_process_lock_rejects_a_second_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_path = tmp_path / "attempt.json"
    attempt_path.write_text("attempt\n", encoding="utf-8")
    monkeypatch.setattr(runner, "ATTEMPT_PATH", attempt_path)

    first_runner = attempt_path.open("rb")
    try:
        fcntl.flock(first_runner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (
            pytest.raises(runner.OutcomeEvaluationError, match="already active"),
            runner.exclusive_runner_lock(),
        ):
            pytest.fail("second runner unexpectedly acquired the lock")
    finally:
        first_runner.close()


def test_paid_runner_source_never_names_the_answer_file() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")

    assert "nba_development_" + "answers.jsonl" not in source
