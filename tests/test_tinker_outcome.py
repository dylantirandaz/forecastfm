"""Offline tests for label-only Tinker training and inference."""

import asyncio
import json
from dataclasses import replace
from math import log
from pathlib import Path
from typing import cast

import pytest
import tinker
from examples import train_tinker_outcome_sft
from examples.tinker_outcome_inference import (
    score_outcome_case,
    score_symmetric_outcome_case,
    score_symmetric_outcome_messages,
)
from examples.train_tinker_outcome_sft import OutcomeDataset
from tinker_cookbook import renderers

from forecastfm.integrity import file_sha256
from forecastfm.models import TrainingExample
from forecastfm.nba_data import side_swap_nba_example
from forecastfm.outcome import OPPONENT_LABEL, OUTCOME_INPUT_SCHEMA_VERSION, TEAM_LABEL
from forecastfm.run_lock import RunLockError
from forecastfm.tinker_data import (
    build_outcome_forecast_record,
    build_outcome_training_record,
    write_outcome_training_jsonl,
)
from tests.helpers import make_nba_training_example


class FakeRenderer:
    """Render every target-free conversation to one fixed token prefix."""

    def build_generation_prompt(
        self,
        messages: list[renderers.Message],
        role: str = "assistant",
        prefill: str | None = None,
    ) -> tinker.ModelInput:
        assert [message["role"] for message in messages] == ["system", "user"]
        assert role == "assistant"
        assert prefill is None
        return tinker.ModelInput.from_ints([1, 2, 3])


class FakeLogprobClient:
    """Return configured next-token scores while retaining exact requests."""

    def __init__(self, values: dict[int, float | None]) -> None:
        self.values = values
        self.calls: list[tuple[int, ...]] = []

    async def compute_logprobs_async(
        self,
        prompt: tinker.ModelInput,
    ) -> list[float | None]:
        tokens = tuple(prompt.to_ints())
        self.calls.append(tokens)
        return [None] * (len(tokens) - 1) + [self.values[tokens[-1]]]


class FakeTokenCodec:
    """Provide the two exact label tokens required by the preflight."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return {TEAM_LABEL: [10], OPPONENT_LABEL: [20]}[text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        return {10: TEAM_LABEL, 20: OPPONENT_LABEL}[token_ids[0]]


def _renderer() -> renderers.Renderer:
    return cast(renderers.Renderer, FakeRenderer())


def _accept_outcome_lock(_root: Path, _path: Path) -> dict[str, object]:
    return {}


def _fake_get_tokenizer(_path: str) -> FakeTokenCodec:
    return FakeTokenCodec()


def _fake_get_renderer(
    _renderer_name: str,
    _tokenizer: object,
    *,
    model_name: str | None = None,
) -> renderers.Renderer:
    assert model_name is not None
    return _renderer()


def _configure_prerequisites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_path = tmp_path / "nba_train_outcome.jsonl"
    pairs: list[TrainingExample] = []
    for index in range(train_tinker_outcome_sft.BATCH_SIZE // 2):
        template = make_nba_training_example()
        original = replace(
            template,
            case=replace(
                template.case,
                question=replace(template.case.question, question_id=f"nba-example-{index}"),
            ),
        )
        pairs.extend((original, side_swap_nba_example(original)))
    write_outcome_training_jsonl(
        pairs,
        data_path,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "outcome_input_schema_version": OUTCOME_INPUT_SCHEMA_VERSION,
        "outputs": {data_path.name: file_sha256(data_path)},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(train_tinker_outcome_sft, "DATA_PATH", data_path)
    monkeypatch.setattr(train_tinker_outcome_sft, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(train_tinker_outcome_sft, "LOG_PATH", tmp_path / "new-log")
    monkeypatch.setattr(
        train_tinker_outcome_sft,
        "verify_outcome_training_lock",
        _accept_outcome_lock,
    )
    monkeypatch.setattr(
        train_tinker_outcome_sft,
        "require_tokenizer_snapshot",
        lambda: Path("/pinned/tokenizer"),
    )
    monkeypatch.setattr(train_tinker_outcome_sft, "get_tokenizer", _fake_get_tokenizer)
    monkeypatch.setattr(train_tinker_outcome_sft.renderers, "get_renderer", _fake_get_renderer)
    monkeypatch.setenv("TINKER_API_KEY", "test-key")


def test_outcome_datum_trains_only_the_realized_winner_token() -> None:
    original = make_nba_training_example("team_wins")
    swapped = side_swap_nba_example(original)
    records = (
        build_outcome_training_record(original),
        build_outcome_training_record(swapped),
    )
    dataset = OutcomeDataset(records, _renderer(), (10, 20), batch_size=2, max_length=32)

    team_datum, opponent_datum = dataset.get_batch(0)

    assert team_datum.model_input.to_ints() == [1, 2, 3]
    assert team_datum.loss_fn_inputs["target_tokens"].data == [2, 3, 10]
    assert team_datum.loss_fn_inputs["weights"].data == [0.0, 0.0, 1.0]
    assert opponent_datum.loss_fn_inputs["target_tokens"].data == [2, 3, 20]


def test_outcome_runner_accepts_verified_local_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_prerequisites(tmp_path, monkeypatch)

    train_tinker_outcome_sft.require_prerequisites()


def test_outcome_runner_rejects_an_unsealed_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_prerequisites(tmp_path, monkeypatch)

    def reject_lock(_root: Path, _path: Path) -> dict[str, object]:
        raise RunLockError("outcome lock differs")

    monkeypatch.setattr(train_tinker_outcome_sft, "verify_outcome_training_lock", reject_lock)

    with pytest.raises(RunLockError, match="differs"):
        train_tinker_outcome_sft.require_prerequisites()


def test_outcome_runner_refuses_to_resume_an_existing_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_prerequisites(tmp_path, monkeypatch)
    log_path = tmp_path / "existing-log"
    log_path.mkdir()
    monkeypatch.setattr(train_tinker_outcome_sft, "LOG_PATH", log_path)

    with pytest.raises(FileExistsError, match="already exists"):
        train_tinker_outcome_sft.require_prerequisites()


def test_candidate_inference_makes_exactly_two_calls() -> None:
    client = FakeLogprobClient({10: log(0.4), 20: log(0.1)})

    result = asyncio.run(
        score_outcome_case(
            client,
            _renderer(),
            make_nba_training_example().case,
            (10, 20),
        )
    )

    assert result.prediction.distribution.probability_for("team_wins") == pytest.approx(0.8)
    assert result.valid_label_mass == pytest.approx(0.5)
    assert len(client.calls) == 2
    assert {tokens[-1] for tokens in client.calls} == {10, 20}
    assert {tokens[:-1] for tokens in client.calls} == {(1, 2, 3)}


def test_candidate_inference_rejects_a_missing_logprob() -> None:
    client = FakeLogprobClient({10: None, 20: log(0.1)})

    with pytest.raises(RuntimeError, match="missing or non-finite"):
        asyncio.run(
            score_outcome_case(
                client,
                _renderer(),
                make_nba_training_example().case,
                (10, 20),
            )
        )


def test_symmetric_inference_averages_original_and_swapped_orientation() -> None:
    client = FakeLogprobClient({10: log(0.4), 20: log(0.1)})

    result = asyncio.run(
        score_symmetric_outcome_case(
            client,
            _renderer(),
            make_nba_training_example().case,
            (10, 20),
        )
    )

    assert result.original.prediction.distribution.probability_for("team_wins") == pytest.approx(
        0.8
    )
    assert result.swapped.prediction.distribution.probability_for("team_wins") == pytest.approx(0.8)
    assert result.prediction.distribution.probability_for("team_wins") == pytest.approx(0.5)
    assert len(client.calls) == 4


def test_symmetric_inference_accepts_frozen_target_free_messages() -> None:
    client = FakeLogprobClient({10: log(0.4), 20: log(0.1)})
    original = make_nba_training_example().case
    swapped = side_swap_nba_example(make_nba_training_example()).case
    original_record = build_outcome_forecast_record(original)
    swapped_record = build_outcome_forecast_record(swapped)

    result = asyncio.run(
        score_symmetric_outcome_messages(
            client,
            _renderer(),
            original_record["messages"],
            swapped_record["messages"],
            (10, 20),
        )
    )

    assert result.prediction.distribution.probability_for("team_wins") == pytest.approx(0.5)
    assert len(client.calls) == 4
