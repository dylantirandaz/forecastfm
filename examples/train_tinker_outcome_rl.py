"""Paid Mantic-style RL run for NBA outcome forecasting under the frozen protocol.

Implements prospective/RL_RUN_PROTOCOL.md: policy-gradient training on realized outcomes with
a Brier reward computed from TEAM/OTHER candidate logprobabilities, GRPO-style group-centered
advantages without standard-deviation division, and the built-in importance-sampling loss.
Decision 2a = A (no health-derived values). The run writes a create-only run lock before any
client exists, renders and length-checks every prompt before client creation, journals every
step to an append-only log, checkpoints at 25/50/75/100 for the scaling plot, and enforces a
hard call cap. Nothing here authorizes spend by itself: the protocol requires the owner's GO,
which is recorded in the run lock.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import tinker
import torch
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

from forecastfm.integrity import bytes_sha256, canonical_json, canonical_sha256
from forecastfm.json_utils import (
    parse_json_object,
    require_float,
    require_string,
    required_field,
)
from forecastfm.local_config import read_tinker_api_key

DATASET_DIR = Path("data/processed/rl_dataset_v2")
PROTOCOL_PATH = Path("prospective/RL_RUN_PROTOCOL.md")
ARTIFACTS_DIR = Path("artifacts/tinker")
TEAM_TOKEN = 197467
OTHER_TOKEN = 60669
RENDERER_NAME = "gpt_oss_low_reasoning"
BASE_MODEL = "openai/gpt-oss-120b"
TEMPLATE_VERSION = "rl-prompt-v2"
FINAL_CHANNEL_MARKER = "<|channel|>final<|message|>"
PROBABILITY_PATTERN = re.compile(r"(0(?:\.\d+)?|1(?:\.0+)?)")


class NbaRlRunError(RuntimeError):
    """Raised when the RL run violates its frozen contract."""


@dataclass(frozen=True, slots=True)
class RlRunConfig:
    """Frozen numerical choices for one RL run."""

    rank: int = 16
    learning_rate: float = 1e-5
    steps: int = 100
    batch_size: int = 64
    group_size: int = 8
    temperature: float = 1.0
    max_tokens: int = 192
    seed: int = 20260720
    max_calls: int = 8_000
    concurrency: int = 32
    max_prompt_tokens: int = 1_024
    max_steps: int | None = None

    def estimated_calls(self) -> int:
        """Return the estimated sampling calls for the configured run."""
        steps = self.max_steps or self.steps
        return steps * self.batch_size * (1 + self.group_size)

    def canonical_payload(self) -> dict[str, object]:
        """Return the exact config covered by the run-lock hash."""
        return {
            "rank": self.rank,
            "learning_rate": self.learning_rate,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "group_size": self.group_size,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "max_calls": self.max_calls,
            "concurrency": self.concurrency,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_steps": self.max_steps,
            "renderer": RENDERER_NAME,
            "base_model": BASE_MODEL,
            "template_version": TEMPLATE_VERSION,
        }


@dataclass(frozen=True, slots=True)
class RlQuestion:
    """One sealed question with its answer and rendered orientations."""

    question_id: str
    system: str
    user: str
    winner: str
    season: int


@dataclass(slots=True)
class _Rollout:
    completion_tokens: list[int]
    completion_logprobs: list[float]
    stated_probability: float | None
    reward: float
    advantage: float = 0.0


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one frozen RL run after writing its immutable run lock."""
    args = _parse_arguments(argv)
    config = RlRunConfig(max_steps=args.max_steps)
    questions = _load_questions()
    if config.estimated_calls() > config.max_calls:
        raise NbaRlRunError(
            f"estimated {config.estimated_calls()} calls exceed the {config.max_calls} cap"
        )
    rendered = _render_all(questions, config)
    artifact_dir = _write_run_lock(config, len(questions))
    journal = artifact_dir / "journal.jsonl"
    context = _RunContext(
        questions=questions,
        rendered=rendered,
        artifact_dir=artifact_dir,
        journal=journal,
        tokenizer=get_tokenizer(BASE_MODEL),
    )
    result = asyncio.run(_run(config, context))
    _write_experiment_seal(config, artifact_dir, result)
    print(json.dumps(result, indent=2))
    return 0


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=None, help="bounded smoke run length")
    return parser.parse_args(argv)


def _load_questions() -> list[RlQuestion]:
    prompts_path = DATASET_DIR / "prompts.jsonl"
    answers_path = DATASET_DIR / "answers.jsonl"
    answers: dict[str, str] = {}
    for line in answers_path.read_text(encoding="utf-8").splitlines():
        payload = parse_json_object(line)
        answers[require_string(required_field(payload, "question_id"), "question_id")] = (
            require_string(required_field(payload, "winner"), "winner")
        )
    questions: list[RlQuestion] = []
    for line in prompts_path.read_text(encoding="utf-8").splitlines():
        payload = parse_json_object(line)
        question_id = require_string(required_field(payload, "question_id"), "question_id")
        winner = answers.get(question_id)
        if winner is None:
            raise NbaRlRunError(f"prompt without sealed answer: {question_id}")
        if require_string(required_field(payload, "orientation"), "orientation") != "original":
            continue
        questions.append(
            RlQuestion(
                question_id=question_id,
                system=require_string(required_field(payload, "system"), "system"),
                user=require_string(required_field(payload, "user"), "user"),
                winner=winner,
                season=int(require_float(required_field(payload, "season"), "season")),
            )
        )
    if not questions:
        raise NbaRlRunError("sealed dataset is empty")
    return questions


def _render_all(questions: list[RlQuestion], config: RlRunConfig) -> dict[str, tinker.ModelInput]:
    tokenizer = get_tokenizer(BASE_MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer)
    rendered: dict[str, tinker.ModelInput] = {}
    for question in questions:
        messages = [
            renderers.Message(role="system", content=question.system),
            renderers.Message(role="user", content=question.user),
        ]
        model_input = renderer.build_generation_prompt(messages)
        length = len(model_input.to_ints())
        if length > config.max_prompt_tokens:
            raise NbaRlRunError(
                f"prompt {question.question_id} renders to {length} tokens "
                f"(cap {config.max_prompt_tokens})"
            )
        rendered[question.question_id] = model_input
    return rendered


def _write_run_lock(config: RlRunConfig, question_count: int) -> Path:
    artifact_dir = ARTIFACTS_DIR / f"outcome_rl_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    if artifact_dir.exists():
        raise NbaRlRunError("artifact directory already exists")
    artifact_dir.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "kind": "forecastfm_nba_rl_run_lock",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": bytes_sha256(PROTOCOL_PATH.read_bytes()),
        "dataset_manifest_sha256": bytes_sha256((DATASET_DIR / "manifest.json").read_bytes()),
        "config": config.canonical_payload(),
        "question_count": question_count,
        "owner_go": "recorded 2026-07-20 (2a=A, 2b acknowledged)",
    }
    lock_bytes = (canonical_json(payload) + "\n").encode("utf-8")
    (artifact_dir / "run_lock.json").write_bytes(lock_bytes)
    return artifact_dir


@dataclass(slots=True)
class _RunContext:
    """Shared immutable inputs for one RL run."""

    questions: list[RlQuestion]
    rendered: dict[str, tinker.ModelInput]
    artifact_dir: Path
    journal: Path
    tokenizer: object


async def _run(
    config: RlRunConfig,
    context: _RunContext,
) -> dict[str, object]:
    api_key = read_tinker_api_key(Path(".env"))
    os.environ["TINKER_API_KEY"] = api_key
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=BASE_MODEL, rank=config.rank, seed=config.seed
    )
    steps = config.max_steps or config.steps
    call_count = 0
    checkpoints: list[str] = []
    step_metrics: list[dict[str, object]] = []
    for step in range(1, steps + 1):
        sampler_future = await training_client.save_weights_for_sampler_async(
            name=f"rl-sampler-{step}"
        )
        sampler_weights = await sampler_future.result_async()
        sampler = service_client.create_sampling_client(model_path=sampler_weights.path)
        if step in {25, 50, 75, 100}:
            checkpoints.append(sampler_weights.path)
        batch = _batch_for_step(context.questions, step, config)
        rollouts, calls = await _rollout_batch(
            batch, context.rendered, sampler, config, context.tokenizer
        )
        call_count += calls
        if call_count > config.max_calls:
            _journal(
                context.journal,
                {"step": step, "aborted": "call cap exceeded", "calls": call_count},
            )
            raise NbaRlRunError(f"call cap exceeded at step {step}: {call_count}")
        _assign_advantages(rollouts)
        datums = [
            _datum(rollout, context.rendered[rollout_question])
            for rollout_question, rollout in rollouts
            for rollout in rollout
        ]
        fb_future = await training_client.forward_backward_async(datums, "importance_sampling")
        await fb_future.result_async()
        optim_future = await training_client.optim_step_async(
            tinker.AdamParams(learning_rate=config.learning_rate)
        )
        await optim_future.result_async()
        metrics = _step_metrics(step, rollouts, call_count)
        _journal(context.journal, metrics)
        step_metrics.append(metrics)
    final_future = await training_client.save_weights_for_sampler_async("rl-final")
    final_saved = await final_future.result_async()
    checkpoints.append(final_saved.path)
    return {
        "steps": steps,
        "calls": call_count,
        "checkpoints": checkpoints,
        "final_sampler_path": checkpoints[-1],
        "metrics": step_metrics,
    }


def _batch_for_step(
    questions: list[RlQuestion], step: int, config: RlRunConfig
) -> list[RlQuestion]:
    start = ((step - 1) * config.batch_size) % len(questions)
    rotated = questions[start:] + questions[:start]
    return rotated[: config.batch_size]


async def _rollout_batch(
    batch: list[RlQuestion],
    rendered: dict[str, tinker.ModelInput],
    sampler: tinker.SamplingClient,
    config: RlRunConfig,
    tokenizer: object,
) -> tuple[list[tuple[str, list[_Rollout]]], int]:
    semaphore = asyncio.Semaphore(config.concurrency)

    async def one(question: RlQuestion) -> tuple[str, list[_Rollout]]:
        async with semaphore:
            response = await sampler.sample_async(
                rendered[question.question_id],
                num_samples=config.group_size,
                sampling_params=tinker.SamplingParams(
                    max_tokens=config.max_tokens,
                    seed=config.seed,
                    temperature=config.temperature,
                ),
            )
        rollouts = await _rollouts_from_response(question, response, tokenizer)
        return question.question_id, rollouts

    results = await asyncio.gather(*(one(question) for question in batch))
    return list(results), len(batch)


async def _rollouts_from_response(
    question: RlQuestion,
    response: tinker.SampleResponse,
    tokenizer: object,
) -> list[_Rollout]:
    target = 1.0 if question.winner == "TEAM" else 0.0
    rollouts: list[_Rollout] = []
    for sequence in response.sequences:
        tokens = list(sequence.tokens)
        logprobs = list(sequence.logprobs or [])
        if len(logprobs) != len(tokens):
            raise NbaRlRunError("sampled sequence is missing logprobs")
        stated = _parse_stated_probability(_decode(tokenizer, tokens))
        reward = 0.0 if stated is None else 1.0 - (stated - target) ** 2
        rollouts.append(
            _Rollout(
                completion_tokens=tokens,
                completion_logprobs=logprobs,
                stated_probability=stated,
                reward=reward,
            )
        )
    return rollouts


def _parse_stated_probability(text: str) -> float | None:
    """Extract the stated win probability from the completion's final channel."""
    tail = text.rsplit(FINAL_CHANNEL_MARKER, 1)[-1]
    match = PROBABILITY_PATTERN.search(tail)
    if match is None:
        return None
    value = float(match.group(1))
    return value if 0.0 <= value <= 1.0 else None


def _decode(tokenizer: object, tokens: list[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise NbaRlRunError("tokenizer cannot decode completions")
    return str(decode(tokens))


def _assign_advantages(groups: list[tuple[str, list[_Rollout]]]) -> None:
    for _question_id, rollouts in groups:
        mean_reward = sum(rollout.reward for rollout in rollouts) / len(rollouts)
        for rollout in rollouts:
            rollout.advantage = rollout.reward - mean_reward


def _datum(rollout: _Rollout, model_input: tinker.ModelInput) -> tinker.Datum:
    prompt_tokens = model_input.to_ints()
    completion = rollout.completion_tokens
    full = prompt_tokens + completion
    target_len = len(full) - 1
    sampled_logprobs = [0.0] * (len(prompt_tokens) - 1) + rollout.completion_logprobs
    advantages = [0.0] * (len(prompt_tokens) - 1) + [rollout.advantage] * len(completion)
    mask = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion)
    if not (len(sampled_logprobs) == len(advantages) == len(mask) == target_len):
        raise NbaRlRunError("datum length mismatch")
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(full[:-1]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_torch(torch.tensor(full[1:])),
            "logprobs": tinker.TensorData.from_torch(torch.tensor(sampled_logprobs)),
            "advantages": tinker.TensorData.from_torch(torch.tensor(advantages)),
        },
    )


def _step_metrics(
    step: int, groups: list[tuple[str, list[_Rollout]]], call_count: int
) -> dict[str, object]:
    flat = [rollout for _question_id, rollouts in groups for rollout in rollouts]
    mean_reward = sum(rollout.reward for rollout in flat) / len(flat)
    stated = [
        rollout.stated_probability for rollout in flat if rollout.stated_probability is not None
    ]
    mean_p = sum(stated) / len(stated) if stated else 0.0
    variance = sum((value - mean_p) ** 2 for value in stated) / len(stated) if stated else 0.0
    parse_rate = len(stated) / len(flat)
    return {
        "step": step,
        "mean_reward": mean_reward,
        "mean_probability_team": mean_p,
        "stated_probability_variance": variance,
        "parse_rate": parse_rate,
        "calls": call_count,
    }


def _journal(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(record) + "\n")


def _write_experiment_seal(
    config: RlRunConfig, artifact_dir: Path, result: dict[str, object]
) -> None:
    payload = {
        "schema_version": 1,
        "kind": "forecastfm_nba_rl_experiment_seal",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": canonical_sha256(config.canonical_payload()),
        "final_sampler_path": result["final_sampler_path"],
        "steps": result["steps"],
        "calls": result["calls"],
        "checkpoints": result["checkpoints"],
    }
    (artifact_dir / "experiment_seal.json").write_text(
        canonical_json(payload) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
