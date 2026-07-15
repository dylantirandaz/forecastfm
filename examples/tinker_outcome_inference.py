"""Deterministic Tinker inference for the two outcome-label tokens."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite
from typing import Protocol

import tinker
from tinker_cookbook import renderers

from forecastfm.models import Distribution, ForecastCase, ForecastPrediction
from forecastfm.nba_data import side_swap_nba_case
from forecastfm.outcome import (
    build_outcome_messages,
    prediction_from_logprobs,
    symmetric_team_probability,
)
from forecastfm.prompting import ChatMessage


class CandidateLogprobClient(Protocol):
    """Tinker operation needed for deterministic candidate scoring."""

    async def compute_logprobs_async(
        self,
        prompt: tinker.ModelInput,
    ) -> list[float | None]:
        """Return prompt-token log-probabilities."""
        ...


@dataclass(frozen=True, slots=True)
class OutcomeLogprobForecast:
    """One binary forecast plus its unnormalized label diagnostics."""

    prediction: ForecastPrediction
    team_logprob: float
    opponent_logprob: float
    valid_label_mass: float
    prompt_tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SymmetricOutcomeForecast:
    """Original, swapped, and symmetry-averaged forecasts for one game."""

    prediction: ForecastPrediction
    original: OutcomeLogprobForecast
    swapped: OutcomeLogprobForecast


async def score_outcome_case(
    client: CandidateLogprobClient,
    renderer: renderers.Renderer,
    case: ForecastCase,
    label_token_ids: tuple[int, int],
) -> OutcomeLogprobForecast:
    """Score both fixed labels without sampling generated text."""
    return await score_outcome_messages(
        client,
        renderer,
        build_outcome_messages(case),
        label_token_ids,
    )


async def score_outcome_messages(
    client: CandidateLogprobClient,
    renderer: renderers.Renderer,
    messages: Sequence[ChatMessage],
    label_token_ids: tuple[int, int],
) -> OutcomeLogprobForecast:
    """Score one already-frozen, target-free orientation."""
    prompt = renderer.build_generation_prompt(_renderer_messages(messages))
    team_token, opponent_token = label_token_ids
    team_logprob, opponent_logprob = await asyncio.gather(
        _label_logprob(client, prompt, team_token),
        _label_logprob(client, prompt, opponent_token),
    )
    valid_label_mass = exp(team_logprob) + exp(opponent_logprob)
    if valid_label_mass > 1.000001:
        raise RuntimeError("candidate label probability mass exceeds one")
    return OutcomeLogprobForecast(
        prediction=prediction_from_logprobs(team_logprob, opponent_logprob),
        team_logprob=team_logprob,
        opponent_logprob=opponent_logprob,
        valid_label_mass=valid_label_mass,
        prompt_tokens=tuple(prompt.to_ints()),
    )


async def score_symmetric_outcome_case(
    client: CandidateLogprobClient,
    renderer: renderers.Renderer,
    case: ForecastCase,
    label_token_ids: tuple[int, int],
) -> SymmetricOutcomeForecast:
    """Score both orientations and average complementary probabilities."""
    return await score_symmetric_outcome_messages(
        client,
        renderer,
        build_outcome_messages(case),
        build_outcome_messages(side_swap_nba_case(case)),
        label_token_ids,
    )


async def score_symmetric_outcome_messages(
    client: CandidateLogprobClient,
    renderer: renderers.Renderer,
    original_messages: Sequence[ChatMessage],
    swapped_messages: Sequence[ChatMessage],
    label_token_ids: tuple[int, int],
) -> SymmetricOutcomeForecast:
    """Score one frozen original/swap prompt pair exactly once."""
    original, swapped = await asyncio.gather(
        score_outcome_messages(client, renderer, original_messages, label_token_ids),
        score_outcome_messages(client, renderer, swapped_messages, label_token_ids),
    )
    original_team = original.prediction.distribution.probability_for("team_wins")
    swapped_team = swapped.prediction.distribution.probability_for("team_wins")
    team_probability = symmetric_team_probability(original_team, swapped_team)
    prediction = ForecastPrediction(
        distribution=Distribution(
            outcomes=original.prediction.distribution.outcomes,
            probabilities=(team_probability, 1.0 - team_probability),
        )
    )
    return SymmetricOutcomeForecast(
        prediction=prediction,
        original=original,
        swapped=swapped,
    )


async def _label_logprob(
    client: CandidateLogprobClient,
    prompt: tinker.ModelInput,
    token_id: int,
) -> float:
    full_prompt = prompt.append_int(token_id)
    values = await client.compute_logprobs_async(full_prompt)
    if len(values) != full_prompt.length:
        raise RuntimeError("Tinker returned an unexpected log-probability count")
    value = values[-1]
    if value is None or not isfinite(value):
        raise RuntimeError("Tinker returned a missing or non-finite label log-probability")
    if value > 0.000001:
        raise RuntimeError("Tinker returned a positive label log-probability")
    return value


def _renderer_messages(messages: Sequence[ChatMessage]) -> list[renderers.Message]:
    return [
        renderers.Message(role=message["role"], content=message["content"]) for message in messages
    ]
