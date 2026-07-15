"""Tests for realized-winner labels and binary probability conversion."""

from dataclasses import replace
from math import log
from pathlib import Path

import pytest

from forecastfm.nba_data import side_swap_nba_example
from forecastfm.outcome import (
    OPPONENT_LABEL,
    OUTCOME_SYSTEM_PROMPT,
    TEAM_LABEL,
    OutcomeForecastError,
    build_outcome_messages,
    label_for_example,
    require_label_token_ids,
    symmetric_team_probability,
    team_probability_from_logprobs,
)
from forecastfm.tinker_data import (
    build_outcome_training_record,
    read_outcome_training_jsonl,
    write_outcome_training_jsonl,
)
from tests.helpers import make_nba_training_example


class FakeTokenCodec:
    """Encode fixed labels without importing a vendor tokenizer."""

    def __init__(self, tokens: dict[str, list[int]]) -> None:
        self.tokens = tokens

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return self.tokens[text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert not skip_special_tokens
        return next(label for label, tokens in self.tokens.items() if tokens == token_ids)


def test_realized_winner_controls_label_instead_of_elo_teacher() -> None:
    example = make_nba_training_example("team_wins")

    assert example.target.distribution.predicted_outcome() == "opponent_wins"
    assert label_for_example(example) == TEAM_LABEL
    assert label_for_example(replace(example, realized_outcome="opponent_wins")) == OPPONENT_LABEL


def test_outcome_messages_are_target_free() -> None:
    messages = build_outcome_messages(make_nba_training_example().case)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == OUTCOME_SYSTEM_PROMPT
    assert "realized_outcome" not in messages[1]["content"]
    assert "0.2" not in messages[1]["content"]


def test_outcome_label_requires_a_realized_winner() -> None:
    with pytest.raises(OutcomeForecastError, match="realized"):
        label_for_example(make_nba_training_example(None))


def test_outcome_training_record_contains_label_but_no_answer_message() -> None:
    record = build_outcome_training_record(make_nba_training_example("team_wins"))

    assert record["label"] == TEAM_LABEL
    assert [message["role"] for message in record["messages"]] == ["system", "user"]
    assert "0.2" not in str(record["messages"])


def test_outcome_training_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.jsonl"
    examples = (
        make_nba_training_example("team_wins"),
        make_nba_training_example("opponent_wins"),
    )

    write_outcome_training_jsonl(examples, path)

    records = read_outcome_training_jsonl(path)
    assert [record["label"] for record in records] == [TEAM_LABEL, OPPONENT_LABEL]


def test_labels_must_be_distinct_exact_single_tokens() -> None:
    tokenizer = FakeTokenCodec({TEAM_LABEL: [7], OPPONENT_LABEL: [8]})

    assert require_label_token_ids(tokenizer) == (7, 8)

    with pytest.raises(OutcomeForecastError, match="not one token"):
        require_label_token_ids(FakeTokenCodec({TEAM_LABEL: [7], OPPONENT_LABEL: [8, 9]}))


def test_label_logprobs_produce_stable_probabilities() -> None:
    assert team_probability_from_logprobs(0.0, 0.0) == pytest.approx(0.5)
    assert team_probability_from_logprobs(log(4.0), 0.0) == pytest.approx(0.8)
    assert team_probability_from_logprobs(-10_000.0, -10_000.0) == pytest.approx(0.5)


def test_swapping_label_scores_complements_probability() -> None:
    original = team_probability_from_logprobs(-0.3, -1.2)
    swapped = team_probability_from_logprobs(-1.2, -0.3)

    assert original == pytest.approx(1.0 - swapped)


def test_side_swap_is_an_involution() -> None:
    original = make_nba_training_example("team_wins")
    swapped = side_swap_nba_example(original)

    assert swapped.case.prior.probabilities == (0.6, 0.4)
    assert swapped.case.evidence[0].text == "Venue for the listed team: away."
    assert swapped.target.distribution.probabilities == (0.8, 0.2)
    assert swapped.realized_outcome == "opponent_wins"
    assert side_swap_nba_example(swapped) == original


def test_symmetric_probability_averages_the_complemented_swap() -> None:
    assert symmetric_team_probability(0.7, 0.4) == pytest.approx(0.65)
