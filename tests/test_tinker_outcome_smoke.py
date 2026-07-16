"""Offline tests for the bounded live candidate smoke inputs."""

from dataclasses import replace
from pathlib import Path

from examples.smoke_tinker_outcome_candidates import load_smoke_pair

from forecastfm.nba_data import side_swap_nba_example
from forecastfm.tinker_data import write_outcome_training_jsonl
from tests.helpers import make_nba_training_example


def test_smoke_pair_uses_training_messages_without_labels(tmp_path: Path) -> None:
    template = make_nba_training_example()
    original = replace(
        template,
        case=replace(
            template.case,
            question=replace(template.case.question, question_id="nba-smoke-0"),
        ),
    )
    path = tmp_path / "training.jsonl"
    write_outcome_training_jsonl((original, side_swap_nba_example(original)), path)

    pair = load_smoke_pair(path)

    assert pair[0]["question_id"] == "nba-smoke-0"
    assert pair[1]["question_id"] == "nba-smoke-0-side-swap"
    assert set(pair[0]) == {"question_id", "messages"}
    assert [message["role"] for message in pair[0]["messages"]] == ["system", "user"]
