"""Tests for the answer-free modern NBA evaluation protocol."""

from pathlib import Path

from forecastfm.json_utils import (
    parse_json_object,
    require_list,
    require_object,
    required_field,
)

PROTOCOL_PATH = Path(__file__).parents[1] / "evaluation/outcome_v2_open_modern/protocol.json"
PINNED_COMMIT = "f6d5b2e1d6da2889345d381c41431f9a4ee208dd"
PINNED_RAPTOR_COMMIT = "4c1ff5e3aef1816ae04af63218015066e186c147"


def _integer_list(value: object, field_name: str) -> tuple[int, ...]:
    values = require_list(value, field_name)
    result: list[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{field_name} must contain integers")
        result.append(item)
    return tuple(result)


def test_open_modern_protocol_is_frozen_before_download() -> None:
    protocol = parse_json_object(PROTOCOL_PATH.read_text(encoding="utf-8"))
    source = require_object(required_field(protocol, "source"), "source")
    player_source = require_object(
        required_field(protocol, "lagged_player_source"),
        "lagged_player_source",
    )
    splits = require_object(required_field(protocol, "splits"), "splits")
    evaluation = require_object(required_field(protocol, "evaluation"), "evaluation")

    train = _integer_list(required_field(splits, "train_seasons"), "train_seasons")
    validation = _integer_list(
        required_field(splits, "validation_seasons"),
        "validation_seasons",
    )
    test = _integer_list(
        required_field(splits, "untouched_test_seasons"),
        "untouched_test_seasons",
    )

    assert required_field(protocol, "status") == "answer_free_protocol"
    assert required_field(source, "commit") == PINNED_COMMIT
    assert required_field(source, "downloaded_before_protocol_freeze") is False
    assert required_field(player_source, "commit") == PINNED_RAPTOR_COMMIT
    assert required_field(player_source, "downloaded_before_protocol_freeze") is False
    assert train == (2016, 2017, 2018, 2019)
    assert validation == (2020,)
    assert test == (2021, 2022)
    assert max(train) < min(validation) < min(test)
    all_seasons = (*train, *validation, *test)
    assert len(set(all_seasons)) == len(all_seasons)
    assert required_field(evaluation, "exact_cohort_coverage") is True
    assert required_field(evaluation, "missing_or_malformed_forecast_realized_probability") == 1e-15
