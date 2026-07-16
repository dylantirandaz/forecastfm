"""Tests for the real-data outcome-v2 build boundary."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import exp, log, sqrt

import pytest
from examples import build_outcome_v2_dataset

from forecastfm.json_utils import (
    parse_json_object,
    require_list,
    require_object,
    required_field,
)
from forecastfm.models import Distribution
from forecastfm.nba_data import SOURCE_SHA256
from forecastfm.nba_v2 import NbaV2Example, NbaV2Features, side_swap_nba_v2_example
from forecastfm.outcome import OUTCOME_INPUT_SCHEMA_VERSION
from tests.helpers import make_nba_training_example


def _example(
    question_id: str,
    season: int,
    vector: tuple[float, ...],
) -> NbaV2Example:
    team_probability = 1.0 / (1.0 + exp(-vector[0]))
    probabilities = (team_probability, 1.0 - team_probability)
    training = make_nba_training_example()
    forecast_at = datetime(season - 1, 10, 1, tzinfo=UTC)
    question = replace(
        training.case.question,
        question_id=question_id,
        forecast_at=forecast_at,
        resolves_at=forecast_at + timedelta(days=2),
    )
    case = replace(
        training.case,
        question=question,
        prior=Distribution(outcomes=question.outcomes, probabilities=probabilities),
        prior_as_of=forecast_at,
        evidence=(),
    )
    training = replace(
        training,
        case=case,
        target_information_cutoff=forecast_at,
    )
    features = NbaV2Features(
        venue_adjusted_elo_probabilities=probabilities,
        venue_adjusted_elo_log_odds=vector[0],
        rest_days_difference=vector[1],
        back_to_back_difference=vector[2],
        games_last_7_difference=vector[3],
        road_games_last_7_difference=vector[4],
        trailing_10_win_rate_difference=vector[5],
        trailing_10_margin_difference=vector[6],
        trailing_10_opponent_elo_difference=vector[7],
        trailing_10_history_difference=vector[8],
    )
    return NbaV2Example(training_example=training, features=features, season=season)


def test_training_only_rms_scaling_preserves_side_swap_antisymmetry() -> None:
    first = _example("first", 2009, (0.5, 1.0, 2.0, 3.0, 4.0, 0.2, 6.0, 70.0, 8.0))
    second = _example(
        "second",
        2009,
        (-0.5, -3.0, -4.0, -5.0, -6.0, -0.4, -8.0, -90.0, -10.0),
    )

    scales = build_outcome_v2_dataset.rms_feature_scales((first, second))
    original = build_outcome_v2_dataset.scale_features(first, scales)
    swapped = build_outcome_v2_dataset.scale_features(
        side_swap_nba_v2_example(first),
        scales,
    )

    assert scales[0] == pytest.approx(0.5)
    assert scales[1] == pytest.approx(sqrt(5.0))
    assert swapped == pytest.approx(tuple(-value for value in original))


def test_split_uses_source_season_metadata() -> None:
    vector = (log(0.4 / 0.6), 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0)
    examples = tuple(
        _example(f"season-{season}", season, vector)
        for season in (2009, 2010, 2011, 2012, 2013, 2014, 2015)
    )

    splits = build_outcome_v2_dataset.split_examples(examples)

    assert tuple(example.season for example in splits.train) == (2009,)
    assert tuple(example.season for example in splits.validation) == (2010, 2011, 2012)
    assert tuple(example.season for example in splits.historical_test) == (2013, 2014, 2015)


def test_prompt_orbit_deduplication_keeps_only_the_first_chronological_row() -> None:
    vector = (log(0.4 / 0.6), 1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0)
    first = _example("first", 2008, vector)
    duplicate = _example("duplicate", 2009, vector)

    kept, removed = build_outcome_v2_dataset.deduplicate_prompt_orbits((first, duplicate))

    assert kept == (first,)
    assert removed == {2009: 1}


def test_checked_in_manifest_preserves_the_failed_rl_gate() -> None:
    manifest = parse_json_object(build_outcome_v2_dataset.MANIFEST_PATH.read_text(encoding="utf-8"))
    source = require_object(required_field(manifest, "source"), "source")
    evaluation = require_object(required_field(manifest, "evaluation"), "evaluation")
    historical = require_object(
        required_field(evaluation, "historical_test"),
        "historical_test",
    )
    splits = require_object(required_field(manifest, "splits"), "splits")
    historical_split = require_object(
        required_field(splits, "historical_test"),
        "historical_test split",
    )
    rich = require_object(required_field(historical, "rich_vs_raw_elo"), "rich_vs_raw_elo")
    seasons = require_list(required_field(rich, "seasons"), "seasons")
    season_records = tuple(require_object(value, "season") for value in seasons)
    rl = require_object(required_field(manifest, "rl"), "rl")

    assert required_field(source, "sha256") == SOURCE_SHA256
    assert required_field(manifest, "outcome_input_schema_version") == (
        OUTCOME_INPUT_SCHEMA_VERSION
    )
    assert required_field(evaluation, "historical_gate_passes_raw_elo") is False
    assert required_field(evaluation, "historical_gate_passes_recalibrated_elo") is False
    assert tuple(required_field(record, "season") for record in season_records) == (
        2013,
        2014,
        2015,
    )
    assert tuple(required_field(record, "passes") for record in season_records) == (
        False,
        True,
        False,
    )
    assert required_field(historical_split, "question_ids_sha256") == (
        "8a0199f95e6fd5be3f59787329e6c2128e50905ef3f16ccfeaee0b2ce06097b6"
    )
    assert (
        required_field(historical_split, "full_cohort_sha256")
        == (build_outcome_v2_dataset.EXPECTED_HISTORICAL_TEST_COHORT[1])
    )
    assert required_field(evaluation, "valid_probability_contract") == (
        "strictly between zero and one"
    )
    assert required_field(evaluation, "failed_forecast_realized_probability") == 1e-15
    assert required_field(rl, "ready") is False
    assert required_field(rl, "paid_tinker_job_launched") is False
