"""Exact multi-season aggregation of externally committed outcome-v2 forecasts."""

from __future__ import annotations

import os
import re
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from forecastfm.github_actions_receipt import GitHubActionsReceipt
from forecastfm.integrity import bytes_sha256, canonical_json, canonical_sha256
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.nba_elo_replay import NbaEloReplayRow
from forecastfm.nba_evaluation_gate import NBA_EVALUATION_FORECAST_SCHEMA_VERSION
from forecastfm.nba_feature_rows import (
    NbaFeatureRowError,
    NbaRichFeatureRow,
    read_nba_feature_rows_jsonl_bytes,
)
from forecastfm.outcome_v2_coverage import (
    OutcomeV2CoverageError,
    OutcomeV2ScheduleCoverageReceiptArtifacts,
    VerifiedOutcomeV2ScheduleCoverage,
    verify_outcome_v2_schedule_coverage_receipt,
)
from forecastfm.outcome_v2_inference import (
    OutcomeV2GenerationLock,
    OutcomeV2InferenceError,
    binary_forecasts_from_inference_records,
    read_outcome_v2_inference_records_jsonl_bytes,
)
from forecastfm.outcome_v2_metrics import BinaryForecast
from forecastfm.outcome_v2_rolling import (
    OutcomeV2ProspectivePlanArtifacts,
    OutcomeV2ProspectiveReceiptArtifacts,
    OutcomeV2RollingError,
    VerifiedOutcomeV2ProspectiveBatch,
    verify_outcome_v2_prospective_batch_receipt,
    verify_outcome_v2_prospective_plan,
)

OUTCOME_V2_ROLLING_AGGREGATE_SCHEMA_VERSION = 1

_KIND = "forecastfm_outcome_v2_prospective_multi_season_forecast_seal"
_STATUS = "complete_answer_free_forecasts_aggregated"
_PROOF_STATUS = "provider_remote_execution_and_canonical_elo_replay_required"
_REQUIRED_SEPARATELY = "required_separately"
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_SEAL_KEYS = {
    "batch_seal_sha256s",
    "canonical_elo_replay",
    "coverage_policy_sha256",
    "coverage_receipt_run_ids",
    "coverage_receipt_sha256s",
    "coverage_seal_sha256s",
    "created_at",
    "failed_record_count",
    "feature_rows_sha256",
    "forecast_count",
    "generation_lock_sha256s",
    "kind",
    "plan_receipt_run_id",
    "plan_receipt_sha256",
    "plan_sha256",
    "prospective_proof_status",
    "provider_authenticity",
    "provider_conformance_report_sha256s",
    "question_ids_sha256",
    "remote_execution_attestation",
    "schema_version",
    "schedule_rows_sha256",
    "seasons",
    "source_game_ids_sha256",
    "status",
    "terminal_receipt_run_ids",
    "terminal_receipt_sha256s",
    "terminal_records_sha256s",
    "forecasts_sha256",
}

type JsonObject = dict[str, object]


class OutcomeV2AggregationError(ValueError):
    """Raised when rolling batches do not exactly cover their frozen schedules."""


@dataclass(frozen=True, slots=True)
class OutcomeV2RollingAggregationArtifacts:
    """One plan, all season coverage receipts, and every terminal forecast batch."""

    plan: OutcomeV2ProspectivePlanArtifacts
    plan_path: Path
    coverages: tuple[OutcomeV2ScheduleCoverageReceiptArtifacts, ...]
    batches: tuple[OutcomeV2ProspectiveReceiptArtifacts, ...]

    def __post_init__(self) -> None:
        if not self.coverages:
            raise OutcomeV2AggregationError("rolling aggregation requires schedule coverage")
        if not self.batches:
            raise OutcomeV2AggregationError("rolling aggregation requires forecast batches")


@dataclass(frozen=True, slots=True)
class OutcomeV2RollingAggregateFiles:
    """Create-only answer-free outputs produced by rolling aggregation."""

    schedule_path: Path
    forecasts_path: Path
    seal_path: Path


@dataclass(frozen=True, slots=True)
class OutcomeV2RollingAggregate:
    """Ordered provider schedule, derived forecasts, and their terminal seal."""

    schedule: tuple[NbaEloReplayRow, ...]
    feature_rows: tuple[NbaRichFeatureRow, ...]
    forecasts: tuple[BinaryForecast, ...]
    canonical_seal_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        record = _seal_record(self.canonical_seal_bytes)
        _validate_aggregate_rows(record, self.schedule, self.forecasts)
        _validate_aggregate_features(record, self.schedule, self.feature_rows)
        _validate_aggregate_bytes(record, self.schedule, self.forecasts)

    @property
    def sha256(self) -> str:
        """Return the digest of the exact aggregate seal bytes."""
        return bytes_sha256(self.canonical_seal_bytes)

    def to_record(self) -> JsonObject:
        """Return a fresh strict record decoded from the aggregate seal."""
        return _seal_record(self.canonical_seal_bytes)


@dataclass(frozen=True, slots=True)
class _BatchData:
    verified: VerifiedOutcomeV2ProspectiveBatch
    rows: tuple[NbaRichFeatureRow, ...]
    forecasts: tuple[BinaryForecast, ...]
    generation_lock_sha256: str
    terminal_records_sha256: str


def _validate_aggregate_rows(
    record: Mapping[str, object],
    schedule: Sequence[NbaEloReplayRow],
    forecasts: Sequence[BinaryForecast],
) -> None:
    if _positive_integer(record, "forecast_count") != len(schedule):
        raise OutcomeV2AggregationError("aggregate count differs from its schedule")
    if len(forecasts) != len(schedule):
        raise OutcomeV2AggregationError("aggregate schedule and forecast counts differ")
    schedule_ids = tuple(row.question_id for row in schedule)
    if schedule_ids != tuple(row.question_id for row in forecasts):
        raise OutcomeV2AggregationError("aggregate schedule and forecast order differs")
    _require_unique(schedule_ids, "aggregate question ID")
    if canonical_sha256(list(schedule_ids)) != _string(record, "question_ids_sha256"):
        raise OutcomeV2AggregationError("aggregate question ID hash is invalid")
    source_ids = [row.source_game_id for row in schedule]
    if canonical_sha256(source_ids) != _string(record, "source_game_ids_sha256"):
        raise OutcomeV2AggregationError("aggregate source game ID hash is invalid")
    seasons = tuple(sorted({row.season for row in schedule}))
    if seasons != _integer_tuple(record, "seasons"):
        raise OutcomeV2AggregationError("aggregate schedule seasons are invalid")
    failed_count = sum(row.team_probability is None for row in forecasts)
    if failed_count != _nonnegative_integer(record, "failed_record_count"):
        raise OutcomeV2AggregationError("aggregate failed forecast count is invalid")


def _validate_aggregate_bytes(
    record: Mapping[str, object],
    schedule: Sequence[NbaEloReplayRow],
    forecasts: Sequence[BinaryForecast],
) -> None:
    if bytes_sha256(_schedule_jsonl_bytes(schedule)) != _string(
        record,
        "schedule_rows_sha256",
    ):
        raise OutcomeV2AggregationError("aggregate schedule hash is invalid")
    if bytes_sha256(_forecast_jsonl_bytes(forecasts)) != _string(record, "forecasts_sha256"):
        raise OutcomeV2AggregationError("aggregate forecast hash is invalid")


def _validate_aggregate_features(
    record: Mapping[str, object],
    schedule: Sequence[NbaEloReplayRow],
    feature_rows: Sequence[NbaRichFeatureRow],
) -> None:
    if len(feature_rows) != len(schedule):
        raise OutcomeV2AggregationError("aggregate feature-row count differs from its schedule")
    if tuple(row.question_id for row in feature_rows) != tuple(row.question_id for row in schedule):
        raise OutcomeV2AggregationError("aggregate feature-row order differs from its schedule")
    expected = canonical_sha256([row.row_sha256 for row in feature_rows])
    if expected != _string(record, "feature_rows_sha256"):
        raise OutcomeV2AggregationError("aggregate feature-row hash is invalid")


def build_outcome_v2_rolling_aggregate(
    artifacts: OutcomeV2RollingAggregationArtifacts,
    created_at: datetime,
    token: str | None = None,
) -> OutcomeV2RollingAggregate:
    """Require exact schedule union and derive every forecast from sealed raw scores."""
    _require_utc(created_at, "created_at")
    plan = verify_outcome_v2_prospective_plan(artifacts.plan, artifacts.plan_path)
    plan_record = plan.to_record()
    planned_seasons = _integer_tuple(plan_record, "seasons")
    coverages = _verify_coverages(artifacts.coverages, token)
    batches = _verify_batches(artifacts.batches, token)
    _require_shared_plan(plan.sha256, coverages, batches)
    ordered_schedule = _ordered_schedule_rows(planned_seasons, coverages)
    ordered_rows, ordered_forecasts = _align_exact_batch_union(ordered_schedule, batches)
    _require_coverage_before_inputs(coverages, ordered_rows)
    _require_aggregate_time(created_at, coverages, batches)

    schedule_bytes = _schedule_jsonl_bytes(ordered_schedule)
    forecast_bytes = _forecast_jsonl_bytes(ordered_forecasts)
    plan_receipt = _one_plan_receipt(batches)
    coverage_records = tuple(item.seal.to_record() for item in coverages)
    batch_records = tuple(item.verified.seal.to_record() for item in batches)
    record: JsonObject = {
        "schema_version": OUTCOME_V2_ROLLING_AGGREGATE_SCHEMA_VERSION,
        "kind": _KIND,
        "status": _STATUS,
        "created_at": _utc_text(created_at, "created_at"),
        "plan_sha256": plan.sha256,
        "plan_receipt_sha256": plan_receipt.sha256,
        "plan_receipt_run_id": _receipt_run_id(plan_receipt),
        "coverage_policy_sha256": _string(plan_record, "coverage_policy_sha256"),
        "coverage_seal_sha256s": [item.seal.sha256 for item in coverages],
        "coverage_receipt_sha256s": [item.receipt.sha256 for item in coverages],
        "coverage_receipt_run_ids": [_receipt_run_id(item.receipt) for item in coverages],
        "provider_conformance_report_sha256s": [
            _string(record, "provider_conformance_report_sha256") for record in coverage_records
        ],
        "batch_seal_sha256s": [item.verified.seal.sha256 for item in batches],
        "terminal_receipt_sha256s": [item.verified.receipt.sha256 for item in batches],
        "terminal_receipt_run_ids": [_receipt_run_id(item.verified.receipt) for item in batches],
        "generation_lock_sha256s": [item.generation_lock_sha256 for item in batches],
        "terminal_records_sha256s": [item.terminal_records_sha256 for item in batches],
        "question_ids_sha256": canonical_sha256([row.question_id for row in ordered_schedule]),
        "source_game_ids_sha256": canonical_sha256(
            [row.source_game_id for row in ordered_schedule]
        ),
        "feature_rows_sha256": canonical_sha256([row.row_sha256 for row in ordered_rows]),
        "schedule_rows_sha256": bytes_sha256(schedule_bytes),
        "forecasts_sha256": bytes_sha256(forecast_bytes),
        "seasons": list(planned_seasons),
        "forecast_count": len(ordered_forecasts),
        "failed_record_count": sum(
            _nonnegative_integer(record, "failed_record_count") for record in batch_records
        ),
        "provider_authenticity": _REQUIRED_SEPARATELY,
        "remote_execution_attestation": _REQUIRED_SEPARATELY,
        "canonical_elo_replay": _REQUIRED_SEPARATELY,
        "prospective_proof_status": _PROOF_STATUS,
    }
    seal_bytes = canonical_json(record).encode("utf-8")
    return OutcomeV2RollingAggregate(
        ordered_schedule,
        ordered_rows,
        ordered_forecasts,
        seal_bytes,
    )


def write_outcome_v2_rolling_aggregate(
    paths: OutcomeV2RollingAggregateFiles,
    aggregate: OutcomeV2RollingAggregate,
) -> str:
    """Create the schedule, forecasts, and terminal aggregate seal without replacement."""
    _write_once(
        paths.schedule_path,
        _schedule_jsonl_bytes(aggregate.schedule),
        "aggregate schedule",
    )
    _write_once(
        paths.forecasts_path,
        _forecast_jsonl_bytes(aggregate.forecasts),
        "aggregate forecasts",
    )
    _write_once(paths.seal_path, aggregate.canonical_seal_bytes, "aggregate seal")
    return aggregate.sha256


def verify_outcome_v2_rolling_aggregate(
    artifacts: OutcomeV2RollingAggregationArtifacts,
    paths: OutcomeV2RollingAggregateFiles,
    token: str | None = None,
) -> OutcomeV2RollingAggregate:
    """Rebuild all live commitments and require exact aggregate output bytes."""
    actual_seal = _read_bytes(paths.seal_path, "aggregate seal")
    record = _seal_record(actual_seal)
    expected = build_outcome_v2_rolling_aggregate(
        artifacts,
        _parse_utc(_string(record, "created_at"), "created_at"),
        token,
    )
    if actual_seal != expected.canonical_seal_bytes:
        raise OutcomeV2AggregationError("aggregate seal differs from current commitments")
    if _read_bytes(paths.schedule_path, "aggregate schedule") != _schedule_jsonl_bytes(
        expected.schedule
    ):
        raise OutcomeV2AggregationError("aggregate schedule differs from verified coverage")
    if _read_bytes(paths.forecasts_path, "aggregate forecasts") != _forecast_jsonl_bytes(
        expected.forecasts
    ):
        raise OutcomeV2AggregationError("aggregate forecasts differ from terminal records")
    return expected


def _verify_coverages(
    artifacts: Sequence[OutcomeV2ScheduleCoverageReceiptArtifacts],
    token: str | None,
) -> tuple[VerifiedOutcomeV2ScheduleCoverage, ...]:
    try:
        verified = tuple(
            verify_outcome_v2_schedule_coverage_receipt(item, token) for item in artifacts
        )
    except OutcomeV2CoverageError as error:
        raise OutcomeV2AggregationError("cannot verify schedule coverage") from error
    _require_unique(
        (item.seal.sha256 for item in verified),
        "coverage seal",
    )
    _require_unique(
        (item.receipt.sha256 for item in verified),
        "coverage receipt",
    )
    _require_unique(
        (_receipt_run_id(item.receipt) for item in verified),
        "coverage receipt run ID",
    )
    return tuple(
        sorted(verified, key=lambda item: _positive_integer(item.seal.to_record(), "season"))
    )


def _verify_batches(
    artifacts: Sequence[OutcomeV2ProspectiveReceiptArtifacts],
    token: str | None,
) -> tuple[_BatchData, ...]:
    values: list[_BatchData] = []
    try:
        for item in artifacts:
            verified = verify_outcome_v2_prospective_batch_receipt(item, token)
            seal = verified.seal.to_record()
            feature_bytes = _read_bytes(item.batch.forecast.feature_rows_path, "batch features")
            generation_bytes = _read_bytes(
                item.batch.forecast.generation_lock_path,
                "batch generation lock",
            )
            records_bytes = _read_bytes(
                item.batch.forecast.inference_records_path,
                "batch terminal records",
            )
            _require_digest(
                feature_bytes,
                _string(seal, "evaluation_feature_rows_sha256"),
                "batch features",
            )
            _require_digest(
                generation_bytes,
                _string(seal, "evaluation_generation_lock_sha256"),
                "batch generation lock",
            )
            _require_digest(
                records_bytes,
                _string(seal, "evaluation_inference_records_sha256"),
                "batch terminal records",
            )
            rows = read_nba_feature_rows_jsonl_bytes(feature_bytes)
            generation_lock = OutcomeV2GenerationLock(generation_bytes)
            records = read_outcome_v2_inference_records_jsonl_bytes(
                records_bytes,
                generation_lock,
            )
            forecasts = binary_forecasts_from_inference_records(records, generation_lock)
            values.append(
                _BatchData(
                    verified=verified,
                    rows=rows,
                    forecasts=forecasts,
                    generation_lock_sha256=generation_lock.sha256,
                    terminal_records_sha256=bytes_sha256(records_bytes),
                )
            )
    except (
        NbaFeatureRowError,
        OutcomeV2AggregationError,
        OutcomeV2InferenceError,
        OutcomeV2RollingError,
    ) as error:
        raise OutcomeV2AggregationError("cannot verify terminal forecast batches") from error
    batches = tuple(values)
    _require_unique(
        (_string(item.verified.seal.to_record(), "batch_id") for item in batches),
        "batch ID",
    )
    _require_unique((item.verified.seal.sha256 for item in batches), "batch seal")
    _require_unique(
        (item.generation_lock_sha256 for item in batches),
        "generation lock",
    )
    _require_unique(
        (item.verified.receipt.sha256 for item in batches),
        "terminal receipt",
    )
    _require_unique(
        (_receipt_run_id(item.verified.receipt) for item in batches),
        "terminal receipt run ID",
    )
    return tuple(
        sorted(
            batches,
            key=lambda item: (
                _string(item.verified.seal.to_record(), "earliest_forecast_cutoff"),
                _string(item.verified.seal.to_record(), "batch_id"),
            ),
        )
    )


def _require_shared_plan(
    plan_sha256: str,
    coverages: Sequence[VerifiedOutcomeV2ScheduleCoverage],
    batches: Sequence[_BatchData],
) -> None:
    if any(_string(item.seal.to_record(), "plan_sha256") != plan_sha256 for item in coverages):
        raise OutcomeV2AggregationError("schedule coverage uses a different plan")
    if any(
        _string(item.verified.seal.to_record(), "plan_sha256") != plan_sha256 for item in batches
    ):
        raise OutcomeV2AggregationError("forecast batch uses a different plan")


def _ordered_schedule_rows(
    planned_seasons: tuple[int, ...],
    coverages: Sequence[VerifiedOutcomeV2ScheduleCoverage],
) -> tuple[NbaEloReplayRow, ...]:
    by_season: dict[int, tuple[NbaEloReplayRow, ...]] = {}
    for coverage in coverages:
        season = _positive_integer(coverage.seal.to_record(), "season")
        if season in by_season:
            raise OutcomeV2AggregationError("planned season has multiple coverage seals")
        by_season[season] = coverage.rows
    if tuple(sorted(by_season)) != planned_seasons:
        raise OutcomeV2AggregationError("coverage seasons differ from the prospective plan")
    rows = tuple(row for season in planned_seasons for row in by_season[season])
    _require_unique((row.question_id for row in rows), "schedule question ID")
    _require_unique((row.source_game_id for row in rows), "schedule source game ID")
    return rows


def _align_exact_batch_union(
    schedule: tuple[NbaEloReplayRow, ...],
    batches: Sequence[_BatchData],
) -> tuple[tuple[NbaRichFeatureRow, ...], tuple[BinaryForecast, ...]]:
    batch_rows = tuple(row for batch in batches for row in batch.rows)
    batch_forecasts = tuple(forecast for batch in batches for forecast in batch.forecasts)
    _require_unique((row.question_id for row in batch_rows), "batch question ID")
    _require_unique((row.row_sha256 for row in batch_rows), "batch feature row")
    _require_unique((row.question_id for row in batch_forecasts), "forecast question ID")
    schedule_ids = {row.question_id for row in schedule}
    batch_ids = {row.question_id for row in batch_rows}
    if schedule_ids != batch_ids:
        missing_count = len(schedule_ids - batch_ids)
        extra_count = len(batch_ids - schedule_ids)
        raise OutcomeV2AggregationError(
            f"batch union differs from schedule; missing={missing_count}, extra={extra_count}"
        )
    rows_by_id = {row.question_id: row for row in batch_rows}
    forecasts_by_id = {row.question_id: row for row in batch_forecasts}
    if set(forecasts_by_id) != schedule_ids:
        raise OutcomeV2AggregationError("derived forecast union differs from schedule")
    for expected in schedule:
        actual = rows_by_id[expected.question_id]
        if (
            actual.source_game_id,
            actual.team_id,
            actual.opponent_id,
            actual.site,
            actual.season,
            actual.forecast_cutoff,
            actual.scheduled_tipoff,
        ) != (
            expected.source_game_id,
            expected.team_id,
            expected.opponent_id,
            expected.site,
            expected.season,
            expected.forecast_cutoff,
            expected.scheduled_tipoff,
        ):
            raise OutcomeV2AggregationError("batch identity differs from committed schedule")
    return (
        tuple(rows_by_id[row.question_id] for row in schedule),
        tuple(forecasts_by_id[row.question_id] for row in schedule),
    )


def _require_coverage_before_inputs(
    coverages: Sequence[VerifiedOutcomeV2ScheduleCoverage],
    rows: Sequence[NbaRichFeatureRow],
) -> None:
    earliest_by_season: dict[int, datetime] = {}
    for row in rows:
        earliest_by_season[row.season] = min(
            earliest_by_season.get(row.season, row.input_available_at),
            row.input_available_at,
        )
    for coverage in coverages:
        record = coverage.seal.to_record()
        season = _positive_integer(record, "season")
        deadline = _parse_utc(
            _string(record, "commitment_deadline"),
            "commitment_deadline",
        )
        if deadline > earliest_by_season[season]:
            raise OutcomeV2AggregationError("schedule was not committed before season inputs")


def _require_aggregate_time(
    created_at: datetime,
    coverages: Sequence[VerifiedOutcomeV2ScheduleCoverage],
    batches: Sequence[_BatchData],
) -> None:
    external_times = tuple(item.externally_committed_at for item in coverages) + tuple(
        item.verified.externally_committed_at for item in batches
    )
    if max(external_times) > created_at:
        raise OutcomeV2AggregationError("aggregate cannot predate an external commitment")


def _one_plan_receipt(batches: Sequence[_BatchData]) -> GitHubActionsReceipt:
    receipts = tuple(item.verified.plan_receipt for item in batches)
    first = receipts[0]
    if any(receipt.canonical_bytes != first.canonical_bytes for receipt in receipts[1:]):
        raise OutcomeV2AggregationError("forecast batches do not share one plan receipt")
    return first


def _schedule_jsonl_bytes(rows: Sequence[NbaEloReplayRow]) -> bytes:
    return "".join(f"{canonical_json(row.canonical_payload())}\n" for row in rows).encode("utf-8")


def _forecast_jsonl_bytes(rows: Sequence[BinaryForecast]) -> bytes:
    payloads = (
        {
            "schema_version": NBA_EVALUATION_FORECAST_SCHEMA_VERSION,
            "question_id": row.question_id,
            "team_probability": row.team_probability,
            "failure_reason": row.failure_reason,
        }
        for row in rows
    )
    return "".join(f"{canonical_json(payload)}\n" for payload in payloads).encode("utf-8")


def _seal_record(value: bytes) -> JsonObject:
    try:
        text = value.decode("utf-8")
        record = parse_json_object(text)
        require_exact_keys(record, _SEAL_KEYS, "rolling aggregate seal")
        _validate_seal_record(record, text)
    except (JsonFormatError, UnicodeError) as error:
        raise OutcomeV2AggregationError("invalid rolling aggregate seal") from error
    return record


def _validate_seal_record(record: JsonObject, text: str) -> None:
    if text != canonical_json(record):
        raise OutcomeV2AggregationError("aggregate seal must use canonical JSON bytes")
    if _integer(record, "schema_version") != OUTCOME_V2_ROLLING_AGGREGATE_SCHEMA_VERSION:
        raise OutcomeV2AggregationError("unsupported rolling aggregate schema")
    _require_value(record, "kind", _KIND)
    _require_value(record, "status", _STATUS)
    _parse_utc(_string(record, "created_at"), "created_at")
    _validate_seal_hashes(record)
    seasons = _integer_tuple(record, "seasons")
    if len(seasons) < 2 or seasons != tuple(sorted(set(seasons))):
        raise OutcomeV2AggregationError("aggregate requires increasing multi-season coverage")
    count = _positive_integer(record, "forecast_count")
    if _nonnegative_integer(record, "failed_record_count") > count:
        raise OutcomeV2AggregationError("failed forecast count exceeds aggregate count")
    _validate_seal_lists(record, len(seasons))
    _require_value(record, "provider_authenticity", _REQUIRED_SEPARATELY)
    _require_value(record, "remote_execution_attestation", _REQUIRED_SEPARATELY)
    _require_value(record, "canonical_elo_replay", _REQUIRED_SEPARATELY)
    _require_value(record, "prospective_proof_status", _PROOF_STATUS)


def _validate_seal_hashes(record: Mapping[str, object]) -> None:
    for field_name in (
        "plan_sha256",
        "plan_receipt_sha256",
        "coverage_policy_sha256",
        "question_ids_sha256",
        "source_game_ids_sha256",
        "feature_rows_sha256",
        "schedule_rows_sha256",
        "forecasts_sha256",
    ):
        _require_hash(_string(record, field_name), field_name)
    _positive_integer(record, "plan_receipt_run_id")


def _validate_seal_lists(record: Mapping[str, object], season_count: int) -> None:
    coverage_hashes = _hash_tuple(record, "coverage_seal_sha256s")
    if len(coverage_hashes) != season_count:
        raise OutcomeV2AggregationError("coverage seal count differs from seasons")
    _matching_hash_tuple_length(record, "coverage_receipt_sha256s", season_count)
    _matching_hash_tuple_length(
        record,
        "provider_conformance_report_sha256s",
        season_count,
    )
    _matching_integer_tuple_length(record, "coverage_receipt_run_ids", season_count)
    batch_count = len(_hash_tuple(record, "batch_seal_sha256s"))
    for field_name in (
        "terminal_receipt_sha256s",
        "generation_lock_sha256s",
        "terminal_records_sha256s",
    ):
        _matching_hash_tuple_length(record, field_name, batch_count)
    _matching_integer_tuple_length(record, "terminal_receipt_run_ids", batch_count)


def _matching_hash_tuple_length(
    record: Mapping[str, object],
    field_name: str,
    expected: int,
) -> None:
    if len(_hash_tuple(record, field_name)) != expected:
        raise OutcomeV2AggregationError(f"{field_name} has the wrong length")


def _matching_integer_tuple_length(
    record: Mapping[str, object],
    field_name: str,
    expected: int,
) -> None:
    values = _integer_tuple(record, field_name)
    if len(values) != expected or any(value <= 0 for value in values):
        raise OutcomeV2AggregationError(f"{field_name} has invalid values")
    _require_unique(values, field_name)


def _hash_tuple(record: Mapping[str, object], field_name: str) -> tuple[str, ...]:
    values = _string_tuple(record, field_name)
    if not values:
        raise OutcomeV2AggregationError(f"{field_name} must not be empty")
    for value in values:
        _require_hash(value, field_name)
    _require_unique(values, field_name)
    return values


def _write_once(path: Path, value: bytes, description: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
    except FileExistsError:
        raise
    except OSError as error:
        raise OutcomeV2AggregationError(f"cannot write {description}") from error


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise OutcomeV2AggregationError(f"cannot read {description}") from error


def _require_digest(value: bytes, expected: str, description: str) -> None:
    if bytes_sha256(value) != expected:
        raise OutcomeV2AggregationError(f"{description} differs from its batch seal")


def _receipt_run_id(receipt: GitHubActionsReceipt) -> int:
    return _positive_integer(_object(receipt.to_record(), "run"), "id")


def _require_unique[T: Hashable](values: Iterable[T], description: str) -> None:
    items = tuple(values)
    if len(set(items)) != len(items):
        raise OutcomeV2AggregationError(f"duplicate {description}")


def _require_hash(value: str, field_name: str) -> None:
    if _HASH_PATTERN.fullmatch(value) is None:
        raise OutcomeV2AggregationError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_value(record: Mapping[str, object], field_name: str, expected: object) -> None:
    if required_field(record, field_name) != expected:
        raise OutcomeV2AggregationError(f"unexpected {field_name}")


def _object(record: Mapping[str, object], field_name: str) -> JsonObject:
    return require_object(required_field(record, field_name), field_name)


def _string(record: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = required_field(record, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutcomeV2AggregationError(f"{field_name} must be an integer")
    return value


def _positive_integer(record: Mapping[str, object], field_name: str) -> int:
    value = _integer(record, field_name)
    if value <= 0:
        raise OutcomeV2AggregationError(f"{field_name} must be positive")
    return value


def _nonnegative_integer(record: Mapping[str, object], field_name: str) -> int:
    value = _integer(record, field_name)
    if value < 0:
        raise OutcomeV2AggregationError(f"{field_name} must be non-negative")
    return value


def _integer_tuple(record: Mapping[str, object], field_name: str) -> tuple[int, ...]:
    values = require_list(required_field(record, field_name), field_name)
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise OutcomeV2AggregationError(f"{field_name}[{index}] must be an integer")
        result.append(value)
    return tuple(result)


def _string_tuple(record: Mapping[str, object], field_name: str) -> tuple[str, ...]:
    values = require_list(required_field(record, field_name), field_name)
    return tuple(
        require_string(value, f"{field_name}[{index}]") for index, value in enumerate(values)
    )


def _utc_text(value: datetime, field_name: str) -> str:
    _require_utc(value, field_name)
    return value.isoformat().replace("+00:00", "Z")


def _parse_utc(value: str, field_name: str) -> datetime:
    if not value.endswith("Z"):
        raise OutcomeV2AggregationError(f"{field_name} must use canonical UTC Z notation")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        raise OutcomeV2AggregationError(f"{field_name} must be an ISO 8601 datetime") from error
    _require_utc(parsed, field_name)
    if _utc_text(parsed, field_name) != value:
        raise OutcomeV2AggregationError(f"{field_name} must use canonical UTC notation")
    return parsed


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise OutcomeV2AggregationError(f"{field_name} must be a UTC datetime")
    if value.astimezone(UTC) != value:
        raise OutcomeV2AggregationError(f"{field_name} must be a UTC datetime")
