"""Post-outcome canonical Elo scoring for a verified rolling forecast aggregate.

This layer structurally links claimed finals to retained snapshot bytes and uses snapshot
availability for Elo chronology. A licensed connector must still authenticate the
provider payload and derive each score before it writes the canonical resolution file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

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
from forecastfm.nba_elo_replay import NbaEloReplayError, replay_nba_elo_states
from forecastfm.nba_evaluation_gate import (
    NbaEvaluationAnswer,
    NbaEvaluationCohortInput,
    nba_evaluation_answers_jsonl_bytes,
    nba_evaluation_cohort_jsonl_bytes,
    write_nba_evaluation_answers_jsonl,
    write_nba_evaluation_cohort_jsonl,
)
from forecastfm.nba_resolutions import (
    NbaResolution,
    NbaResolutionError,
    read_nba_resolutions_jsonl_bytes,
)
from forecastfm.nba_snapshot_pack import (
    NbaSnapshotIndex,
    SnapshotPackError,
    load_snapshot_pack_bytes,
)
from forecastfm.outcome_v2_aggregation import (
    OutcomeV2RollingAggregate,
    OutcomeV2RollingAggregateFiles,
    OutcomeV2RollingAggregationArtifacts,
    verify_outcome_v2_rolling_aggregate,
)
from forecastfm.outcome_v2_config import outcome_v2_elo_recipe
from forecastfm.outcome_v2_metrics import BinaryForecast
from forecastfm.outcome_v2_rolling import verify_outcome_v2_prospective_plan

OUTCOME_V2_ROLLING_SCORE_SCHEMA_VERSION = 1

_KIND = "forecastfm_outcome_v2_rolling_score"
_STATUS = "canonical_replay_from_snapshot_bound_claimed_resolutions"
_REQUIRED_SEPARATELY = "required_separately"
_SCORING_TIME_STATUS = "local_claim_only"
_PROOF_STATUS = "provider_resolution_authenticity_and_remote_execution_required"
_GAME_DATE_TIMEZONE = "America/New_York"
_GAME_DATE_RULE = {
    "source": "scheduled_tipoff",
    "timezone": _GAME_DATE_TIMEZONE,
}
_HASH_CHARACTERS = frozenset("0123456789abcdef")
_SEAL_KEYS = {
    "schema_version",
    "kind",
    "status",
    "scored_at",
    "scoring_time_attestation",
    "aggregate_seal_sha256",
    "evaluation_policy_sha256",
    "calibration_sha256",
    "snapshot_pack_sha256",
    "resolutions_sha256",
    "resolution_availability_sha256",
    "elo_recipe",
    "game_date_rule",
    "game_date_rule_sha256",
    "cohort_sha256",
    "answers_sha256",
    "forecasts_sha256",
    "question_ids_sha256",
    "source_game_ids_sha256",
    "seasons",
    "game_count",
    "failed_forecast_count",
    "provider_resolution_authenticity",
    "provider_score_derivation",
    "remote_execution_attestation",
    "prospective_proof_status",
}
type JsonObject = dict[str, object]


class OutcomeV2RollingScoreError(ValueError):
    """Raised when rolling scoring differs from its canonical proof boundary."""


@dataclass(frozen=True, slots=True)
class OutcomeV2RollingScoringInputs:
    """Canonical generic-gate inputs created only after terminal outcomes exist."""

    cohort: tuple[NbaEvaluationCohortInput, ...]
    answers: tuple[NbaEvaluationAnswer, ...]
    forecasts: tuple[BinaryForecast, ...]


@dataclass(frozen=True, slots=True)
class OutcomeV2RollingScoringFiles:
    """Create-only canonical scoring inputs and their provenance seal."""

    cohort_path: Path
    answers_path: Path
    seal_path: Path


@dataclass(frozen=True, slots=True)
class OutcomeV2RollingResolutionArtifacts:
    """Claimed finals and retained bytes used to validate snapshot linkage."""

    snapshot_pack_path: Path
    resolutions_path: Path


@dataclass(frozen=True, slots=True)
class OutcomeV2RollingScoringArtifacts:
    """Answer-free aggregation plus separately retained claimed finals."""

    aggregation: OutcomeV2RollingAggregationArtifacts
    aggregate_files: OutcomeV2RollingAggregateFiles
    resolutions: OutcomeV2RollingResolutionArtifacts
    calibration_path: Path


@dataclass(frozen=True, slots=True)
class OutcomeV2RollingScoringSeal:
    """Canonical provenance binding for one post-outcome scoring boundary."""

    canonical_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _seal_record(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        """Return the digest of the exact canonical seal bytes."""
        return bytes_sha256(self.canonical_bytes)

    def to_record(self) -> JsonObject:
        """Return a fresh strict record decoded from the seal bytes."""
        return _seal_record(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class _LoadedResolutions:
    index: NbaSnapshotIndex
    rows: tuple[NbaResolution, ...]
    snapshot_pack_sha256: str
    resolutions_sha256: str


@dataclass(frozen=True, slots=True)
class _VerifiedScoring:
    aggregate: OutcomeV2RollingAggregate
    inputs: OutcomeV2RollingScoringInputs
    loaded: _LoadedResolutions
    availability_sha256: str
    evaluation_policy_sha256: str
    calibration_sha256: str


def build_outcome_v2_rolling_scoring_inputs(
    artifacts: OutcomeV2RollingScoringArtifacts,
    scored_at: datetime,
    token: str | None = None,
) -> OutcomeV2RollingScoringInputs:
    """Reverify receipts and retained finals, then independently replay canonical Elo."""
    return _verify_scoring(
        artifacts,
        scored_at,
        token,
    ).inputs


def write_outcome_v2_rolling_scoring_inputs(
    paths: OutcomeV2RollingScoringFiles,
    inputs: OutcomeV2RollingScoringInputs,
) -> None:
    """Write canonical gate inputs without replacing an existing scoring boundary."""
    paths.cohort_path.parent.mkdir(parents=True, exist_ok=True)
    paths.answers_path.parent.mkdir(parents=True, exist_ok=True)
    write_nba_evaluation_cohort_jsonl(paths.cohort_path, inputs.cohort)
    write_nba_evaluation_answers_jsonl(paths.answers_path, inputs.answers)


def build_outcome_v2_rolling_scoring_seal(
    artifacts: OutcomeV2RollingScoringArtifacts,
    scoring_files: OutcomeV2RollingScoringFiles,
    scored_at: datetime,
    token: str | None = None,
) -> OutcomeV2RollingScoringSeal:
    """Bind canonical gate inputs to live receipts, claimed finals, and the Elo recipe."""
    verified = _verify_scoring(
        artifacts,
        scored_at,
        token,
    )
    cohort_bytes = nba_evaluation_cohort_jsonl_bytes(verified.inputs.cohort)
    answer_bytes = nba_evaluation_answers_jsonl_bytes(verified.inputs.answers)
    _require_file_bytes(scoring_files.cohort_path, cohort_bytes, "rolling scoring cohort")
    _require_file_bytes(scoring_files.answers_path, answer_bytes, "rolling scoring answers")

    aggregate_record = verified.aggregate.to_record()
    recipe = outcome_v2_elo_recipe()
    schedule = verified.aggregate.schedule
    record: JsonObject = {
        "schema_version": OUTCOME_V2_ROLLING_SCORE_SCHEMA_VERSION,
        "kind": _KIND,
        "status": _STATUS,
        "scored_at": _utc_text(scored_at, "scored_at"),
        "scoring_time_attestation": _SCORING_TIME_STATUS,
        "aggregate_seal_sha256": verified.aggregate.sha256,
        "evaluation_policy_sha256": verified.evaluation_policy_sha256,
        "calibration_sha256": verified.calibration_sha256,
        "snapshot_pack_sha256": verified.loaded.snapshot_pack_sha256,
        "resolutions_sha256": verified.loaded.resolutions_sha256,
        "resolution_availability_sha256": verified.availability_sha256,
        "elo_recipe": {
            "config": recipe.canonical_payload(),
            "sha256": recipe.recipe_sha256,
        },
        "game_date_rule": _GAME_DATE_RULE,
        "game_date_rule_sha256": canonical_sha256(_GAME_DATE_RULE),
        "cohort_sha256": bytes_sha256(cohort_bytes),
        "answers_sha256": bytes_sha256(answer_bytes),
        "forecasts_sha256": _string(aggregate_record, "forecasts_sha256"),
        "question_ids_sha256": canonical_sha256([row.question_id for row in schedule]),
        "source_game_ids_sha256": canonical_sha256([row.source_game_id for row in schedule]),
        "seasons": sorted({row.season for row in schedule}),
        "game_count": len(schedule),
        "failed_forecast_count": sum(
            forecast.team_probability is None for forecast in verified.inputs.forecasts
        ),
        "provider_resolution_authenticity": _REQUIRED_SEPARATELY,
        "provider_score_derivation": _REQUIRED_SEPARATELY,
        "remote_execution_attestation": _REQUIRED_SEPARATELY,
        "prospective_proof_status": _PROOF_STATUS,
    }
    return OutcomeV2RollingScoringSeal(canonical_json(record).encode("utf-8"))


def write_outcome_v2_rolling_scoring_seal(
    path: Path,
    seal: OutcomeV2RollingScoringSeal,
) -> str:
    """Create the exact scoring seal without replacing a prior claim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as file:
            file.write(seal.canonical_bytes)
    except FileExistsError as error:
        raise OutcomeV2RollingScoreError("rolling scoring seal already exists") from error
    except OSError as error:
        raise OutcomeV2RollingScoreError("cannot write rolling scoring seal") from error
    return seal.sha256


def verify_outcome_v2_rolling_scoring_seal(
    artifacts: OutcomeV2RollingScoringArtifacts,
    scoring_files: OutcomeV2RollingScoringFiles,
    token: str | None = None,
) -> OutcomeV2RollingScoringSeal:
    """Live-rebuild every scoring input and require the persisted provenance seal."""
    actual = OutcomeV2RollingScoringSeal(
        _read_bytes(scoring_files.seal_path, "rolling scoring seal")
    )
    scored_at = _parse_utc(_string(actual.to_record(), "scored_at"), "scored_at")
    expected = build_outcome_v2_rolling_scoring_seal(
        artifacts,
        scoring_files,
        scored_at,
        token,
    )
    if actual.canonical_bytes != expected.canonical_bytes:
        raise OutcomeV2RollingScoreError("rolling scoring seal differs from current artifacts")
    return actual


def _verify_scoring(
    artifacts: OutcomeV2RollingScoringArtifacts,
    scored_at: datetime,
    token: str | None,
) -> _VerifiedScoring:
    aggregate = verify_outcome_v2_rolling_aggregate(
        artifacts.aggregation,
        artifacts.aggregate_files,
        token,
    )
    plan = verify_outcome_v2_prospective_plan(
        artifacts.aggregation.plan,
        artifacts.aggregation.plan_path,
    )
    plan_record = plan.to_record()
    calibration_bytes = _read_bytes(artifacts.calibration_path, "rolling calibration rows")
    calibration_sha256 = bytes_sha256(calibration_bytes)
    if calibration_sha256 != _string(plan_record, "calibration_sha256"):
        raise OutcomeV2RollingScoreError(
            "rolling calibration differs from the externally committed plan"
        )
    loaded = _load_resolution_artifacts(artifacts.resolutions)
    _require_scoring_time(scored_at, loaded.rows)
    replay_resolutions, availability_sha256 = _snapshot_timed_resolutions(
        loaded.rows,
        loaded.index,
    )
    inputs = _scoring_inputs_from_verified_aggregate(
        aggregate,
        loaded.rows,
        replay_resolutions,
    )
    return _VerifiedScoring(
        aggregate,
        inputs,
        loaded,
        availability_sha256,
        _string(plan_record, "evaluation_policy_sha256"),
        calibration_sha256,
    )


def _scoring_inputs_from_verified_aggregate(
    aggregate: OutcomeV2RollingAggregate,
    resolutions: tuple[NbaResolution, ...],
    replay_resolutions: tuple[NbaResolution, ...],
) -> OutcomeV2RollingScoringInputs:
    try:
        states = replay_nba_elo_states(
            aggregate.schedule,
            replay_resolutions,
            outcome_v2_elo_recipe(),
        )
    except NbaEloReplayError as error:
        raise OutcomeV2RollingScoreError("cannot replay canonical rolling Elo states") from error
    for feature_row, state in zip(aggregate.feature_rows, states, strict=True):
        if feature_row.question_id != state.question_id:
            raise OutcomeV2RollingScoreError("feature row differs from canonical Elo identity")
        if feature_row.elo_team_win_probability != state.team_win_probability:
            raise OutcomeV2RollingScoreError("feature row differs from canonical Elo probability")
        if feature_row.elo_opponent_win_probability != 1.0 - state.team_win_probability:
            raise OutcomeV2RollingScoreError(
                "feature row opponent probability differs from canonical Elo"
            )
    resolution_by_id = {row.question_id: row for row in resolutions}
    cohort = tuple(
        NbaEvaluationCohortInput(
            question_id=row.question_id,
            season=row.season,
            game_date=_league_game_date(row.scheduled_tipoff),
            raw_elo_team_probability=state.team_win_probability,
        )
        for row, state in zip(aggregate.schedule, states, strict=True)
    )
    answers = tuple(
        NbaEvaluationAnswer(
            question_id=row.question_id,
            realized_team_win=resolution_by_id[row.question_id].team_won,
        )
        for row in aggregate.schedule
    )
    return OutcomeV2RollingScoringInputs(cohort, answers, aggregate.forecasts)


def _load_resolution_artifacts(
    artifacts: OutcomeV2RollingResolutionArtifacts,
) -> _LoadedResolutions:
    snapshot_bytes = _read_bytes(artifacts.snapshot_pack_path, "rolling snapshot pack")
    resolution_bytes = _read_bytes(artifacts.resolutions_path, "rolling resolutions")
    try:
        index = load_snapshot_pack_bytes(snapshot_bytes)
        resolutions = read_nba_resolutions_jsonl_bytes(
            resolution_bytes,
            snapshot_index=index,
        )
    except (NbaResolutionError, SnapshotPackError) as error:
        raise OutcomeV2RollingScoreError("invalid rolling resolution artifacts") from error
    return _LoadedResolutions(
        index,
        resolutions,
        bytes_sha256(snapshot_bytes),
        bytes_sha256(resolution_bytes),
    )


def _snapshot_timed_resolutions(
    resolutions: tuple[NbaResolution, ...],
    snapshot_index: NbaSnapshotIndex,
) -> tuple[tuple[NbaResolution, ...], str]:
    timed: list[NbaResolution] = []
    lineage: list[JsonObject] = []
    for resolution in resolutions:
        try:
            snapshot = snapshot_index.latest_eligible(
                resolution.source_id,
                resolution.resolved_at,
            )
        except SnapshotPackError as error:
            raise OutcomeV2RollingScoreError(
                "cannot select final-score snapshot availability"
            ) from error
        if snapshot is None:
            raise OutcomeV2RollingScoreError("resolution has no eligible final-score snapshot")
        available_at = snapshot.metadata.available_at
        timed.append(replace(resolution, resolved_at=available_at))
        lineage.append(
            {
                "question_id": resolution.question_id,
                "source_game_id": resolution.source_game_id,
                "source_id": resolution.source_id,
                "snapshot_metadata_sha256": resolution.snapshot_metadata_sha256,
                "available_at": _utc_text(available_at, "snapshot available_at"),
            }
        )
    return tuple(timed), canonical_sha256(lineage)


def _require_scoring_time(
    scored_at: datetime,
    resolutions: tuple[NbaResolution, ...],
) -> None:
    _require_utc(scored_at, "scored_at")
    if any(resolution.resolved_at > scored_at for resolution in resolutions):
        raise OutcomeV2RollingScoreError("scoring cannot precede a declared final resolution")


def _seal_record(value: bytes) -> JsonObject:
    try:
        text = value.decode("utf-8")
        record = parse_json_object(text)
        require_exact_keys(record, _SEAL_KEYS, "rolling scoring seal")
        if text != canonical_json(record):
            raise OutcomeV2RollingScoreError("rolling scoring seal must use canonical JSON bytes")
        _validate_seal_record(record)
    except (JsonFormatError, UnicodeError) as error:
        raise OutcomeV2RollingScoreError("invalid rolling scoring seal") from error
    return record


def _validate_seal_record(record: Mapping[str, object]) -> None:
    if _integer(record, "schema_version") != OUTCOME_V2_ROLLING_SCORE_SCHEMA_VERSION:
        raise OutcomeV2RollingScoreError("unsupported rolling scoring seal schema")
    _require_value(record, "kind", _KIND)
    _require_value(record, "status", _STATUS)
    _parse_utc(_string(record, "scored_at"), "scored_at")
    _require_value(record, "scoring_time_attestation", _SCORING_TIME_STATUS)
    for field_name in (
        "aggregate_seal_sha256",
        "evaluation_policy_sha256",
        "calibration_sha256",
        "snapshot_pack_sha256",
        "resolutions_sha256",
        "resolution_availability_sha256",
        "game_date_rule_sha256",
        "cohort_sha256",
        "answers_sha256",
        "forecasts_sha256",
        "question_ids_sha256",
        "source_game_ids_sha256",
    ):
        _require_hash(_string(record, field_name), field_name)
    recipe = require_object(required_field(record, "elo_recipe"), "elo_recipe")
    require_exact_keys(recipe, {"config", "sha256"}, "elo_recipe")
    _require_hash(_string(recipe, "sha256"), "elo_recipe.sha256")
    rule = require_object(required_field(record, "game_date_rule"), "game_date_rule")
    require_exact_keys(rule, {"source", "timezone"}, "game_date_rule")
    seasons = _integer_list(record, "seasons")
    if len(seasons) < 2 or seasons != sorted(set(seasons)):
        raise OutcomeV2RollingScoreError("rolling scoring requires increasing seasons")
    game_count = _positive_integer(record, "game_count")
    if _nonnegative_integer(record, "failed_forecast_count") > game_count:
        raise OutcomeV2RollingScoreError("failed forecast count exceeds game count")
    _require_value(record, "provider_resolution_authenticity", _REQUIRED_SEPARATELY)
    _require_value(record, "provider_score_derivation", _REQUIRED_SEPARATELY)
    _require_value(record, "remote_execution_attestation", _REQUIRED_SEPARATELY)
    _require_value(record, "prospective_proof_status", _PROOF_STATUS)


def _league_game_date(scheduled_tipoff: datetime) -> date:
    _require_utc(scheduled_tipoff, "scheduled_tipoff")
    return scheduled_tipoff.astimezone(ZoneInfo(_GAME_DATE_TIMEZONE)).date()


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise OutcomeV2RollingScoreError(f"cannot read {description}") from error


def _require_file_bytes(path: Path, expected: bytes, description: str) -> None:
    if _read_bytes(path, description) != expected:
        raise OutcomeV2RollingScoreError(f"{description} differs from verified scoring inputs")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise OutcomeV2RollingScoreError(f"{field_name} must be in UTC")


def _utc_text(value: datetime, field_name: str) -> str:
    _require_utc(value, field_name)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str, field_name: str) -> datetime:
    if not value.endswith("Z"):
        raise OutcomeV2RollingScoreError(f"{field_name} must use canonical UTC notation")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise OutcomeV2RollingScoreError(f"{field_name} must be an ISO 8601 datetime") from error
    _require_utc(parsed, field_name)
    if value != _utc_text(parsed, field_name):
        raise OutcomeV2RollingScoreError(f"{field_name} must use canonical UTC notation")
    return parsed.astimezone(UTC)


def _string(record: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = required_field(record, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise OutcomeV2RollingScoreError(f"{field_name} must be an integer")
    return value


def _positive_integer(record: Mapping[str, object], field_name: str) -> int:
    value = _integer(record, field_name)
    if value <= 0:
        raise OutcomeV2RollingScoreError(f"{field_name} must be positive")
    return value


def _nonnegative_integer(record: Mapping[str, object], field_name: str) -> int:
    value = _integer(record, field_name)
    if value < 0:
        raise OutcomeV2RollingScoreError(f"{field_name} must be nonnegative")
    return value


def _integer_list(record: Mapping[str, object], field_name: str) -> list[int]:
    values = require_list(required_field(record, field_name), field_name)
    return [_positive_integer({field_name: value}, field_name) for value in values]


def _require_hash(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in _HASH_CHARACTERS for character in value):
        raise OutcomeV2RollingScoreError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_value(record: Mapping[str, object], field_name: str, expected: str) -> None:
    if _string(record, field_name) != expected:
        raise OutcomeV2RollingScoreError(f"rolling scoring seal has invalid {field_name}")
