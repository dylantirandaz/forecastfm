"""Target-free richer NBA feature rows with point-in-time provenance."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import isclose, isfinite
from pathlib import Path

from forecastfm.integrity import canonical_json, canonical_sha256
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_float,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_evidence import NbaEvidenceBundle, evidence_bundle_sha256
from forecastfm.nba_rich import (
    NBA_RICH_FEATURE_NAMES,
    NBA_RICH_SCHEMA_SHA256,
    NbaRichFeatures,
    local_rich_features_from_bundle,
    tinker_rich_features_from_bundle,
)

_HASH_CHARACTERS = frozenset("0123456789abcdef")
_PRIMARY_STATE_OFFSET = timedelta(minutes=60)
NBA_PRIMARY_STATE_ID = "T-60"
_ROW_KEYS = {
    "elo_available_at",
    "elo_opponent_win_probability",
    "elo_state_sha256",
    "elo_team_win_probability",
    "evidence_bundle_sha256",
    "feature_schema_sha256",
    "features",
    "forecast_cutoff",
    "input_available_at",
    "question_id",
    "scheduled_tipoff",
    "season",
    "state_id",
}
_FEATURE_KEYS = {"name", "value"}


class NbaFeatureRowError(ValueError):
    """Raised when a richer NBA feature row violates its causal contract."""


@dataclass(frozen=True, slots=True)
class NbaEloPriorInput:
    """One timestamped Elo prior bound to its exact pre-cutoff state digest."""

    team_win_probability: float
    available_at: datetime
    state_sha256: str

    def __post_init__(self) -> None:
        _require_elo_probabilities(
            self.team_win_probability,
            1.0 - self.team_win_probability,
        )
        _require_utc(self.available_at, "elo.available_at")
        _require_sha256(self.state_sha256, "elo.state_sha256")


@dataclass(frozen=True, slots=True)
class NbaRichFeatureRow:
    """One immutable, target-free input row for a richer NBA forecast."""

    question_id: str
    season: int
    forecast_cutoff: datetime
    scheduled_tipoff: datetime
    elo_team_win_probability: float
    elo_opponent_win_probability: float
    elo_available_at: datetime
    elo_state_sha256: str
    rich_features: NbaRichFeatures
    evidence_bundle_sha256: str
    input_available_at: datetime
    state_id: str = field(default=NBA_PRIMARY_STATE_ID, init=False)
    feature_schema_sha256: str = field(default=NBA_RICH_SCHEMA_SHA256, init=False)

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise NbaFeatureRowError("question_id must not be empty")
        _require_positive_season(self.season)
        _require_utc(self.forecast_cutoff, "forecast_cutoff")
        _require_utc(self.scheduled_tipoff, "scheduled_tipoff")
        _require_utc(self.elo_available_at, "elo_available_at")
        _require_utc(self.input_available_at, "input_available_at")
        if self.forecast_cutoff >= self.scheduled_tipoff:
            raise NbaFeatureRowError("forecast_cutoff must precede scheduled_tipoff")
        if self.scheduled_tipoff - self.forecast_cutoff != _PRIMARY_STATE_OFFSET:
            raise NbaFeatureRowError("primary supervised state must be exactly T-60")
        _require_elo_probabilities(
            self.elo_team_win_probability,
            self.elo_opponent_win_probability,
        )
        if self.elo_available_at > self.forecast_cutoff:
            raise NbaFeatureRowError("Elo cannot be newer than the forecast cutoff")
        _require_sha256(self.elo_state_sha256, "elo_state_sha256")
        if self.input_available_at < self.elo_available_at:
            raise NbaFeatureRowError("input_available_at cannot predate Elo")
        if self.input_available_at > self.forecast_cutoff:
            raise NbaFeatureRowError("inputs cannot be newer than the forecast cutoff")
        _require_sha256(self.evidence_bundle_sha256, "evidence_bundle_sha256")

    @property
    def feature_items(self) -> tuple[tuple[str, float], ...]:
        """Return the exact standard features in their frozen schema order."""
        return tuple(zip(NBA_RICH_FEATURE_NAMES, self.rich_features.vector, strict=True))

    def canonical_payload(self) -> dict[str, object]:
        """Return the small canonical payload covered by ``row_sha256``."""
        return {
            "question_id": self.question_id,
            "season": self.season,
            "state_id": self.state_id,
            "forecast_cutoff": _utc_text(self.forecast_cutoff),
            "scheduled_tipoff": _utc_text(self.scheduled_tipoff),
            "elo_team_win_probability": self.elo_team_win_probability,
            "elo_opponent_win_probability": self.elo_opponent_win_probability,
            "elo_available_at": _utc_text(self.elo_available_at),
            "elo_state_sha256": self.elo_state_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "features": [{"name": name, "value": value} for name, value in self.feature_items],
            "evidence_bundle_sha256": self.evidence_bundle_sha256,
            "input_available_at": _utc_text(self.input_available_at),
        }

    @property
    def row_sha256(self) -> str:
        """Hash the exact target-free row deterministically."""
        return canonical_sha256(self.canonical_payload())

    def side_swap(self) -> NbaRichFeatureRow:
        """Exchange sides while retaining the exact causal inputs and timestamps."""
        return NbaRichFeatureRow(
            question_id=_side_swap_question_id(self.question_id),
            season=self.season,
            forecast_cutoff=self.forecast_cutoff,
            scheduled_tipoff=self.scheduled_tipoff,
            elo_team_win_probability=self.elo_opponent_win_probability,
            elo_opponent_win_probability=self.elo_team_win_probability,
            elo_available_at=self.elo_available_at,
            elo_state_sha256=self.elo_state_sha256,
            rich_features=self.rich_features.side_swap(),
            evidence_bundle_sha256=self.evidence_bundle_sha256,
            input_available_at=self.input_available_at,
        )


def write_nba_feature_rows_jsonl(
    path: Path,
    rows: Iterable[NbaRichFeatureRow],
) -> None:
    """Create immutable, nonempty original T-60 rows as canonical JSONL."""
    checked_rows = _require_original_rows(tuple(rows))
    text = "".join(f"{canonical_json(row.canonical_payload())}\n" for row in checked_rows)
    try:
        with path.open("x", encoding="utf-8") as file:
            file.write(text)
    except FileExistsError as error:
        raise NbaFeatureRowError(
            "NBA feature-row file already exists; sealed rows cannot be replaced"
        ) from error


def read_nba_feature_rows_jsonl(path: Path) -> tuple[NbaRichFeatureRow, ...]:
    """Read and reconstruct strict canonical original T-60 feature rows."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise NbaFeatureRowError("cannot read NBA feature-row JSONL") from error

    rows: list[NbaRichFeatureRow] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            rows.append(_feature_row_from_payload(parse_json_object(line)))
        except ValueError as error:
            raise NbaFeatureRowError(f"invalid NBA feature row at line {line_number}") from error

    checked_rows = _require_original_rows(tuple(rows))
    canonical_text = "".join(f"{canonical_json(row.canonical_payload())}\n" for row in checked_rows)
    if text != canonical_text:
        raise NbaFeatureRowError("NBA feature rows must use canonical JSONL encoding")
    return checked_rows


def build_local_rich_feature_row(
    bundle: NbaEvidenceBundle,
    *,
    season: int,
    elo: NbaEloPriorInput,
    action_at: datetime,
) -> NbaRichFeatureRow:
    """Materialize one row under the evidence bundle's local-processing rights."""
    _require_elo_available_for_action(elo, action_at)
    features = local_rich_features_from_bundle(bundle, action_at=action_at)
    return _build_row(
        bundle,
        season=season,
        elo=elo,
        features=features,
    )


def build_tinker_rich_feature_row(
    bundle: NbaEvidenceBundle,
    *,
    season: int,
    elo: NbaEloPriorInput,
    action_at: datetime,
) -> NbaRichFeatureRow:
    """Materialize one row under the evidence bundle's Tinker-processing rights."""
    _require_elo_available_for_action(elo, action_at)
    features = tinker_rich_features_from_bundle(bundle, action_at=action_at)
    return _build_row(
        bundle,
        season=season,
        elo=elo,
        features=features,
    )


def _build_row(
    bundle: NbaEvidenceBundle,
    *,
    season: int,
    elo: NbaEloPriorInput,
    features: NbaRichFeatures,
) -> NbaRichFeatureRow:
    if bundle.game.question_id.endswith(SIDE_SWAP_SUFFIX):
        raise NbaFeatureRowError("source question_id cannot use the side-swap suffix")
    latest_evidence_at = max(record.available_at for record in bundle.records)
    return NbaRichFeatureRow(
        question_id=bundle.game.question_id,
        season=season,
        forecast_cutoff=bundle.game.forecast_deadline,
        scheduled_tipoff=bundle.game.scheduled_tipoff,
        elo_team_win_probability=elo.team_win_probability,
        elo_opponent_win_probability=1.0 - elo.team_win_probability,
        elo_available_at=elo.available_at,
        elo_state_sha256=elo.state_sha256,
        rich_features=features,
        evidence_bundle_sha256=evidence_bundle_sha256(bundle),
        input_available_at=max(elo.available_at, latest_evidence_at),
    )


def _feature_row_from_payload(payload: dict[str, object]) -> NbaRichFeatureRow:
    require_exact_keys(payload, _ROW_KEYS, "NBA feature row")
    state_id = require_string(required_field(payload, "state_id"), "state_id")
    if state_id != NBA_PRIMARY_STATE_ID:
        raise JsonFormatError(f"state_id must equal {NBA_PRIMARY_STATE_ID}")
    schema_sha256 = require_string(
        required_field(payload, "feature_schema_sha256"),
        "feature_schema_sha256",
    )
    if schema_sha256 != NBA_RICH_SCHEMA_SHA256:
        raise JsonFormatError("feature_schema_sha256 differs from the current schema")
    return NbaRichFeatureRow(
        question_id=require_string(required_field(payload, "question_id"), "question_id"),
        season=_require_json_integer(payload, "season"),
        forecast_cutoff=_parse_datetime(payload, "forecast_cutoff"),
        scheduled_tipoff=_parse_datetime(payload, "scheduled_tipoff"),
        elo_team_win_probability=require_float(
            required_field(payload, "elo_team_win_probability"),
            "elo_team_win_probability",
        ),
        elo_opponent_win_probability=require_float(
            required_field(payload, "elo_opponent_win_probability"),
            "elo_opponent_win_probability",
        ),
        elo_available_at=_parse_datetime(payload, "elo_available_at"),
        elo_state_sha256=require_string(
            required_field(payload, "elo_state_sha256"),
            "elo_state_sha256",
        ),
        rich_features=_parse_rich_features(payload),
        evidence_bundle_sha256=require_string(
            required_field(payload, "evidence_bundle_sha256"),
            "evidence_bundle_sha256",
        ),
        input_available_at=_parse_datetime(payload, "input_available_at"),
    )


def _parse_rich_features(payload: dict[str, object]) -> NbaRichFeatures:
    raw_features = require_list(required_field(payload, "features"), "features")
    if len(raw_features) != len(NBA_RICH_FEATURE_NAMES):
        raise JsonFormatError("features must contain the current richer schema exactly")

    values: list[float] = []
    for expected_name, raw_feature in zip(
        NBA_RICH_FEATURE_NAMES,
        raw_features,
        strict=True,
    ):
        feature = require_object(raw_feature, "feature")
        require_exact_keys(feature, _FEATURE_KEYS, "feature")
        name = require_string(required_field(feature, "name"), "feature.name")
        if name != expected_name:
            raise JsonFormatError("feature names or order differ from the current schema")
        values.append(require_float(required_field(feature, "value"), f"{name}.value"))

    return NbaRichFeatures.from_vector(tuple(values))


def _require_original_rows(
    rows: tuple[NbaRichFeatureRow, ...],
) -> tuple[NbaRichFeatureRow, ...]:
    if not rows:
        raise NbaFeatureRowError("NBA feature-row JSONL must not be empty")
    question_ids: set[str] = set()
    for row in rows:
        if row.question_id.endswith(SIDE_SWAP_SUFFIX):
            raise NbaFeatureRowError("NBA feature-row JSONL may contain only original rows")
        if row.question_id in question_ids:
            raise NbaFeatureRowError("NBA feature-row JSONL contains a duplicate question ID")
        question_ids.add(row.question_id)
    return rows


def _parse_datetime(payload: dict[str, object], field_name: str) -> datetime:
    value = require_string(required_field(payload, field_name), field_name)
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise JsonFormatError(f"{field_name} must be an ISO-8601 datetime") from error


def _require_json_integer(payload: dict[str, object], field_name: str) -> int:
    value = required_field(payload, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise JsonFormatError(f"{field_name} must be an integer")
    return value


def _require_positive_season(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NbaFeatureRowError("season must be a positive integer")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise NbaFeatureRowError(f"{field_name} must be in UTC")


def _utc_text(value: datetime) -> str:
    _require_utc(value, "datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in _HASH_CHARACTERS for character in value):
        raise NbaFeatureRowError(f"{field_name} must be a lowercase SHA-256 digest")


def _require_elo_probabilities(team: float, opponent: float) -> None:
    if not all(isfinite(value) and 0.0 < value < 1.0 for value in (team, opponent)):
        raise NbaFeatureRowError("Elo probabilities must be finite and interior")
    if not isclose(team + opponent, 1.0, rel_tol=0.0, abs_tol=1e-15):
        raise NbaFeatureRowError("Elo probabilities must sum to one")


def _require_elo_available_for_action(elo: NbaEloPriorInput, action_at: datetime) -> None:
    _require_utc(action_at, "action_at")
    if elo.available_at > action_at:
        raise NbaFeatureRowError("Elo cannot be newer than the protected action")


def _side_swap_question_id(question_id: str) -> str:
    if question_id.endswith(SIDE_SWAP_SUFFIX):
        return question_id.removesuffix(SIDE_SWAP_SUFFIX)
    return f"{question_id}{SIDE_SWAP_SUFFIX}"
