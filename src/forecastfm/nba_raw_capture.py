"""Create-only, caller-asserted envelopes for raw NBA response buffers."""

from __future__ import annotations

import base64
import binascii
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile

from forecastfm.integrity import bytes_sha256, canonical_json
from forecastfm.json_utils import (
    JsonFormatError,
    parse_json_object,
    require_exact_keys,
    require_list,
    require_object,
    require_string,
    required_field,
)

NBA_RAW_CAPTURE_SCHEMA_VERSION = 1
NBA_RAW_CAPTURE_KIND = "forecastfm_nba_raw_http_capture"
NBA_RAW_CAPTURE_CLAIM = "local_retrieval_only"
NBA_RAW_CAPTURE_CLOCK_AUTHORITY = "caller_asserted_local_system_clock_untrusted"
NBA_RAW_CAPTURE_NOT_ESTABLISHED = "not_established"
NBA_RAW_CAPTURE_REQUEST_IDENTITY_AUTHORITY = "caller_asserted_unverified"
NBA_RAW_CAPTURE_BODY_REPRESENTATION = "caller_supplied_http_response_entity_bytes"
NBA_RAW_CAPTURE_MAX_BODY_BYTES = 32 * 1024 * 1024
NBA_RAW_CAPTURE_MAX_ARTIFACT_BYTES = 48 * 1024 * 1024
NBA_RAW_CAPTURE_PROOF_SCOPE = (
    "exact caller-supplied response-entity bytes and selected normalized response fields are bound "
    "to a caller-asserted request identity and UTC time; network transport, provider identity, "
    "wire bytes, publication time, archive authenticity, revision completeness, rights, historical "
    "cutoff eligibility, and model authorization are not proven"
)

_TOP_LEVEL_KEYS = {
    "body_base64",
    "body_length",
    "body_representation",
    "body_sha256",
    "claim",
    "clock_authority",
    "host",
    "http_status",
    "kind",
    "method",
    "operation",
    "path",
    "proof_scope",
    "provider_publication_time",
    "provider_identity",
    "request_identity_authority",
    "response_headers",
    "retrieved_at",
    "revision_authenticity",
    "schema_version",
    "transport_authenticity",
}
_PAIR_KEYS = {"name", "value"}
_SAFE_RESPONSE_HEADERS = frozenset(
    {
        "cache-control",
        "content-type",
        "date",
        "etag",
        "last-modified",
        "x-request-id",
    }
)
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_HOST_PATTERN = re.compile(
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?"
)
_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_HEADER_NAME_PATTERN = re.compile(r"[a-z0-9!#$%&'*+.^_`|~-]+")
_MAX_OPERATION_CHARACTERS = 100
_MAX_PATH_CHARACTERS = 2048
_MAX_HEADER_VALUE_CHARACTERS = 4096
_MAX_SELECTED_RESPONSE_HEADERS = 32
_MAX_BODY_BASE64_CHARACTERS = ((NBA_RAW_CAPTURE_MAX_BODY_BYTES + 2) // 3) * 4

type HeaderPairs = tuple[tuple[str, str], ...]
type JsonObject = dict[str, object]


class NbaRawCaptureError(ValueError):
    """Raised when a raw local provider capture is unsafe or inconsistent."""


class NbaRawCapturePublicationUncertainError(NbaRawCaptureError):
    """Raised when publication may have succeeded but directory sync failed."""


@dataclass(frozen=True, slots=True)
class NbaRawCaptureRequest:
    """Syntactically safe caller assertion about one completed GET request."""

    operation: str
    host: str
    path: str

    def __post_init__(self) -> None:
        _require_operation(self.operation)
        _require_host(self.host)
        _require_path(self.path)


@dataclass(frozen=True, slots=True)
class NbaRawCapture:
    """One exact response buffer with explicitly local-only provenance."""

    request: NbaRawCaptureRequest
    retrieved_at: datetime
    response_headers: HeaderPairs
    body: bytes = field(repr=False)
    http_status: int = 200

    def __post_init__(self) -> None:
        _require_utc(self.retrieved_at, "retrieved_at")
        if isinstance(self.http_status, bool) or self.http_status != 200:
            raise NbaRawCaptureError("raw captures require HTTP status 200")
        if type(self.body) is not bytes or not self.body:
            raise NbaRawCaptureError("raw capture body must be nonempty immutable bytes")
        if len(self.body) > NBA_RAW_CAPTURE_MAX_BODY_BYTES:
            raise NbaRawCaptureError("raw capture body exceeds the local artifact limit")
        _require_selected_headers(self.response_headers)

    @classmethod
    def from_response(
        cls,
        request: NbaRawCaptureRequest,
        retrieved_at: datetime,
        response_headers: Sequence[tuple[str, str]],
        body: bytes,
        http_status: int = 200,
    ) -> NbaRawCapture:
        """Select allowlisted metadata from one caller-supplied completed response."""
        return cls(
            request=request,
            retrieved_at=retrieved_at,
            response_headers=_select_response_headers(response_headers),
            body=bytes(body),
            http_status=http_status,
        )

    @property
    def canonical_bytes(self) -> bytes:
        """Return the complete canonical capture artifact bytes."""
        return f"{canonical_json(self.canonical_payload())}\n".encode()

    @property
    def sha256(self) -> str:
        """Hash the exact self-contained artifact bytes."""
        return bytes_sha256(self.canonical_bytes)

    def canonical_payload(self) -> JsonObject:
        """Return the narrow caller-asserted local capture record."""
        body_sha256 = bytes_sha256(self.body)
        return {
            "schema_version": NBA_RAW_CAPTURE_SCHEMA_VERSION,
            "kind": NBA_RAW_CAPTURE_KIND,
            "claim": NBA_RAW_CAPTURE_CLAIM,
            "proof_scope": NBA_RAW_CAPTURE_PROOF_SCOPE,
            "clock_authority": NBA_RAW_CAPTURE_CLOCK_AUTHORITY,
            "request_identity_authority": NBA_RAW_CAPTURE_REQUEST_IDENTITY_AUTHORITY,
            "transport_authenticity": NBA_RAW_CAPTURE_NOT_ESTABLISHED,
            "provider_identity": NBA_RAW_CAPTURE_NOT_ESTABLISHED,
            "provider_publication_time": NBA_RAW_CAPTURE_NOT_ESTABLISHED,
            "revision_authenticity": NBA_RAW_CAPTURE_NOT_ESTABLISHED,
            "method": "GET",
            "operation": self.request.operation,
            "host": self.request.host,
            "path": self.request.path,
            "retrieved_at": _utc_text(self.retrieved_at),
            "http_status": self.http_status,
            "response_headers": _pairs_payload(self.response_headers),
            "body_representation": NBA_RAW_CAPTURE_BODY_REPRESENTATION,
            "body_length": len(self.body),
            "body_sha256": body_sha256,
            "body_base64": base64.b64encode(self.body).decode("ascii"),
        }


def write_nba_raw_capture(capture: NbaRawCapture, path: Path, storage_root: Path) -> None:
    """Create one artifact under a caller-selected existing restricted storage root."""
    _require_storage_target(path, storage_root)
    payload = capture.canonical_bytes
    partial_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".part",
            delete=False,
        ) as file:
            partial_path = Path(file.name)
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(partial_path, path)
        except FileExistsError as error:
            raise NbaRawCaptureError("raw capture already exists") from error
        try:
            _fsync_directory(path.parent)
        except OSError as error:
            raise NbaRawCapturePublicationUncertainError(
                "raw capture may have been published; verify the target before retrying"
            ) from error
    except NbaRawCaptureError:
        raise
    except OSError as error:
        raise NbaRawCaptureError("cannot publish raw capture") from error
    finally:
        if partial_path is not None:
            partial_path.unlink(missing_ok=True)


def load_nba_raw_capture(path: Path) -> NbaRawCapture:
    """Read and verify one canonical raw-capture artifact from a single byte buffer."""
    try:
        value = path.read_bytes()
    except OSError as error:
        raise NbaRawCaptureError("cannot read raw capture") from error
    return load_nba_raw_capture_bytes(value)


def load_nba_raw_capture_bytes(value: bytes) -> NbaRawCapture:
    """Verify canonical raw-capture bytes without another filesystem read."""
    if len(value) > NBA_RAW_CAPTURE_MAX_ARTIFACT_BYTES:
        raise NbaRawCaptureError("raw capture exceeds the local artifact limit")
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise NbaRawCaptureError("raw capture must use UTF-8") from error
    if not text or not text.endswith("\n") or text.endswith("\n\n"):
        raise NbaRawCaptureError("raw capture must end with exactly one newline")
    try:
        record = parse_json_object(text[:-1])
        capture = _capture_from_record(record)
    except (JsonFormatError, ValueError) as error:
        raise NbaRawCaptureError("invalid raw capture record") from error
    if capture.canonical_bytes != value:
        raise NbaRawCaptureError("raw capture is not canonical")
    return capture


def _capture_from_record(record: Mapping[str, object]) -> NbaRawCapture:
    require_exact_keys(record, _TOP_LEVEL_KEYS, "raw capture")
    if _integer(record, "schema_version") != NBA_RAW_CAPTURE_SCHEMA_VERSION:
        raise NbaRawCaptureError("raw capture schema_version is unsupported")
    _require_constant(record, "kind", NBA_RAW_CAPTURE_KIND)
    _require_constant(record, "claim", NBA_RAW_CAPTURE_CLAIM)
    _require_constant(record, "proof_scope", NBA_RAW_CAPTURE_PROOF_SCOPE)
    _require_constant(record, "clock_authority", NBA_RAW_CAPTURE_CLOCK_AUTHORITY)
    _require_constant(
        record,
        "request_identity_authority",
        NBA_RAW_CAPTURE_REQUEST_IDENTITY_AUTHORITY,
    )
    _require_constant(record, "transport_authenticity", NBA_RAW_CAPTURE_NOT_ESTABLISHED)
    _require_constant(record, "provider_identity", NBA_RAW_CAPTURE_NOT_ESTABLISHED)
    _require_constant(
        record,
        "provider_publication_time",
        NBA_RAW_CAPTURE_NOT_ESTABLISHED,
    )
    _require_constant(record, "revision_authenticity", NBA_RAW_CAPTURE_NOT_ESTABLISHED)
    _require_constant(record, "body_representation", NBA_RAW_CAPTURE_BODY_REPRESENTATION)
    _require_constant(record, "method", "GET")

    body = _decode_body(_string(record, "body_base64"))
    if _integer(record, "body_length") != len(body):
        raise NbaRawCaptureError("raw capture body length differs")
    body_sha256 = _string(record, "body_sha256")
    _require_sha256(body_sha256, "body_sha256")
    if body_sha256 != bytes_sha256(body):
        raise NbaRawCaptureError("raw capture body hash differs")

    return NbaRawCapture(
        request=NbaRawCaptureRequest(
            operation=_string(record, "operation"),
            host=_string(record, "host"),
            path=_string(record, "path"),
        ),
        retrieved_at=_datetime(record, "retrieved_at"),
        response_headers=_pairs(record, "response_headers"),
        body=body,
        http_status=_integer(record, "http_status"),
    )


def _select_response_headers(headers: Sequence[tuple[str, str]]) -> HeaderPairs:
    selected: list[tuple[str, str]] = []
    for raw_name, value in headers:
        name = raw_name.lower()
        _require_header_name(name)
        if name not in _SAFE_RESPONSE_HEADERS:
            continue
        _require_header_value(value)
        selected.append((name, value))
        if len(selected) > _MAX_SELECTED_RESPONSE_HEADERS:
            raise NbaRawCaptureError("too many selected response headers")
    return tuple(sorted(selected))


def _require_selected_headers(headers: HeaderPairs) -> None:
    if len(headers) > _MAX_SELECTED_RESPONSE_HEADERS:
        raise NbaRawCaptureError("too many selected response headers")
    if tuple(sorted(headers)) != headers:
        raise NbaRawCaptureError("selected response headers must be sorted")
    for name, value in headers:
        _require_header_name(name)
        if name not in _SAFE_RESPONSE_HEADERS:
            raise NbaRawCaptureError("response header is not allowlisted")
        _require_header_value(value)


def _require_header_name(value: str) -> None:
    if not value or value != value.lower() or _HEADER_NAME_PATTERN.fullmatch(value) is None:
        raise NbaRawCaptureError("response header name is unsafe")


def _require_header_value(value: str) -> None:
    if (
        not value
        or len(value) > _MAX_HEADER_VALUE_CHARACTERS
        or value != value.strip()
        or not value.isascii()
        or any(ord(character) < 32 or ord(character) > 126 for character in value)
    ):
        raise NbaRawCaptureError("response header value is unsafe")


def _require_operation(value: str) -> None:
    if len(value) > _MAX_OPERATION_CHARACTERS or _NAME_PATTERN.fullmatch(value) is None:
        raise NbaRawCaptureError("operation must be a lowercase identifier")


def _require_host(value: str) -> None:
    if len(value) > 253 or value != value.lower() or _HOST_PATTERN.fullmatch(value) is None:
        raise NbaRawCaptureError("host must be a lowercase DNS-shaped name")


def _require_path(value: str) -> None:
    if (
        len(value) > _MAX_PATH_CHARACTERS
        or not value.startswith("/")
        or value.startswith("//")
        or value.endswith("/")
        or not value.isascii()
        or any(ord(character) < 33 or ord(character) > 126 for character in value)
        or any(character in value for character in ("\\", "?", "#", "%", "@", "&", "="))
        or "://" in value
        or any(segment in {"", ".", ".."} for segment in value.split("/")[1:])
    ):
        raise NbaRawCaptureError("path must be a safe absolute path without a query")


def _pairs_payload(pairs: HeaderPairs) -> list[JsonObject]:
    return [{"name": name, "value": value} for name, value in pairs]


def _pairs(record: Mapping[str, object], field_name: str) -> HeaderPairs:
    values = require_list(required_field(record, field_name), field_name)
    pairs: list[tuple[str, str]] = []
    for index, value in enumerate(values):
        pair = require_object(value, f"{field_name}[{index}]")
        require_exact_keys(pair, _PAIR_KEYS, f"{field_name}[{index}]")
        pairs.append(
            (
                require_string(required_field(pair, "name"), "name"),
                require_string(required_field(pair, "value"), "value"),
            )
        )
    return tuple(pairs)


def _string(record: Mapping[str, object], field_name: str) -> str:
    return require_string(required_field(record, field_name), field_name)


def _integer(record: Mapping[str, object], field_name: str) -> int:
    value = required_field(record, field_name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise NbaRawCaptureError(f"{field_name} must be an integer")
    return value


def _datetime(record: Mapping[str, object], field_name: str) -> datetime:
    text = _string(record, field_name)
    if not text.endswith("Z"):
        raise NbaRawCaptureError(f"{field_name} must use canonical UTC")
    try:
        value = datetime.fromisoformat(f"{text[:-1]}+00:00")
    except ValueError as error:
        raise NbaRawCaptureError(f"{field_name} must be a datetime") from error
    _require_utc(value, field_name)
    if _utc_text(value) != text:
        raise NbaRawCaptureError(f"{field_name} must use canonical UTC")
    return value


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise NbaRawCaptureError(f"{field_name} must be in UTC")


def _utc_text(value: datetime) -> str:
    _require_utc(value, "datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _require_sha256(value: str, field_name: str) -> None:
    if _HASH_PATTERN.fullmatch(value) is None:
        raise NbaRawCaptureError(f"{field_name} must be a lowercase SHA-256 digest")


def _decode_body(value: str) -> bytes:
    if len(value) > _MAX_BODY_BASE64_CHARACTERS:
        raise NbaRawCaptureError("body_base64 exceeds the local artifact limit")
    try:
        body = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise NbaRawCaptureError("body_base64 is invalid") from error
    if base64.b64encode(body).decode("ascii") != value:
        raise NbaRawCaptureError("body_base64 is not canonical")
    return body


def _require_constant(record: Mapping[str, object], field_name: str, expected: object) -> None:
    if required_field(record, field_name) != expected:
        raise NbaRawCaptureError(f"raw capture {field_name} is unsupported")


def _require_storage_target(path: Path, storage_root: Path) -> None:
    root = storage_root.absolute()
    try:
        root_stat = storage_root.stat()
        resolved_root = storage_root.resolve()
    except OSError as error:
        raise NbaRawCaptureError("cannot validate storage_root") from error
    if (
        not storage_root.exists()
        or not storage_root.is_dir()
        or storage_root.is_symlink()
        or resolved_root != root
        or root_stat.st_uid != os.getuid()
        or stat.S_IMODE(root_stat.st_mode) & 0o077 != 0
    ):
        raise NbaRawCaptureError(
            "storage_root must be an owned, private, existing nonsymlink directory"
        )
    target = path.absolute()
    if target.parent != root or path.name.startswith(".") or path.suffix != ".json":
        raise NbaRawCaptureError(
            "raw capture path must be one JSON file directly under storage_root"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
