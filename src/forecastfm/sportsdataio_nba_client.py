"""One-attempt HTTPS retrieval for registered SportsDataIO NBA requests."""

from __future__ import annotations

import ssl
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.client import HTTPSConnection
from math import isfinite
from typing import Protocol, cast, final

from forecastfm.nba_raw_capture import (
    NBA_RAW_CAPTURE_MAX_BODY_BYTES,
    NbaRawCapture,
    NbaRawCaptureError,
    NbaRawCaptureRequest,
)
from forecastfm.sportsdataio_nba_openapi import (
    SPORTSDATAIO_NBA_HOST,
    SportsDataIONbaRequestError,
    require_registered_nba_request,
)

SPORTSDATAIO_API_KEY_HEADER = "Ocp-Apim-Subscription-Key"
SPORTSDATAIO_NBA_USER_AGENT = "forecastfm-sportsdataio-nba/0.1"
SPORTSDATAIO_MAX_TIMEOUT_SECONDS = 60.0

_MAX_RESPONSE_HEADERS = 256
_MAX_RESPONSE_HEADER_NAME_CHARACTERS = 256
_MAX_RESPONSE_HEADER_VALUE_CHARACTERS = 8192
_HTTP_HEADER_NAME_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!#$%&'*+-.^_`|~"
)

type HeaderPairs = tuple[tuple[str, str], ...]


class SportsDataIONbaClientError(ValueError):
    """Raised when a registered NBA retrieval cannot produce a safe local capture."""


@dataclass(frozen=True, slots=True)
class SportsDataIOHttpResponse:
    """One bounded response entity returned by a transport implementation."""

    status: int
    headers: HeaderPairs = field(repr=False)
    body: bytes = field(repr=False)


class SportsDataIOTransport(Protocol):
    """Minimal one-attempt transport seam used by deterministic tests."""

    def get(
        self,
        path: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_body_bytes: int,
    ) -> SportsDataIOHttpResponse:
        """Perform exactly one fixed-host HTTPS GET without redirects or retries."""
        ...


@final
class SportsDataIOHttpsTransport:
    """Stdlib HTTPS transport pinned to the SportsDataIO NBA API host."""

    __slots__ = ()

    def get(
        self,
        path: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_body_bytes: int,
    ) -> SportsDataIOHttpResponse:
        """Perform one certificate-verified request and read one bounded entity."""
        response = _try_https_get(path, headers, timeout_seconds, max_body_bytes)
        if response is None:
            raise SportsDataIONbaClientError("SportsDataIO retrieval failed")
        return response


def _try_https_get(
    path: str,
    headers: Mapping[str, str],
    timeout_seconds: float,
    max_body_bytes: int,
) -> SportsDataIOHttpResponse | None:
    """Return ``None`` for an untrusted transport exception without retaining it."""
    try:
        context = _verified_tls_context()
        connection = HTTPSConnection(
            SPORTSDATAIO_NBA_HOST,
            timeout=timeout_seconds,
            context=context,
        )
        try:
            connection.set_debuglevel(0)
            connection.request("GET", path, headers=headers)
            response = connection.getresponse()
            if type(response.status) is not int or response.status != 200:
                raise SportsDataIONbaClientError("SportsDataIO did not return HTTP 200")
            response_headers = _validated_response_headers(tuple(response.getheaders()))
            content_length = _require_response_framing(response_headers, max_body_bytes)
            body = response.read(max_body_bytes + 1)
            if type(body) is not bytes:
                raise SportsDataIONbaClientError("SportsDataIO response body is invalid")
            _require_body_size(body, max_body_bytes)
            if content_length is not None and content_length != len(body):
                raise SportsDataIONbaClientError("SportsDataIO response length differs")
            return SportsDataIOHttpResponse(
                status=response.status,
                headers=response_headers,
                body=body,
            )
        finally:
            connection.close()
    except Exception:
        return None


@final
class SportsDataIONbaClient:
    """Credential-holding client that emits only narrow local-capture envelopes."""

    __slots__ = ("_api_key", "_timeout_seconds", "_transport", "_utc_now")

    def __init__(
        self,
        api_key: str,
        transport: SportsDataIOTransport | None = None,
        utc_now: Callable[[], datetime] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        """Freeze one credential, transport, clock, and bounded timeout."""
        _require_api_key(api_key)
        _require_timeout(timeout_seconds)
        self._api_key = api_key
        self._transport = transport if transport is not None else SportsDataIOHttpsTransport()
        self._utc_now = utc_now if utc_now is not None else _system_utc_now
        self._timeout_seconds = timeout_seconds

    def capture(self, request: NbaRawCaptureRequest) -> NbaRawCapture:
        """Make one request and return its exact secret-screened response envelope."""
        if type(request) is not NbaRawCaptureRequest:
            raise SportsDataIONbaClientError("NBA request is outside the fixed registry")
        try:
            require_registered_nba_request(request)
        except SportsDataIONbaRequestError:
            raise SportsDataIONbaClientError("NBA request is outside the fixed registry") from None
        if _request_contains(request, self._api_key):
            raise SportsDataIONbaClientError("NBA request conflicts with the local credential")

        response = _try_transport_get(
            self._transport,
            request.path,
            _request_headers(self._api_key),
            self._timeout_seconds,
            NBA_RAW_CAPTURE_MAX_BODY_BYTES,
        )
        if response is None:
            raise SportsDataIONbaClientError("SportsDataIO retrieval failed")

        response = _validated_safe_response(
            response,
            self._api_key,
            NBA_RAW_CAPTURE_MAX_BODY_BYTES,
        )
        retrieved_at = _completion_time(self._utc_now)
        try:
            capture = NbaRawCapture.from_response(
                request=request,
                retrieved_at=retrieved_at,
                response_headers=response.headers,
                body=response.body,
                http_status=response.status,
            )
        except NbaRawCaptureError:
            raise SportsDataIONbaClientError("SportsDataIO response cannot be captured") from None
        if self._api_key.encode("ascii") in capture.canonical_bytes:
            raise SportsDataIONbaClientError("SportsDataIO response reflected the credential")
        return capture


def _request_headers(api_key: str) -> Mapping[str, str]:
    return {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "Connection": "close",
        "User-Agent": SPORTSDATAIO_NBA_USER_AGENT,
        SPORTSDATAIO_API_KEY_HEADER: api_key,
    }


def _request_contains(request: NbaRawCaptureRequest, value: str) -> bool:
    return value in request.operation or value in request.host or value in request.path


def _try_transport_get(
    transport: SportsDataIOTransport,
    path: str,
    headers: Mapping[str, str],
    timeout_seconds: float,
    max_body_bytes: int,
) -> SportsDataIOHttpResponse | None:
    """Discard an untrusted transport exception before the caller raises."""
    try:
        return transport.get(path, headers, timeout_seconds, max_body_bytes)
    except Exception:
        return None


def _verified_tls_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    if not context.check_hostname or context.verify_mode != ssl.CERT_REQUIRED:
        raise SportsDataIONbaClientError("certificate verification is unavailable")
    return context


def _validated_safe_response(
    response: object,
    api_key: str,
    max_body_bytes: int,
) -> SportsDataIOHttpResponse:
    if type(response) is not SportsDataIOHttpResponse:
        raise SportsDataIONbaClientError("SportsDataIO response is invalid")
    if type(response.status) is not int or response.status != 200:
        raise SportsDataIONbaClientError("SportsDataIO did not return HTTP 200")
    if type(response.body) is not bytes:
        raise SportsDataIONbaClientError("SportsDataIO response body is invalid")
    _require_body_size(response.body, max_body_bytes)
    headers = _validated_response_headers(response.headers)
    content_length = _require_response_framing(headers, max_body_bytes)
    if content_length is not None and content_length != len(response.body):
        raise SportsDataIONbaClientError("SportsDataIO response length differs")
    if api_key.encode("ascii") in response.body or _headers_contain(headers, api_key):
        raise SportsDataIONbaClientError("SportsDataIO response reflected the credential")
    return SportsDataIOHttpResponse(status=response.status, headers=headers, body=response.body)


def _require_response_framing(
    headers: HeaderPairs,
    max_body_bytes: int,
) -> int | None:
    content_encoding = _header_values(headers, "content-encoding")
    if content_encoding:
        raise SportsDataIONbaClientError("SportsDataIO response encoding is unsupported")

    content_lengths = _header_values(headers, "content-length")
    if len(content_lengths) > 1:
        raise SportsDataIONbaClientError("SportsDataIO returned ambiguous response length")
    content_length = None
    if content_lengths:
        content_length = _canonical_content_length(content_lengths[0], max_body_bytes)

    transfer_encodings = _header_values(headers, "transfer-encoding")
    if len(transfer_encodings) > 1:
        raise SportsDataIONbaClientError("SportsDataIO returned ambiguous transfer encoding")
    if transfer_encodings and (transfer_encodings[0] != "chunked" or content_length is not None):
        raise SportsDataIONbaClientError("SportsDataIO transfer framing is unsupported")
    return content_length


def _validated_response_headers(headers: object) -> HeaderPairs:
    if type(headers) is not tuple:
        raise SportsDataIONbaClientError("SportsDataIO response headers are invalid")
    raw_headers = cast(tuple[object, ...], headers)
    if len(raw_headers) > _MAX_RESPONSE_HEADERS:
        raise SportsDataIONbaClientError("SportsDataIO response headers are invalid")
    validated: list[tuple[str, str]] = []
    for raw_pair in raw_headers:
        if type(raw_pair) is not tuple:
            raise SportsDataIONbaClientError("SportsDataIO response headers are invalid")
        pair = cast(tuple[object, ...], raw_pair)
        if len(pair) != 2:
            raise SportsDataIONbaClientError("SportsDataIO response headers are invalid")
        name, value = pair
        if (
            type(name) is not str
            or not name
            or len(name) > _MAX_RESPONSE_HEADER_NAME_CHARACTERS
            or any(character not in _HTTP_HEADER_NAME_CHARACTERS for character in name)
            or type(value) is not str
            or len(value) > _MAX_RESPONSE_HEADER_VALUE_CHARACTERS
            or not value.isascii()
            or any(ord(character) < 32 or ord(character) > 126 for character in value)
        ):
            raise SportsDataIONbaClientError("SportsDataIO response headers are invalid")
        validated.append((name, value))
    return tuple(validated)


def _canonical_content_length(value: str, max_body_bytes: int) -> int:
    if (
        not value
        or len(value) > len(str(max_body_bytes))
        or not value.isascii()
        or not value.isdecimal()
    ):
        raise SportsDataIONbaClientError("SportsDataIO response length is invalid")
    result = int(value)
    if value != str(result) or result > max_body_bytes:
        raise SportsDataIONbaClientError("SportsDataIO response length is invalid")
    return result


def _header_values(headers: HeaderPairs, expected_name: str) -> tuple[str, ...]:
    return tuple(value for name, value in headers if name.lower() == expected_name)


def _headers_contain(headers: HeaderPairs, value: str) -> bool:
    return any(value in name or value in header_value for name, header_value in headers)


def _require_body_size(body: bytes, max_body_bytes: int) -> None:
    if not body or len(body) > max_body_bytes:
        raise SportsDataIONbaClientError("SportsDataIO response body size is invalid")


def _require_api_key(value: str) -> None:
    if (
        type(value) is not str
        or not value
        or len(value) > 512
        or not value.isascii()
        or any(ord(character) < 33 or ord(character) > 126 for character in value)
    ):
        raise SportsDataIONbaClientError("SportsDataIO API key is invalid")


def _require_timeout(value: float) -> None:
    if (
        type(value) not in {int, float}
        or not isfinite(float(value))
        or not 0.0 < float(value) <= SPORTSDATAIO_MAX_TIMEOUT_SECONDS
    ):
        raise SportsDataIONbaClientError("SportsDataIO timeout is invalid")


def _completion_time(utc_now: Callable[[], datetime]) -> datetime:
    value = _try_utc_now(utc_now)
    if value is None:
        raise SportsDataIONbaClientError("local retrieval time is unavailable")
    if type(value) is not datetime or value.tzinfo is not UTC:
        raise SportsDataIONbaClientError("local retrieval time must be in UTC")
    return value


def _try_utc_now(utc_now: Callable[[], datetime]) -> datetime | None:
    """Discard an untrusted clock exception before the caller raises."""
    try:
        return utc_now()
    except Exception:
        return None


def _system_utc_now() -> datetime:
    return datetime.now(UTC)
