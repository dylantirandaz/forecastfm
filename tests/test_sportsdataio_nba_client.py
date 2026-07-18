"""Tests for the fixed-host SportsDataIO NBA capture client."""

from __future__ import annotations

import ssl
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, timezone, tzinfo
from typing import cast, override

import pytest

import forecastfm.sportsdataio_nba_client as client_module
from forecastfm.nba_raw_capture import (
    NBA_RAW_CAPTURE_MAX_BODY_BYTES,
    NBA_RAW_CAPTURE_NOT_ESTABLISHED,
    NbaRawCaptureRequest,
)
from forecastfm.sportsdataio_nba_client import (
    SPORTSDATAIO_API_KEY_HEADER,
    SPORTSDATAIO_MAX_TIMEOUT_SECONDS,
    SPORTSDATAIO_NBA_USER_AGENT,
    SportsDataIOHttpResponse,
    SportsDataIOHttpsTransport,
    SportsDataIONbaClient,
    SportsDataIONbaClientError,
)
from forecastfm.sportsdataio_nba_openapi import (
    SPORTSDATAIO_NBA_HOST,
    SportsDataIONbaSeason,
    depth_charts_request,
    games_by_date_final_request,
    games_request,
    injured_players_request,
    player_game_stats_by_date_request,
    starting_lineups_by_date_request,
    team_game_stats_by_season_request,
    transactions_by_date_request,
)

_API_KEY = "test-key-not-a-real-credential-123"
_BODY = b'[{"GameID":1}]'
_NOW = datetime(2026, 7, 17, 12, 30, tzinfo=UTC)


class _RequestSubclass(NbaRawCaptureRequest):
    pass


class _HostileTzInfo(tzinfo):
    @override
    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        raise RuntimeError(f"timezone repeated {_API_KEY}")

    @override
    def dst(self, dt: datetime | None) -> timedelta | None:
        return timedelta(0)

    @override
    def tzname(self, dt: datetime | None) -> str | None:
        return "hostile"


def _transport_calls() -> list[tuple[str, dict[str, str], float, int]]:
    return []


def _header_list() -> list[tuple[str, str]]:
    return []


def _integer_list() -> list[int]:
    return []


def _request_list() -> list[tuple[str, str, dict[str, str]]]:
    return []


def _factory_calls() -> list[tuple[str, float, ssl.SSLContext]]:
    return []


@dataclass(slots=True)
class _FakeTransport:
    response: SportsDataIOHttpResponse = field(
        default_factory=lambda: SportsDataIOHttpResponse(
            status=200,
            headers=(("Content-Length", str(len(_BODY))),),
            body=_BODY,
        )
    )
    error: Exception | None = None
    events: list[str] | None = None
    calls: list[tuple[str, dict[str, str], float, int]] = field(default_factory=_transport_calls)

    def get(
        self,
        path: str,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_body_bytes: int,
    ) -> SportsDataIOHttpResponse:
        """Record one deterministic transport call."""
        self.calls.append((path, dict(headers), timeout_seconds, max_body_bytes))
        if self.events is not None:
            self.events.append("transport")
        if self.error is not None:
            raise self.error
        return self.response


def _fixed_utc_now() -> datetime:
    return _NOW


def _client(
    transport: _FakeTransport,
    *,
    api_key: str = _API_KEY,
    utc_now: Callable[[], datetime] = _fixed_utc_now,
    timeout_seconds: float = 30.0,
) -> SportsDataIONbaClient:
    return SportsDataIONbaClient(
        api_key,
        transport=transport,
        utc_now=utc_now,
        timeout_seconds=timeout_seconds,
    )


def _response(
    *,
    status: int = 200,
    headers: tuple[tuple[str, str], ...] = (),
    body: bytes = _BODY,
) -> SportsDataIOHttpResponse:
    return SportsDataIOHttpResponse(status=status, headers=headers, body=body)


def _assert_secret_free(error: BaseException, secret: str) -> None:
    current: BaseException | None = error
    while current is not None:
        assert secret not in str(current)
        assert secret not in repr(current)
        current = current.__cause__ or current.__context__


def test_client_returns_narrow_capture_after_one_registered_get() -> None:
    transport = _FakeTransport(
        response=_response(
            headers=(
                ("Content-Length", str(len(_BODY))),
                ("Content-Type", "application/json"),
                ("X-Request-ID", "request-1"),
                ("Server", "ignored"),
            )
        )
    )
    request = games_request(SportsDataIONbaSeason(2025))

    capture = _client(transport).capture(request)

    assert transport.calls == [
        (
            request.path,
            {
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Connection": "close",
                "User-Agent": SPORTSDATAIO_NBA_USER_AGENT,
                SPORTSDATAIO_API_KEY_HEADER: _API_KEY,
            },
            30.0,
            NBA_RAW_CAPTURE_MAX_BODY_BYTES,
        )
    ]
    assert capture.request == request
    assert capture.retrieved_at == _NOW
    assert capture.body == _BODY
    assert capture.response_headers == (
        ("content-type", "application/json"),
        ("x-request-id", "request-1"),
    )
    payload = capture.canonical_payload()
    assert payload["transport_authenticity"] == NBA_RAW_CAPTURE_NOT_ESTABLISHED
    assert payload["provider_identity"] == NBA_RAW_CAPTURE_NOT_ESTABLISHED
    assert payload["provider_publication_time"] == NBA_RAW_CAPTURE_NOT_ESTABLISHED
    assert payload["revision_authenticity"] == NBA_RAW_CAPTURE_NOT_ESTABLISHED
    assert _API_KEY.encode() not in capture.canonical_bytes


def test_client_uses_the_clock_only_after_transport_completion() -> None:
    events: list[str] = []
    transport = _FakeTransport(events=events)

    def utc_now() -> datetime:
        events.append("clock")
        return _NOW

    _client(transport, utc_now=utc_now).capture(depth_charts_request())

    assert events == ["transport", "clock"]


@pytest.mark.parametrize(
    "nba_request",
    [
        games_request(SportsDataIONbaSeason(2025)),
        games_by_date_final_request(date(2026, 1, 2)),
        depth_charts_request(),
        transactions_by_date_request(date(2026, 1, 2)),
        starting_lineups_by_date_request(date(2026, 1, 2)),
        injured_players_request(),
        team_game_stats_by_season_request(SportsDataIONbaSeason(2025), 7, "all"),
        player_game_stats_by_date_request(date(2026, 1, 2)),
    ],
)
def test_client_accepts_each_registered_request(nba_request: NbaRawCaptureRequest) -> None:
    transport = _FakeTransport()

    _client(transport).capture(nba_request)

    assert [call[0] for call in transport.calls] == [nba_request.path]


@pytest.mark.parametrize(
    "nba_request",
    [
        NbaRawCaptureRequest("games", "example.com", "/v3/nba/scores/json/Games/2025"),
        NbaRawCaptureRequest("other", SPORTSDATAIO_NBA_HOST, "/v3/nba/scores/json/Games/2025"),
        NbaRawCaptureRequest(
            "games", SPORTSDATAIO_NBA_HOST, "/v3/nba/scores/json/Games/2025/extra"
        ),
    ],
)
def test_client_rejects_unregistered_request_before_transport(
    nba_request: NbaRawCaptureRequest,
) -> None:
    transport = _FakeTransport()

    with pytest.raises(SportsDataIONbaClientError, match="fixed registry"):
        _client(transport).capture(nba_request)

    assert transport.calls == []


def test_client_rejects_key_text_in_request_before_transport() -> None:
    transport = _FakeTransport()
    request = games_request(SportsDataIONbaSeason(2025))

    with pytest.raises(SportsDataIONbaClientError, match="conflicts with the local credential"):
        _client(transport, api_key="2025").capture(request)

    assert transport.calls == []


def test_client_rejects_request_subclass_before_reading_or_transport() -> None:
    transport = _FakeTransport()
    request = _RequestSubclass(
        operation="games",
        host=SPORTSDATAIO_NBA_HOST,
        path="/v3/nba/scores/json/Games/2025",
    )

    with pytest.raises(SportsDataIONbaClientError, match="fixed registry"):
        _client(transport).capture(request)

    assert transport.calls == []


@pytest.mark.parametrize(
    "api_key",
    [
        "",
        " ",
        " leading",
        "trailing ",
        "line\nbreak",
        "non-ascii-π",
        "x" * 513,
        cast(str, 7),
        cast(str, b"bytes"),
    ],
)
def test_client_rejects_invalid_key_before_transport(api_key: str) -> None:
    transport = _FakeTransport()

    with pytest.raises(SportsDataIONbaClientError, match="API key is invalid") as caught:
        _client(transport, api_key=api_key)

    assert transport.calls == []
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "timeout_seconds",
    [
        0.0,
        -1.0,
        SPORTSDATAIO_MAX_TIMEOUT_SECONDS + 0.1,
        float("inf"),
        float("-inf"),
        float("nan"),
        cast(float, True),
        cast(float, "30"),
    ],
)
def test_client_rejects_invalid_timeout_before_transport(timeout_seconds: float) -> None:
    transport = _FakeTransport()

    with pytest.raises(SportsDataIONbaClientError, match="timeout is invalid"):
        _client(transport, timeout_seconds=timeout_seconds)

    assert transport.calls == []


def test_client_discards_secret_bearing_transport_exception() -> None:
    transport = _FakeTransport(error=RuntimeError(f"provider echoed {_API_KEY}"))

    with pytest.raises(SportsDataIONbaClientError, match="retrieval failed") as caught:
        _client(transport).capture(depth_charts_request())

    assert len(transport.calls) == 1
    _assert_secret_free(caught.value, _API_KEY)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_client_rejects_wrong_transport_response_type() -> None:
    response = cast(SportsDataIOHttpResponse, object())
    transport = _FakeTransport(response=response)

    with pytest.raises(SportsDataIONbaClientError, match="response is invalid"):
        _client(transport).capture(depth_charts_request())

    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    ("headers", "body"),
    [
        ((), _API_KEY.encode()),
        (((f"x-{_API_KEY}", "value"),), _BODY),
        ((("x-reflection", f"prefix-{_API_KEY}-suffix"),), _BODY),
    ],
)
def test_client_rejects_credential_reflection(
    headers: tuple[tuple[str, str], ...],
    body: bytes,
) -> None:
    transport = _FakeTransport(response=_response(headers=headers, body=body))

    with pytest.raises(SportsDataIONbaClientError, match="reflected the credential") as caught:
        _client(transport).capture(depth_charts_request())

    _assert_secret_free(caught.value, _API_KEY)


def test_client_rejects_credential_text_created_by_base64_artifact() -> None:
    api_key = "YWJj"
    transport = _FakeTransport(response=_response(body=b"abc"))

    with pytest.raises(SportsDataIONbaClientError, match="reflected the credential"):
        _client(transport, api_key=api_key).capture(depth_charts_request())


@pytest.mark.parametrize("status", [cast(int, True), cast(int, 200.0), 201, 301, 307, 400, 500])
def test_client_rejects_every_non_200_without_retry(status: int) -> None:
    transport = _FakeTransport(response=_response(status=status))

    with pytest.raises(SportsDataIONbaClientError, match="did not return HTTP 200"):
        _client(transport).capture(depth_charts_request())

    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "headers",
    [
        cast(tuple[tuple[str, str], ...], []),
        cast(tuple[tuple[str, str], ...], (["Content-Type", "application/json"],)),
        cast(tuple[tuple[str, str], ...], ((7, "value"),)),
        (("Bad Header", "value"),),
        (("X-Test", "line\nbreak"),),
        (("X-Test", "non-ascii-π"),),
        (("X-Test", "x" * 8193),),
        tuple((f"X-Test-{index}", "value") for index in range(257)),
    ],
)
def test_client_rejects_malformed_or_unbounded_response_headers(
    headers: tuple[tuple[str, str], ...],
) -> None:
    transport = _FakeTransport(response=_response(headers=headers))

    with pytest.raises(SportsDataIONbaClientError, match="headers are invalid"):
        _client(transport).capture(depth_charts_request())

    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "headers",
    [
        (("Content-Encoding", "identity"),),
        (("Content-Encoding", "gzip"),),
        (("Content-Encoding", "identity"), ("Content-Encoding", "identity")),
        (("Content-Length", ""),),
        (("Content-Length", "+15"),),
        (("Content-Length", "015"),),
        (("Content-Length", "15, 15"),),
        (("Content-Length", "99999999999"),),
        (("Content-Length", "15"), ("Content-Length", "15")),
        (("Transfer-Encoding", "gzip"),),
        (("Transfer-Encoding", "Chunked"),),
        (("Transfer-Encoding", "chunked"), ("Transfer-Encoding", "chunked")),
        (("Transfer-Encoding", "chunked"), ("Content-Length", "15")),
    ],
)
def test_client_rejects_ambiguous_or_unsupported_framing(
    headers: tuple[tuple[str, str], ...],
) -> None:
    transport = _FakeTransport(response=_response(headers=headers))

    with pytest.raises(SportsDataIONbaClientError):
        _client(transport).capture(depth_charts_request())

    assert len(transport.calls) == 1


def test_client_accepts_exact_chunked_entity_framing() -> None:
    transport = _FakeTransport(response=_response(headers=(("Transfer-Encoding", "chunked"),)))

    capture = _client(transport).capture(depth_charts_request())

    assert capture.body == _BODY


def test_client_requires_declared_length_to_match_entity() -> None:
    transport = _FakeTransport(response=_response(headers=(("Content-Length", "13"),)))

    with pytest.raises(SportsDataIONbaClientError, match="length differs"):
        _client(transport).capture(depth_charts_request())


@pytest.mark.parametrize("body", [b"", b"x" * (NBA_RAW_CAPTURE_MAX_BODY_BYTES + 1)])
def test_client_rejects_empty_or_oversized_entity(body: bytes) -> None:
    transport = _FakeTransport(response=_response(body=body))

    with pytest.raises(SportsDataIONbaClientError, match="body size is invalid"):
        _client(transport).capture(depth_charts_request())


def test_client_rejects_mutable_entity() -> None:
    body = cast(bytes, bytearray(_BODY))
    transport = _FakeTransport(response=_response(body=body))

    with pytest.raises(SportsDataIONbaClientError, match="body is invalid"):
        _client(transport).capture(depth_charts_request())


@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 7, 17, 12, 30),  # noqa: DTZ001 - intentionally naive test value.
        datetime(2026, 7, 17, 12, 30, tzinfo=timezone(timedelta(hours=1))),
        cast(datetime, "not-a-datetime"),
    ],
)
def test_client_requires_exact_utc_completion_time(value: datetime) -> None:
    transport = _FakeTransport()

    with pytest.raises(SportsDataIONbaClientError, match="must be in UTC"):
        _client(transport, utc_now=lambda: value).capture(depth_charts_request())

    assert len(transport.calls) == 1


def test_client_discards_secret_bearing_clock_exception() -> None:
    transport = _FakeTransport()

    def broken_clock() -> datetime:
        raise RuntimeError(f"clock repeated {_API_KEY}")

    with pytest.raises(SportsDataIONbaClientError, match="time is unavailable") as caught:
        _client(transport, utc_now=broken_clock).capture(depth_charts_request())

    _assert_secret_free(caught.value, _API_KEY)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_client_does_not_call_hostile_timezone_methods() -> None:
    transport = _FakeTransport()
    value = datetime(2026, 7, 17, 12, 30, tzinfo=_HostileTzInfo())

    with pytest.raises(SportsDataIONbaClientError, match="must be in UTC") as caught:
        _client(transport, utc_now=lambda: value).capture(depth_charts_request())

    _assert_secret_free(caught.value, _API_KEY)


def test_client_and_response_reprs_hide_secrets_and_payloads() -> None:
    response = _response(headers=(("X-Secret", _API_KEY),), body=_API_KEY.encode())
    transport = _FakeTransport(response=response)
    client = _client(transport)

    assert _API_KEY not in repr(client)
    assert _API_KEY not in repr(response)
    assert _BODY.decode() not in repr(response)
    assert repr(response) == "SportsDataIOHttpResponse(status=200)"


@dataclass(slots=True)
class _FakeHttpResponse:
    status: int = 200
    headers: list[tuple[str, str]] = field(default_factory=_header_list)
    body: bytes = _BODY
    reads: list[int] = field(default_factory=_integer_list)

    def getheaders(self) -> list[tuple[str, str]]:
        """Return one frozen response-header snapshot source."""
        return list(self.headers)

    def read(self, amount: int) -> bytes:
        """Record and return the caller-bounded entity read."""
        self.reads.append(amount)
        return self.body


@dataclass(slots=True)
class _FakeHttpsConnection:
    response: _FakeHttpResponse
    request_error: Exception | None = None
    requests: list[tuple[str, str, dict[str, str]]] = field(default_factory=_request_list)
    responses: int = 0
    closes: int = 0
    debug_levels: list[int] = field(default_factory=_integer_list)

    def set_debuglevel(self, level: int) -> None:
        """Record the explicit suppression of credential-bearing HTTP debug output."""
        self.debug_levels.append(level)

    def request(
        self,
        method: str,
        url: str,
        body: object | None = None,
        headers: Mapping[str, str] | None = None,
        *,
        encode_chunked: bool = False,
    ) -> None:
        """Record the one request shape accepted by ``HTTPSConnection``."""
        assert body is None
        assert not encode_chunked
        self.requests.append((method, url, dict(headers or {})))
        if self.request_error is not None:
            raise self.request_error

    def getresponse(self) -> _FakeHttpResponse:
        """Return the configured response once."""
        self.responses += 1
        return self.response

    def close(self) -> None:
        """Record deterministic cleanup."""
        self.closes += 1


@dataclass(slots=True)
class _FakeConnectionFactory:
    connection: _FakeHttpsConnection
    calls: list[tuple[str, float, ssl.SSLContext]] = field(default_factory=_factory_calls)

    def __call__(
        self,
        host: str,
        *,
        timeout: float,
        context: ssl.SSLContext,
    ) -> _FakeHttpsConnection:
        self.calls.append((host, timeout, context))
        return self.connection


def test_https_transport_pins_verified_tls_host_and_one_bounded_get(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_response = _FakeHttpResponse(
        headers=[("Content-Length", str(len(_BODY)))],
    )
    connection = _FakeHttpsConnection(raw_response)
    factory = _FakeConnectionFactory(connection)
    monkeypatch.setattr(client_module, "HTTPSConnection", factory)
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "Connection": "close",
        "User-Agent": SPORTSDATAIO_NBA_USER_AGENT,
        SPORTSDATAIO_API_KEY_HEADER: _API_KEY,
    }

    response = SportsDataIOHttpsTransport().get("/fixed", headers, 7.0, 100)

    assert response.body == _BODY
    assert len(factory.calls) == 1
    host, timeout, context = factory.calls[0]
    assert host == SPORTSDATAIO_NBA_HOST
    assert timeout == 7.0
    assert context.check_hostname
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert connection.debug_levels == [0]
    assert connection.requests == [("GET", "/fixed", headers)]
    assert connection.responses == 1
    assert raw_response.reads == [101]
    assert connection.closes == 1


@pytest.mark.parametrize("status", [201, 301, 307, 400, 500])
def test_https_transport_rejects_non_200_before_read_and_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    raw_response = _FakeHttpResponse(status=status)
    connection = _FakeHttpsConnection(raw_response)
    factory = _FakeConnectionFactory(connection)
    monkeypatch.setattr(client_module, "HTTPSConnection", factory)

    with pytest.raises(SportsDataIONbaClientError, match="retrieval failed"):
        SportsDataIOHttpsTransport().get("/fixed", {}, 7.0, 100)

    assert len(factory.calls) == 1
    assert len(connection.requests) == 1
    assert connection.responses == 1
    assert raw_response.reads == []
    assert connection.closes == 1


def test_https_transport_rejects_oversized_declared_length_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_response = _FakeHttpResponse(headers=[("Content-Length", "101")])
    connection = _FakeHttpsConnection(raw_response)
    monkeypatch.setattr(client_module, "HTTPSConnection", _FakeConnectionFactory(connection))

    with pytest.raises(SportsDataIONbaClientError, match="retrieval failed"):
        SportsDataIOHttpsTransport().get("/fixed", {}, 7.0, 100)

    assert raw_response.reads == []
    assert connection.closes == 1


def test_https_transport_closes_and_discards_secret_bearing_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _FakeHttpsConnection(
        _FakeHttpResponse(),
        request_error=RuntimeError(f"request repeated {_API_KEY}"),
    )
    monkeypatch.setattr(client_module, "HTTPSConnection", _FakeConnectionFactory(connection))

    with pytest.raises(SportsDataIONbaClientError, match="retrieval failed") as caught:
        SportsDataIOHttpsTransport().get("/fixed", {}, 7.0, 100)

    assert connection.closes == 1
    _assert_secret_free(caught.value, _API_KEY)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_https_transport_rejects_mutable_body_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = cast(bytes, bytearray(_BODY))
    connection = _FakeHttpsConnection(_FakeHttpResponse(body=body))
    monkeypatch.setattr(client_module, "HTTPSConnection", _FakeConnectionFactory(connection))

    with pytest.raises(SportsDataIONbaClientError, match="retrieval failed"):
        SportsDataIOHttpsTransport().get("/fixed", {}, 7.0, 100)

    assert connection.closes == 1
