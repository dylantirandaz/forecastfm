"""Tests for local-only raw NBA provider captures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

import forecastfm.nba_raw_capture as raw_capture
from forecastfm.integrity import bytes_sha256, canonical_json
from forecastfm.nba_raw_capture import (
    NBA_RAW_CAPTURE_BODY_REPRESENTATION,
    NBA_RAW_CAPTURE_CLAIM,
    NBA_RAW_CAPTURE_CLOCK_AUTHORITY,
    NBA_RAW_CAPTURE_NOT_ESTABLISHED,
    NBA_RAW_CAPTURE_REQUEST_IDENTITY_AUTHORITY,
    NbaRawCapture,
    NbaRawCaptureError,
    NbaRawCapturePublicationUncertainError,
    NbaRawCaptureRequest,
    load_nba_raw_capture,
    load_nba_raw_capture_bytes,
    write_nba_raw_capture,
)

_RETRIEVED_AT = datetime(2026, 7, 17, 18, 30, tzinfo=UTC)


def _request() -> NbaRawCaptureRequest:
    return NbaRawCaptureRequest(
        operation="games_by_date_final",
        host="api.sportsdata.io",
        path="/v3/nba/scores/json/GamesByDateFinal/2026-JUL-17",
    )


def _capture(body: bytes = b'[{"GameID":1}]') -> NbaRawCapture:
    return NbaRawCapture.from_response(
        request=_request(),
        retrieved_at=_RETRIEVED_AT,
        response_headers=(
            ("X-Unknown", "omitted"),
            ("Content-Type", "application/json"),
            ("Set-Cookie", "private=value"),
            ("Date", "Fri, 17 Jul 2026 18:30:00 GMT"),
        ),
        body=body,
    )


def _encoded(record: object) -> bytes:
    return f"{canonical_json(record)}\n".encode()


def test_raw_capture_round_trip_is_exact_and_narrow() -> None:
    capture = _capture(b"\x00\xffNBA\n")

    loaded = load_nba_raw_capture_bytes(capture.canonical_bytes)
    record = loaded.canonical_payload()

    assert loaded == capture
    assert loaded.body == b"\x00\xffNBA\n"
    assert loaded.sha256 == bytes_sha256(loaded.canonical_bytes)
    assert record["claim"] == NBA_RAW_CAPTURE_CLAIM
    assert record["clock_authority"] == NBA_RAW_CAPTURE_CLOCK_AUTHORITY
    assert record["request_identity_authority"] == NBA_RAW_CAPTURE_REQUEST_IDENTITY_AUTHORITY
    assert record["body_representation"] == NBA_RAW_CAPTURE_BODY_REPRESENTATION
    assert record["transport_authenticity"] == NBA_RAW_CAPTURE_NOT_ESTABLISHED
    assert record["provider_identity"] == NBA_RAW_CAPTURE_NOT_ESTABLISHED
    assert record["provider_publication_time"] == NBA_RAW_CAPTURE_NOT_ESTABLISHED
    assert record["revision_authenticity"] == NBA_RAW_CAPTURE_NOT_ESTABLISHED
    assert "available_at" not in record
    assert "provider_published_at" not in record
    assert "authorization" not in record


def test_response_metadata_uses_only_sorted_allowlisted_headers() -> None:
    capture = _capture()

    assert capture.response_headers == (
        ("content-type", "application/json"),
        ("date", "Fri, 17 Jul 2026 18:30:00 GMT"),
    )
    assert "Set-Cookie" not in capture.canonical_bytes.decode()
    assert "private=value" not in capture.canonical_bytes.decode()
    assert "X-Unknown" not in capture.canonical_bytes.decode()


@pytest.mark.parametrize(
    "host",
    [
        "API.sportsdata.io",
        "api.sportsdata.io:443",
        "127.0.0.1",
        "localhost",
        "user@api.sportsdata.io",
        "api.sportsdata.io/path",
    ],
)
def test_request_rejects_unsafe_hosts(host: str) -> None:
    with pytest.raises(NbaRawCaptureError, match="DNS-shaped"):
        NbaRawCaptureRequest("games", host, "/v3/nba/scores/json/Games/2026")


@pytest.mark.parametrize(
    "path",
    [
        "v3/nba",
        "//evil.example/path",
        "/v3//nba",
        "/v3/../secret",
        "/v3/./nba",
        "/v3/nba/",
        "/v3/nba?key=secret",
        "/v3/nba&key=secret",
        "/v3/nba=secret",
        "/v3/nba#fragment",
        "/v3/%2e%2e/secret",
        "/v3\\nba",
        "/https://evil.example",
        "/v3/nba\r\nInjected:value",
        "/v3/nba\x7f",
    ],
)
def test_request_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(NbaRawCaptureError, match="safe absolute path"):
        NbaRawCaptureRequest("games", "api.sportsdata.io", path)


@pytest.mark.parametrize("operation", ["", "Games", "games-by-date", "games/path"])
def test_request_rejects_unsafe_operation(operation: str) -> None:
    with pytest.raises(NbaRawCaptureError, match="lowercase identifier"):
        NbaRawCaptureRequest(operation, "api.sportsdata.io", "/v3/nba")


def test_request_host_is_only_a_syntactic_caller_assertion() -> None:
    request = NbaRawCaptureRequest("games", "evil.example", "/v3/nba/Games/2026")
    capture = NbaRawCapture.from_response(request, _RETRIEVED_AT, (), b"body")

    assert capture.request.host == "evil.example"
    assert capture.canonical_payload()["provider_identity"] == NBA_RAW_CAPTURE_NOT_ESTABLISHED


def test_capture_rejects_non_200_empty_mutable_and_non_utc_inputs() -> None:
    with pytest.raises(NbaRawCaptureError, match="HTTP status 200"):
        NbaRawCapture.from_response(_request(), _RETRIEVED_AT, (), b"error", 302)
    with pytest.raises(NbaRawCaptureError, match="nonempty"):
        NbaRawCapture.from_response(_request(), _RETRIEVED_AT, (), b"")
    with pytest.raises(NbaRawCaptureError, match="immutable bytes"):
        NbaRawCapture(
            request=_request(),
            retrieved_at=_RETRIEVED_AT,
            response_headers=(),
            body=cast(bytes, bytearray(b"mutable")),
        )
    with pytest.raises(NbaRawCaptureError, match="must be in UTC"):
        NbaRawCapture.from_response(
            _request(),
            _RETRIEVED_AT.replace(tzinfo=None),
            (),
            b"body",
        )
    with pytest.raises(NbaRawCaptureError, match="must be in UTC"):
        NbaRawCapture.from_response(
            _request(),
            datetime(2026, 7, 17, 13, 30, tzinfo=timezone(-timedelta(hours=5))),
            (),
            b"body",
        )


def test_capture_rejects_header_injection_without_echoing_value() -> None:
    secret = "do-not-echo"
    with pytest.raises(NbaRawCaptureError, match="name is unsafe") as error:
        NbaRawCapture.from_response(
            _request(),
            _RETRIEVED_AT,
            ((f"X-Bad\r\n{secret}", "value"),),
            b"body",
        )
    assert secret not in str(error.value)

    with pytest.raises(NbaRawCaptureError, match="value is unsafe") as error:
        NbaRawCapture.from_response(
            _request(),
            _RETRIEVED_AT,
            (("Content-Type", f"application/json\r\n{secret}"),),
            b"body",
        )
    assert secret not in str(error.value)


def test_write_is_create_only_and_loads_one_buffer(tmp_path: Path) -> None:
    path = tmp_path / "capture.json"
    first = _capture()
    write_nba_raw_capture(first, path, tmp_path)
    original = path.read_bytes()

    assert original == first.canonical_bytes
    assert load_nba_raw_capture(path) == first
    assert path.stat().st_mode & 0o777 == 0o600

    with pytest.raises(NbaRawCaptureError, match="already exists"):
        write_nba_raw_capture(_capture(b"different"), path, tmp_path)
    assert path.read_bytes() == original
    assert not tuple(tmp_path.glob(".*.part"))


def test_writer_rejects_paths_outside_the_explicit_storage_root(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    with pytest.raises(NbaRawCaptureError, match="directly under storage_root"):
        write_nba_raw_capture(_capture(), outside, tmp_path)
    with pytest.raises(NbaRawCaptureError, match="directly under storage_root"):
        write_nba_raw_capture(_capture(), nested / "capture.json", tmp_path)
    with pytest.raises(NbaRawCaptureError, match="directly under storage_root"):
        write_nba_raw_capture(_capture(), tmp_path / ".capture.json", tmp_path)
    with pytest.raises(NbaRawCaptureError, match="directly under storage_root"):
        write_nba_raw_capture(_capture(), tmp_path / "capture.txt", tmp_path)


def test_writer_rejects_a_symlink_storage_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(NbaRawCaptureError, match="nonsymlink"):
        write_nba_raw_capture(_capture(), link / "capture.json", link)


def test_writer_rejects_nonprivate_storage_permissions(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)

    with pytest.raises(NbaRawCaptureError, match="owned, private"):
        write_nba_raw_capture(_capture(), shared / "capture.json", shared)


def test_directory_sync_failure_reports_uncertain_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "capture.json"

    def fail_sync(_path: Path) -> None:
        raise OSError("simulated sync failure")

    monkeypatch.setattr("forecastfm.nba_raw_capture._fsync_directory", fail_sync)
    with pytest.raises(NbaRawCapturePublicationUncertainError, match="may have been published"):
        write_nba_raw_capture(_capture(), path, tmp_path)

    assert load_nba_raw_capture(path) == _capture()
    assert not tuple(tmp_path.glob(".*.part"))


def test_capture_and_loader_enforce_resource_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(raw_capture, "NBA_RAW_CAPTURE_MAX_BODY_BYTES", 4)
    with pytest.raises(NbaRawCaptureError, match="body exceeds"):
        _capture(b"12345")

    monkeypatch.setattr(raw_capture, "NBA_RAW_CAPTURE_MAX_ARTIFACT_BYTES", 4)
    with pytest.raises(NbaRawCaptureError, match="artifact limit"):
        load_nba_raw_capture_bytes(b"12345")


def test_loader_rejects_missing_non_utf8_and_noncanonical_bytes(tmp_path: Path) -> None:
    with pytest.raises(NbaRawCaptureError, match="cannot read"):
        load_nba_raw_capture(tmp_path / "missing.json")
    with pytest.raises(NbaRawCaptureError, match="UTF-8"):
        load_nba_raw_capture_bytes(b"\xff\n")
    with pytest.raises(NbaRawCaptureError, match="exactly one newline"):
        load_nba_raw_capture_bytes(_capture().canonical_bytes.rstrip(b"\n"))
    with pytest.raises(NbaRawCaptureError, match="exactly one newline"):
        load_nba_raw_capture_bytes(_capture().canonical_bytes + b"\n")

    pretty = canonical_json(_capture().canonical_payload()).replace(",", ", ", 1)
    with pytest.raises(NbaRawCaptureError, match="not canonical"):
        load_nba_raw_capture_bytes(f"{pretty}\n".encode())


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("body_length", 999),
        ("body_sha256", "0" * 64),
        ("http_status", 301),
        ("claim", "provider_verified"),
        ("provider_publication_time", "2026-07-17T18:00:00Z"),
        ("provider_identity", "verified"),
        ("transport_authenticity", "verified"),
        ("request_identity_authority", "provider_verified"),
        ("body_representation", "wire_bytes"),
        ("revision_authenticity", "verified"),
    ],
)
def test_loader_rejects_tampering(field_name: str, replacement: object) -> None:
    record = _capture().canonical_payload()
    record[field_name] = replacement

    with pytest.raises(NbaRawCaptureError, match="invalid raw capture"):
        load_nba_raw_capture_bytes(_encoded(record))


def test_loader_rejects_body_substitution_and_noncanonical_base64() -> None:
    record = _capture().canonical_payload()
    record["body_base64"] = "ZGlmZmVyZW50"
    with pytest.raises(NbaRawCaptureError, match="invalid raw capture"):
        load_nba_raw_capture_bytes(_encoded(record))

    record = _capture().canonical_payload()
    record["body_base64"] = "YWJj=="
    with pytest.raises(NbaRawCaptureError, match="invalid raw capture"):
        load_nba_raw_capture_bytes(_encoded(record))


def test_loader_rejects_shape_changes_duplicate_keys_and_bool_schema() -> None:
    record = _capture().canonical_payload()
    del record["proof_scope"]
    with pytest.raises(NbaRawCaptureError, match="invalid raw capture"):
        load_nba_raw_capture_bytes(_encoded(record))

    duplicate = (
        _capture()
        .canonical_bytes.decode()
        .replace(
            '"schema_version":1',
            '"schema_version":1,"schema_version":1',
            1,
        )
    )
    with pytest.raises(NbaRawCaptureError, match="invalid raw capture"):
        load_nba_raw_capture_bytes(duplicate.encode())

    record = _capture().canonical_payload()
    record["schema_version"] = True
    with pytest.raises(NbaRawCaptureError, match="invalid raw capture"):
        load_nba_raw_capture_bytes(_encoded(record))


def test_loader_rejects_noncanonical_time_and_selected_headers() -> None:
    record = _capture().canonical_payload()
    record["retrieved_at"] = "2026-07-17T18:30:00+00:00"
    with pytest.raises(NbaRawCaptureError, match="invalid raw capture"):
        load_nba_raw_capture_bytes(_encoded(record))

    record = _capture().canonical_payload()
    record["response_headers"] = [{"name": "set-cookie", "value": "private=value"}]
    with pytest.raises(NbaRawCaptureError, match="invalid raw capture"):
        load_nba_raw_capture_bytes(_encoded(record))
