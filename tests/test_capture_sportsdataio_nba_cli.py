from __future__ import annotations

import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import examples.capture_sportsdataio_nba as cli
import pytest

from forecastfm.nba_raw_capture import (
    NbaRawCapture,
    NbaRawCaptureRequest,
    load_nba_raw_capture,
)
from forecastfm.sportsdataio_nba_openapi import SPORTSDATAIO_NBA_HOST

_API_KEY = "test-key-not-a-real-credential"
_BODY = b'[{"GameID":1}]'
_PATH = "/v3/nba/scores/json/Games/2026"
_NOW = datetime(2026, 7, 17, 18, 30, tzinfo=UTC)


@dataclass(slots=True)
class _FakeClient:
    factory: _FakeClientFactory

    def capture(self, request: NbaRawCaptureRequest) -> NbaRawCapture:
        self.factory.requests.append(request)
        if self.factory.error is not None:
            raise self.factory.error
        return NbaRawCapture.from_response(
            request=request,
            retrieved_at=_NOW,
            response_headers=(
                ("Content-Type", "application/json"),
                ("X-Request-ID", "request-1"),
            ),
            body=_BODY,
        )


class _FakeClientFactory:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.api_keys: list[str] = []
        self.requests: list[NbaRawCaptureRequest] = []

    def __call__(self, api_key: str) -> _FakeClient:
        self.api_keys.append(api_key)
        return _FakeClient(self)


def _private_root(tmp_path: Path) -> Path:
    root = tmp_path / "captures"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    return root


def _write_config(path: Path) -> None:
    path.write_text(f'SPORTSDATAIO_API_KEY="{_API_KEY}"\n', encoding="utf-8")
    path.chmod(0o600)


def _arguments(root: Path, output: Path, config: Path | None = None) -> list[str]:
    arguments = [
        "games",
        _PATH,
        "--storage-root",
        str(root),
        "--output",
        str(output),
    ]
    if config is not None:
        arguments.extend(("--config", str(config)))
    return arguments


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    factory: _FakeClientFactory,
) -> None:
    monkeypatch.setattr(cli, "SportsDataIONbaClient", factory)


def test_cli_captures_once_to_a_new_private_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _private_root(tmp_path)
    output = root / "games-2026.json"
    config = tmp_path / "custom.env"
    _write_config(config)
    factory = _FakeClientFactory()
    _install_fake_client(monkeypatch, factory)

    result = cli.main(_arguments(root, output, config))

    assert result == 0
    assert factory.api_keys == [_API_KEY]
    assert factory.requests == [NbaRawCaptureRequest("games", SPORTSDATAIO_NBA_HOST, _PATH)]
    capture = load_nba_raw_capture(output)
    assert capture.body == _BODY
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    streams = capsys.readouterr()
    assert streams.out == f"capture={output} sha256={capture.sha256}\n"
    assert streams.err == ""
    assert _API_KEY not in streams.out
    assert _BODY.decode() not in streams.out
    assert "X-Request-ID" not in streams.out


def test_cli_uses_ignored_config_path_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path / ".sportsdataio.env")
    root = _private_root(tmp_path)
    output = root / "games.json"
    factory = _FakeClientFactory()
    _install_fake_client(monkeypatch, factory)

    assert cli.main(_arguments(root, output)) == 0

    assert factory.api_keys == [_API_KEY]
    assert len(factory.requests) == 1


@pytest.mark.parametrize(
    ("operation", "path"),
    [
        ("unknown", _PATH),
        ("games", "/v3/nba/scores/json/Games/1945"),
        ("depth_charts", _PATH),
    ],
)
def test_cli_rejects_unregistered_requests_before_loading_the_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    path: str,
) -> None:
    root = _private_root(tmp_path)
    output = root / "capture.json"
    factory = _FakeClientFactory()
    _install_fake_client(monkeypatch, factory)
    arguments = _arguments(root, output)
    arguments[0:2] = [operation, path]

    assert cli.main(arguments) == 1

    assert factory.api_keys == []
    assert not output.exists()


@pytest.mark.parametrize("root_mode", [0o755, 0o710, 0o701])
def test_cli_rejects_nonprivate_storage_before_loading_the_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_mode: int,
) -> None:
    root = _private_root(tmp_path)
    root.chmod(root_mode)
    output = root / "capture.json"
    factory = _FakeClientFactory()
    _install_fake_client(monkeypatch, factory)

    assert cli.main(_arguments(root, output)) == 1

    assert factory.api_keys == []
    assert not output.exists()


def test_cli_does_not_create_a_missing_storage_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "missing"
    factory = _FakeClientFactory()
    _install_fake_client(monkeypatch, factory)

    assert cli.main(_arguments(root, root / "capture.json")) == 1

    assert factory.api_keys == []
    assert not root.exists()


@pytest.mark.parametrize(
    "output_name",
    [".hidden.json", "capture.txt", "nested/capture.json", "../outside.json"],
)
def test_cli_rejects_outputs_outside_one_direct_json_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_name: str,
) -> None:
    root = _private_root(tmp_path)
    output = root / output_name
    factory = _FakeClientFactory()
    _install_fake_client(monkeypatch, factory)

    assert cli.main(_arguments(root, output)) == 1

    assert factory.api_keys == []
    assert not output.exists()


def test_cli_never_overwrites_or_recaptures_an_existing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _private_root(tmp_path)
    output = root / "capture.json"
    config = tmp_path / "key.env"
    _write_config(config)
    factory = _FakeClientFactory()
    _install_fake_client(monkeypatch, factory)
    arguments = _arguments(root, output, config)

    assert cli.main(arguments) == 0
    original = output.read_bytes()
    assert cli.main(arguments) == 1

    assert len(factory.requests) == 1
    assert output.read_bytes() == original


def test_cli_failure_output_cannot_reflect_client_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _private_root(tmp_path)
    output = root / "capture.json"
    config = tmp_path / "key.env"
    _write_config(config)
    reflected = f"{_API_KEY} {_BODY.decode()} X-Secret: reflected"
    factory = _FakeClientFactory(RuntimeError(reflected))
    _install_fake_client(monkeypatch, factory)

    assert cli.main(_arguments(root, output, config)) == 1

    streams = capsys.readouterr()
    assert streams.out == ""
    assert streams.err == "SportsDataIO capture failed.\n"
    assert _API_KEY not in streams.err
    assert _BODY.decode() not in streams.err
    assert "X-Secret" not in streams.err
    assert len(factory.requests) == 1
    assert not output.exists()


def test_cli_missing_config_error_does_not_print_its_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _private_root(tmp_path)
    output = root / "capture.json"
    config = tmp_path / "config-path-must-stay-private"
    factory = _FakeClientFactory()
    _install_fake_client(monkeypatch, factory)

    assert cli.main(_arguments(root, output, config)) == 1

    streams = capsys.readouterr()
    assert streams.out == ""
    assert streams.err == "SportsDataIO capture failed.\n"
    assert config.name not in streams.err
    assert factory.api_keys == []
    assert not output.exists()
