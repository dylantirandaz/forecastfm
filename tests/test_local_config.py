"""Tests for ignored local credential loaders."""

from pathlib import Path

import pytest

from forecastfm.local_config import read_sportsdataio_api_key, read_tinker_api_key


def _write_private_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def _write_private_bytes(path: Path, content: bytes) -> None:
    path.write_bytes(content)
    path.chmod(0o600)


def test_read_api_key_accepts_quotes(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text('TINKER_API_KEY="secret-value"\n', encoding="utf-8")

    assert read_tinker_api_key(path) == "secret-value"


@pytest.mark.parametrize("content", ["", "TINKER_API_KEY=", "OTHER_KEY=value"])
def test_read_api_key_rejects_missing_or_empty_value(tmp_path: Path, content: str) -> None:
    path = tmp_path / ".env"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(RuntimeError):
        read_tinker_api_key(path)


@pytest.mark.parametrize(
    "content",
    [
        "SPORTSDATAIO_API_KEY=test-key",
        "SPORTSDATAIO_API_KEY=test-key\n",
        "SPORTSDATAIO_API_KEY=test-key\r\n",
        'SPORTSDATAIO_API_KEY="test-key"\n',
        "SPORTSDATAIO_API_KEY='test-key'\n",
    ],
)
def test_read_sportsdataio_api_key_accepts_one_exact_assignment(
    tmp_path: Path,
    content: str,
) -> None:
    path = tmp_path / ".sportsdataio.env"
    _write_private_text(path, content)

    assert read_sportsdataio_api_key(path) == "test-key"


def test_read_sportsdataio_api_key_accepts_safe_maximum_length(tmp_path: Path) -> None:
    path = tmp_path / ".sportsdataio.env"
    expected = "a" * 512
    _write_private_text(path, f"SPORTSDATAIO_API_KEY={expected}\n")

    assert read_sportsdataio_api_key(path) == expected


@pytest.mark.parametrize(
    "content",
    [
        "",
        "SPORTSDATAIO_API_KEY=",
        'SPORTSDATAIO_API_KEY=""',
        "SPORTSDATAIO_API_KEY='abc\"",
        "SPORTSDATAIO_API_KEY=\"abc'",
        'SPORTSDATAIO_API_KEY="abc',
        'SPORTSDATAIO_API_KEY=abc"',
        "SPORTSDATAIO_API_KEY",
        "OTHER_KEY=test-key",
        " SPORTSDATAIO_API_KEY=test-key",
        "SPORTSDATAIO_API_KEY =test-key",
        "SPORTSDATAIO_API_KEY=test-key ",
        "SPORTSDATAIO_API_KEY=test-key\n\n",
        "SPORTSDATAIO_API_KEY=test-key\nOTHER_KEY=value",
        "SPORTSDATAIO_API_KEY=first\nSPORTSDATAIO_API_KEY=second",
        "# comment\nSPORTSDATAIO_API_KEY=test-key",
        "export SPORTSDATAIO_API_KEY=test-key",
    ],
)
def test_read_sportsdataio_api_key_rejects_nonexact_content(
    tmp_path: Path,
    content: str,
) -> None:
    path = tmp_path / ".sportsdataio.env"
    _write_private_text(path, content)

    with pytest.raises(RuntimeError):
        read_sportsdataio_api_key(path)


@pytest.mark.parametrize(
    "unsafe_value",
    [
        "contains space",
        "contains\ttab",
        "contains\x00null",
        "contains\x1fcontrol",
        "non-ascii-\N{SNOWMAN}",
        "a" * 513,
    ],
)
def test_read_sportsdataio_api_key_rejects_unsafe_values(
    tmp_path: Path,
    unsafe_value: str,
) -> None:
    path = tmp_path / ".sportsdataio.env"
    _write_private_text(path, f"SPORTSDATAIO_API_KEY={unsafe_value}")

    with pytest.raises(RuntimeError):
        read_sportsdataio_api_key(path)


def test_read_sportsdataio_api_key_missing_error_does_not_reveal_path(tmp_path: Path) -> None:
    path_secret = "path-must-stay-private"
    path = tmp_path / path_secret

    with pytest.raises(RuntimeError) as caught:
        read_sportsdataio_api_key(path)

    assert path_secret not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_read_sportsdataio_api_key_rejects_non_utf8_content(tmp_path: Path) -> None:
    path = tmp_path / ".sportsdataio.env"
    _write_private_bytes(path, b"SPORTSDATAIO_API_KEY=\xff")

    with pytest.raises(RuntimeError) as caught:
        read_sportsdataio_api_key(path)

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_read_sportsdataio_api_key_error_does_not_reveal_value(tmp_path: Path) -> None:
    path = tmp_path / ".sportsdataio.env"
    value_secret = "value-must-stay-private"
    _write_private_text(
        path,
        f"SPORTSDATAIO_API_KEY={value_secret}\nOTHER_KEY=unexpected",
    )

    with pytest.raises(RuntimeError) as caught:
        read_sportsdataio_api_key(path)

    assert value_secret not in str(caught.value)
    assert value_secret not in repr(caught.value)


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o604])
def test_read_sportsdataio_api_key_rejects_nonprivate_file(
    tmp_path: Path,
    mode: int,
) -> None:
    path = tmp_path / ".sportsdataio.env"
    _write_private_text(path, "SPORTSDATAIO_API_KEY=test-key")
    path.chmod(mode)

    with pytest.raises(RuntimeError, match="private regular file"):
        read_sportsdataio_api_key(path)


def test_read_sportsdataio_api_key_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.env"
    _write_private_text(target, "SPORTSDATAIO_API_KEY=test-key")
    link = tmp_path / ".sportsdataio.env"
    link.symlink_to(target)

    with pytest.raises(RuntimeError):
        read_sportsdataio_api_key(link)


def test_read_sportsdataio_api_key_rejects_directory(tmp_path: Path) -> None:
    path = tmp_path / ".sportsdataio.env"
    path.mkdir(mode=0o700)

    with pytest.raises(RuntimeError, match="private regular file"):
        read_sportsdataio_api_key(path)


def test_read_sportsdataio_api_key_rejects_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / ".sportsdataio.env"
    _write_private_text(path, "x" * 1025)

    with pytest.raises(RuntimeError, match="private regular file"):
        read_sportsdataio_api_key(path)
