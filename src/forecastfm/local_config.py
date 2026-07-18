"""Read ignored local credentials without a dotenv dependency."""

import os
import stat
from contextlib import suppress
from pathlib import Path
from typing import Literal

_SPORTSDATAIO_API_KEY_NAME = "SPORTSDATAIO_API_KEY"
_SPORTSDATAIO_API_KEY_MAX_LENGTH = 512
_SPORTSDATAIO_CONFIG_MAX_BYTES = 1024

type _LocalReadStatus = Literal["ok", "missing", "unavailable", "unsafe"]


def read_tinker_api_key(path: Path) -> str:
    """Read one TINKER_API_KEY assignment without a dotenv dependency."""
    try:
        name, separator, raw_value = path.read_text(encoding="utf-8").strip().partition("=")
    except FileNotFoundError as error:
        raise RuntimeError(f"local config is missing: {path}") from error
    if name != "TINKER_API_KEY" or not separator:
        raise RuntimeError(".env must contain exactly one TINKER_API_KEY assignment")

    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    if not value:
        raise RuntimeError("TINKER_API_KEY is empty in .env")
    return value


def read_sportsdataio_api_key(path: Path) -> str:
    """Read one exact, header-safe SportsDataIO API-key assignment."""
    content, status = _try_read_private_utf8(path)
    if content is None:
        if status == "missing":
            raise RuntimeError("local SportsDataIO config is missing")
        if status == "unsafe":
            raise RuntimeError("local SportsDataIO config must be a private regular file")
        raise RuntimeError("local SportsDataIO config cannot be read")

    line = _remove_one_line_ending(content)
    name, separator, raw_value = line.partition("=")
    if name != _SPORTSDATAIO_API_KEY_NAME or not separator:
        raise RuntimeError("local SportsDataIO config must contain one exact assignment")

    value = _remove_optional_quotes(raw_value)
    if not _is_safe_sportsdataio_api_key(value):
        raise RuntimeError("local SportsDataIO API key is invalid")
    return value


def _try_read_private_utf8(path: Path) -> tuple[str | None, _LocalReadStatus]:
    """Read one small owned file without following its final symlink."""
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077 != 0
            or metadata.st_size > _SPORTSDATAIO_CONFIG_MAX_BYTES
        ):
            return None, "unsafe"
        raw = _read_bounded(descriptor)
        if len(raw) > _SPORTSDATAIO_CONFIG_MAX_BYTES:
            return None, "unsafe"
        return raw.decode("utf-8"), "ok"
    except FileNotFoundError:
        return None, "missing"
    except (OSError, UnicodeError):
        return None, "unavailable"
    finally:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)


def _read_bounded(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    remaining = _SPORTSDATAIO_CONFIG_MAX_BYTES + 1
    while remaining > 0:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _remove_one_line_ending(content: str) -> str:
    if content.endswith("\r\n"):
        line = content[:-2]
    elif content.endswith("\n"):
        line = content[:-1]
    else:
        line = content
    if "\r" in line or "\n" in line:
        raise RuntimeError("local SportsDataIO config must contain one exact assignment")
    return line


def _remove_optional_quotes(raw_value: str) -> str:
    if not raw_value:
        return raw_value
    first = raw_value[0]
    last = raw_value[-1]
    if first not in {'"', "'"}:
        if last in {'"', "'"}:
            raise RuntimeError("local SportsDataIO config has invalid quoting")
        return raw_value
    if len(raw_value) < 2 or last != first:
        raise RuntimeError("local SportsDataIO config has invalid quoting")
    return raw_value[1:-1]


def _is_safe_sportsdataio_api_key(value: str) -> bool:
    return (
        0 < len(value) <= _SPORTSDATAIO_API_KEY_MAX_LENGTH
        and value.isascii()
        and all(33 <= ord(character) <= 126 for character in value)
    )
