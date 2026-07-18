"""Capture one registered SportsDataIO NBA response into private local storage."""

from __future__ import annotations

import argparse
import os
import stat
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from forecastfm.local_config import read_sportsdataio_api_key
from forecastfm.nba_raw_capture import (
    NbaRawCaptureRequest,
    write_nba_raw_capture,
)
from forecastfm.sportsdataio_nba_client import SportsDataIONbaClient
from forecastfm.sportsdataio_nba_openapi import (
    SPORTSDATAIO_NBA_HOST,
    require_registered_nba_request,
)

DEFAULT_CONFIG_PATH = Path(".sportsdataio.env")


@dataclass(frozen=True, slots=True)
class _Arguments:
    operation: str
    path: str
    config: Path
    storage_root: Path
    output: Path


def main(argv: Sequence[str] | None = None) -> int:
    """Capture one registered response without printing response or credential data."""
    arguments = _parse_arguments(argv)
    try:
        _require_unused_private_target(arguments.storage_root, arguments.output)
        request = require_registered_nba_request(
            NbaRawCaptureRequest(
                operation=arguments.operation,
                host=SPORTSDATAIO_NBA_HOST,
                path=arguments.path,
            )
        )
        api_key = read_sportsdataio_api_key(arguments.config)
        capture = SportsDataIONbaClient(api_key).capture(request)
        write_nba_raw_capture(capture, arguments.output, arguments.storage_root)
    except Exception:
        print("SportsDataIO capture failed.", file=sys.stderr)
        return 1

    print(f"capture={arguments.output} sha256={capture.sha256}")
    return 0


def _parse_arguments(argv: Sequence[str] | None) -> _Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", help="registered operation identifier")
    parser.add_argument("path", help="exact registered NBA API path")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="ignored local key file (default: .sportsdataio.env)",
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        required=True,
        help="existing private output directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new .json file directly under storage root",
    )
    namespace = parser.parse_args(argv)
    return _Arguments(
        operation=cast(str, namespace.operation),
        path=cast(str, namespace.path),
        config=cast(Path, namespace.config),
        storage_root=cast(Path, namespace.storage_root),
        output=cast(Path, namespace.output),
    )


def _require_unused_private_target(storage_root: Path, output: Path) -> None:
    root = storage_root.absolute()
    try:
        root_stat = storage_root.stat()
        resolved_root = storage_root.resolve(strict=True)
    except OSError:
        raise RuntimeError("capture storage is unavailable") from None
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or storage_root.is_symlink()
        or resolved_root != root
        or root_stat.st_uid != os.getuid()
        or stat.S_IMODE(root_stat.st_mode) & 0o077 != 0
    ):
        raise RuntimeError("capture storage is unavailable")

    target = output.absolute()
    if target.parent != root or output.name.startswith(".") or output.suffix != ".json":
        raise RuntimeError("capture output is unavailable")
    try:
        output.lstat()
    except FileNotFoundError:
        return
    except OSError:
        raise RuntimeError("capture output is unavailable") from None
    raise RuntimeError("capture output is unavailable")


if __name__ == "__main__":
    raise SystemExit(main())
