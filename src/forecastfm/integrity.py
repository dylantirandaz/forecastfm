"""Small deterministic hashing helpers for committed experiment artifacts."""

import hashlib
import json
from pathlib import Path


def canonical_json(value: object) -> str:
    """Serialize JSON deterministically for hashing and ledger records."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(value: object) -> str:
    """Hash a value's canonical UTF-8 JSON representation."""
    return text_sha256(canonical_json(value))


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
