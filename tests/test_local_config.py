"""Tests for the ignored local Tinker-key loader."""

from pathlib import Path

import pytest
from examples.run_tinker_sft_local import read_api_key


def test_read_api_key_accepts_quotes(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text('TINKER_API_KEY="secret-value"\n', encoding="utf-8")

    assert read_api_key(path) == "secret-value"


@pytest.mark.parametrize("content", ["", "TINKER_API_KEY=", "OTHER_KEY=value"])
def test_read_api_key_rejects_missing_or_empty_value(tmp_path: Path, content: str) -> None:
    path = tmp_path / ".env"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(RuntimeError):
        read_api_key(path)
