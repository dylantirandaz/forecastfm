"""Load the ignored local API key, then run the frozen Tinker trainer."""

import os
from pathlib import Path

from examples import train_tinker_sft

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV_PATH = PROJECT_ROOT / ".env"


def read_api_key(path: Path) -> str:
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


def main() -> None:
    """Prefer an exported key, otherwise load the ignored local config."""
    if not os.environ.get("TINKER_API_KEY"):
        os.environ["TINKER_API_KEY"] = read_api_key(LOCAL_ENV_PATH)
    train_tinker_sft.main()


if __name__ == "__main__":
    main()
