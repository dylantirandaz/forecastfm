"""Load the ignored local API key, then run the frozen Tinker trainer."""

import os
from pathlib import Path

from examples import train_tinker_sft

from forecastfm.local_config import read_tinker_api_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV_PATH = PROJECT_ROOT / ".env"


read_api_key = read_tinker_api_key


def main() -> None:
    """Prefer an exported key, otherwise load the ignored local config."""
    if not os.environ.get("TINKER_API_KEY"):
        os.environ["TINKER_API_KEY"] = read_api_key(LOCAL_ENV_PATH)
    train_tinker_sft.main()


if __name__ == "__main__":
    main()
