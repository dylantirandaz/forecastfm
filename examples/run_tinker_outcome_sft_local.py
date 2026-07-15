"""Load the ignored local API key, then run outcome-v1 training."""

import os
from pathlib import Path

from examples import train_tinker_outcome_sft
from examples.run_tinker_sft_local import read_api_key

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ENV_PATH = PROJECT_ROOT / ".env"


def main() -> None:
    """Prefer an exported key, otherwise load the ignored local config."""
    if not os.environ.get("TINKER_API_KEY"):
        os.environ["TINKER_API_KEY"] = read_api_key(LOCAL_ENV_PATH)
    train_tinker_outcome_sft.main()


if __name__ == "__main__":
    main()
