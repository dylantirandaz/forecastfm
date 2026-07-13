"""Run the published validation-canary v2 protocol exactly once."""

from pathlib import Path

from examples import run_tinker_canary

from forecastfm.canary import V2_PROTOCOL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANARY_DIRECTORY = PROJECT_ROOT / "evaluation" / "validation_canary_v2"
CONFIG = run_tinker_canary.CanaryRunConfig(
    directory=CANARY_DIRECTORY,
    protocol=V2_PROTOCOL,
    entrypoint_path=Path(__file__).resolve(),
    require_published_commitments=True,
)


async def run() -> None:
    """Run the v2 canary through the shared immutable runner."""
    await run_tinker_canary.run(CONFIG)


def main() -> None:
    """Load the local key and start the v2 canary."""
    run_tinker_canary.main(CONFIG)


if __name__ == "__main__":
    main()
