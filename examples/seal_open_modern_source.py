"""Seal the pinned open-modern NBA source into safe development artifacts."""

from pathlib import Path

from forecastfm.open_modern import seal_open_modern_source

PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_PATH = PROJECT_ROOT / "data/raw/outcome_v2_open_modern/nba_games.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data/processed/outcome_v2_open_modern"
DEVELOPMENT_PATH = OUTPUT_DIRECTORY / "development.csv"
TEST_INPUTS_PATH = OUTPUT_DIRECTORY / "test_inputs.csv"
PROTOCOL_PATH = PROJECT_ROOT / "evaluation/outcome_v2_open_modern/protocol.json"
SEAL_PATH = PROJECT_ROOT / "evaluation/outcome_v2_open_modern/source_seal.json"


def main() -> None:
    """Create target-free test inputs without printing any test answers."""
    result = seal_open_modern_source(
        SOURCE_PATH,
        DEVELOPMENT_PATH,
        TEST_INPUTS_PATH,
        PROTOCOL_PATH,
        SEAL_PATH,
    )
    print(
        f"Sealed {result.development_count} development rows and "
        f"{result.test_input_count} target-free test inputs."
    )
    print(f"Seal SHA-256: {result.seal_sha256}")


if __name__ == "__main__":
    main()
