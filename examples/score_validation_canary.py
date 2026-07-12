"""Score sealed base and adapter generations on the frozen validation canary."""

import json
from dataclasses import asdict
from pathlib import Path

from forecastfm.canary import load_sealed_generations, score_primary
from forecastfm.canary_history import score_historical
from forecastfm.integrity import file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANARY_DIRECTORY = PROJECT_ROOT / "evaluation" / "validation_canary"
MANIFEST_PATH = CANARY_DIRECTORY / "manifest.json"
PROMPTS_PATH = CANARY_DIRECTORY / "prompts.jsonl"
RAW_DIRECTORY = CANARY_DIRECTORY / "raw"
BASE_PATH = RAW_DIRECTORY / "base.jsonl"
ADAPTER_PATH = RAW_DIRECTORY / "adapter.jsonl"
SEAL_PATH = RAW_DIRECTORY / "manifest.json"
ANSWERS_PATH = PROJECT_ROOT / "data" / "processed" / "nba_elo_validation_answers.jsonl"
SCORES_PATH = CANARY_DIRECTORY / "scores.json"


def main() -> None:
    """Verify the output seal before either answer-free or historical scoring."""
    generations = load_sealed_generations(
        SEAL_PATH,
        MANIFEST_PATH,
        PROMPTS_PATH,
        BASE_PATH,
        ADAPTER_PATH,
    )
    primary = score_primary(generations)
    historical = score_historical(generations, ANSWERS_PATH)
    report = {
        "schema_version": 1,
        "kind": "forecastfm_validation_canary_scores",
        "warning": (
            "Historical scores are contamination-prone diagnostics, not prospective evidence."
        ),
        "commitments": {
            "generation_seal_sha256": file_sha256(SEAL_PATH),
            "canary_manifest_sha256": file_sha256(MANIFEST_PATH),
        },
        "primary_answer_free": asdict(primary),
        "secondary_historical": asdict(historical),
    }
    try:
        with SCORES_PATH.open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")
    except FileExistsError as error:
        raise RuntimeError(f"refusing to replace frozen scores: {SCORES_PATH}") from error
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
