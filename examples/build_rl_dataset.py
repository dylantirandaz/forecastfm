"""Build and seal the rl-prompt-v1 RL training question set (protocol prerequisite 5).

Rebuilds every regular-season game for labels 2022-2026 from local retained data via the
private prototype helpers (carryover MOV Elo replay with warmup from 2016-17, RAPM player
values, the 11 standard features), then writes three create-only artifacts under
data/processed/rl_dataset/: prompts.jsonl (both orientations per game, no outcomes),
answers.jsonl (winner labels in a separate file), and manifest.json (per-season counts,
template hash, file hashes, chronological split note, decision-2a=A disclosure). Existing
artifacts are never replaced; rerun only after removing them deliberately.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

if __package__ in {None, ""}:  # direct `python examples/build_rl_dataset.py` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples import run_private_prototype as prototype

from forecastfm.nba_prototype_dataset import PrototypeGameRow
from forecastfm.nba_rl_dataset import seal_rl_dataset, verify_sealed_dataset

OUTPUT_DIR = Path("data/processed/rl_dataset")
RL_SEASONS = (2022, 2023, 2024, 2025, 2026)


def build_rows() -> tuple[list[PrototypeGameRow], list[str]]:
    """Rebuild the prototype rows for every RL season from local data."""
    notes: list[str] = []
    injury_snapshots = prototype.load_injury_index(prototype.INJURY_ARCHIVE)
    schedule = prototype.build_schedule(injury_snapshots)
    joined_by_season = {}
    for season in RL_SEASONS:
        joined, season_notes = prototype.load_season(
            season, prototype.SEASON_FILES[season], schedule
        )
        joined_by_season[season] = joined
        notes.extend(season_notes)
    replay = prototype.elo_replay(joined_by_season, notes)
    rows: list[PrototypeGameRow] = []
    for season in RL_SEASONS:
        season_rows, _features, season_notes = prototype.season_rows(
            season, joined_by_season[season], replay, injury_snapshots
        )
        rows.extend(season_rows)
        notes.extend(season_notes)
    return rows, notes


def main(argv: Sequence[str] | None = None) -> int:
    """Seal the RL question set and confirm the sealed hashes reproduce."""
    del argv
    rows, notes = build_rows()
    manifest = seal_rl_dataset(rows, OUTPUT_DIR)
    verify_sealed_dataset(OUTPUT_DIR)
    print(json.dumps(manifest, indent=2))
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
