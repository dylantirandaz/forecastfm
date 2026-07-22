"""Emit pre-tipoff forecasts from collector-frozen prospective evidence.

Private research use. For each upcoming game in the prospective collector's schedule state
that has a completed T-60 capture, this emitter rebuilds the exact rl-prompt-v2.1 evidence
pack from strictly pregame inputs (the same pipeline as ``examples/build_rl_dataset.py``),
optionally samples a frozen RL sampler for its stated win probability, and writes one
create-only forecast row per game under
``data/processed/prospective_2026_27/rl_forecasts``. Every row binds the capture-ledger
SHA-256 of the injury-report PDF it consumed, so each forecast provably derives from bytes
frozen before tipoff.

Commands:

- ``plan`` lists games eligible for emission (offline, no writes).
- ``run`` emits forecasts for eligible games; ``--sampler PATH`` enables the RL arm
  (a Tinker sampler-weights path such as the run's ``rl-final`` artifact).
- ``report`` joins emitted forecasts with completed scores and prints per-arm log loss
  and Brier score. Preseason rows are reported separately and never enter the 2026-27 gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from math import log
from pathlib import Path
from typing import cast

if __package__ in {None, ""}:  # direct `python examples/emit_rl_forecasts.py` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tinker
from examples import run_private_prototype as prototype
from examples.train_tinker_outcome_rl import parse_stated_probability
from nbainjuries import injury  # pyright: ignore[reportMissingTypeStubs]
from tinker_cookbook import renderers, tokenizer_utils

from forecastfm.integrity import bytes_sha256, canonical_json
from forecastfm.json_utils import (
    parse_json_object,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.local_config import read_tinker_api_key
from forecastfm.nba_arenas import game_arena, is_neutral_site
from forecastfm.nba_feature_builder import (
    GameFeatures,
    PlayerValueInputs,
    build_game_features,
)
from forecastfm.nba_injury_report import InjuryReportRow, matchup_teams, rows_from_report_records
from forecastfm.nba_mov_elo import EloGameResult, MovEloRecipe, MovEloReplay
from forecastfm.nba_pbp import PbpGame, TeamGameStats, read_pbp_games
from forecastfm.nba_prototype_dataset import PrototypeGameRow, build_prototype_rows
from forecastfm.nba_rapm import fit_season_ratings_by_name
from forecastfm.nba_rl_dataset import build_prompt_v21
from forecastfm.nba_season_games import ScheduleEntry, SeasonGame

type JsonObject = dict[str, object]

STORAGE_ROOT = Path("data/raw/prospective")
STATE_PATH = STORAGE_ROOT / "state" / "schedule.json"
LEDGER_PATH = STORAGE_ROOT / "state" / "capture-ledger.jsonl"
OUTPUT_DIR = Path("data/processed/prospective_2026_27")
FORECAST_DIR = OUTPUT_DIR / "rl_forecasts"
SEASON_LABEL = 2027
SEASON_OPEN = date(2026, 10, 21)
SYNTHETIC_GAME_ID_BASE = 900_000_000
EMITTER_SCHEMA_VERSION = 1
RL_BASE_MODEL = "moonshotai/Kimi-K2.5"
RL_RENDERER = "kimi_k25"
RL_MAX_TOKENS = 2048
RL_SEASONS = (2022, 2023, 2024, 2025, 2026)


class EmitterError(RuntimeError):
    """Raised when the emitter cannot build an honest forecast."""


@dataclass(frozen=True, slots=True)
class UpcomingGame:
    """One upcoming game from the collector's schedule state."""

    event_id: str
    away: str
    home: str
    game_date_et: date
    tipoff: datetime
    t60_pdf_path: Path
    t60_pdf_sha256: str
    t60_scoreboard_sha256: str


@dataclass(frozen=True, slots=True)
class _Arguments:
    command: str
    sampler: str | None
    limit: int | None


@dataclass(slots=True)
class _RlArm:
    """The optional RL sampling arm: client, renderer, and tokenizer together."""

    client: tinker.SamplingClient
    renderer: object
    tokenizer: object


@dataclass(slots=True)
class _Context:
    """Rebuilt pipeline state shared across the games being emitted."""

    features_by_id: dict[int, GameFeatures]
    rows_by_id: dict[int, PrototypeGameRow]
    synthetic_ids: dict[str, int]


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the ``plan``, ``run``, and ``report`` subcommands."""
    args = _parse_arguments(argv)
    if args.command == "plan":
        return _cmd_plan()
    if args.command == "report":
        return _cmd_report()
    return _cmd_run(args)


def _parse_arguments(argv: Sequence[str] | None) -> _Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "report"])
    parser.add_argument("--sampler", default=None, help="Tinker sampler-weights path")
    parser.add_argument("--limit", type=int, default=None)
    parsed = parser.parse_args(argv)
    return _Arguments(
        command=cast("str", parsed.command),
        sampler=cast("str | None", parsed.sampler),
        limit=cast("int | None", parsed.limit),
    )


def _cmd_plan() -> int:
    games = _eligible_games()
    print(
        canonical_json(
            {
                "eligible": len(games),
                "games": [
                    {
                        "event_id": game.event_id,
                        "matchup": f"{game.away} @ {game.home}",
                        "tipoff": game.tipoff.isoformat(),
                    }
                    for game in games
                ],
            }
        )
    )
    return 0


def _cmd_run(args: _Arguments) -> int:
    games = _eligible_games()
    if args.limit is not None:
        games = games[: args.limit]
    pending = [game for game in games if not (FORECAST_DIR / f"{game.event_id}.json").exists()]
    if not pending:
        print("no eligible games without forecasts")
        return 0
    context = _build_context(pending)
    rl_arm = _open_rl_arm(args.sampler) if args.sampler is not None else None
    emitted: list[str] = []
    for game in pending:
        row = _emit_one(game, context, rl_arm)
        target = FORECAST_DIR / f"{game.event_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(canonical_json(row) + "\n", encoding="utf-8")
        temporary.replace(target)
        emitted.append(game.event_id)
    print(canonical_json({"emitted": len(emitted), "event_ids": emitted}))
    return 0


def _cmd_report() -> int:
    rows = _read_forecast_rows()
    if not rows:
        print("no forecasts emitted yet")
        return 0
    outcomes = _completed_outcomes()
    scored: dict[str, dict[str, list[tuple[float, int]]]] = {"regular": {}, "preseason": {}}
    for row in rows:
        key = (str(row["away"]), str(row["home"]), str(row["game_date_et"]))
        outcome = outcomes.get(key)
        if outcome is None:
            continue
        bucket = "preseason" if row.get("preseason") is True else "regular"
        arms = require_object(required_field(row, "arms"), "arms")
        for arm, value in arms.items():
            if isinstance(value, int | float):
                scored[bucket].setdefault(arm, []).append((float(value), outcome))
    report: JsonObject = {}
    for bucket, arms in scored.items():
        arm_reports: JsonObject = {}
        for arm, pairs in arms.items():
            log_loss = sum(-math.log(max(p if y else 1.0 - p, 1e-15)) for p, y in pairs)
            brier = sum((p - y) ** 2 for p, y in pairs)
            arm_reports[arm] = {
                "games": len(pairs),
                "log_loss": round(log_loss / len(pairs), 5),
                "brier": round(brier / len(pairs), 5),
            }
        report[bucket] = arm_reports
    print(canonical_json(report))
    return 0


def _eligible_games() -> list[UpcomingGame]:
    """Return schedule-state games that have a T-60 capture in the ledger."""
    if not STATE_PATH.exists() or not LEDGER_PATH.exists():
        return []
    state = parse_json_object(STATE_PATH.read_text(encoding="utf-8"))
    captures = _t60_captures()
    games: list[UpcomingGame] = []
    for entry in require_list(required_field(state, "games"), "games"):
        game = require_object(entry, "game")
        if game.get("removed") is True:
            continue
        event_id = require_string(required_field(game, "event_id"), "event_id")
        capture = captures.get(event_id)
        if capture is None:
            continue
        pdf_path, pdf_sha256, scoreboard_sha256 = capture
        games.append(
            UpcomingGame(
                event_id=event_id,
                away=require_string(required_field(game, "away"), "away"),
                home=require_string(required_field(game, "home"), "home"),
                game_date_et=date.fromisoformat(
                    require_string(required_field(game, "game_date_et"), "game_date_et")
                ),
                tipoff=datetime.fromisoformat(
                    require_string(required_field(game, "tipoff_utc"), "tipoff_utc")
                ),
                t60_pdf_path=pdf_path,
                t60_pdf_sha256=pdf_sha256,
                t60_scoreboard_sha256=scoreboard_sha256,
            )
        )
    return sorted(games, key=lambda game: game.tipoff)


def _t60_captures() -> dict[str, tuple[Path, str, str]]:
    """Index completed T-60 capture payloads by event id from the capture ledger."""
    captures: dict[str, tuple[Path, str, str]] = {}
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        event = parse_json_object(line)
        if event.get("event_type") != "capture":
            continue
        payload = require_object(required_field(event, "payload"), "payload")
        if payload.get("cutoff") != "t_60":
            continue
        event_id = require_string(required_field(payload, "event_id"), "event_id")
        records = require_object(required_field(payload, "captures"), "captures")
        injury = require_object(required_field(records, "injury_report_pdf"), "injury_report_pdf")
        scoreboard = require_object(required_field(records, "scoreboard"), "scoreboard")
        if injury.get("status") == "missing":
            continue
        pdf_rel = require_string(required_field(injury, "path"), "path")
        captures[event_id] = (
            STORAGE_ROOT / pdf_rel,
            require_string(required_field(injury, "sha256"), "sha256"),
            require_string(required_field(scoreboard, "sha256"), "sha256"),
        )
    return captures


def _replay_with_upcoming(
    seasons: list[list[EloGameResult]],
    upcoming_ids: set[int],
    recipe: MovEloRecipe | None = None,
) -> MovEloReplay:
    """Replay carryover MOV Elo, recording pre-game state for upcoming games.

    Mirrors ``replay_mov_elo`` exactly for completed games. Games whose id is in
    ``upcoming_ids`` record their pre-game ratings and home probability but apply no
    rating update, so several same-day fixtures all read the state after the last
    completed game. The consistency check against the sealed v2.1 dataset pins this
    math to the frozen baseline's.
    """
    settings = recipe or MovEloRecipe()
    ratings: dict[str, float] = {}
    recorded: dict[tuple[int, str], float] = {}
    probabilities: dict[int, float] = {}
    for season_games in seasons:
        ratings = {
            team: settings.initial_rating + settings.carryover * (rating - settings.initial_rating)
            for team, rating in ratings.items()
        }
        for game in season_games:
            home_rating = ratings.get(game.home_abbreviation, settings.initial_rating)
            away_rating = ratings.get(game.away_abbreviation, settings.initial_rating)
            advantage = 0.0 if game.neutral else settings.home_advantage
            difference = home_rating + advantage - away_rating
            expected_home = 1.0 / (1.0 + 10.0 ** (-difference / settings.rating_scale))
            recorded[(game.game_id, game.home_abbreviation)] = home_rating
            recorded[(game.game_id, game.away_abbreviation)] = away_rating
            probabilities[game.game_id] = expected_home
            if game.game_id in upcoming_ids:
                continue
            margin = abs(game.home_score - game.away_score)
            home_won = game.home_score > game.away_score
            winner_edge = difference if home_won else -difference
            multiplier = log(margin + 1.0) * 2.2 / (0.001 * winner_edge + 2.2)
            shift = settings.k_factor * multiplier * ((1.0 if home_won else 0.0) - expected_home)
            ratings[game.home_abbreviation] = home_rating + shift
            ratings[game.away_abbreviation] = away_rating - shift
    return MovEloReplay(ratings=recorded, home_probabilities=probabilities)


def _build_context(games: list[UpcomingGame], season_label: int = SEASON_LABEL) -> _Context:
    """Rebuild history plus season-to-date and append synthetic upcoming games."""
    notes: list[str] = []
    injury_snapshots = prototype.load_injury_index(prototype.INJURY_ARCHIVE)
    schedule = prototype.build_schedule(injury_snapshots)
    joined_by_season: dict[int, list[SeasonGame]] = {}
    for season in RL_SEASONS:
        joined, season_notes = prototype.load_season(
            season, prototype.SEASON_FILES[season], schedule
        )
        joined_by_season[season] = joined
        notes.extend(season_notes)
    synthetic: list[SeasonGame] = []
    synthetic_ids: dict[str, int] = {}
    for index, game in enumerate(games):
        game_id = SYNTHETIC_GAME_ID_BASE + index
        synthetic_ids[game.event_id] = game_id
        synthetic.append(_synthetic_game(game, game_id, season_label))
    season_games = [*_load_season_to_date(schedule, season_label), *synthetic]
    joined_by_season[season_label] = season_games
    sequences = _replay_sequences(joined_by_season, notes)
    replay = _replay_with_upcoming(sequences, set(synthetic_ids.values()))
    rapm = fit_season_ratings_by_name(prototype.RAPM_PRIOR_FILES, season_label, failures=notes)
    features, feature_notes = build_game_features(
        season_games,
        replay.ratings,
        injury_snapshots,
        player_values=PlayerValueInputs(flat=rapm),
    )
    notes.extend(feature_notes)
    rows = build_prototype_rows(season_games, features, replay.home_probabilities)
    return _Context(
        features_by_id={entry.game_id: entry for entry in features},
        rows_by_id={row.game_id: row for row in rows},
        synthetic_ids=synthetic_ids,
    )


def _replay_sequences(
    joined_by_season: dict[int, list[SeasonGame]], notes: list[str]
) -> list[list[EloGameResult]]:
    """Build the warmup-plus-joined Elo sequences exactly as ``elo_replay`` does."""
    sequences: list[list[EloGameResult]] = []
    for path in (prototype.WARMUP_FILES[season] for season in sorted(prototype.WARMUP_FILES)):
        failures: list[str] = []
        games = list(read_pbp_games(path, failures))
        notes.extend(failures)
        sequences.append(
            [
                EloGameResult(
                    game_id=game.game_id,
                    home_abbreviation=game.home_abbreviation,
                    away_abbreviation=game.away_abbreviation,
                    home_score=game.home_score,
                    away_score=game.away_score,
                    neutral=False,
                )
                for game in games
            ]
        )
    sequences.extend(
        [
            EloGameResult(
                game_id=game.game_id,
                home_abbreviation=game.home_abbreviation,
                away_abbreviation=game.away_abbreviation,
                home_score=game.home_score,
                away_score=game.away_score,
                neutral=is_neutral_site(
                    game.game_date, game.away_abbreviation, game.home_abbreviation
                ),
            )
            for game in joined_by_season[season]
        ]
        for season in sorted(joined_by_season)
    )
    return sequences


def _load_season_to_date(
    schedule: list[ScheduleEntry], season_label: int = SEASON_LABEL
) -> list[SeasonGame]:
    """Load completed games for the emitted season from its rolling ESPN CSV."""
    path = prototype.SEASON_FILES.get(season_label)
    if path is None or not path.exists():
        return []
    joined, _notes = prototype.load_season(season_label, path, schedule)
    return joined


def _synthetic_game(
    game: UpcomingGame, game_id: int, season_label: int = SEASON_LABEL
) -> SeasonGame:
    """Build a placeholder SeasonGame for one upcoming fixture.

    Feature construction for a game reads only strictly prior games, so placeholder scores
    and empty play-by-play never influence its own evidence pack; every synthetic game is
    the final entry for its season inside one emission run.
    """
    placeholder = PbpGame(
        game_id=game_id,
        away_abbreviation=game.away,
        home_abbreviation=game.home,
        away_score=0,
        home_score=0,
        team_stats=(
            TeamGameStats(
                team_abbreviation=game.away,
                points=0,
                field_goals_attempted=0,
                free_throws_attempted=0,
                offensive_rebounds=0,
                turnovers=0,
                starters=(),
            ),
            TeamGameStats(
                team_abbreviation=game.home,
                points=0,
                field_goals_attempted=0,
                free_throws_attempted=0,
                offensive_rebounds=0,
                turnovers=0,
                starters=(),
            ),
        ),
        player_lines=(),
        player_names={},
    )
    return SeasonGame(
        game_id=game_id,
        season_label=season_label,
        game_date=game.game_date_et,
        tipoff=game.tipoff,
        away_abbreviation=game.away,
        home_abbreviation=game.home,
        away_score=0,
        home_score=0,
        arena=game_arena(game.game_date_et, game.away, game.home, game.tipoff),
        pbp=placeholder,
    )


def _emit_one(game: UpcomingGame, context: _Context, rl_arm: _RlArm | None) -> JsonObject:
    """Build the evidence pack for one game and return its forecast row."""
    game_id = context.synthetic_ids[game.event_id]
    row = context.rows_by_id.get(game_id)
    features = context.features_by_id.get(game_id)
    if row is None or features is None:
        raise EmitterError(f"no prototype row built for {game.event_id}")
    injury_rows = _captured_game_rows(game)
    system, user = build_prompt_v21(row, features, injury_rows, swapped=False)
    arms: JsonObject = {"elo": round(row.elo_home_probability, 6)}
    if rl_arm is not None:
        arms["rl_stated"] = _sample_rl(rl_arm, system, user, seed=game_id % (2**31))
    return {
        "schema_version": EMITTER_SCHEMA_VERSION,
        "event_id": game.event_id,
        "away": game.away,
        "home": game.home,
        "matchup": f"{game.away} @ {game.home}",
        "game_date_et": game.game_date_et.isoformat(),
        "tipoff_utc": game.tipoff.isoformat(),
        "preseason": game.game_date_et < SEASON_OPEN,
        "arms": arms,
        "evidence": {
            "injury_pdf_sha256": game.t60_pdf_sha256,
            "scoreboard_sha256": game.t60_scoreboard_sha256,
        },
        "prompt_sha256": bytes_sha256(f"{system}\n{user}".encode()),
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _captured_game_rows(game: UpcomingGame) -> tuple[InjuryReportRow, ...]:
    """Parse the captured T-60 PDF and keep only this game's rows."""
    payload = game.t60_pdf_path.read_bytes()
    if bytes_sha256(payload) != game.t60_pdf_sha256:
        raise EmitterError(f"captured PDF hash mismatch for {game.event_id}")
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / game.t60_pdf_path.name.split("-", 3)[-1]
        target.write_bytes(payload)
        slot = game.tipoff.astimezone(UTC).replace(tzinfo=None)
        raw_json = injury.get_reportdata(  # pyright: ignore[reportUnknownMemberType]
            slot, local=True, localdir=tmp
        )
    records = cast("list[dict[str, object]]", json.loads(cast("str", raw_json)))
    parsed = rows_from_report_records(records, game.tipoff)
    return tuple(
        row
        for row in parsed.rows
        if row.game_date == game.game_date_et
        and matchup_teams(row.matchup) == (game.away, game.home)
    )


def _open_rl_arm(sampler_path: str) -> _RlArm:
    """Open the frozen RL sampler and its renderer/tokenizer."""
    api_key = read_tinker_api_key(Path(".env"))
    os.environ["TINKER_API_KEY"] = api_key
    service_client = tinker.ServiceClient(api_key=api_key)
    tokenizer = tokenizer_utils.get_tokenizer(RL_BASE_MODEL)
    return _RlArm(
        client=service_client.create_sampling_client(model_path=sampler_path),
        renderer=renderers.get_renderer(RL_RENDERER, tokenizer),
        tokenizer=tokenizer,
    )


def _sample_rl(arm: _RlArm, system: str, user: str, *, seed: int) -> float | None:
    """Sample the frozen RL sampler once and parse its stated probability."""
    conversation = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    build_prompt = getattr(arm.renderer, "build_generation_prompt", None)
    if not callable(build_prompt):
        raise EmitterError("RL renderer cannot build generation prompts")
    response = arm.client.sample(
        prompt=cast("tinker.ModelInput", build_prompt(conversation)),
        sampling_params=tinker.SamplingParams(max_tokens=RL_MAX_TOKENS, temperature=1.0, seed=seed),
        num_samples=1,
    ).result()
    decode = getattr(arm.tokenizer, "decode", None)
    if not callable(decode):
        raise EmitterError("RL tokenizer cannot decode completions")
    return parse_stated_probability(str(decode(list(response.sequences[0].tokens))))


def _read_forecast_rows() -> list[JsonObject]:
    if not FORECAST_DIR.exists():
        return []
    return [
        parse_json_object(path.read_text(encoding="utf-8"))
        for path in sorted(FORECAST_DIR.glob("*.json"))
    ]


def _completed_outcomes() -> dict[tuple[str, str, str], int]:
    """Map (away, home, game_date) to the home-win outcome for completed games."""
    outcomes: dict[tuple[str, str, str], int] = {}
    path = prototype.SEASON_FILES.get(SEASON_LABEL)
    if path is None or not path.exists():
        return outcomes
    schedule = prototype.build_schedule(prototype.load_injury_index(prototype.INJURY_ARCHIVE))
    joined, _notes = prototype.load_season(SEASON_LABEL, path, schedule)
    for game in joined:
        outcomes[(game.away_abbreviation, game.home_abbreviation, game.game_date.isoformat())] = (
            1 if game.home_won else 0
        )
    return outcomes


if __name__ == "__main__":
    raise SystemExit(main())
