"""Assemble per-game side features from play-by-play, injury reports, and Elo state.

This is the binding layer between the raw derivations and the frozen rich schema: it normalizes
player names across the two sources (play-by-play uses ``First Last``, the official report uses
``Last, First``), selects the latest pre-T-60 report snapshot per game, and computes the two
local-only availability aggregates per side. Reported players with no matching play-by-play
history contribute zero, per the frozen no-history rule; their names are counted and disclosed
in the manifest. Games with no pre-cutoff report at all get no health features and are excluded
from the availability ablation only.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from forecastfm.nba_injury_report import (
    UNAVAILABLE_STATUSES,
    InjuryReportRow,
    matchup_teams,
)
from forecastfm.nba_season_games import SeasonGame
from forecastfm.nba_team_history import GameContext, NbaTeamHistory, TeamSideFeatures

NBA_FEATURE_BUILDER_SCHEMA_VERSION = 1

TEAM_NAME_TO_ABBREVIATION: dict[str, str] = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "LA Clippers": "LAC",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


class NbaFeatureBuilderError(ValueError):
    """Raised when feature assembly violates its causal contract."""


@dataclass(frozen=True, slots=True)
class GameFeatures:
    """Both sides' standard features and optional health aggregates for one game."""

    game_id: int
    away: TeamSideFeatures
    home: TeamSideFeatures
    health: tuple[tuple[float, float], tuple[float, float]] | None


@dataclass(frozen=True, slots=True)
class InjurySnapshot:
    """One retained report snapshot for one date."""

    report_time: datetime
    rows: tuple[InjuryReportRow, ...]


def normalize_player_name(name: str) -> tuple[str, ...]:
    """Reduce a player name to a sorted tuple of casefolded, accent-free tokens."""
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    collapsed = stripped.replace(".", "").replace("'", "")
    tokens = "".join(character if character.isalnum() else " " for character in collapsed).split()
    return tuple(sorted(token.casefold() for token in tokens))


def load_injury_index(archive_root: Path) -> list[InjurySnapshot]:
    """Load every archived report snapshot, ordered by report time.

    Evening reports list games for the following day as well as the same day, so snapshots are
    indexed by their own report time only; game dates come from each row itself.
    """
    snapshots: list[InjurySnapshot] = []
    for rows_path in sorted(archive_root.glob("*/*.rows.jsonl")):
        rows = _read_rows(rows_path)
        if rows:
            snapshots.append(InjurySnapshot(report_time=rows[0].report_time, rows=rows))
    snapshots.sort(key=lambda snapshot: snapshot.report_time)
    return snapshots


def _read_rows(rows_path: Path) -> tuple[InjuryReportRow, ...]:
    rows: list[InjuryReportRow] = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        clock_hour, clock_minute = (int(part) for part in payload["game_clock_et"].split(":"))
        rows.append(
            InjuryReportRow(
                report_time=datetime.fromisoformat(payload["report_time"]),
                game_date=date.fromisoformat(payload["game_date"]),
                game_clock_et=(clock_hour, clock_minute),
                matchup=str(payload["matchup"]),
                team=str(payload["team"]),
                player_name=str(payload["player_name"]),
                status=str(payload["status"]),
            )
        )
    return tuple(rows)


def schedule_from_injury_index(
    snapshots: Iterable[InjurySnapshot],
) -> list[tuple[date, str, str, tuple[int, int]]]:
    """Extract (date, away, home, clock) entries from report rows' own game dates."""
    seen: set[tuple[date, str, str, tuple[int, int]]] = set()
    schedule: list[tuple[date, str, str, tuple[int, int]]] = []
    for snapshot in snapshots:
        for row in snapshot.rows:
            away, home = matchup_teams(row.matchup)
            key = (row.game_date, away, home, row.game_clock_et)
            if key not in seen:
                seen.add(key)
                schedule.append(key)
    schedule.sort()
    return schedule


def build_game_features(
    games: Iterable[SeasonGame],
    elo_ratings: Mapping[tuple[int, str], float],
    injury_snapshots: list[InjurySnapshot],
) -> tuple[list[GameFeatures], list[str]]:
    """Compute per-side features for one season in strict tipoff order.

    State resets at the start of each call, so one call covers exactly one season. Returns the
    per-game features plus notes for games without any pre-cutoff report snapshot.
    """
    histories: dict[str, NbaTeamHistory] = {}
    features: list[GameFeatures] = []
    notes: list[str] = []
    for game in games:
        away_history = histories.setdefault(
            game.away_abbreviation, NbaTeamHistory(game.away_abbreviation)
        )
        home_history = histories.setdefault(
            game.home_abbreviation, NbaTeamHistory(game.home_abbreviation)
        )
        away_context = GameContext(game.game_date, game.tipoff, False, game.arena)
        home_context = GameContext(game.game_date, game.tipoff, True, game.arena)
        health = _health_for_game(game, injury_snapshots, away_history, home_history, notes)
        features.append(
            GameFeatures(
                game_id=game.game_id,
                away=away_history.features_for(away_context),
                home=home_history.features_for(home_context),
                health=health,
            )
        )
        away_elo = elo_ratings[(game.game_id, game.away_abbreviation)]
        home_elo = elo_ratings[(game.game_id, game.home_abbreviation)]
        away_history.record_game(game.pbp, away_context, home_elo)
        home_history.record_game(game.pbp, home_context, away_elo)
    return features, notes


def _health_for_game(
    game: SeasonGame,
    snapshots: list[InjurySnapshot],
    away_history: NbaTeamHistory,
    home_history: NbaTeamHistory,
    notes: list[str],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    cutoff = game.tipoff - timedelta(minutes=60)
    selected = next(
        (
            snapshot
            for snapshot in reversed(snapshots)
            if snapshot.report_time.astimezone(UTC) <= cutoff
            and any(
                row.game_date == game.game_date
                and matchup_teams(row.matchup) == (game.away_abbreviation, game.home_abbreviation)
                for row in snapshot.rows
            )
        ),
        None,
    )
    if selected is None:
        notes.append(f"game {game.game_id} has no report snapshot at or before its T-60 cutoff")
        return None
    away = _side_health(selected.rows, game.away_abbreviation, away_history, game, notes)
    home = _side_health(selected.rows, game.home_abbreviation, home_history, game, notes)
    return (away, home)


def _side_health(
    rows: tuple[InjuryReportRow, ...],
    team_abbreviation: str,
    history: NbaTeamHistory,
    game: SeasonGame,
    notes: list[str],
) -> tuple[float, float]:
    side_rows = [
        row
        for row in rows
        if TEAM_NAME_TO_ABBREVIATION.get(row.team, "") == team_abbreviation
        and matchup_teams(row.matchup) == (game.away_abbreviation, game.home_abbreviation)
    ]
    names = game.pbp.player_names
    prior_minutes = {
        _name_key(names[player_id]): minutes
        for player_id, minutes in history.prior_game_minutes().items()
        if player_id in names
    }
    values = {
        _name_key(names[player_id]): value
        for player_id, value in history.rolling_values().items()
        if player_id in names
    }
    total_minutes = 0.0
    total_value = 0.0
    for row in side_rows:
        if row.status not in UNAVAILABLE_STATUSES:
            continue
        key = _name_key(row.player_name)
        if key not in prior_minutes and key not in values:
            notes.append(
                f"game {game.game_id} team {team_abbreviation} has an unavailable row "
                "with no play-by-play history; contributing zero"
            )
        minutes = prior_minutes.get(key, 0.0)
        total_minutes += minutes
        total_value += minutes * values.get(key, 0.0)
    return (total_minutes, total_value)


def _name_key(name: str) -> str:
    return " ".join(normalize_player_name(name))
