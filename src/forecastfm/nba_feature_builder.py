"""Assemble per-game side features from play-by-play, injury reports, and Elo state.

This is the binding layer between the raw derivations and the frozen rich schema: it normalizes
player names across the two sources (play-by-play uses ``First Last``, the official report uses
``Last, First``), selects the latest pre-T-60 report snapshot per game, and computes the two
local-only availability aggregates per side. Availability minutes are median expected minutes
over each player's last ten appearances (a disclosed prototype variant of the frozen
prior-game-minutes definition, so season-long absences are priced rather than zeroed).
Reported players with no matching play-by-play history contribute zero, per the frozen
no-history rule; their names are counted and disclosed in the manifest. Games with no
pre-cutoff report at all get no health features and are excluded from the availability
ablation only.

Two availability pricing policies are disclosed. The default is the frozen binary rule:
Out and Doubtful count as fully unavailable (weight 1.0) and every other status as fully
available (weight 0.0). The variant passes a ``StatusPlayRates`` (from
``forecastfm.nba_status_rates``) through ``build_game_features``; each listed player then
contributes with effective unavailability weight ``1 - play_rate[status]`` applied to
both health aggregates (minutes and value). The variant is off unless the rates are
explicitly supplied.

A disclosed prototype variant also computes each side's projected rotation value from the
same selected snapshot: the minutes-weighted player value of the rotation pool (players
with at least three appearances in the team's last ten games), with players listed Out or
Doubtful excluded from the numerator but not the denominator. It is None exactly when no
pre-cutoff snapshot exists.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from forecastfm.nba_injury_report import (
    UNAVAILABLE_STATUSES,
    InjuryReportRow,
    matchup_teams,
)
from forecastfm.nba_pbp import normalize_player_name
from forecastfm.nba_season_games import SeasonGame
from forecastfm.nba_team_history import GameContext, NbaTeamHistory, TeamSideFeatures

if TYPE_CHECKING:
    from forecastfm.nba_status_rates import StatusPlayRates

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
    """Both sides' standard features and optional health aggregates for one game.

    ``projected_rotation`` is a disclosed prototype variant (away, home) computed from the
    same selected pre-T-60 snapshot as ``health``; it defaults to None so existing
    constructors keep working.
    """

    game_id: int
    away: TeamSideFeatures
    home: TeamSideFeatures
    health: tuple[tuple[float, float], tuple[float, float]] | None
    projected_rotation: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class InjurySnapshot:
    """One retained report snapshot for one date."""

    report_time: datetime
    rows: tuple[InjuryReportRow, ...]


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
    """Extract (date, away, home, clock) entries from report rows' own game dates.

    Postponed games are listed on their original date and then vanish from later same-day
    snapshots, so an entry survives only when the game's date still lists it in that date's
    final retained snapshot.
    """
    by_report_date: dict[date, list[InjurySnapshot]] = {}
    for snapshot in snapshots:
        by_report_date.setdefault(snapshot.report_time.date(), []).append(snapshot)
    latest_clock: dict[tuple[date, str, str], tuple[int, int]] = {}
    for report_date, day_snapshots in sorted(by_report_date.items()):
        final_snapshot = max(day_snapshots, key=lambda snapshot: snapshot.report_time)
        for row in final_snapshot.rows:
            if row.game_date != report_date:
                continue
            away, home = matchup_teams(row.matchup)
            latest_clock[(row.game_date, away, home)] = row.game_clock_et
    return [(day, away, home, clock) for (day, away, home), clock in sorted(latest_clock.items())]


@dataclass(frozen=True, slots=True)
class PlayerValueInputs:
    """Optional player-value maps: a flat season map and/or a per-(game, team) map."""

    flat: Mapping[str, float] | None = None
    by_game: Mapping[tuple[int, str], Mapping[str, float]] | None = None


def injury_rows_for_game(
    game: SeasonGame,
    snapshots: list[InjurySnapshot],
) -> tuple[InjuryReportRow, ...]:
    """Return the selected pre-T-60 snapshot's rows for one game's matchup, or none.

    Uses the same selection rule as the health-feature path: the latest snapshot at or before
    tipoff minus 60 minutes that contains the matchup. Returns an empty tuple when no such
    snapshot exists.
    """
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
        return ()
    return tuple(
        row
        for row in selected.rows
        if row.game_date == game.game_date
        and matchup_teams(row.matchup) == (game.away_abbreviation, game.home_abbreviation)
    )


def build_game_features(
    games: Iterable[SeasonGame],
    elo_ratings: Mapping[tuple[int, str], float],
    injury_snapshots: list[InjurySnapshot],
    player_values: PlayerValueInputs | None = None,
    status_rates: StatusPlayRates | None = None,
) -> tuple[list[GameFeatures], list[str]]:
    """Compute per-side features for one season in strict tipoff order.

    State resets at the start of each call, so one call covers exactly one season. Returns the
    per-game features plus notes for games without any pre-cutoff report snapshot. When
    ``status_rates`` is supplied, the disclosed empirical play-rate variant prices every
    listed status by ``1 - play_rate`` instead of the frozen binary rule. When
    ``player_ratings_by_game`` is supplied it takes precedence per (game, team), supporting
    time-varying ratings such as the Kalman filter; ``player_ratings`` is the flat fallback.
    """
    histories: dict[str, NbaTeamHistory] = {}
    features: list[GameFeatures] = []
    notes: list[str] = []
    health_context = HealthContext(
        injury_snapshots,
        notes,
        player_values.flat if player_values is not None else None,
        status_rates,
    )
    for game in games:
        away_history = histories.setdefault(
            game.away_abbreviation, NbaTeamHistory(game.away_abbreviation)
        )
        home_history = histories.setdefault(
            game.home_abbreviation, NbaTeamHistory(game.home_abbreviation)
        )
        away_context = GameContext(game.game_date, game.tipoff, False, game.arena)
        home_context = GameContext(game.game_date, game.tipoff, True, game.arena)
        away_values = _values_for(player_values, game, False)
        home_values = _values_for(player_values, game, True)
        pregame = _pregame_report(game, health_context, away_history, home_history)
        if pregame is None:
            health, projected_rotation = None, None
        else:
            health, projected_rotation = pregame
        features.append(
            GameFeatures(
                game_id=game.game_id,
                away=away_history.features_for(away_context, away_values),
                home=home_history.features_for(home_context, home_values),
                health=health,
                projected_rotation=projected_rotation,
            )
        )
        away_elo = elo_ratings[(game.game_id, game.away_abbreviation)]
        home_elo = elo_ratings[(game.game_id, game.home_abbreviation)]
        away_history.record_game(game.pbp, away_context, home_elo)
        home_history.record_game(game.pbp, home_context, away_elo)
    return features, notes


def _values_for(
    player_values: PlayerValueInputs | None,
    game: SeasonGame,
    home: bool,
) -> Mapping[str, float] | None:
    """Return the player-value map for one side, per-game map first, flat fallback."""
    if player_values is None:
        return None
    team = game.home_abbreviation if home else game.away_abbreviation
    if player_values.by_game is not None:
        return player_values.by_game.get((game.game_id, team), player_values.flat)
    return player_values.flat


class HealthContext:
    """Shared health-assembly inputs for one season pass."""

    def __init__(
        self,
        snapshots: list[InjurySnapshot],
        notes: list[str],
        player_ratings: Mapping[str, float] | None,
        status_rates: StatusPlayRates | None = None,
    ) -> None:
        """Initialize the shared health context."""
        self.snapshots = snapshots
        self.notes = notes
        self.player_ratings = player_ratings
        self.status_rates = status_rates


def _pregame_report(
    game: SeasonGame,
    context: HealthContext,
    away_history: NbaTeamHistory,
    home_history: NbaTeamHistory,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], tuple[float, float]] | None:
    """Compute both snapshot-derived variants from the same selected pre-T-60 snapshot."""
    cutoff = game.tipoff - timedelta(minutes=60)
    selected = next(
        (
            snapshot
            for snapshot in reversed(context.snapshots)
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
        context.notes.append(
            f"game {game.game_id} has no report snapshot at or before its T-60 cutoff"
        )
        return None
    away = side_health(selected.rows, game.away_abbreviation, away_history, game, context)
    home = side_health(selected.rows, game.home_abbreviation, home_history, game, context)
    projected = (
        side_projected_rotation(selected.rows, game.away_abbreviation, away_history, game, context),
        side_projected_rotation(selected.rows, game.home_abbreviation, home_history, game, context),
    )
    return (away, home), projected


def _side_rows(
    rows: tuple[InjuryReportRow, ...],
    team_abbreviation: str,
    game: SeasonGame,
) -> list[InjuryReportRow]:
    return [
        row
        for row in rows
        if TEAM_NAME_TO_ABBREVIATION.get(row.team, "") == team_abbreviation
        and matchup_teams(row.matchup) == (game.away_abbreviation, game.home_abbreviation)
    ]


def _side_values(
    history: NbaTeamHistory,
    game: SeasonGame,
    context: HealthContext,
) -> Mapping[str, float]:
    if context.player_ratings is not None:
        return context.player_ratings
    names = game.pbp.player_names
    return {
        _name_key(names[player_id]): value
        for player_id, value in history.rolling_values().items()
        if player_id in names
    }


def side_projected_rotation(
    rows: tuple[InjuryReportRow, ...],
    team_abbreviation: str,
    history: NbaTeamHistory,
    game: SeasonGame,
    context: HealthContext,
) -> float:
    """Compute one side's projected-rotation value from the selected report rows."""
    unavailable = frozenset(
        _name_key(row.player_name)
        for row in _side_rows(rows, team_abbreviation, game)
        if row.status in UNAVAILABLE_STATUSES
    )
    return history.projected_rotation_value(_side_values(history, game, context), unavailable)


def side_health(
    rows: tuple[InjuryReportRow, ...],
    team_abbreviation: str,
    history: NbaTeamHistory,
    game: SeasonGame,
    context: HealthContext,
) -> tuple[float, float]:
    """Compute one side's two availability aggregates from the selected report rows."""
    side_rows = _side_rows(rows, team_abbreviation, game)
    names = game.pbp.player_names
    expected = history.expected_minutes()
    prior_minutes = {
        _name_key(names[player_id]): minutes
        for player_id, minutes in expected.items()
        if player_id in names
    }
    values = _side_values(history, game, context)
    total_minutes = 0.0
    total_value = 0.0
    for row in side_rows:
        weight = _unavailability_weight(row.status, context.status_rates)
        if weight <= 0.0:
            continue
        key = _name_key(row.player_name)
        if key not in prior_minutes and key not in values:
            context.notes.append(
                f"game {game.game_id} team {team_abbreviation} has an unavailable row "
                "with no play-by-play history; contributing zero"
            )
        minutes = prior_minutes.get(key, 0.0)
        total_minutes += weight * minutes
        total_value += weight * minutes * values.get(key, 0.0)
    return (total_minutes, total_value)


def _unavailability_weight(status: str, status_rates: StatusPlayRates | None) -> float:
    """Return the effective unavailability weight for one reported status.

    The default frozen binary rule prices Out and Doubtful as fully unavailable and every
    other status as fully available; the disclosed empirical variant prices each status by
    one minus its play rate.
    """
    if status_rates is None:
        return 1.0 if status in UNAVAILABLE_STATUSES else 0.0
    rate = status_rates.rates.get(status)
    if rate is None:
        raise NbaFeatureBuilderError(f"status play rates are missing status {status!r}")
    return 1.0 - rate


def _name_key(name: str) -> str:
    return " ".join(normalize_player_name(name))
