"""Sequential revision-stream features from successive injury-report snapshots.

The static T-60 model reads one pre-cutoff availability state per game. This module asks the
follow-up question: does the *stream* of status revisions between reports carry signal of its
own? For each game and each horizon (360, 120, 60, 15 minutes before tipoff) it selects the
latest retained snapshot at or before tipoff minus the horizon that lists the game's matchup —
the same containment-fallback rule as ``nba_feature_builder._health_for_game`` — and compares it
against the immediately previous retained snapshot on the same report date that also lists the
matchup. Status classes are ordered Available < Probable < Questionable < Doubtful < Out; a
player listed in only one of the two snapshots counts against the implicit ``Available`` class
of the unlisted state. When no previous same-date snapshot exists, deltas are zero and
``minutes_since_last_change`` saturates at the horizon (no change observed inside the window).
Deltas are weighted by causal RAPM ratings keyed by normalized player name (the same
space-joined sorted-token key as ``nba_rapm.fit_season_ratings_by_name``); unrated players
contribute zero. Model rows are home-minus-away differences of the two sides' deltas, with the
recency difference taken in log1p minutes. All times are timezone-aware; report times are
America/New_York, tipoffs UTC.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from math import log1p

from forecastfm.nba_feature_builder import TEAM_NAME_TO_ABBREVIATION, InjurySnapshot
from forecastfm.nba_injury_report import KNOWN_STATUSES, InjuryReportRow, matchup_teams
from forecastfm.nba_pbp import normalize_player_name
from forecastfm.nba_prototype_dataset import question_id_for

NBA_REVISION_SCHEMA_VERSION = 1
REVISION_HORIZON_MINUTES: tuple[int, ...] = (360, 120, 60, 15)

_STATUS_ORDER = ("Available", "Probable", "Questionable", "Doubtful", "Out")
STATUS_RANK: dict[str, int] = {status: rank for rank, status in enumerate(_STATUS_ORDER)}

REVISION_FEATURE_NAMES: tuple[str, ...] = (
    "downgrade_value_diff",
    "upgrade_value_diff",
    "minutes_since_last_change_log1p_diff",
    "changes_count_diff",
)

MatchupKey = tuple[date, str, str]  # (game date, away, home)


class NbaRevisionError(ValueError):
    """Raised when revision-stream inputs violate their contract."""


if frozenset(STATUS_RANK) != KNOWN_STATUSES:
    raise NbaRevisionError("status ordering must cover exactly the known report statuses")


@dataclass(frozen=True, slots=True)
class RevisionGame:
    """The game identity the revision stream needs: matchup, tipoff, and answer."""

    game_id: int
    season: int
    game_date: date
    tipoff: datetime
    away_abbreviation: str
    home_abbreviation: str
    home_won: bool


@dataclass(frozen=True, slots=True)
class RevisionGameContext:
    """Matchup identity, RAPM weights, and one horizon's timing for delta computation."""

    away_abbreviation: str
    home_abbreviation: str
    player_ratings: Mapping[str, float]
    horizon_time: datetime
    horizon_minutes: int


@dataclass(frozen=True, slots=True)
class SideRevisionDeltas:
    """One side's revision deltas between the selected and previous same-date snapshots."""

    downgrade_value: float
    upgrade_value: float
    minutes_since_last_change: float
    changes_count: int


@dataclass(frozen=True, slots=True)
class RevisionGameRow:
    """One home-perspective revision row for one game at one horizon."""

    question_id: str
    game_id: int
    season: int
    game_date: date
    horizon_minutes: int
    elo_home_probability: float
    features: tuple[float, ...]
    home_won: bool


@dataclass(frozen=True, slots=True)
class RevisionBuildResult:
    """All retained rows plus the count of (game, horizon) pairs with no snapshot."""

    rows: tuple[RevisionGameRow, ...]
    skipped: int
    games_total: int


def status_rank(status: str) -> int:
    """Return the availability-class rank of one report status (worse is higher)."""
    try:
        return STATUS_RANK[status]
    except KeyError as exc:
        raise NbaRevisionError(f"unknown report status: {status!r}") from exc


def select_snapshot_at_horizon(
    snapshots: list[InjurySnapshot],
    game: RevisionGame,
    horizon_minutes: int,
) -> InjurySnapshot | None:
    """Return the latest snapshot at or before tipoff minus the horizon listing the matchup.

    Mirrors ``nba_feature_builder._health_for_game``: snapshots without a row for the game are
    skipped, so selection falls back to an earlier snapshot that does contain the matchup.
    """
    cutoff = game.tipoff - timedelta(minutes=horizon_minutes)
    return next(
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


def side_revision_deltas(
    current: InjurySnapshot,
    previous: InjurySnapshot | None,
    team_abbreviation: str,
    context: RevisionGameContext,
) -> SideRevisionDeltas:
    """Compute one side's RAPM-weighted status-revision deltas for one game horizon."""
    if previous is None:
        return SideRevisionDeltas(0.0, 0.0, float(context.horizon_minutes), 0)
    current_map = _status_map(current.rows, team_abbreviation, context)
    previous_map = _status_map(previous.rows, team_abbreviation, context)
    downgrade = 0.0
    upgrade = 0.0
    changes = 0
    for player in current_map.keys() | previous_map.keys():
        old_rank = previous_map.get(player, STATUS_RANK["Available"])
        new_rank = current_map.get(player, STATUS_RANK["Available"])
        if new_rank == old_rank:
            continue
        changes += 1
        weight = context.player_ratings.get(player, 0.0)
        if new_rank > old_rank:
            downgrade += weight
        else:
            upgrade += weight
    if changes == 0:
        minutes = float(context.horizon_minutes)
    else:
        age = context.horizon_time - current.report_time.astimezone(UTC)
        minutes = max(0.0, age.total_seconds() / 60.0)
    return SideRevisionDeltas(downgrade, upgrade, minutes, changes)


def revision_features(
    away: SideRevisionDeltas,
    home: SideRevisionDeltas,
) -> tuple[float, ...]:
    """Return the home-minus-away revision feature vector in REVISION_FEATURE_NAMES order."""
    return (
        home.downgrade_value - away.downgrade_value,
        home.upgrade_value - away.upgrade_value,
        log1p(home.minutes_since_last_change) - log1p(away.minutes_since_last_change),
        float(home.changes_count - away.changes_count),
    )


def build_revision_rows(
    games: Iterable[RevisionGame],
    snapshots: list[InjurySnapshot],
    home_probabilities: Mapping[int, float],
    player_ratings: Mapping[str, float],
    horizons: tuple[int, ...] = REVISION_HORIZON_MINUTES,
) -> RevisionBuildResult:
    """Build revision rows for one season's games at every horizon.

    (Game, horizon) pairs with no snapshot containing the matchup at or before the horizon
    cutoff are skipped and counted, never silently filled.
    """
    index = _MatchupIndex(snapshots)
    game_list = list(games)
    rows: list[RevisionGameRow] = []
    skipped = 0
    for game in game_list:
        game_rows, game_skipped = _rows_for_game(
            game, index, home_probabilities, player_ratings, horizons
        )
        rows.extend(game_rows)
        skipped += game_skipped
    return RevisionBuildResult(rows=tuple(rows), skipped=skipped, games_total=len(game_list))


def _rows_for_game(
    game: RevisionGame,
    index: _MatchupIndex,
    home_probabilities: Mapping[int, float],
    player_ratings: Mapping[str, float],
    horizons: tuple[int, ...],
) -> tuple[list[RevisionGameRow], int]:
    key: MatchupKey = (game.game_date, game.away_abbreviation, game.home_abbreviation)
    rows: list[RevisionGameRow] = []
    skipped = 0
    for horizon in horizons:
        selected = index.select(key, game.tipoff - timedelta(minutes=horizon))
        if selected is None:
            skipped += 1
            continue
        previous = index.previous_same_date(key, selected)
        context = RevisionGameContext(
            away_abbreviation=game.away_abbreviation,
            home_abbreviation=game.home_abbreviation,
            player_ratings=player_ratings,
            horizon_time=game.tipoff - timedelta(minutes=horizon),
            horizon_minutes=horizon,
        )
        away = side_revision_deltas(selected, previous, game.away_abbreviation, context)
        home = side_revision_deltas(selected, previous, game.home_abbreviation, context)
        rows.append(
            RevisionGameRow(
                question_id=question_id_for(game.game_id),
                game_id=game.game_id,
                season=game.season,
                game_date=game.game_date,
                horizon_minutes=horizon,
                elo_home_probability=home_probabilities[game.game_id],
                features=revision_features(away, home),
                home_won=game.home_won,
            )
        )
    return rows, skipped


def _status_map(
    rows: tuple[InjuryReportRow, ...],
    team_abbreviation: str,
    context: RevisionGameContext,
) -> dict[str, int]:
    matchup = (context.away_abbreviation, context.home_abbreviation)
    statuses: dict[str, int] = {}
    for row in rows:
        if TEAM_NAME_TO_ABBREVIATION.get(row.team, "") != team_abbreviation:
            continue
        if matchup_teams(row.matchup) != matchup:
            continue
        statuses[_name_key(row.player_name)] = status_rank(row.status)
    return statuses


def _name_key(name: str) -> str:
    return " ".join(normalize_player_name(name))


class _MatchupIndex:
    """Per-matchup snapshot lists in report-time order, with UTC keys for bisection."""

    def __init__(self, snapshots: list[InjurySnapshot]) -> None:
        by_matchup: dict[MatchupKey, list[InjurySnapshot]] = {}
        for snapshot in snapshots:
            keys = {(row.game_date, *matchup_teams(row.matchup)) for row in snapshot.rows}
            for key in keys:
                by_matchup.setdefault(key, []).append(snapshot)
        self._by_matchup = by_matchup
        self._times = {
            key: [snapshot.report_time.astimezone(UTC) for snapshot in entries]
            for key, entries in by_matchup.items()
        }

    def select(self, key: MatchupKey, cutoff: datetime) -> InjurySnapshot | None:
        """Return the latest snapshot listing the matchup at or before the cutoff."""
        entries = self._by_matchup.get(key)
        if not entries:
            return None
        position = bisect_right(self._times[key], cutoff) - 1
        return entries[position] if position >= 0 else None

    def previous_same_date(
        self,
        key: MatchupKey,
        selected: InjurySnapshot,
    ) -> InjurySnapshot | None:
        """Return the immediately previous same-report-date snapshot listing the matchup."""
        entries = self._by_matchup[key]
        position = entries.index(selected)
        report_date = selected.report_time.date()
        for candidate in reversed(entries[:position]):
            if candidate.report_time.date() == report_date:
                return candidate
        return None
