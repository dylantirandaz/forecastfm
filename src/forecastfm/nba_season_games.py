"""Join play-by-play derivations with injury-report schedules into dated season games.

Play-by-play events carry no game date; the official injury reports carry every game day's
matchups and ET tip times. This module pairs them within one season: games sharing the same
away/home matchup are assigned to that matchup's report dates in game-ID order, which is
chronological within a season. Pair counts must agree exactly; mismatches raise and are reported,
never silently dropped. Neutral-site games are identified through the arena table.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time

from forecastfm.nba_arenas import NbaArena, game_arena
from forecastfm.nba_injury_report import ET_ZONE
from forecastfm.nba_pbp import PbpGame


class NbaSeasonGamesError(ValueError):
    """Raised when the schedule join cannot be made exactly."""


@dataclass(frozen=True, slots=True)
class SeasonGame:
    """One dated, located, fully derived regular-season game."""

    game_id: int
    season_label: int
    game_date: date
    tipoff: datetime
    away_abbreviation: str
    home_abbreviation: str
    away_score: int
    home_score: int
    arena: NbaArena
    pbp: PbpGame

    @property
    def home_won(self) -> bool:
        """Return whether the home team won."""
        return self.home_score > self.away_score


@dataclass(frozen=True, slots=True)
class ScheduleEntry:
    """One dated game from the injury-report schedule, with its ET tip time."""

    game_date: date
    away_abbreviation: str
    home_abbreviation: str
    tip_clock: tuple[int, int]

    def tipoff(self) -> datetime:
        """Return the scheduled tipoff as an aware UTC datetime."""
        hour, minute = self.tip_clock
        local = datetime.combine(self.game_date, time(hour, minute), tzinfo=ET_ZONE)
        return local.astimezone(UTC)


def join_season_games(
    pbp_games: list[PbpGame],
    schedule: list[ScheduleEntry],
) -> tuple[list[SeasonGame], list[str]]:
    """Join one season's play-by-play games with its report schedule.

    Returns the joined games sorted by (date, tipoff, game ID) plus human-readable notes for
    games present on exactly one side. Raises when a matchup pair's game counts disagree.
    """
    pbp_by_pair = _index_pbp(pbp_games)
    schedule_by_pair = _index_schedule(schedule)
    joined: list[SeasonGame] = [
        _join_one(game, entry)
        for pair in sorted(set(pbp_by_pair) | set(schedule_by_pair))
        for game, entry in _paired(pair, pbp_by_pair, schedule_by_pair)
    ]
    joined.sort(key=lambda game: (game.game_date, game.tipoff, game.game_id))
    schedule_keys = {(e.game_date, e.away_abbreviation, e.home_abbreviation) for e in schedule}
    joined_keys = {(g.game_date, g.away_abbreviation, g.home_abbreviation) for g in joined}
    joined_ids = {joined_game.game_id for joined_game in joined}
    notes: list[str] = [
        f"play-by-play game {game.game_id} has no schedule match"
        for game in pbp_games
        if game.game_id not in joined_ids
    ]
    notes.extend(
        f"scheduled game {key[1]}@{key[2]} on {key[0]} has no play-by-play match"
        for key in sorted(schedule_keys - joined_keys)
    )
    return joined, notes


def _paired(
    pair: tuple[str, str],
    pbp_by_pair: dict[tuple[str, str], list[PbpGame]],
    schedule_by_pair: dict[tuple[str, str], list[ScheduleEntry]],
) -> list[tuple[PbpGame, ScheduleEntry]]:
    games = pbp_by_pair.get(pair, [])
    entries = schedule_by_pair.get(pair, [])
    if len(games) != len(entries):
        raise NbaSeasonGamesError(
            f"matchup {pair[0]}@{pair[1]} has {len(games)} play-by-play games "
            f"but {len(entries)} scheduled dates"
        )
    return list(zip(games, entries, strict=True))


def _join_one(game: PbpGame, entry: ScheduleEntry) -> SeasonGame:
    return SeasonGame(
        game_id=game.game_id,
        season_label=game.season_label,
        game_date=entry.game_date,
        tipoff=entry.tipoff(),
        away_abbreviation=game.away_abbreviation,
        home_abbreviation=game.home_abbreviation,
        away_score=game.away_score,
        home_score=game.home_score,
        arena=game_arena(
            entry.game_date,
            game.away_abbreviation,
            game.home_abbreviation,
            entry.tipoff(),
        ),
        pbp=game,
    )


def _index_pbp(pbp_games: list[PbpGame]) -> dict[tuple[str, str], list[PbpGame]]:
    index: dict[tuple[str, str], list[PbpGame]] = {}
    for game in pbp_games:
        pair = (game.away_abbreviation, game.home_abbreviation)
        index.setdefault(pair, []).append(game)
    for games in index.values():
        games.sort(key=lambda game: game.game_id)
    return index


def _index_schedule(schedule: list[ScheduleEntry]) -> dict[tuple[str, str], list[ScheduleEntry]]:
    index: dict[tuple[str, str], list[ScheduleEntry]] = {}
    for entry in schedule:
        pair = (entry.away_abbreviation, entry.home_abbreviation)
        index.setdefault(pair, []).append(entry)
    for entries in index.values():
        entries.sort(key=lambda entry: (entry.game_date, entry.tip_clock))
    return index
