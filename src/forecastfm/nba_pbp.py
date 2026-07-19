"""Derive games, scores, rotation, and rating inputs from stats.nba.com play-by-play CSVs.

Reads the shufinskiy/nba_data ``nbastats`` per-season CSV (pinned by commit and SHA-256) in one
streaming pass, one game at a time, and emits validated per-game derivations: home/away identity,
final score, inferred starters, player minutes from substitution stints, team counting stats, and
per-player plus-minus from score changes. Everything is strictly postgame information; causality
comes from lagging these derivations by one or more games downstream.

Starters and period-start lineups are inferred per period as the players appearing in that
period's events minus those whose first substitution of the period was an entry. Between-period
lineup changes are not recorded as substitution events, so a player who left during the break can
appear only as a phantom later exit; lineups larger than five are repaired by dropping such
phantom exits (fewest period events, lowest player id) and lineups smaller than five reject the
game. Home/away sides come from majority actor votes on home/visitor description events.
Possessions use the standard estimate FGA + 0.44*FTA - OREB + TOV; team rebounds count as
defensive. All conventions are disclosed here and in the dataset manifest.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

NBA_PBP_SCHEMA_VERSION = 1
NBASTATS_HEADER: tuple[str, ...] = (
    "GAME_ID",
    "EVENTNUM",
    "EVENTMSGTYPE",
    "EVENTMSGACTIONTYPE",
    "PERIOD",
    "WCTIMESTRING",
    "PCTIMESTRING",
    "HOMEDESCRIPTION",
    "NEUTRALDESCRIPTION",
    "VISITORDESCRIPTION",
    "SCORE",
    "SCOREMARGIN",
    "PERSON1TYPE",
    "PLAYER1_ID",
    "PLAYER1_NAME",
    "PLAYER1_TEAM_ID",
    "PLAYER1_TEAM_CITY",
    "PLAYER1_TEAM_NICKNAME",
    "PLAYER1_TEAM_ABBREVIATION",
    "PERSON2TYPE",
    "PLAYER2_ID",
    "PLAYER2_NAME",
    "PLAYER2_TEAM_ID",
    "PLAYER2_TEAM_CITY",
    "PLAYER2_TEAM_NICKNAME",
    "PLAYER2_TEAM_ABBREVIATION",
    "PERSON3TYPE",
    "PLAYER3_ID",
    "PLAYER3_NAME",
    "PLAYER3_TEAM_ID",
    "PLAYER3_TEAM_CITY",
    "PLAYER3_TEAM_NICKNAME",
    "PLAYER3_TEAM_ABBREVIATION",
    "VIDEO_AVAILABLE_FLAG",
)
_PERIOD_SECONDS = 720
_OVERTIME_SECONDS = 300
_REGULAR_SEASON_TYPE = "002"
_TEAM_SECONDS_TOLERANCE = 5


class NbaPbpError(ValueError):
    """Raised when one play-by-play game fails structural validation."""


@dataclass(frozen=True, slots=True)
class PlayerGameLine:
    """One player's postgame participation line in one game."""

    player_id: int
    team_abbreviation: str
    seconds_played: int
    plus_minus: int

    @property
    def minutes_played(self) -> float:
        """Return minutes on court as a float."""
        return self.seconds_played / 60.0


@dataclass(frozen=True, slots=True)
class TeamGameStats:
    """One team's postgame counting stats in one game."""

    team_abbreviation: str
    points: int
    field_goals_attempted: int
    free_throws_attempted: int
    offensive_rebounds: int
    turnovers: int
    starters: tuple[int, ...]

    @property
    def possessions(self) -> float:
        """Return the standard play-by-play possession estimate."""
        return (
            self.field_goals_attempted
            + 0.44 * self.free_throws_attempted
            - self.offensive_rebounds
            + self.turnovers
        )


@dataclass(frozen=True, slots=True)
class StintRecord:
    """One uninterrupted interval with constant on-court lineups."""

    home_players: tuple[int, ...]
    away_players: tuple[int, ...]
    seconds: int
    home_points: int
    away_points: int
    home_possessions: float
    away_possessions: float


@dataclass(frozen=True, slots=True)
class PbpGame:
    """One validated postgame derivation from a regular-season play-by-play stream."""

    game_id: int
    away_abbreviation: str
    home_abbreviation: str
    away_score: int
    home_score: int
    team_stats: tuple[TeamGameStats, TeamGameStats]
    player_lines: tuple[PlayerGameLine, ...]
    player_names: dict[int, str]
    stints: tuple[StintRecord, ...] = ()

    @property
    def season_label(self) -> int:
        """Return the ending-year season label (2022 for the 2021-22 season)."""
        digits = str(self.game_id).zfill(10)
        return 2000 + int(digits[3:5]) + 1


class _TeamState:
    """Mutable per-team accumulation inside one game."""

    def __init__(self, abbreviation: str) -> None:
        """Initialize an empty team accumulator."""
        self.abbreviation = abbreviation
        self.fga = 0
        self.fta = 0
        self.oreb = 0
        self.tov = 0
        self.starters: set[int] = set()
        self.on_court: set[int] = set()


class _GameBuilder:
    """Mutable per-game accumulation state."""

    def __init__(self, game_id: int) -> None:
        """Initialize an empty game accumulator."""
        self.game_id = game_id
        self.home_abbreviation = ""
        self.away_abbreviation = ""
        self.home_votes: dict[str, int] = {}
        self.away_votes: dict[str, int] = {}
        self.away_score = -1
        self.home_score = -1
        self.period = 1
        self.teams: dict[str, _TeamState] = {}
        self.player_team: dict[int, str] = {}
        self.stint_open: dict[int, int] = {}
        self.seconds_played: dict[int, int] = {}
        self.plus_minus: dict[int, int] = {}
        self.last_miss_team = ""
        self.buffered: list[list[str]] = []
        self.period_participants: dict[str, set[int]] = {}
        self.period_first_sub: dict[int, str] = {}
        self.period_event_counts: dict[int, int] = {}
        self.player_names: dict[int, str] = {}
        self.stints: list[StintRecord] = []
        self.stint_start_remaining = 0
        self.stint_start_away_score = 0
        self.stint_start_home_score = 0
        self.stint_counts: dict[str, list[int]] = {}


def read_pbp_games(path: Path, failures: list[str] | None = None) -> Iterator[PbpGame]:
    """Stream validated per-game derivations from one nbastats season CSV.

    Games failing structural validation are discarded and recorded as reason strings in
    ``failures`` when a list is supplied; they are never silently repaired.
    """
    state = _ReaderState()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for record in reader:
            game_id = int(record[0])
            if state.builder is not None and game_id != state.builder.game_id:
                yield from _finalize_or_record(state.builder, failures)
                state.builder = None
            builder = _step_builder(state, game_id, record, failures)
            if builder is not None:
                try:
                    _consume(builder, record)
                except NbaPbpError as error:
                    if failures is not None:
                        failures.append(str(error))
                    state.discarded.add(game_id)
                    state.builder = None
    if state.builder is not None:
        yield from _finalize_or_record(state.builder, failures)


class _ReaderState:
    """Mutable streaming state for one season pass."""

    def __init__(self) -> None:
        """Initialize an empty reader state."""
        self.builder: _GameBuilder | None = None
        self.discarded: set[int] = set()
        self.reported_types: set[int] = set()


def _step_builder(
    state: _ReaderState,
    game_id: int,
    record: list[str],
    failures: list[str] | None,
) -> _GameBuilder | None:
    if state.builder is not None or game_id in state.discarded:
        return state.builder
    if not _is_regular(game_id):
        if failures is not None and game_id not in state.reported_types:
            state.reported_types.add(game_id)
            failures.append(f"game {game_id} is not a regular-season game")
        state.discarded.add(game_id)
        return None
    state.builder = _GameBuilder(game_id=game_id)
    return state.builder


def _finalize_or_record(builder: _GameBuilder, failures: list[str] | None) -> Iterator[PbpGame]:
    try:
        yield _finalize(builder)
    except NbaPbpError as error:
        if failures is not None:
            failures.append(str(error))


def _is_regular(game_id: int) -> bool:
    return str(game_id).zfill(10)[:3] == _REGULAR_SEASON_TYPE


def _consume(builder: _GameBuilder, record: list[str]) -> None:
    period = int(record[4])
    if period != builder.period:
        _finish_period(builder)
        builder.period = period
    builder.buffered.append(record)
    _observe_period(builder, record)


def _observe_period(builder: _GameBuilder, record: list[str]) -> None:
    _learn_sides(builder, record)
    for player_id, team_abbr, player_name in _participants(record):
        builder.teams.setdefault(team_abbr, _TeamState(abbreviation=team_abbr))
        builder.period_participants.setdefault(team_abbr, set()).add(player_id)
        builder.player_team.setdefault(player_id, team_abbr)
        builder.period_event_counts[player_id] = builder.period_event_counts.get(player_id, 0) + 1
        if player_name:
            builder.player_names.setdefault(player_id, player_name)
    if int(record[2]) == 8:
        team_abbr = record[18]
        if team_abbr:
            out_id, in_id = int(record[13]), int(record[20])
            builder.period_first_sub.setdefault(out_id, "out")
            builder.period_first_sub.setdefault(in_id, "in")


def _finish_period(builder: _GameBuilder) -> None:
    _resolve_sides(builder)
    previous_on_court = {abbr: set(team.on_court) for abbr, team in builder.teams.items()}
    for team_abbr, team in builder.teams.items():
        team.on_court = _resolve_lineup(builder, team_abbr, previous_on_court.get(team_abbr, set()))
        if builder.period == 1:
            team.starters = set(team.on_court)
        length = _PERIOD_SECONDS if builder.period <= 4 else _OVERTIME_SECONDS
        for player_id in team.on_court:
            builder.stint_open[player_id] = length
    _open_stint_interval(builder, _PERIOD_SECONDS if builder.period <= 4 else _OVERTIME_SECONDS)
    buffered, builder.buffered = builder.buffered, []
    builder.period_participants = {}
    builder.period_first_sub = {}
    builder.period_event_counts = {}
    for record in buffered:
        _consume_resolved(builder, record)
    _close_all_stints(builder, 0)
    _close_stint_interval(builder, 0)


def _resolve_lineup(
    builder: _GameBuilder,
    team_abbr: str,
    previous_on_court: set[int],
) -> set[int]:
    participants = builder.period_participants.get(team_abbr, set())
    lineup = {
        player_id for player_id in participants if builder.period_first_sub.get(player_id) != "in"
    }
    while len(lineup) > 5:
        phantom = min(
            lineup,
            key=lambda player_id: (
                builder.period_first_sub.get(player_id) != "out",
                builder.period_event_counts.get(player_id, 0),
                player_id,
            ),
        )
        lineup.discard(phantom)
    if len(lineup) < 5:
        returners = {
            player_id
            for player_id in previous_on_court
            if builder.period_first_sub.get(player_id) == "in"
        }
        for player_id in sorted(previous_on_court - lineup - returners):
            lineup.add(player_id)
            if len(lineup) == 5:
                break
    if len(lineup) != 5:
        raise NbaPbpError(
            f"game {builder.game_id} team {team_abbr} has {len(lineup)} players "
            f"on court at period {builder.period} start"
        )
    return lineup


def _consume_resolved(builder: _GameBuilder, record: list[str]) -> None:
    msg_type = int(record[2])
    remaining = _parse_clock(record[6])
    if msg_type == 8:
        _close_stint_interval(builder, remaining)
        _handle_substitution(builder, record, remaining)
        _open_stint_interval(builder, remaining)
        return
    team = builder.teams.get(record[18])
    if team is not None:
        _count_team_stat(builder, team, msg_type)
    _track_misses_and_rebounds(builder, msg_type, record)
    _update_score(builder, record)


def _open_stint_interval(builder: _GameBuilder, remaining: int) -> None:
    builder.stint_start_remaining = remaining
    builder.stint_start_away_score = max(builder.away_score, 0)
    builder.stint_start_home_score = max(builder.home_score, 0)
    builder.stint_counts = {abbr: [0, 0, 0, 0] for abbr in builder.teams}


def _close_stint_interval(builder: _GameBuilder, remaining: int) -> None:
    seconds = builder.stint_start_remaining - remaining
    if seconds <= 0:
        return
    home_abbr, away_abbr = builder.home_abbreviation, builder.away_abbreviation
    if not home_abbr or not away_abbr:
        return
    home_counts = builder.stint_counts.get(home_abbr, [0, 0, 0, 0])
    away_counts = builder.stint_counts.get(away_abbr, [0, 0, 0, 0])
    builder.stints.append(
        StintRecord(
            home_players=tuple(sorted(builder.teams[home_abbr].on_court)),
            away_players=tuple(sorted(builder.teams[away_abbr].on_court)),
            seconds=seconds,
            home_points=max(builder.home_score, 0) - builder.stint_start_home_score,
            away_points=max(builder.away_score, 0) - builder.stint_start_away_score,
            home_possessions=_stint_possessions(home_counts),
            away_possessions=_stint_possessions(away_counts),
        )
    )


def _stint_possessions(counts: list[int]) -> float:
    fga, fta, oreb, tov = counts
    return fga + 0.44 * fta - oreb + tov


def _track_misses_and_rebounds(builder: _GameBuilder, msg_type: int, record: list[str]) -> None:
    if msg_type == 2 or (msg_type == 3 and "MISS" in _descriptions(record)):
        builder.last_miss_team = record[18]
    elif msg_type == 1:
        builder.last_miss_team = ""
    elif msg_type == 4:
        _handle_rebound(builder, record[18])


def _learn_sides(builder: _GameBuilder, record: list[str]) -> None:
    actor_abbr = record[18]
    if not actor_abbr:
        return
    builder.teams.setdefault(actor_abbr, _TeamState(abbreviation=actor_abbr))
    if record[7].strip():
        builder.home_votes[actor_abbr] = builder.home_votes.get(actor_abbr, 0) + 1
    elif record[9].strip():
        builder.away_votes[actor_abbr] = builder.away_votes.get(actor_abbr, 0) + 1


def _resolve_sides(builder: _GameBuilder) -> None:
    if builder.home_votes:
        builder.home_abbreviation = _majority(builder.home_votes)
    if builder.away_votes:
        builder.away_abbreviation = _majority(builder.away_votes)


def _majority(votes: dict[str, int]) -> str:
    return max(votes, key=lambda abbr: votes[abbr])


def _handle_substitution(builder: _GameBuilder, record: list[str], remaining: int) -> None:
    team = builder.teams.get(record[18])
    if team is None:
        raise NbaPbpError(f"game {builder.game_id} has a substitution for an unknown team")
    out_id, in_id = int(record[13]), int(record[20])
    if out_id not in team.on_court and len(team.on_court) >= 5:
        ghost = min(
            team.on_court,
            key=lambda player_id: (builder.period_event_counts.get(player_id, 0), player_id),
        )
        _close_stint(builder, ghost, remaining)
        team.on_court.discard(ghost)
    _close_stint(builder, out_id, remaining)
    team.on_court.discard(out_id)
    team.on_court.add(in_id)
    _close_stint(builder, in_id, remaining)
    builder.stint_open[in_id] = remaining
    builder.player_team[in_id] = team.abbreviation


def _count_team_stat(builder: _GameBuilder, team: _TeamState, msg_type: int) -> None:
    index = 0
    if msg_type in (1, 2):
        team.fga += 1
        index = 0
    elif msg_type == 3:
        team.fta += 1
        index = 1
    elif msg_type == 5:
        team.tov += 1
        index = 3
    else:
        return
    counts = builder.stint_counts.get(team.abbreviation)
    if counts is not None:
        counts[index] += 1


def _handle_rebound(builder: _GameBuilder, team_abbr: str) -> None:
    team = builder.teams.get(team_abbr)
    if team is not None and team_abbr == builder.last_miss_team:
        team.oreb += 1
        counts = builder.stint_counts.get(team_abbr)
        if counts is not None:
            counts[2] += 1
    builder.last_miss_team = ""


def _update_score(builder: _GameBuilder, record: list[str]) -> None:
    raw = record[10].strip()
    if not raw:
        return
    away_text, home_text = (part.strip() for part in raw.split("-", 1))
    away_score, home_score = int(away_text), int(home_text)
    previous_away, previous_home = max(builder.away_score, 0), max(builder.home_score, 0)
    delta_away, delta_home = away_score - previous_away, home_score - previous_home
    if delta_away or delta_home:
        _attribute_plus_minus(builder, delta_away, delta_home)
    builder.away_score, builder.home_score = away_score, home_score


def _attribute_plus_minus(builder: _GameBuilder, delta_away: int, delta_home: int) -> None:
    if not builder.away_abbreviation or not builder.home_abbreviation:
        raise NbaPbpError(f"game {builder.game_id} scored before home/away identity")
    scoring: tuple[tuple[str, int], ...] = ()
    if delta_away:
        scoring = ((builder.away_abbreviation, delta_away),)
    elif delta_home:
        scoring = ((builder.home_abbreviation, delta_home),)
    for team_abbr, delta in scoring:
        for side_abbr, sign in ((team_abbr, delta), (_opponent(builder, team_abbr), -delta)):
            for player_id in builder.teams[side_abbr].on_court:
                builder.plus_minus[player_id] = builder.plus_minus.get(player_id, 0) + sign


def _close_stint(builder: _GameBuilder, player_id: int, remaining: int) -> None:
    start_remaining = builder.stint_open.pop(player_id, None)
    if start_remaining is not None:
        builder.seconds_played[player_id] = builder.seconds_played.get(player_id, 0) + (
            start_remaining - remaining
        )


def _close_all_stints(builder: _GameBuilder, remaining: int) -> None:
    for player_id in list(builder.stint_open):
        _close_stint(builder, player_id, remaining)


def _finalize(builder: _GameBuilder) -> PbpGame:
    _finish_period(builder)
    if not builder.home_abbreviation or not builder.away_abbreviation:
        raise NbaPbpError(f"game {builder.game_id} is missing home or away identity")
    home_score = builder.home_score
    away_score = builder.away_score
    if home_score <= 0 or away_score <= 0 or home_score == away_score:
        raise NbaPbpError(f"game {builder.game_id} has an invalid final score")
    team_stats = (
        _team_stats(builder, builder.away_abbreviation),
        _team_stats(builder, builder.home_abbreviation),
    )
    player_lines = _player_lines(builder)
    return PbpGame(
        game_id=builder.game_id,
        away_abbreviation=builder.away_abbreviation,
        home_abbreviation=builder.home_abbreviation,
        away_score=builder.away_score,
        home_score=builder.home_score,
        team_stats=team_stats,
        player_lines=player_lines,
        player_names=dict(builder.player_names),
        stints=tuple(builder.stints),
    )


def _team_stats(builder: _GameBuilder, abbreviation: str) -> TeamGameStats:
    team = builder.teams[abbreviation]
    if len(team.starters) != 5:
        raise NbaPbpError(
            f"game {builder.game_id} team {abbreviation} has {len(team.starters)} starters"
        )
    is_away = abbreviation == builder.away_abbreviation
    return TeamGameStats(
        team_abbreviation=abbreviation,
        points=builder.away_score if is_away else builder.home_score,
        field_goals_attempted=team.fga,
        free_throws_attempted=team.fta,
        offensive_rebounds=team.oreb,
        turnovers=team.tov,
        starters=tuple(sorted(team.starters)),
    )


def _player_lines(builder: _GameBuilder) -> tuple[PlayerGameLine, ...]:
    lines: list[PlayerGameLine] = []
    totals = {builder.away_abbreviation: 0, builder.home_abbreviation: 0}
    for player_id, seconds in sorted(builder.seconds_played.items()):
        team_abbr = builder.player_team.get(player_id)
        if team_abbr is None:
            raise NbaPbpError(f"game {builder.game_id} has a player with no team: {player_id}")
        totals[team_abbr] += seconds
        lines.append(
            PlayerGameLine(
                player_id=player_id,
                team_abbreviation=team_abbr,
                seconds_played=seconds,
                plus_minus=builder.plus_minus.get(player_id, 0),
            )
        )
    expected = _expected_team_seconds(builder)
    for team_abbr, total in totals.items():
        if abs(total - expected) > _TEAM_SECONDS_TOLERANCE:
            raise NbaPbpError(
                f"game {builder.game_id} team {team_abbr} played {total}s of {expected}s"
            )
    return tuple(lines)


def _expected_team_seconds(builder: _GameBuilder) -> int:
    overtime_periods = max(builder.period - 4, 0)
    return 5 * (4 * _PERIOD_SECONDS + overtime_periods * _OVERTIME_SECONDS)


def _opponent(builder: _GameBuilder, team_abbr: str) -> str:
    if team_abbr == builder.away_abbreviation:
        return builder.home_abbreviation
    return builder.away_abbreviation


def _participants(record: list[str]) -> list[tuple[int, str, str]]:
    participants: list[tuple[int, str, str]] = []
    for id_index, name_index, team_index in ((13, 14, 18), (20, 21, 25), (27, 28, 32)):
        raw_id, team_abbr = record[id_index].strip(), record[team_index].strip()
        if raw_id and raw_id != "0" and team_abbr:
            participants.append((int(raw_id), team_abbr, record[name_index].strip()))
    return participants


def _descriptions(record: list[str]) -> str:
    return " ".join(record[index] for index in (7, 8, 9))


def _parse_clock(raw: str) -> int:
    minutes, seconds = raw.strip().split(":", 1)
    return int(minutes) * 60 + int(seconds)
