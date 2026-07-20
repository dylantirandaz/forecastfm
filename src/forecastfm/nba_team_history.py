"""Per-team chronological state for strictly pregame feature derivation.

One ``NbaTeamHistory`` tracks a single team inside a single season and answers the standard
rich-schema side features using only games that already tipped off. State resets every season.
Openers use the disclosed defaults from the frozen schema: rest and travel zero, continuity 1.0,
rolling value zero, schedule strength 1500. Possessions played by a player in one game are
approximated as team possessions times the player's share of team seconds; the approximation is
disclosed in the dataset manifest.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from forecastfm.nba_arenas import NbaArena, great_circle_miles, travel_time_zone_change
from forecastfm.nba_pbp import PbpGame, normalize_player_name

NBA_TEAM_HISTORY_SCHEMA_VERSION = 1
ROLLING_WINDOW_GAMES = 10
MIN_PROJECTED_ROTATION_APPEARANCES = 3
OPENER_SCHEDULE_STRENGTH = 1500.0
OPENER_CONTINUITY = 1.0

_SECONDS_PER_GAME_TEAM = 5 * 48 * 60


@dataclass(frozen=True, slots=True)
class GameContext:
    """The pregame facts of one scheduled game from one team's perspective."""

    game_date: date
    tipoff: datetime
    home: bool
    arena: NbaArena


@dataclass(frozen=True, slots=True)
class TeamSideFeatures:
    """The 11 standard rich-schema features for one team before one game."""

    rest_days: float
    back_to_back: float
    games_last_7: float
    road_games_last_7: float
    travel_miles: float
    travel_time_zones: float
    roster_continuity: float
    expected_lineup_continuity: float
    rolling_team_net_rating: float
    rolling_player_value: float
    schedule_strength: float


@dataclass(frozen=True, slots=True)
class _PlayedGame:
    game_date: date
    tipoff: datetime
    home: bool
    arena: NbaArena
    opponent_abbreviation: str
    points_for: int
    points_against: int
    possessions: float
    starters: tuple[int, ...]
    minutes: dict[int, float]
    plus_minus: dict[int, int]
    seconds_by_player: dict[int, int]
    player_names: dict[int, str]
    opponent_elo: float


def _name_key(name: str) -> str:
    return " ".join(normalize_player_name(name))


class NbaTeamHistory:
    """Rolling strictly-pregame state for one team in one season."""

    def __init__(self, team_abbreviation: str) -> None:
        """Initialize empty season state for one team."""
        self.team_abbreviation = team_abbreviation
        self._games: list[_PlayedGame] = []

    def features_for(
        self,
        context: GameContext,
        player_values: Mapping[str, float] | None = None,
    ) -> TeamSideFeatures:
        """Compute the standard side features from strictly prior games.

        When ``player_values`` is supplied (for example causal RAPM keyed by normalized player
        name), it replaces the raw rolling plus-minus values; unseen players contribute zero
        per the no-history rule.
        """
        prior = self._games[-1] if self._games else None
        window = self._games[-ROLLING_WINDOW_GAMES:]
        return TeamSideFeatures(
            rest_days=self._rest_days(context.game_date, prior),
            back_to_back=self._back_to_back(context.game_date, prior),
            games_last_7=self._games_in_window(context.tipoff, road_only=False),
            road_games_last_7=self._games_in_window(context.tipoff, road_only=True),
            travel_miles=self._travel_miles(context.arena, prior),
            travel_time_zones=self._travel_zones(context.arena, context.tipoff, prior),
            roster_continuity=self._roster_continuity(),
            expected_lineup_continuity=self._lineup_continuity(),
            rolling_team_net_rating=self._net_rating(window),
            rolling_player_value=self._player_value(player_values),
            schedule_strength=self._schedule_strength(window),
        )

    def record_game(
        self,
        game: PbpGame,
        context: GameContext,
        opponent_elo: float,
    ) -> None:
        """Append one completed game to the team's history."""
        stats = next(s for s in game.team_stats if s.team_abbreviation == self.team_abbreviation)
        lines = [
            line for line in game.player_lines if line.team_abbreviation == self.team_abbreviation
        ]
        opponent = game.home_abbreviation if not context.home else game.away_abbreviation
        self._games.append(
            _PlayedGame(
                game_date=context.game_date,
                tipoff=context.tipoff,
                home=context.home,
                arena=context.arena,
                opponent_abbreviation=opponent,
                points_for=stats.points,
                points_against=game.home_score if not context.home else game.away_score,
                possessions=stats.possessions,
                starters=stats.starters,
                minutes={line.player_id: line.minutes_played for line in lines},
                plus_minus={line.player_id: line.plus_minus for line in lines},
                seconds_by_player={line.player_id: line.seconds_played for line in lines},
                player_names=dict(game.player_names),
                opponent_elo=opponent_elo,
            )
        )

    def prior_game_minutes(self) -> dict[int, float]:
        """Return player minutes from the team's most recent game, empty for openers."""
        if not self._games:
            return {}
        return self._games[-1].minutes

    def expected_minutes(self) -> dict[int, float]:
        """Return median minutes over each player's last ten appearances this season.

        Appearance-based (missed games are skipped, not zeroed); genuinely no-history
        players are absent and contribute zero downstream.
        """
        appearances: dict[int, list[float]] = {}
        for game in self._games:
            for player_id, minutes in game.minutes.items():
                appearances.setdefault(player_id, []).append(minutes)
        return {
            player_id: statistics.median(values[-ROLLING_WINDOW_GAMES:])
            for player_id, values in appearances.items()
        }

    def projected_rotation_value(
        self,
        player_values: Mapping[str, float],
        unavailable_names: frozenset[str],
    ) -> float:
        """Minutes-weighted player value of the projected available rotation.

        The pool is players with at least three appearances in the team's last ten games;
        expected minutes are the median of those in-window appearances. The denominator is
        the whole pool's expected minutes before unavailable players (normalized name keys)
        are excluded, so absences lower the value. No history or an empty pool yields zero.
        """
        window = self._games[-ROLLING_WINDOW_GAMES:]
        appearances: dict[int, list[float]] = {}
        names: dict[int, str] = {}
        for game in window:
            for player_id, minutes in game.minutes.items():
                appearances.setdefault(player_id, []).append(minutes)
                name = game.player_names.get(player_id)
                if name is not None:
                    names[player_id] = name
        total_minutes = 0.0
        weighted = 0.0
        for player_id, values in appearances.items():
            if len(values) < MIN_PROJECTED_ROTATION_APPEARANCES:
                continue
            expected = statistics.median(values)
            total_minutes += expected
            name = names.get(player_id)
            if name is None or _name_key(name) in unavailable_names:
                continue
            weighted += expected * player_values.get(_name_key(name), 0.0)
        if total_minutes <= 0.0:
            return 0.0
        return weighted / total_minutes

    def rolling_values(self) -> dict[int, float]:
        """Return per-player rolling per-100 plus-minus over each player's ten prior games."""
        values: dict[int, float] = {}
        accum: dict[int, list[tuple[int, int, float]]] = {}
        for game in self._games:
            team_poss = game.possessions
            for player_id, seconds in game.seconds_by_player.items():
                played = team_poss * seconds / _SECONDS_PER_GAME_TEAM
                entries = accum.setdefault(player_id, [])
                entries.append((game.plus_minus.get(player_id, 0), seconds, played))
        for player_id, entries in accum.items():
            recent = entries[-ROLLING_WINDOW_GAMES:]
            total_pm = sum(pm for pm, _, _ in recent)
            total_played = sum(played for _, _, played in recent)
            if total_played > 0.0:
                values[player_id] = total_pm / total_played * 100.0
        return values

    def _rest_days(self, game_date: date, prior: _PlayedGame | None) -> float:
        if prior is None:
            return 0.0
        return float(max((game_date - prior.game_date).days - 1, 0))

    def _back_to_back(self, game_date: date, prior: _PlayedGame | None) -> float:
        if prior is None:
            return 0.0
        return 1.0 if (game_date - prior.game_date).days == 1 else 0.0

    def _games_in_window(self, tipoff: datetime, *, road_only: bool) -> float:
        start = tipoff - timedelta(days=7)
        return float(
            sum(
                1
                for game in self._games
                if start <= game.tipoff < tipoff and (not road_only or not game.home)
            )
        )

    def _travel_miles(self, arena: NbaArena, prior: _PlayedGame | None) -> float:
        if prior is None:
            return 0.0
        return great_circle_miles(prior.arena, arena)

    def _travel_zones(self, arena: NbaArena, tipoff: datetime, prior: _PlayedGame | None) -> float:
        if prior is None:
            return 0.0
        return travel_time_zone_change(prior.arena, arena, tipoff)

    def _roster_continuity(self) -> float:
        if len(self._games) < 2:
            return OPENER_CONTINUITY
        prior_game = self._games[-1]
        window = self._games[-(ROLLING_WINDOW_GAMES + 1) : -1]
        window_players = {player_id for game in window for player_id in game.minutes}
        total_minutes = sum(prior_game.minutes.values())
        if total_minutes <= 0.0:
            return OPENER_CONTINUITY
        shared = sum(
            minutes
            for player_id, minutes in prior_game.minutes.items()
            if player_id in window_players
        )
        return shared / total_minutes

    def _lineup_continuity(self) -> float:
        if len(self._games) < 2:
            return OPENER_CONTINUITY
        prior_starters = set(self._games[-1].starters)
        earlier_starters = set(self._games[-2].starters)
        if not prior_starters:
            return OPENER_CONTINUITY
        return len(prior_starters & earlier_starters) / len(prior_starters)

    def _net_rating(self, window: list[_PlayedGame]) -> float:
        if not window:
            return 0.0
        net_points = sum(game.points_for - game.points_against for game in window)
        possessions = sum(game.possessions for game in window)
        if possessions <= 0.0:
            return 0.0
        return net_points / possessions * 100.0

    def _player_value(self, player_values: Mapping[str, float] | None = None) -> float:
        if not self._games:
            return 0.0
        last_game = self._games[-1]
        if player_values is None:
            weights = last_game.minutes
            values = self.rolling_values()
            total_weight = sum(weights.values())
            if total_weight <= 0.0:
                return 0.0
            weighted = sum(weights[player_id] * values.get(player_id, 0.0) for player_id in weights)
            return weighted / total_weight
        total_weight = 0.0
        weighted = 0.0
        for player_id, minutes in last_game.minutes.items():
            name = last_game.player_names.get(player_id)
            if name is None:
                continue
            total_weight += minutes
            weighted += minutes * player_values.get(_name_key(name), 0.0)
        if total_weight <= 0.0:
            return 0.0
        return weighted / total_weight

    def _schedule_strength(self, window: list[_PlayedGame]) -> float:
        if not window:
            return OPENER_SCHEDULE_STRENGTH
        return sum(game.opponent_elo for game in window) / len(window)
