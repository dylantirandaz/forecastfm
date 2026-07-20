"""Empirical play rates per injury-report status class (disclosed prototype variant).

The frozen availability policy prices Out and Doubtful as certainly unavailable and every
other status as certainly available. This module estimates the empirical replacement:
P(player appears in the game | status in the selected pre-T-60 report), using the same
snapshot-selection rule as the feature builder (latest retained snapshot at or before
tipoff minus 60 minutes that contains the game's matchup). A player APPEARS in a game
when they have more than zero seconds in that game's play-by-play player lines, matched
to report rows by normalized name within the listed team.

Disclosure: the estimation window is seasons 2021-22 through 2025-26 (labels 2022-2026),
which includes the opened evaluation seasons. Those seasons were already used for
diagnostics, so the pooled rates are informational; no new claim is made on them. The
variant consumes the rates only through the disclosed ``status_rates`` path in the
feature builder and defaults off.

Measured on that window, the injury-severity ladder is strictly monotone (Out < Doubtful
< Questionable < Probable), but Available (about 0.77) sits BELOW Probable (about 0.91):
the report lists cleared end-of-bench players as Available and they frequently record a
DNP-CD, while Probable is used for rotation players expected to play. Validation
therefore enforces the severity ladder plus Available > Questionable, not the naive
Available > Probable ordering.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import UTC, timedelta

from forecastfm.nba_feature_builder import TEAM_NAME_TO_ABBREVIATION, InjurySnapshot
from forecastfm.nba_injury_report import KNOWN_STATUSES, matchup_teams
from forecastfm.nba_pbp import normalize_player_name
from forecastfm.nba_season_games import SeasonGame

STATUS_ORDER = ("Out", "Doubtful", "Questionable", "Probable", "Available")


class NbaStatusRatesError(ValueError):
    """Raised when empirical status play rates fail structural validation."""


@dataclass(frozen=True, slots=True)
class StatusPlayRates:
    """Empirical per-status appearance probabilities with their listing counts."""

    rates: dict[str, float]
    counts: dict[str, int]


def compute_status_play_rates(
    snapshots: list[InjurySnapshot],
    games: list[SeasonGame],
) -> StatusPlayRates:
    """Estimate P(appears | status) per status class over the supplied games.

    Games without any pre-T-60 snapshot are skipped, matching the health-feature rule.
    Every known status class must end up with at least one listing, all rates must lie in
    [0, 1], the injury-severity ladder must be strictly increasing (Out < Doubtful <
    Questionable < Probable), and Available must exceed Questionable. Violations raise
    rather than silently produce a miscalibrated variant. Available is NOT required to
    exceed Probable: empirically it does not (see the module docstring).
    """
    ordered = sorted(snapshots, key=lambda snapshot: snapshot.report_time)
    listings = dict.fromkeys(STATUS_ORDER, 0)
    appearances = dict.fromkeys(STATUS_ORDER, 0)
    for game in games:
        selected = _selected_snapshot(ordered, game)
        if selected is None:
            continue
        appeared = _appeared_keys(game)
        matchup = (game.away_abbreviation, game.home_abbreviation)
        for row in selected.rows:
            if row.status not in KNOWN_STATUSES or matchup_teams(row.matchup) != matchup:
                continue
            team = TEAM_NAME_TO_ABBREVIATION.get(row.team, "")
            if team not in matchup:
                continue
            listings[row.status] += 1
            if (team, _name_key(row.player_name)) in appeared:
                appearances[row.status] += 1
    result = StatusPlayRates(
        rates={
            status: appearances[status] / listings[status]
            for status in STATUS_ORDER
            if listings[status] > 0
        },
        counts=dict(listings),
    )
    _validate(result)
    return result


def _selected_snapshot(
    ordered: list[InjurySnapshot],
    game: SeasonGame,
) -> InjurySnapshot | None:
    """Return the latest pre-T-60 snapshot containing the game's matchup, if any."""
    cutoff = game.tipoff - timedelta(minutes=60)
    matchup = (game.away_abbreviation, game.home_abbreviation)
    return next(
        (
            snapshot
            for snapshot in reversed(ordered)
            if snapshot.report_time.astimezone(UTC) <= cutoff
            and any(
                row.game_date == game.game_date and matchup_teams(row.matchup) == matchup
                for row in snapshot.rows
            )
        ),
        None,
    )


def _appeared_keys(game: SeasonGame) -> frozenset[tuple[str, str]]:
    """Return (team, normalized name) keys for players with more than zero seconds.

    Rare play-by-play lines carry a placeholder player ID with no name; they can never
    match a report row, so they are skipped rather than failing the whole game.
    """
    keys: set[tuple[str, str]] = set()
    for line in game.pbp.player_lines:
        if line.seconds_played <= 0:
            continue
        name = game.pbp.player_names.get(line.player_id)
        if name is not None:
            keys.add((line.team_abbreviation, _name_key(name)))
    return frozenset(keys)


def _name_key(name: str) -> str:
    return " ".join(normalize_player_name(name))


def _validate(result: StatusPlayRates) -> None:
    """Enforce full coverage, probability bounds, and the empirical status ordering."""
    for status in STATUS_ORDER:
        if result.counts.get(status, 0) <= 0:
            raise NbaStatusRatesError(f"status {status} has no listings in the estimation window")
        rate = result.rates[status]
        if not 0.0 <= rate <= 1.0:
            raise NbaStatusRatesError(f"status {status} play rate {rate} is outside [0, 1]")
    for lower, higher in itertools.pairwise(STATUS_ORDER[:4]):
        if not result.rates[lower] < result.rates[higher]:
            raise NbaStatusRatesError(
                f"status play rates are not strictly increasing: {lower} "
                f"{result.rates[lower]:.4f} is not below {higher} {result.rates[higher]:.4f}"
            )
    if not result.rates["Questionable"] < result.rates["Available"]:
        raise NbaStatusRatesError(
            "Available play rate "
            f"{result.rates['Available']:.4f} is not above Questionable "
            f"{result.rates['Questionable']:.4f}"
        )
