"""Convert ESPN NBA summary JSON into nbastats-format play-by-play CSV rows.

ESPN's free API is the 2025-26 source because stats.nba.com and cdn.nba.com are blocked from
this environment and shufinskiy's republished archive stops at 2024-25. ESPN athlete IDs live in
a different ID space than NBA stats IDs, so downstream player identity must key on normalized
names, never raw IDs. Conversion decisions: substitution participants are (in, out) in ESPN and
become (out, in) in nbastats convention; missed free throws get the uppercase ``MISS`` prefix
the downstream rebound tracker expects; scores use the ``away - home`` column format. Game IDs
are synthetic, sequential in chronological order, and marked as non-official.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from forecastfm.json_utils import (
    parse_json_object,
    require_float,
    require_list,
    require_object,
    require_string,
    required_field,
)
from forecastfm.nba_pbp import NBASTATS_HEADER

NBA_ESPN_SCHEMA_VERSION = 1

ESPN_TO_NBA_ABBREVIATION: dict[str, str] = {
    "GS": "GSW",
    "NY": "NYK",
    "NO": "NOP",
    "SA": "SAS",
    "UTAH": "UTA",
    "PHO": "PHX",
    "WSH": "WAS",
}

NBA_TEAM_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "ATL",
        "BOS",
        "BKN",
        "CHA",
        "CHI",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GSW",
        "HOU",
        "IND",
        "LAC",
        "LAL",
        "MEM",
        "MIA",
        "MIL",
        "MIN",
        "NOP",
        "NYK",
        "OKC",
        "ORL",
        "PHI",
        "PHX",
        "POR",
        "SAC",
        "SAS",
        "TOR",
        "UTA",
        "WAS",
    }
)

_TYPE_MAP = {
    "made_shot": "1",
    "missed_shot": "2",
    "free_throw": "3",
    "rebound": "4",
    "turnover": "5",
    "foul": "6",
    "substitution": "8",
    "other": "13",
}


@dataclass(frozen=True, slots=True)
class EspnGameRef:
    """One ESPN event reference from a scoreboard document."""

    event_id: str
    date_utc: str
    away_abbreviation: str
    home_abbreviation: str


@dataclass(frozen=True, slots=True)
class EspnConvertedGame:
    """One converted game: nbastats rows plus identity and final score."""

    rows: list[list[str]]
    away_abbreviation: str
    home_abbreviation: str
    away_score: int
    home_score: int


@dataclass(frozen=True, slots=True)
class _ConversionContext:
    """Shared conversion inputs for one game's plays."""

    game_id: int
    athletes: dict[str, tuple[str, str]]
    home_abbr: str
    away_abbr: str
    teams: dict[str, str]


def map_abbreviation(espn_abbreviation: str) -> str:
    """Map an ESPN team abbreviation to the official NBA tricode."""
    return ESPN_TO_NBA_ABBREVIATION.get(espn_abbreviation, espn_abbreviation)


def parse_scoreboard(payload: bytes) -> list[EspnGameRef]:
    """Extract event references from one ESPN scoreboard document."""
    return _parse_scoreboard_events(payload, keep_completed=True)


def parse_upcoming_scoreboard(payload: bytes) -> list[EspnGameRef]:
    """Extract scheduled (not-yet-completed) event references from one scoreboard.

    The prospective collector polls future days, whose events are never completed; using
    the completed-only parser there yields an empty schedule, so no cutoffs ever come due.
    """
    return _parse_scoreboard_events(payload, keep_completed=False)


def _parse_scoreboard_events(payload: bytes, *, keep_completed: bool) -> list[EspnGameRef]:
    document = parse_json_object(payload.decode("utf-8"))
    references: list[EspnGameRef] = []
    for event in require_list(required_field(document, "events"), "events"):
        event_object = require_object(event, "event")
        status = require_object(event_object.get("status", {}), "status")
        status_type = require_object(status.get("type", {}), "status.type")
        if (status_type.get("completed") is True) is not keep_completed:
            continue
        away_abbr, home_abbr = _competitors(event_object)
        if away_abbr not in NBA_TEAM_ABBREVIATIONS or home_abbr not in NBA_TEAM_ABBREVIATIONS:
            continue
        references.append(
            EspnGameRef(
                event_id=require_string(required_field(event_object, "id"), "id"),
                date_utc=require_string(required_field(event_object, "date"), "date"),
                away_abbreviation=away_abbr,
                home_abbreviation=home_abbr,
            )
        )
    return references


def convert_summary(payload: bytes, synthetic_game_id: int) -> EspnConvertedGame:
    """Convert one ESPN summary document to nbastats rows plus identity and final score."""
    document = parse_json_object(payload.decode("utf-8"))
    header = require_object(required_field(document, "header"), "header")
    away_abbr, home_abbr = _competitors(header)
    teams = _team_ids(header)
    athletes = _athletes(document)
    context = _ConversionContext(
        game_id=synthetic_game_id,
        athletes=athletes,
        home_abbr=home_abbr,
        away_abbr=away_abbr,
        teams=teams,
    )
    plays = require_list(document.get("plays", []), "plays")
    rows = [_convert_play(require_object(play, "play"), context) for play in plays]
    away_score, home_score = _scores(header)
    return EspnConvertedGame(
        rows=rows,
        away_abbreviation=away_abbr,
        home_abbreviation=home_abbr,
        away_score=away_score,
        home_score=home_score,
    )


def write_nbastats_csv(path: Path, rows: list[list[str]]) -> None:
    """Write converted rows with the standard 34-column nbastats header."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(NBASTATS_HEADER)
        writer.writerows(rows)


def _competitors(parent: dict[str, object]) -> tuple[str, str]:
    competitions = require_list(required_field(parent, "competitions"), "competitions")
    competition = require_object(competitions[0], "competition")
    competitors = require_list(required_field(competition, "competitors"), "competitors")
    abbreviations: dict[str, str] = {}
    for entry in competitors:
        competitor = require_object(entry, "competitor")
        side = require_string(required_field(competitor, "homeAway"), "homeAway")
        team = require_object(required_field(competitor, "team"), "team")
        abbreviations[side] = map_abbreviation(
            require_string(required_field(team, "abbreviation"), "abbreviation")
        )
    return abbreviations["away"], abbreviations["home"]


def _team_ids(header: dict[str, object]) -> dict[str, str]:
    competitions = require_list(required_field(header, "competitions"), "competitions")
    competition = require_object(competitions[0], "competition")
    competitors = require_list(required_field(competition, "competitors"), "competitors")
    teams: dict[str, str] = {}
    for entry in competitors:
        competitor = require_object(entry, "competitor")
        team = require_object(required_field(competitor, "team"), "team")
        team_id = require_string(team.get("id", ""), "id")
        abbreviation = require_string(required_field(team, "abbreviation"), "abbreviation")
        if team_id:
            teams[team_id] = map_abbreviation(abbreviation)
    return teams


def _scores(header: dict[str, object]) -> tuple[int, int]:
    competitions = require_list(required_field(header, "competitions"), "competitions")
    competition = require_object(competitions[0], "competition")
    competitors = require_list(required_field(competition, "competitors"), "competitors")
    scores: dict[str, int] = {}
    for entry in competitors:
        competitor = require_object(entry, "competitor")
        side = require_string(required_field(competitor, "homeAway"), "homeAway")
        raw_score = required_field(competitor, "score")
        if isinstance(raw_score, str):
            scores[side] = int(raw_score)
        else:
            scores[side] = int(require_float(raw_score, "score"))
    return scores["away"], scores["home"]


def _athletes(document: dict[str, object]) -> dict[str, tuple[str, str]]:
    boxscore = require_object(document.get("boxscore", {}), "boxscore")
    athletes: dict[str, tuple[str, str]] = {}
    for team_block in require_list(boxscore.get("players", []), "players"):
        team_object = require_object(team_block, "players[]")
        team = require_object(required_field(team_object, "team"), "team")
        team_abbr = map_abbreviation(
            require_string(required_field(team, "abbreviation"), "abbreviation")
        )
        for stat_block in require_list(team_object.get("statistics", []), "statistics"):
            stat_object = require_object(stat_block, "statistics[]")
            for entry in require_list(stat_object.get("athletes", []), "athletes"):
                entry_object = require_object(entry, "athletes[]")
                athlete = require_object(required_field(entry_object, "athlete"), "athlete")
                athlete_id = require_string(athlete.get("id", ""), "id")
                if not athlete_id:
                    continue
                name = require_string(athlete.get("displayName", ""), "displayName")
                athletes[athlete_id] = (name, team_abbr)
    return athletes


def _convert_play(play: dict[str, object], context: _ConversionContext) -> list[str]:
    text = require_string(play.get("text", ""), "text")
    msg_type = _message_type(play, text)
    acting_abbr = _acting_team(play, context)
    player1, player2 = _players(msg_type, play.get("participants"), context.athletes)
    if msg_type == _TYPE_MAP["substitution"]:
        player1 = (player1[0], player1[1], player1[2] or acting_abbr)
        player2 = (player2[0], player2[1], player2[2] or acting_abbr)
    if msg_type == _TYPE_MAP["free_throw"] and "miss" in text.lower():
        text = f"MISS {text}"
    home_score = int(require_float(play.get("homeScore", 0), "homeScore"))
    away_score = int(require_float(play.get("awayScore", 0), "awayScore"))
    period = require_object(play.get("period", {}), "period")
    clock = require_object(play.get("clock", {}), "clock")
    row = [""] * 34
    row[0] = str(context.game_id)
    row[1] = str(play.get("sequenceNumber", play.get("id", "0")))
    row[2] = msg_type
    row[3] = "0"
    row[4] = str(period.get("number", 1))
    row[6] = _clock_text(require_string(clock.get("displayValue", "12:00"), "displayValue"))
    if acting_abbr == context.home_abbr:
        row[7] = text
    elif acting_abbr == context.away_abbr:
        row[9] = text
    else:
        row[8] = text
    row[10] = f"{away_score} - {home_score}"
    margin = home_score - away_score
    row[11] = str(margin) if margin != 0 else "TIE"
    row[12] = "5" if player1[0] != "0" else "0"
    row[13], row[14], row[18] = player1
    row[19] = "5" if player2[0] != "0" else "0"
    row[20], row[21], row[25] = player2
    return row


def _clock_text(raw: str) -> str:
    """Normalize an ESPN clock to whole-second ``M:SS`` form; raw seconds get a zero minute."""
    text = raw.strip()
    parts = text.split(":")
    if len(parts) == 3:
        total_minutes = int(parts[0]) * 60 + int(parts[1])
        return f"{total_minutes}:{int(float(parts[2])):02d}"
    if len(parts) == 2:
        return f"{int(parts[0])}:{int(float(parts[1])):02d}"
    return f"0:{int(float(text)):02d}"


def _message_type(play: dict[str, object], text: str) -> str:
    play_type = require_object(play.get("type", {}), "type")
    type_text = require_string(play_type.get("text", ""), "type.text").lower()
    lowered = text.lower()
    checks = (
        (_TYPE_MAP["substitution"], "substitution" in type_text or "enters the game" in lowered),
        (_TYPE_MAP["free_throw"], "free throw" in lowered),
        (_TYPE_MAP["rebound"], "rebound" in type_text or "rebound" in lowered),
        (_TYPE_MAP["turnover"], "turnover" in type_text or "turnover" in lowered),
        (_TYPE_MAP["foul"], "foul" in type_text),
    )
    for msg_type, matched in checks:
        if matched:
            return msg_type
    if play.get("shootingPlay") is True:
        made = int(require_float(play.get("scoreValue", 0), "scoreValue")) > 0
        return _TYPE_MAP["made_shot"] if made else _TYPE_MAP["missed_shot"]
    return _TYPE_MAP["other"]


def _acting_team(play: dict[str, object], context: _ConversionContext) -> str:
    team_ref = play.get("team")
    if team_ref is not None:
        team_object = require_object(team_ref, "team")
        team_id = require_string(team_object.get("id", ""), "id")
        if team_id in context.teams:
            return context.teams[team_id]
    participants = play.get("participants")
    if participants is not None:
        entries = require_list(participants, "participants")
        if entries:
            participant = require_object(entries[0], "participant")
            athlete_ref = participant.get("athlete")
            if athlete_ref is not None:
                athlete = require_object(athlete_ref, "athlete")
                athlete_id = require_string(athlete.get("id", ""), "id")
                if athlete_id in context.athletes:
                    return context.athletes[athlete_id][1]
    return ""


def _players(
    msg_type: str,
    participants: object,
    athletes: dict[str, tuple[str, str]],
) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    if participants is None:
        return (("0", "", ""), ("0", "", ""))
    entries = require_list(participants, "participants")
    if not entries:
        return (("0", "", ""), ("0", "", ""))
    first: object = entries[0]
    second: object = entries[1] if len(entries) > 1 else None
    if msg_type == _TYPE_MAP["substitution"] and second is not None:
        first, second = second, first
    return (_player_fields(first, athletes), _player_fields(second, athletes))


def _player_fields(
    participant: object, athletes: dict[str, tuple[str, str]]
) -> tuple[str, str, str]:
    if participant is None:
        return ("0", "", "")
    participant_object = require_object(participant, "participant")
    athlete_ref = participant_object.get("athlete")
    if athlete_ref is None:
        return ("0", "", "")
    athlete = require_object(athlete_ref, "athlete")
    athlete_id = require_string(athlete.get("id", ""), "id")
    name, abbr = athletes.get(athlete_id, ("", ""))
    return (athlete_id if athlete_id else "0", name, abbr)
