"""Parser for the legacy nine-column NBA injury-report PDF layout (Oct 2018 - Dec 2019).

The official report archive switches to the seven-column layout parsed by ``nbainjuries`` in
January 2020. Older snapshots carry nine columns (Game Date, Game Time, Matchup, Team,
Player Name, Category, Reason, Current Status, Previous Status) and are rejected by that
package, so this module parses their extracted text directly. PyPDF2 is imported lazily
inside ``parse_legacy_report`` so the module stays importable without the ``pdf-audit``
extra.

Health-data boundary: rows carry player identifiers attached to health information and remain
local-only, exactly like the rows produced by ``forecastfm.nba_injury_report``.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from forecastfm.nba_feature_builder import TEAM_NAME_TO_ABBREVIATION
from forecastfm.nba_injury_report import (
    InjuryReportRow,
    parse_game_clock,
    parse_game_date,
    parse_report_header_time,
)

if TYPE_CHECKING:
    from PyPDF2 import PdfReader

_TEAM_NAMES = tuple(sorted(TEAM_NAME_TO_ABBREVIATION, key=len, reverse=True))
_NOT_SUBMITTED = "NOT YET SUBMITTED"

_NAME_PATTERN = (
    r"[A-Z][A-Za-z'.\-]*(?:\s+(?:Jr\.|Sr\.|II|III|IV))?,\s+"
    r"[A-Z][A-Za-z'.\-]*(?:\s+[A-Za-z'.\-]+)*?"
)
_ROW_START = re.compile(
    rf"(?P<name>{_NAME_PATTERN})\s+"
    r"(?P<category>G League|Injury/Illness|League Suspension|Not With Team|Rest)"
)
_STATUS_TOKEN = re.compile(r"(?:Out|Doubtful|Questionable|Probable|Available)(?=\s|$)")
_PREVIOUS_TOKEN = re.compile(r"\s*(?:-|Out|Doubtful|Questionable|Probable|Available)(?=\s|$)")
_DATE_AT = re.compile(r"\d{2}/\d{2}/\d{4}")
_CLOCK_AT = re.compile(r"\d{2}:\d{2}\s*\(ET\)")
_MATCHUP_AT = re.compile(r"[A-Z]{3}@[A-Z]{3}")
_PAGE_MARKER = re.compile(r"Page\s+\d+\s+of")
_PAGE_TOTAL = re.compile(r"\d+")
_TEAM_GLUE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(re.escape(name) for name in _TEAM_NAMES) + r")(?=[A-Z])"
)


@dataclass(frozen=True, slots=True)
class LegacyParseResult:
    """Rows parsed from one legacy snapshot plus the count of dropped status-less rows."""

    rows: tuple[InjuryReportRow, ...]
    dropped_rows: int


@dataclass(frozen=True, slots=True)
class _GameContext:
    game_date: date | None
    game_clock_et: tuple[int, int] | None
    matchup: str | None
    team: str | None


def parse_legacy_report(pdf_path: Path) -> LegacyParseResult:
    """Parse one legacy nine-column report PDF into canonical injury rows."""
    try:
        reader: PdfReader = importlib.import_module("PyPDF2").PdfReader(str(pdf_path))
    except ImportError as exc:
        raise RuntimeError(
            "parse_legacy_report requires PyPDF2; install the 'pdf-audit' extra"
        ) from exc
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return parse_legacy_text(text, parse_report_header_time(text))


def parse_legacy_text(text: str, report_time: datetime) -> LegacyParseResult:
    """Parse extracted legacy-report text into rows stamped with ``report_time``.

    A row carries the game context (date, clock, matchup, team) most recently seen to its
    left; game times without a date repeat the previous game's date, matching the source
    layout. Rows whose reason zone contains no status token are dropped and counted.
    """
    content = _separate_glued_teams(" ".join(_content_lines(text)))
    starts = [match.start() for match in _ROW_START.finditer(content)]
    starts.append(len(content))
    rows: list[InjuryReportRow] = []
    dropped = 0
    context = _GameContext(game_date=None, game_clock_et=None, matchup=None, team=None)
    pos = 0
    for index, start in enumerate(starts[:-1]):
        while pos < start:
            context, pos = _consume_header(content, pos, context)
        row, consumed = _parse_segment(content[start : starts[index + 1]], report_time, context)
        if row is None:
            dropped += 1
        else:
            rows.append(row)
        pos = start + consumed
    return LegacyParseResult(rows=tuple(rows), dropped_rows=dropped)


def _content_lines(text: str) -> list[str]:
    lines: list[str] = []
    pending_page_total = False
    for raw in text.splitlines():
        line = raw.strip()
        if pending_page_total and _PAGE_TOTAL.fullmatch(line):
            pending_page_total = False
            continue
        pending_page_total = False
        if not line or line.startswith(("Injury Report:", "Game Date")):
            continue
        if _PAGE_MARKER.fullmatch(line):
            pending_page_total = True
            continue
        lines.append(line)
    return lines


def _consume_header(content: str, pos: int, context: _GameContext) -> tuple[_GameContext, int]:
    if pos >= len(content) or content[pos] == " ":
        return context, pos + 1
    if content.startswith(_NOT_SUBMITTED, pos):
        return context, pos + len(_NOT_SUBMITTED)
    team = _match_team(content, pos)
    date_match = _DATE_AT.match(content, pos)
    clock_match = _CLOCK_AT.match(content, pos)
    matchup_match = _MATCHUP_AT.match(content, pos)
    if team is not None:
        context = replace(context, team=team)
        pos = _advance(content, pos + len(team))
    elif date_match is not None:
        context = replace(context, game_date=parse_game_date(date_match.group(0)))
        pos = date_match.end()
    elif clock_match is not None:
        context = replace(context, game_clock_et=parse_game_clock(clock_match.group(0)))
        pos = _advance(content, clock_match.end())
    elif matchup_match is not None:
        context = replace(context, matchup=matchup_match.group(0))
        pos = _advance(content, matchup_match.end())
    else:
        next_space = content.find(" ", pos)
        pos = len(content) if next_space < 0 else next_space + 1
    return context, pos


def _parse_segment(
    segment: str, report_time: datetime, context: _GameContext
) -> tuple[InjuryReportRow | None, int]:
    """Parse one row segment; return the row (or None when dropped) and chars consumed."""
    start = _ROW_START.match(segment)
    if start is None:
        return None, len(segment)
    status = _STATUS_TOKEN.search(segment, start.end())
    if status is None:
        return None, _next_header_offset(segment, start.end())
    end = status.end()
    previous = _PREVIOUS_TOKEN.match(segment, end)
    if previous is not None:
        end = previous.end()
    parts = _context_parts(context)
    if parts is None:
        return None, end
    game_date, clock, matchup, team = parts
    row = InjuryReportRow(
        report_time=report_time,
        game_date=game_date,
        game_clock_et=clock,
        matchup=matchup,
        team=team,
        player_name=" ".join(start.group("name").split()),
        status=status.group(0),
    )
    return row, end


def _context_parts(context: _GameContext) -> tuple[date, tuple[int, int], str, str] | None:
    if context.game_date is None or context.game_clock_et is None:
        return None
    if context.matchup is None or context.team is None:
        return None
    return (context.game_date, context.game_clock_et, context.matchup, context.team)


def _next_header_offset(segment: str, start: int) -> int:
    offsets = [len(segment)]
    for pattern in (_DATE_AT, _CLOCK_AT, _MATCHUP_AT):
        match = pattern.search(segment, start)
        if match is not None:
            offsets.append(match.start())
    offsets.extend(i for name in _TEAM_NAMES if (i := segment.find(name, start)) >= 0)
    submitted = segment.find(_NOT_SUBMITTED, start)
    if submitted >= 0:
        offsets.append(submitted)
    return min(offsets)


def _separate_glued_teams(content: str) -> str:
    """Insert a space where a full team name is glued to a following capitalized token."""
    return _TEAM_GLUE.sub(r"\1 ", content)


def _match_team(content: str, pos: int) -> str | None:
    for name in _TEAM_NAMES:
        if content.startswith(name, pos):
            return name
    return None


def _advance(content: str, end: int) -> int:
    return end + 1 if end < len(content) and content[end] == " " else end
