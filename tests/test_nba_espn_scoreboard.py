"""Regression tests for scheduled-vs-completed ESPN scoreboard parsing.

The prospective collector polls future days whose events are never completed. The
completed-only parser used for historical season fetches silently yields an empty schedule
there, so no capture cutoff would ever come due. These tests pin both directions.
"""

from __future__ import annotations

import json

from forecastfm.nba_espn import parse_scoreboard, parse_upcoming_scoreboard


def _scoreboard_payload(*, completed: bool) -> bytes:
    return json.dumps(
        {
            "events": [
                {
                    "id": "401700001",
                    "date": "2026-10-21T23:30Z",
                    "status": {"type": {"completed": completed}},
                    "competitions": [
                        {
                            "competitors": [
                                {"homeAway": "home", "team": {"abbreviation": "BOS"}},
                                {"homeAway": "away", "team": {"abbreviation": "NYK"}},
                            ]
                        }
                    ],
                }
            ]
        }
    ).encode("utf-8")


def test_parse_scoreboard_keeps_only_completed_events() -> None:
    assert [ref.event_id for ref in parse_scoreboard(_scoreboard_payload(completed=True))] == [
        "401700001"
    ]
    assert parse_scoreboard(_scoreboard_payload(completed=False)) == []


def test_parse_upcoming_scoreboard_keeps_only_scheduled_events() -> None:
    references = parse_upcoming_scoreboard(_scoreboard_payload(completed=False))
    assert [ref.event_id for ref in references] == ["401700001"]
    assert references[0].away_abbreviation == "NYK"
    assert references[0].home_abbreviation == "BOS"
    assert references[0].date_utc == "2026-10-21T23:30Z"
    assert parse_upcoming_scoreboard(_scoreboard_payload(completed=True)) == []
