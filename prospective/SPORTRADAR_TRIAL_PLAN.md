# Sportradar NBA trial inspection plan

Status: prewritten 2026-07-17, before signup. Approved by the user on 2026-07-17. No Sportradar
request has been made yet. This plan is the only authorized spend of trial requests.

## Decisive question

A change timestamp proves that an object changed, not that every old payload value remains
queryable. The entire trial answers one question:

> Does a Sportradar historical/as-of query return the payload values that were actually published
> on that past day, or only the latest current state?

If every historical query silently returns latest state, Sportradar fails the outcome-v2 T-60 gate
exactly like SportsDataIO, and no money is spent beyond the trial.

## Trial facts to reverify at signup

Documented at research time (2026-07-17), all time-sensitive; confirm on the signup page before
creating an account:

- 30-day trial; approximately 1,000 requests; 1 request/second.
- Which NBA feeds and tiers the trial key enables (Daily Injuries, Daily Change Log, schedule,
  game summary/boxscore, standings, depth charts, play-by-play).
- Trial terms: storage, derived-data, ML, and redistribution restrictions. The trial produces an
  evidence pack for a custom-agreement conversation only. Nothing from the trial is trained on,
  uploaded to Tinker, or redistributed, regardless of what the data shows.

References:

- https://developer.sportradar.com/basketball/docs/nba-ig-historical-data
- https://developer.sportradar.com/basketball/reference/nba-daily-change-log
- https://developer.sportradar.com/basketball/reference/nba-daily-injuries
- https://developer.sportradar.com/basketball/docs/basketball-ig-account-maintenance

## Hard rules

- Total spend cap: 50 requests of the ~1,000. Log every request (UTC time, path, purpose, HTTP
  status, response SHA-256) in a local ledger before the next call.
- Respect 1 request/second; no concurrency.
- Never record the API key in any capture, ledger, log, URL record, or commit. If the API takes
  the key as a query parameter, redact it from the recorded request identity. Prefer a header if
  the API supports one.
- Every response is retained as a create-only `nba_raw_capture.py`-style envelope under an ignored
  `0700` root (`data/raw/sportradar/`), claiming only `local_retrieval_only`.
- No bulk caching, no season-scale pulls, no training use. This is a 50-call inspection.

## Call plan

### Phase 0 — account and access inventory (0 API calls)

- Record trial limits, enabled feeds, base URL, and rate from the account page.
- Prewrite the support question (below) but do not send it until Phase 2 data exists.

### Phase 1 — access sanity (about 5 calls)

- Current season schedule, league hierarchy, today's daily injuries, one live-date daily change
  log, one standings call.
- Purpose: confirm the key works, learn exact response shapes, confirm which feeds the trial
  enables. Abort and record if Daily Injuries or Daily Change Log are not enabled.

### Phase 2 — decisive historical-as-of test (about 15 calls)

Pick three past dates spanning different seasons, each chosen because a well-documented injury
event occurred (a star listed OUT who later returned, or a late scratch reported in contemporary
press). The documented chronology is the ground truth the API response is checked against.

1. Daily Injuries for past date D1. Does the response exist, and does it show the state as
   documented on D1 — or today's roster/status?
2. Repeat for D2 and D3 (different seasons).
3. Daily Change Log for windows covering D1-D3: do injury/lineup resource IDs appear with
   last-modified timestamps consistent with the documented events?
4. Re-issue the D1 Daily Injuries call at least 24 hours later: identical bytes, or did the
   historical payload change (backfill/correction)? Both outcomes are informative; a mutable
   historical payload without revision IDs fails the gate.

### Phase 3 — lineup and depth state (about 10 calls)

- Historical game summary/boxscore for one game with a documented late scratch: does any
  pregame-status field exist, and does it carry publication timestamps?
- Depth chart endpoint: current-only, or does any date/as-of parameter work? Probe one past date.
- One starting-lineup-type endpoint if the trial exposes it; probe one past date.

### Phase 4 — timestamps and completeness (about 10 calls)

- One full-season schedule call: postponements/reschedules visible? Venue identity complete?
- Two or three calls probing update cadence: same live resource polled at least an hour apart on a
  game day, retained as separate captures so intermediate states can be compared.
- Reserved slack for follow-ups raised by Phase 2 results.

## Predeclared verdict criteria

Decide from the retained captures, not impressions:

- **PASS** (proceed to rights negotiation): historical-day injury queries return payload values
  that (a) differ from current state where documentation says they should, (b) match independently
  documented chronology on all three test dates, and (c) carry stable IDs, with either immutable
  historical payloads or explicit revision identifiers.
- **FAIL** (do not spend): historical queries return latest/current state, historical dates error
  out, payloads contradict documented chronology in a latest-state pattern, or historical payloads
  mutate with no revision identity.
- **INCONCLUSIVE** (one bounded follow-up, still inside the 50-call cap): mixed or ambiguous
  evidence; write down exactly which additional call resolves it before making it.

## Support question to send after Phase 2

> For NBA Daily Injuries and lineup-related feeds: when I query a past date, does the API return
> every payload value as originally published that day, or the latest current state? Are old
> values, corrections, and deletions (tombstones) retained and queryable? Is historical
> completeness guaranteed back to a specific season? Is there an enterprise archive product that
> exposes immutable publication timestamps and full revision history for injuries, projected
> lineups, and depth charts?

## If the data passes

Data quality passing is necessary, not sufficient. Before any production use the signed agreement
must still satisfy `prospective/DATA_RIGHTS.md` and
`prospective/OUTCOME_V2_VENDOR_REQUIREMENTS.md`: retention, revisions, feature engineering,
training, Tinker third-party processing, derived-weight ownership, post-termination use,
publication of aggregates, and upstream NBA licensor warranties.
