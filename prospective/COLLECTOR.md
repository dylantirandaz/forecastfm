# Prospective 2026-27 collector

Status, 2026-07-19: offseason build. Nothing is live to poll until the 2026-27 regular season
opens in October 2026. The collector is implemented in
[`examples/collect_prospective.py`](../examples/collect_prospective.py) and is exercised only
through the dry-run checklist below until opening night.

## Why this exists

The SportsDataIO historical ladder closed with a rejection for the T-60 use case (see
[OUTCOME_V2_VENDOR_REQUIREMENTS.md](OUTCOME_V2_VENDOR_REQUIREMENTS.md)): no vendor has yet
proven reconstructable point-in-time injury and lineup history. That document's fallback is
explicit — while the historical search continues, "append-only prospective polling begins."
This collector produces the provably untouched future evidence that the outcome-v2 SFT/RL
gates and any prospective-win claim require: at each pre-tipoff cutoff it freezes the exact
injury-report PDF and scoreboard bytes this machine could have seen, hashes them, and chains
them so any later substitution is detectable.

## Sources and trust claims

- **Schedule and day state**: the ESPN scoreboard endpoint already used by
  `examples/fetch_espn_season.py`
  (`https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD`).
- **Availability state**: the official NBA injury-report PDFs on `ak-static.cms.nba.com`,
  with URLs built by `nbainjuries._util._gen_url` (the `pdf-audit` extra, pinned
  `nbainjuries==1.1.1`). For 2026-27 timestamps this yields the 15-minute-granularity form
  `Injury-Report_YYYY-MM-DD_HH_MMPM.pdf`.

Every capture is `local_retrieval_only` in the sense of `nba_raw_capture.py`: it proves what
bytes this machine retrieved and the UTC time it retrieved them. It does not prove HTTPS
transport, provider identity, provider publication chronology, or rights. The evidentiary
weight comes from the create-only store plus the hash chain — and only becomes externally
timestamped once the chain head is published per [README.md](README.md) ("locally
tamper-evident, not externally timestamped" until then). Named player-health rows remain
local-only under the standing health-data boundary; nothing here uploads anything anywhere.

## Schedule poll

The first non-dry `run` of each America/New_York day performs the daily schedule poll:

1. Fetch the ESPN scoreboard for today, tomorrow, and the day after (three documents).
2. Write each raw payload create-only under `data/raw/prospective/<poll-date-et>/` as
   `poll-scoreboard-<date>.json`.
3. Reconcile the parsed events against the operational schedule state
   (`state/schedule.json`): new games are added; changed tipoffs and vanished games produce
   amendments (below).
4. Append one `schedule_poll` event to the capture ledger binding the poll date, the three
   payload paths and SHA-256s, and the parsed game list.

## Cutoff rules

- Tipoff comes from the ESPN event's UTC `date`. Each game gets three cutoffs expressed in
  `America/New_York`: **T-6h**, **T-60** (the primary prospective state), and **T-15**.
  They are distinct immutable states and are never pooled or substituted for one another.
- A cutoff becomes due when `cutoff <= now + 2 minutes` and `now < tipoff`. The execution
  target is the cutoff ±2 minutes; each capture event records `cutoff_scheduled`, the true
  UTC `retrieved_at`, and a `within_window` boolean.
- Late captures are still written and chained with their honest `retrieved_at`. Under the
  `available_at` contract in [README.md](README.md), a capture whose retrieval time is after
  a state's cutoff is simply ineligible for that state; nothing is backdated or deleted.
- Cutoffs of games already past tipoff are skipped, not fabricated.

### Injury-report snapshot selection

At a cutoff, the collector must name "the current official injury report" deterministically:

1. Round the cutoff (ET) down to the previous 15-minute mark and step backward in 15-minute
   increments for 12 steps (3 hours), trying each URL.
2. Continue with the fixed publication slots 11:30, 13:30, 15:30, 17:30, 19:30, and 21:30 ET
   on the cutoff day and the prior day, newest first.
3. Stop at the first HTTP 200; 403/404 advance to the next candidate. At most 20 candidates
   are tried. Downloads use three bounded attempts with linear backoff (2 s, 4 s), the same
   policy as `build_nba_injury_archive.py`.
4. If every candidate is absent, the capture event records
   `{"kind": "injury_report_pdf", "status": "missing"}` — a chained statement that no report
   was retrievable at the cutoff, which is itself evidence.

The scoreboard document for the game date is captured alongside the PDF at every cutoff.

## Storage layout

```text
data/raw/prospective/                 # created 0700; collector refuses group/other access
  state/
    schedule.json                     # mutable operational index (NOT evidence)
    capture-ledger.jsonl              # append-only hash-chained evidence ledger
  <date-et>/                          # one directory per relevant ET date
    poll-scoreboard-<date>.json       # daily schedule poll payloads (under the poll date)
    injury-report-<event>-<cutoff>-Injury-Report_*.pdf
    scoreboard-<event>-<cutoff>.json
    capture-<event>-<cutoff>.json     # sidecar = ledger payload + its event hash
```

Raw payloads are written with `O_EXCL` create-only semantics (mode 0600): an existing file
is never overwritten, only reused. Filenames are deterministic per game and cutoff, so a
run that crashes between the download and the ledger append is idempotent on retry — the
next run reuses the untouched earlier bytes rather than replacing them. Each ledger capture
event records every file's SHA-256, source URL, relative path, and the UTC `retrieved_at`.

## Ledger integration

`ledger.py` is integrated, not reinvented: the capture ledger reuses its exact envelope
(`schema_version`, `sequence`, `event_type`, `recorded_at`, `previous_hash`, `payload`,
`event_hash`), its `GENESIS_HASH`, and `canonical_sha256`/`canonical_json` from
`integrity.py`. `recorded_at` values must be nondecreasing, as in the main ledger. It lives
in a **separate file** (`state/capture-ledger.jsonl`) because `ledger.py`'s auditor admits
only `forecast_batch` and `resolution_batch` events; the collector adds three event types:

- `schedule_poll` — the daily schedule freeze (payloads, hashes, parsed games).
- `capture` — one game at one cutoff: scheduled tipoff and cutoff, true `retrieved_at`,
  `within_window`, and the injury-report and scoreboard capture records.
- `schedule_amendment` — a tipoff move or removal (below).

Verification mirrors `ledger.py`'s audit: recompute each `event_hash` over the body, require
contiguous `sequence`, require each `previous_hash` to equal the prior `event_hash`, and
require monotonic `recorded_at`. When the season's forecast batches are later appended to the
main `ledger.jsonl` with the existing tooling, the cohort's `schedule_source` and
`schedule_snapshot_sha256` should reference the final pre-deadline `poll-scoreboard` payload,
with that poll's retrieval time as `schedule_retrieved`. Publish the capture-ledger head to
the protected append-only remote before the earliest cutoff it covers — the chain is
tamper-evident locally but only becomes externally timestamped on publication.

## Reschedule-amendment rule

If a game's tipoff moves between polls, the collector **keeps every original capture** and
appends a `schedule_amendment` event recording `original_tipoff`, `previous_tipoff`, and
`amended_tipoff`; future cutoffs derive from the amended tipoff. A game that vanishes from
the scoreboard (postponement or cancellation) gets an amendment with `amended_tipoff: null`
and is excluded from further captures. Nothing is ever deleted or rewritten. Note that the
machine-readable v1 coverage policy rejects post-first-input reschedules outright; these
amendment records are the raw material for the future amendment protocol the README names,
not an ad hoc exception.

## Operations

No daemons, no threads, stdlib only plus the pinned `nbainjuries` extra. `run` does the work
due now and exits, so it is cron-friendly:

```cron
*/5 * * * * cd ~/forecastfm && .venv/bin/python examples/collect_prospective.py run >> data/raw/prospective/state/cron.log 2>&1
```

A failed fetch (after the bounded retries) exits non-zero; the next cron invocation retries
and reuses any payloads already written. `plan` is always offline and needs no state.
`run --dry-run` prints the poll and captures it would perform with no network and no writes;
`run --now <ISO-8601 UTC>` overrides the clock for rehearsals.

## October dry-run checklist

Complete before the first regular-season tipoff (2026-10-21):

1. `uv sync --extra pdf-audit`; confirm
   `.venv/bin/python -c "import nbainjuries._util"` succeeds.
2. `.venv/bin/python examples/collect_prospective.py plan --date 2026-10-21` prints a plan
   with no network and no state (verified 2026-07-19).
3. `run --dry-run --now 2026-10-19T12:00:00Z` shows `would_poll` for 10-19…10-21 and no
   captures; confirm the storage root is not created by the dry run.
4. First live poll (before opening night): `run`, then confirm three
   `poll-scoreboard-*.json` payloads, one `schedule_poll` ledger event chaining from the
   genesis hash, and root permissions `0700` with payloads `0600`.
5. Rehearse a cutoff with `run --dry-run --now <T-60 minus one minute>` for the first real
   game; confirm `would_capture` lists t_6h and t_60. After the first real capture, confirm
   the PDF, scoreboard, sidecar, and `capture` event exist, that `within_window` is true,
   and that an immediate re-run captures nothing.
6. Re-verify the chain locally (recompute per the rules above) and publish the
   capture-ledger head to the protected remote before the earliest T-6h cutoff it covers.
7. Install the cron entry above from opening night; confirm `cron.log` shows exit-0 runs.
8. Parse one captured PDF with the `build_nba_injury_archive.py` parser as a smoke check;
   remember that player-health rows stay local-only under the health-data boundary.

This collector changes no authorization: prospective-win and RL claims remain denied until
the production gates in [README.md](README.md) pass on this prospective evidence.
