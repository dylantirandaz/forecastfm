# Outcome-v2 vendor acceptance gate

Status, 2026-07-17: the SportsDataIO ladder below was executed through a four-file sample audit
and a pre-NDA Sales clarification, and is **closed with a rejection for the T-60 historical use
case**. The provider does not retain correction/deletion revision history, historical depth
charts, or reconstructable earlier pregame injury/lineup state. The SportsDataIO-specific rungs
are preserved as the record of that execution; they are not a recommendation to proceed. The
vendor-neutral criteria in this document (required pre-contract sample, coverage, rights in
writing, health-data boundary, rejection conditions) remain the standing acceptance gate for every
later vendor. Sportradar is the next candidate; the bounded inspection plan is
[SPORTRADAR_TRIAL_PLAN.md](SPORTRADAR_TRIAL_PLAN.md), and the decision record is
[DATA_RIGHTS.md](DATA_RIGHTS.md).

Use an evidence ladder instead of a purchase-first workflow. Start with a free Replay package and
endpoint inspection. If that integration check is promising, request SportsDataIO NBA League API
access, Vault, and a separately negotiated raw revision-history export. Vault and Replay alone are
not accepted as proof of point-in-time injuries or projected lineups.

Useful official references:

- [Developer access levels](https://sportsdata.io/developers)
- [Replay](https://sportsdata.io/developers/replay)
- [NBA OpenAPI specification](https://cdn.sportsdata.io/openapi/NBA-openapi-3.1.json)
- [NBA API](https://sportsdata.io/nba-api)
- [NBA workflow guide](https://sportsdata.io/developers/workflow-guide/nba)
- [NBA data dictionary](https://sportsdata.io/developers/data-dictionary/nba)
- [Vault](https://sportsdata.io/vault)
- [Replay and integration tools](https://sportsdata.io/developers/integration-tools)

## Verified access ladder

Public access descriptions establish how a feed may be explored. They do not, by themselves, prove
that the data satisfy outcome-v2's point-in-time or processing-rights requirements.

1. Do not train or evaluate on Free Trial responses. The provider says those responses are
   scrambled and intended only for integration testing.
2. Use Replay as the zero-cost connector prototype if the logged-in NBA package exposes the needed
   endpoints. Replay provides real archived production responses, but its exact NBA sessions and
   endpoint coverage must be inspected. A working Replay connector does not prove complete revision
   history, original publication-time chronology, arbitrary-cutoff reconstruction, schedule
   coverage, or Tinker processing rights. A Replay or archive timestamp must not be relabeled as
   `provider_published_at` or trusted `available_at` unless its semantics are independently proven.
3. Do not use Discovery Lab to claim historical T−60 injury or lineup state. Its public access is
   next-day delayed and aimed at personal or hobby use. It can support limited schema exploration,
   but not the outcome-v2 point-in-time or commercial-rights gate.
4. Consider paid NBA League API and Vault access only after the required pre-contract sample below
   passes and a signed agreement grants the required rights. Public product pages do not establish
   arbitrary-cutoff injury and lineup revision history or permission to send derived rows to Tinker.

No purchase, production-key use, or Tinker upload is authorized by this document. The next safe
step is a free Replay package inspection or a vendor-supplied pre-contract sample.

No access tier opens the production rolling gate by itself. Provider and reviewer authentication,
reviewed connector derivation, final-score parsing, trusted chronology, licensed raw bytes, and
remote-execution attestation remain separate requirements. Prospective-win and RL authorization
remain denied until those proofs exist and the frozen evaluation gate passes.

### Replay package inspection checklist

After login, record whether Replay exposes each required NBA OpenAPI operation and which dated
session was inspected:

- `Games/{season}` and `GamesByDateFinal/{date}` for schedule identity and final scores.
- `DepthCharts` and `TransactionsByDate/{date}` for roster state and changes.
- `StartingLineupsByDate/{date}` and `InjuredPlayers` for pregame availability state.
- `TeamGameStatsBySeason/{season}/{teamid}/{numberofgames}` and
  `PlayerGameStatsByDate/{date}` for strictly prior rolling metrics.

`sportsdataio_nba_openapi.py` now constructs exactly these eight public OpenAPI request identities
from typed values. That registry contains no network client or key and is not evidence that the
logged-in Replay package enables an operation. Actual access must still be recorded during the
inspection.

For a bounded local production-host check, `.sportsdataio.env` is ignored and accepts exactly one
local assignment:

```text
SPORTSDATAIO_API_KEY="your-actual-key"
```

Run `chmod 600 .sportsdataio.env`; the loader requires a bounded, owned, non-symlink regular file
with no group or other permissions. With its default transport, `sportsdataio_nba_client.py`
performs one certificate-verified fixed-host `GET` for a registered path, forces HTTP debug output
off, and rejects redirects and retries, compressed responses, unsafe HTTP framing, and any
response that reflects the API key. The injectable transport is a trusted test seam. A successful
response is returned only as a `local_retrieval_only` raw capture. No network call has been made.
This path does not prove Replay entitlement, provider identity, publication chronology, revision
completeness, rights, data conformance, or model authorization; production and RL remain closed.

For every response, retain the exact request path, response-entity bytes, selected allowlisted
response headers, local UTC retrieval time, and SHA-256. The capture does not retain or prove the
HTTP wire bytes or complete header block. Never place the API key in the URL or captured metadata;
use the documented `Ocp-Apim-Subscription-Key` request header. Repeated responses must be retained
as separate captures so changes can be compared. This inspection answers only whether a connector
can be built. It does not establish complete seasons, original publication times, revision
semantics, or training rights.

Do not relabel the fixed production-host client as a Replay client. Replay's actual host, session
controls, enabled operations, and realistic response-size limits still require logged-in
inspection. Raw responses remain `local_retrieval_only`; they must not be converted into
provider-versioned snapshot records until publication-time and revision-authenticity semantics are
separately established.

The provider-neutral `nba_raw_capture.py` artifact is ready to envelope those responses. It binds
exact caller-supplied response-entity bytes, a caller-asserted local UTC retrieval time, selected
allowlisted response-header fields, a declared request identity, and hashes in one create-only
canonical file under an explicit restricted storage root. The schema has no credential field, but
the generic envelope cannot prove that a caller omitted secrets from its path, body, or header
values. The bounded client uses the fixed production host and endpoint registry and rejects a
response that reflects the API key. Callers must still pin an ignored `data/raw/...` capture
directory owned by the current user with mode `0700`; the generic writer rejects group- or
world-accessible roots. Neither layer supplies provider identity proof, publication chronology,
archive attestation, rights decisions, conformance results, or model authorization.

## Required pre-contract sample

The vendor must supply raw revision history for games containing all of these cases:

- A late scratch or injury-status change.
- A projected lineup that later changed.
- A roster transaction.
- A postponed or rescheduled game.
- Every captured version, correction, and deletion—not only the final record.

Every captured record must include:

- Stable source, entity, and revision IDs.
- Original raw payload bytes and a checksum.
- `effective_at`, provider publication time, and archive capture time in UTC.
- Schema and API versions.
- Correction and deletion markers.

The sample passes only if an independent replay can select exactly the latest information that was
available at T−6h, T−60, and T−15 without reading a later corrected state.

The repository's provider-conformance validator turns this sample into a bounded, deterministic
check. Its reviewed inventory must bind every full revision envelope—not only revision IDs—and
every schedule fact used by Elo or the cohort. A digest match proves agreement with those reviewed
bytes, not who reviewed them or whether the vendor and connector are trustworthy. Those identities,
the agreement, connector code digest, and any trusted timestamp must be verified separately.

## Coverage required

The signed schedule must name seasons, season types, known gaps, and update cadence for:

- Complete schedules, venue identity, postponements, and final scores.
- Rosters, transactions, depth charts, and player/team statistics.
- Injuries and availability changes.
- Projected and confirmed lineups.
- Historical corrections and deletions.

## Rights required in writing

The agreement must explicitly permit:

- Local copying, caching, normalization, joining, feature engineering, evaluation, and backtesting.
- Model training and fine-tuning on licensed or contract-approved derived data.
- Third-party processing of approved derived rows by Thinking Machines Lab's Tinker service and its
  infrastructure subprocessors.
- Retention and use of permitted derived rows, predictions, adapters, checkpoints, weights, and
  aggregate evaluation reports.
- Publication of aggregate model comparisons that do not reconstruct source records.

It must also state:

- Whether raw snapshots may be retained for reproducibility after termination.
- Which derived artifacts may remain usable after termination.
- Whether upstream NBA, news, lineup, or injury licensors impose additional limits.
- That the negotiated data-rights schedule overrides conflicting public or per-record terms for the
  approved use.

The repository rights lock must be able to record `local_processing`, `third_party_processing`, and
`tinker_processing` as allowed for exact named scopes. Unknown permission remains a hard failure.

## Health-data boundary

Named injuries, raw injury text, player IDs attached to health information, and the two
health-derived availability features remain local-only. Standard Tinker rows may contain only
contract-approved, non-health, non-reconstructable features. Any broader upload requires an
explicit vendor amendment and an appropriate Tinker enterprise agreement first.

## Rejection conditions

Reject the historical pack for outcome-v2 if any of these are true:

- “Historical” means only the latest values returned today.
- Publication or capture times are missing or self-inferred.
- Projected-lineup or injury revisions cannot be reconstructed at arbitrary pregame cutoffs.
- The export omits raw bytes, stable revisions, corrections, or deletions.
- Tinker processing rights rely on silence, a public marketing page, or an unknown field.

If the historical revision export fails, the feed may still be used as a schedule/statistics
backbone while append-only prospective polling begins. It must not be represented as a historical
point-in-time lineup or injury source.
