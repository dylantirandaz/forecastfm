# NBA data-rights gate

No modern NBA feed is currently licensed for this project. In particular, nothing is yet cleared
for Tinker upload or commercial redistribution. Keep every permission `unknown` until a signed
agreement says otherwise; the code fails closed.

## Audited source decision

### Decision log, 2026-07-17: SportsDataIO rejected for the T-60 historical task

[SportsDataIO production NBA access](https://sportsdata.io/nba-api) was audited through four
vendor-supplied sample files (starting lineups, box scores, player game logs, season aggregates)
and a pre-NDA Sales clarification, and is rejected as the primary historical T-60 injury/lineup
source. The samples were internally consistent finalized postgame snapshots with no publication,
revision, deletion, or as-of fields. Sales stated (user-supplied correspondence, nonbinding) that
they do not retain every correction/deletion revision, that historical depth charts were not
retained, that historical injury state exists only embedded in the final Box Score record as
status at game start, and that the historical API is not designed to reconstruct an earlier
pregame state. A game-start injury field could support a separate T0 lock-time task only under
contractual publication-timing, completeness, versioning, and rights guarantees; it cannot support
the T-60 task. The project did not proceed to NDA/Sales. The SportsDataIO registry, client, and
capture code remain in the repository as a provider-security and schema boundary and for possible
schedule/statistics, T0, or prospective uses if those are separately proven. Do not represent
SportsDataIO as satisfying outcome-v2 T-60 history.

For the record: its [free developer trial](https://sportsdata.io/developers) is scrambled rather
than real data, so it cannot populate or validate this dataset. Replay can exercise a connector
against real archived responses at zero cost, but the one inspected NBA Replay session covered a
single week (2023-11-21 through 2023-11-28 EST) with package/session-specific keys and no
self-service full-season package, and Replay does not by itself prove complete revision history,
original publication times, or training rights. SportsDataIO's
[historical-data page](https://sportsdata.io/historical-sports-data) describes machine-learning use,
but that description is not a grant of the exact storage, Tinker, derivative-output, or
redistribution rights required here; the signed order form and
[current terms](https://sportsdata.io/terms-of-service) would have controlled.

### Leading trial candidate: Sportradar

[Sportradar's NBA API](https://developer.sportradar.com/basketball/reference/nba-overview) is the
leading trial candidate. Its documentation advertises NBA data back to 2013, official NBA-sourced
data from 2017, a date/current-state Daily Injuries feed, and a Daily Change Log with changed
resource IDs and last-modified timestamps. The unresolved decisive question: a change timestamp
proves that an object changed, not that every old payload value remains queryable. The bounded
trial inspection in [SPORTRADAR_TRIAL_PLAN.md](SPORTRADAR_TRIAL_PLAN.md) must answer that before
any spend. Its standard developer access and
[terms](https://developer.sportradar.com/sportradar-updates/page/terms-and-conditions) do not by
themselves clear this pipeline. Use it only under a custom written agreement covering the intended
machine-learning and third-party-processing uses.

A bring-your-own-license (BYO-license) pack is the fallback: the buyer supplies canonical snapshot
packs plus the signed license and rights attestation, and the provider-neutral pipeline validates
them locally. Supplying bytes, buying commercial access, or possessing a public API key does not
itself establish permission to store history, derive features, use Tinker, train models, or
redistribute a data pack.

For either vendor or a buyer-owned pack, the signed agreement must expressly cover:

- retention of historical and live records and their revisions;
- feature engineering, supervised fine-tuning, RL, calibration, and evaluation;
- Thinking Machines/Tinker as a named third-party processor;
- commercial use and ownership of adapters, weights, forecasts, and aggregate evaluations;
- publication of metrics, scaling plots, audits, and failure analyses without raw licensed rows;
- survival of model and checkpoint rights after the subscription ends;
- injury and availability data, if processed locally; and
- point-in-time revision history, correction timestamps, backfill policy, and upstream-rights
  warranties.

Raw licensed bytes remain outside redistributable and model-facing artifacts. Derived rows are not
assumed redistributable merely because they omit the raw payload.

Before any readiness-true Tinker preflight, create `nba_rights_approval_lock.json` from the exact
reviewed agreement bytes. The lock freezes the agreement SHA-256, provider/license IDs, stable
rights scopes, review decision, and each permission; preflight requires both that lock and the
unchanged private agreement file. A rights scope names a licensed feed or endpoint family and stays
stable across per-game or per-query source IDs. This proves which bytes were reviewed, not that the
agreement is authentic or that the legal interpretation is correct. Those remain human/legal review
responsibilities.

## Point-in-time rule

A live snapshot uses `available_at = retrieved_at`. An attested provider archive uses
`available_at = provider_published_at` and also requires an immutable provider version, exact raw
payload hash, and hash of the provider's revision/backfill attestation. `effective_at` describes
when the underlying fact or scheduled state applies and may be in the future; it is never a
substitute for `available_at`.

The main supervised cutoff is T-60. Optional T-6h and T-15m states are retained separately and use
their own availability gates. A provider archive may support retrospective development, but it is
not proof that this project captured the bytes prospectively.

The data path is raw snapshot pack -> causal evidence bundle -> target-free model rows -> separately
sealed resolutions. Each numeric feature remains bound to its source hash, timestamps, rights
decision, stable rights scope, complete snapshot-metadata hash, and sensitivity. The standard 11
numeric features may enter Tinker only with explicit
third-party and Tinker permissions. The two health-derived availability features remain local-only;
standard Tinker conversion rejects player-health lineage even after numeric aggregation.
