# NBA data-rights gate

No modern NBA feed is currently licensed for this project. In particular, nothing is yet cleared
for Tinker upload or commercial redistribution. Keep every permission `unknown` until a signed
agreement says otherwise; the code fails closed.

## Audited source decision

[SportsDataIO production NBA access](https://sportsdata.io/nba-api) is the best current one-vendor
candidate because its documented [NBA workflow](https://sportsdata.io/developers/workflow-guide/nba)
covers schedules, rosters, injuries, projected and confirmed lineups, statistics, and historical
data. The practical request is a narrow Vault + Leagues agreement through
[SportsDataIO sales](https://sportsdata.io/contact-us). Its
[free developer trial](https://sportsdata.io/developers) is scrambled rather than real data, so it
cannot populate or validate this dataset. SportsDataIO's
[historical-data page](https://sportsdata.io/historical-sports-data) describes machine-learning use,
but that description is not a grant of the exact storage, Tinker, derivative-output, or
redistribution rights required here; the signed order form and
[current terms](https://sportsdata.io/terms-of-service) control.

[Sportradar's NBA API](https://developer.sportradar.com/basketball/reference/nba-overview) is the
official-data alternative, but its standard developer access and
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
