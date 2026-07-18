# Prospective integrity protocol

This directory holds the files that turn a forecast into an auditable pre-event commitment.
All files here are tracked by Git. Forecast records must never be amended, rebased, squashed, or
force-pushed after publication.

The simple data path is:

```text
licensed raw snapshot pack -> causal evidence bundle -> target-free model rows -> sealed resolutions
```

Before licensing and connector review, `nba_raw_capture.py` can retain one exact caller-supplied
response-entity buffer plus selected allowlisted response-header fields in a canonical, create-only
artifact under an explicit restricted storage root. Its request identity and retrieval time are
caller assertions, and its claim is only `local_retrieval_only`. The body, path, and even
allowlisted header values may still be sensitive, so captures belong under ignored private data
storage. The artifact does not prove HTTPS transport or provider identity, set
`provider_published_at` or `available_at`, establish revision history or rights, or enter the
training chain directly. A reviewed provider connector must derive and justify a licensed snapshot
pack separately.

`sportsdataio_nba_openapi.py` constructs the eight request identities needed by the vendor
inspection from typed date, season, team-ID, and game-count values. It only mirrors the public
OpenAPI paths and fixed production hostname. It does not make a request, carry a key, establish
Replay availability, or prove that a response came from that host.

The bounded `sportsdataio_nba_client.py` layer can use one assignment in the ignored
`.sportsdataio.env` file:

```text
SPORTSDATAIO_API_KEY="your-actual-key"
```

Run `chmod 600 .sportsdataio.env`; the loader rejects symlinks, non-regular files, oversized files,
other owners, and any group or other permissions. With its default transport, the client performs
one certificate-verified fixed-host `GET` for a registered path, forces HTTP debug output off, and
rejects redirects, retries, compression, unsafe HTTP framing, and any response that reflects the
key. The injectable transport is a trusted test seam, not provenance. Its output is still only a
`local_retrieval_only` raw capture. No network call has been made. The path does not prove Replay
entitlement, provider identity, publication chronology, revision completeness, rights, data
conformance, or model authorization. Production and RL remain closed.

The raw pack binds the exact provider bytes to a per-entity source ID, stable licensed rights scope,
version, timing, sensitivity, rights, and hash metadata. The causal bundle retains a digest of that
complete metadata, selects only snapshots eligible at one declared cutoff, and records the lineage
of each derived number. Model-facing rows contain no winner, score, or other postgame target.
Resolutions are created and sealed separately only after the forecasts are committed.
Keep raw packs and signed agreements under the ignored `data/` tree, never in `prospective/` or a
Git commit. Only publish contract/artifact hashes and fields the signed agreement permits.
Use [`OUTCOME_V2_VENDOR_REQUIREMENTS.md`](OUTCOME_V2_VENDOR_REQUIREMENTS.md) before accepting a
historical pack or signing Tinker-processing rights.

The provider-neutral code can round-trip bundles, verify that each source is the latest eligible
snapshot, and recompute a frozen numeric row from bundle records. It cannot interpret an opaque
vendor payload. Until a reviewed production connector deterministically derives those records from
the raw bytes, a bundle can still contain self-declared numbers. The offline preflight now binds and
checks the complete sealed chain—snapshot pack, evidence bundles, target-free rows, Elo replay and
states, season map, final-score resolutions, evaluation pack, rights lock, and manifest hashes—but
it cannot replace that provider-specific derivation. A passing preflight also returns the exact
manifest digest for the eventual immutable training-run lock. The paid preparation path derives
its protected action time internally and retains the exact validated training bytes. The
readiness boolean cannot bypass the empty-missing-list, two-untouched-season, or raw and
training-only-recalibrated Elo gates, and training seasons must be disjoint from the named holdouts.
Opened historical pass/fail booleans remain diagnostic metadata and have no readiness authority.
The verifier now recomputes both gates from separate canonical cohort, answer, forecast, and
training-only calibration files. Preflight binds that calibration set exactly to training, then
binds evaluation dates, raw Elo, and answers to a separate deterministic replay and sealed scores.
The production policy requires at least 1,000 games and 20 calendar blocks in every named season.
It still does not prove candidate-model origin, external precommit time, or raw-provider derivation.
Production preflight therefore has an explicit hard failure until reviewed connector and pre-event
commitment verifiers exist. `full_outcome_v2_ready` must remain false. A local immutable run-lock
core now binds a passing proof, exact training bytes, code, configuration, model reference, prompt,
and packages before remote-client creation. The guarded paid entrypoint writes and re-verifies that
lock before late-importing a direct Tinker runtime; the runtime renders the exact retained bytes
before constructing a client. A successful run then creates a separate canonical experiment seal
binding the exact run lock to its permanent Tinker state and sampler paths. The current hard gate
prevents this path from reaching Tinker, and no paid outcome-v2 job has been launched.

After training, `outcome_v2_sft_gate.py` requires a separate answer-free evaluation chain. It does
not reuse the tabular gate's opened seasons: SFT IDs must be disjoint and every SFT season must be
later. Its seal binds strict feature rows and their exact original/side-swapped prompts to the
verified experiment sampler, pre-call generation lock, terminal raw label-logprob records, and
exact derived forecast bytes. The post-SFT report has its own kind and replays the raw- and
training-only-recalibrated Elo gate.

This one-lock path is a retrospective answer-held holdout, not a prospective claim. The separate
`outcome_v2_rolling.py` path freezes a multi-season plan, accepts one-season slate batches, binds
the terminal raw inference records, and enforces one overlapping window:

```text
latest input availability
    <= generation-lock creation
    <= answer-free forecast seal
    <= terminal batch seal
    < earliest T-60 cutoff
```

Two live GitHub Actions receipts close the externally visible timing chain. The plan receipt must
be created before the batch's earliest input; the terminal-seal receipt must be created before its
earliest cutoff. The verifier checks the exact repository, `main` branch, `push` event, numeric
workflow ID, workflow path and bytes, first run attempt, successful status, and artifact bytes at
the full run head SHA. It uses GitHub's server-side run `created_at`; it does not relabel
`updated_at` as completion time or trust client-controlled Git dates.

`outcome_v2_coverage.py` adds one externally receipted schedule seal per planned season. The seal
structurally binds the exact replay rows to a caller-supplied provider-conformance report; it does
not authenticate that report or prove that the rows are the complete NBA schedule.
`outcome_v2_aggregation.py` rejects duplicate batches plus any missing or extra question ID before
binding the verified schedule and forecasts derived from terminal raw records. After outcomes,
`outcome_v2_rolling_score.py` re-verifies that aggregate, independently replays the frozen canonical
Elo recipe, checks the feature-row Elo probabilities, and then creates the separate scoring cohort
and answer inputs. Its create-only scoring seal hashes the exact snapshot-pack and resolution
bytes, uses each bound snapshot's `available_at` for Elo chronology instead of the resolution row's
declared `resolved_at`, and freezes league game dates as scheduled tipoff converted through
`America/New_York`. It also binds the cohort, answers, and forecasts.

The prospective plan commits the original training-only calibration hash and evaluation-policy
hash from the immutable run lock. The scoring seal requires the exact calibration bytes to match
that plan and carries both hashes. `outcome_v2_rolling_gate.py` rejects a substituted calibration
or weakened policy, re-runs the generic multi-season gate, and checks its cohort, answer, forecast,
and calibration hashes plus IDs and seasons. Its terminal wrapper status is
`structural_claim_only`; prospective-win and RL authorization are both `denied`. The
machine-readable v1 coverage policy is regular-season only and rejects cancellations or
reschedules after the first season input; those cases need a future amendment protocol, not an ad
hoc exception.

These receipts are centralized live GitHub evidence, not signatures or transparency-log proofs,
and workflow runs can be deleted. No production plan, coverage seal, batch receipt, or aggregate
exists yet. The local coverage status is deliberately
`structurally_bound_to_claimed_conformance_report`. Provider/reviewer authentication, licensed
production bytes, and remote-call attestation remain separate requirements, so the aggregate
cannot claim complete schedule coverage or complete prospective proof. No provider-backed
production remote run has occurred. The scorer does not parse an opaque final-score payload;
provider authentication and licensed connector derivation of each score remain separate. A local
passing report therefore cannot authorize a prospective win claim or RL. The production rolling
gate remains hard-closed until the reviewed provider parser and authenticity verifier exist.

Before accepting a vendor sample, run its reviewed connector through
`nba_provider_conformance.py`. The validator requires a claimed independently reviewed inventory
that binds every decoded revision field and every schedule fact, then exact-checks archive
revisions, T-6h/T-60/T-15 selection, corrections/deletions, known gaps, Elo replay rows, cohort
games, and the snapshot-pack hash. The report includes a digest of the exact replay rows consumed
by coverage.
A passing report is deliberately bounded: the caller-supplied connector and
inventory digests do not authenticate the reviewer, connector code, vendor, agreement, or time.

The rich tabular candidate is also separated at the answer boundary. Run
`examples.fit_nba_rich_baseline` only on sealed training feature rows and resolutions. Run
`examples.predict_nba_rich_baseline` on the resulting model and strictly later target-free rows;
that module has no answer input and emits a create-only forecast lock. The lock binds exact model,
feature-row, ID, season, and forecast bytes. It still needs external publication before answer
release; a local create-only file is not trusted time. The model bytes stay under the ignored data
tree by default because permission to retain or use derived weights is not permission to publish
them. Only the lock/hash should be committed unless the signed agreement permits redistribution.

The commitment protocol has five layers:

1. `training_lock.json` freezes the committed code revision, exact prompt, dataset hashes,
   tokenizer revision, Tinker versions, training recipe, and decoding policy.
2. `experiment.json` is created only after training and binds the training lock to Tinker's final
   permanent `sampler_path`.
3. A cohort file freezes every game in a declared slate, its schedule snapshot, and each forecast
   deadline.
4. Each causal evidence bundle freezes source hashes, timestamps, rights, sensitivity, and the exact
   numeric records used by one forecast.
5. `ledger.jsonl` hash-chains one complete forecast batch and, later, one complete resolution batch
   for each cohort. Every raw model response and evidence-bundle digest is retained.

The first two locks are created with:

```bash
uv run --extra tinker python examples/freeze_training_lock.py
# Run the separately approved paid training step.
uv run --extra tinker python examples/freeze_experiment.py
```

Both commands refuse to overwrite an existing lock. A materially different model, prompt,
dataset, or decoding policy is a new experiment and must use a new directory or Git history—not
an edit to an old lock.

Verify the chain locally, or against a head copied from the protected remote or timestamp receipt:

```bash
uv run python examples/verify_prospective_ledger.py
uv run python examples/verify_prospective_ledger.py PUBLISHED_HEAD_SHA256
```

## What the hashes prove

The chain detects modification, reordering, deletion, and incomplete cohort coverage once its
head hash has been independently published. It does not prove publication time by itself. Local
Git author dates can be changed, and an unpublished chain can be rebuilt from scratch.

Before the earliest forecast deadline, publish the exact rolling artifact on `main`. The dedicated
workflow gives the verifier a GitHub-hosted run creation time bound to the full commit SHA; the
verifier then fetches the exact artifact and frozen workflow bytes at that SHA. A signed commit
authenticates its author but still does not independently prove when it was created.

This repository has a remote, but a remote by itself is not trusted time. Until a ledger head is
published to a protected append-only branch or an independent timestamp service, its status is
**locally tamper-evident, not externally timestamped**.

## Cutoff and availability contract

The primary supervised and prospective state is **T-60**, exactly 60 minutes before scheduled
tipoff. Optional T-6h and T-15m states may be retained under separate state IDs for sequential
update and latency analysis. They never replace, amend, or get pooled silently with T-60.

`available_at` is the only knowledge-time gate:

- for a live capture, `available_at` equals the time the project retrieved the payload;
- for an attested, immutable provider archive, it equals the provider's publication timestamp; and
- `effective_at` says when a fact or schedule state applies. It may legitimately be after the
  forecast cutoff, so it must never be used as proof that the payload was already knowable.

An archive also needs an immutable provider version and a hash of the provider's revision/backfill
attestation. A record with `available_at` after a state's cutoff is ineligible for that state.

## Model-facing feature contract

Every standard feature is a team-minus-opponent difference with an exact side swap. The stable
11-feature order is `rest_days`, `back_to_back`, `games_last_7`, `road_games_last_7`,
`travel_miles`, `travel_time_zones`, `roster_continuity`, `expected_lineup_continuity`,
`rolling_team_net_rating`, `rolling_player_value`, and `schedule_strength`.

`unavailable_rotation_minutes` and `unavailable_rotation_value` are player-health-derived and
local-only. The standard Tinker path excludes them and rejects any player-health lineage. Elo must
be computed in-house from strictly prior outcomes under a frozen recipe or obtained from a source
whose signed license clears this use. Preflight now recomputes the sealed Elo states with a
deterministic replay, using only results available by each cutoff. The claimed conformance report
binds the replay rows structurally; an authenticated licensed connector and inventory must still
prove schedule completeness and that every team and site field was derived from retained provider
bytes.

No paid training or prospective forecast begins until the raw artifact and rights hashes validate,
the exact reviewed agreement matches its immutable rights-approval lock, the required local,
third-party, and Tinker permissions are explicitly allowed, every stable rights scope is reviewed,
every per-entity source is
cutoff-eligible, model rows are target-free, feature keys and lineage match the frozen schema,
side-swap pairs and cohort coverage are exact, and the simple tabular correction passes the
declared multi-season chronological gates against Elo.

## Forecast rules

- Declare every game in the cohort before generating predictions.
- Bind each submission to the exact `evidence_bundle_sha256`; never substitute evidence after a
  model call.
- Keep T-60, T-6h, and T-15m as distinct immutable forecast states.
- For fixed-token SFT inference, make exactly four logical candidate calls per game: TEAM and OTHER
  for the original and side-swapped prompts. Retain both raw log-probabilities and every failure.
- Disable application retry logic and never select among attempts. Tinker 0.22.7 may still
  retransmit the same logical request ID for up to five minutes after connection or timeout errors
  and HTTP 408, 409, 429, or 5xx responses; disclose that transport behavior.
- Commit the complete cohort before its earliest deadline and before every scheduled tipoff.
- Append resolutions later; never edit forecasts to add outcomes.
- Verify against the externally published head when scoring.
- Report malformed output and missing coverage as failures rather than silently dropping rows.
