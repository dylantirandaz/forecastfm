# Prospective integrity protocol

This directory holds the files that turn a forecast into an auditable pre-event commitment.
All files here are tracked by Git. Forecast records must never be amended, rebased, squashed, or
force-pushed after publication.

The simple data path is:

```text
licensed raw snapshot pack -> causal evidence bundle -> target-free model rows -> sealed resolutions
```

The raw pack binds the exact provider bytes to a per-entity source ID, stable licensed rights scope,
version, timing, sensitivity, rights, and hash metadata. The causal bundle retains a digest of that
complete metadata, selects only snapshots eligible at one declared cutoff, and records the lineage
of each derived number. Model-facing rows contain no winner, score, or other postgame target.
Resolutions are created and sealed separately only after the forecasts are committed.
Keep raw packs and signed agreements under the ignored `data/` tree, never in `prospective/` or a
Git commit. Only publish contract/artifact hashes and fields the signed agreement permits.

The provider-neutral code can round-trip bundles, verify that each source is the latest eligible
snapshot, and recompute a frozen numeric row from bundle records. It cannot interpret an opaque
vendor payload. Until a reviewed production connector deterministically derives those records from
the raw bytes, a bundle can still contain self-declared numbers. The paid preflight now binds and
checks the complete sealed chain—snapshot pack, evidence bundles, target-free rows, Elo states,
season map, final-score resolutions, rights lock, and manifest hashes—but it cannot replace that
provider-specific derivation. A passing preflight also returns the exact manifest digest for the
eventual immutable training-run lock together with the non-future protected action time. The
readiness boolean cannot bypass the empty-missing-list, two-untouched-season, or raw and
training-only-recalibrated Elo gates. Therefore `full_outcome_v2_ready` must remain false.

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

Before the earliest forecast deadline, publish both the Git commit and printed ledger head to a
protected remote branch or an external transparency/timestamp service. Treat public CI creation
time as the latest defensible publication time. A signed commit authenticates its author but still
does not independently prove when it was created.

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
whose signed license clears this use. The sealed Elo-state file recomputes each probability from
its exact ratings, home edge, scale, and recipe digest; a deterministic replay from prior resolved
games is still required to prove that the ratings themselves are causal.

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
- Use one model call and one retained raw response per game; never select among retries.
- Commit the complete cohort before its earliest deadline and before every scheduled tipoff.
- Append resolutions later; never edit forecasts to add outcomes.
- Verify against the externally published head when scoring.
- Report malformed output and missing coverage as failures rather than silently dropping rows.
