# Roadmap

Each milestone must produce a measurable forecasting improvement before the next one begins.

## 1. Offline contract — complete

- Validate timestamped forecast cases and categorical distributions.
- Load a pinned CC-BY real NBA forecast source with an exact hash.
- Select a deterministic, balanced perspective and exclude identity and postgame fields.
- Create chronological, exact-prompt-disjoint train, validation, and test splits.
- Separate target-free evaluation prompts from answer keys.
- Quarantine source rows whose date-only cutoff is ambiguous.
- Score predictions with Brier score and log loss.
- Export strict SFT conversations after conservative health-term screening.
- Enforce Ruff, Pyright strict, and focused unit tests.

## 2. First Tinker SFT smoke test

- Pin the Tinker SDK and Cookbook versions.
- Convert exported conversations with the model's official renderer.
- Prove the pipeline with one `Qwen/Qwen3.5-4B` LoRA step. — complete
- Save the adapter, training metrics, configuration, and data hashes. — complete
- Compare base and adapter on the exact same anonymous, complete cohort.
- Freeze every raw response before loading the answer key.
- Report JSON validity and error against the closed-form Elo oracle.

The smoke test succeeds only if JSON validity and oracle fidelity improve without selective
retries or dropped rows. Historical Brier score, log loss, and calibration are reported only as
contamination-prone diagnostics; they do not establish forecasting improvement.

## 3. Prospective integrity protocol — locally complete

- Freeze the committed code, prompt, dataset, tokenizer, training, and decoding configuration.
- Bind the completed adapter to Tinker's permanent sampler path before inference.
- Declare every game and forecast deadline in a complete cohort manifest.
- Append one atomic forecast batch and one later resolution batch per cohort.
- Retain exact prompts and raw responses and reject retries, missing games, or changed outcomes.
- Verify a canonical SHA-256 chain against a previously published head.

Local completion provides tamper evidence, not trusted time. A fail-closed GitHub Actions receipt
verifier and path-filtered `main` push workflow are now implemented. They bind exact plan or batch
bytes and the frozen workflow bytes at the run's full head SHA, using the server-side run creation
time as centralized existence evidence. No production receipt has been created yet, and GitHub is
not a signed transparency log.

## 4. Realized-outcome optimization — locally implemented

- Map the realized winner to two verified single-token labels.
- Train only the winner token with ordinary cross-entropy.
- Convert candidate-token log-probabilities into a binary forecast without text generation.
- Pair every fit and development game with its exact side swap.
- Keep the old Elo-distillation adapter as a fixed historical baseline.
- Run 32, 128, 512, and 2,048-step checkpoints only after freezing each paid protocol.

Local implementation is not a forecasting result. Advancement requires improved chronological
development log loss and Brier score, acceptable calibration, and side-swap consistency. Legacy
holdout answers already exist locally, so a new prospective cohort is required for a truly unseen
final claim.

## 5. Leakage-safe outcome v2 — in progress

- Acquire a licensed point-in-time snapshot pack from a vendor passing the
  [`prospective/OUTCOME_V2_VENDOR_REQUIREMENTS.md`](prospective/OUTCOME_V2_VENDOR_REQUIREMENTS.md)
  acceptance gate, or a buyer-owned licensed snapshot pack under the same gate. SportsDataIO was
  audited through samples and a Sales clarification and was rejected for the T-60 historical use
  case on 2026-07-17; Sportradar is the next trial candidate. No modern source is currently
  cleared.
- A provider-neutral create-only raw-response artifact now binds exact caller-supplied response
  entity bytes, selected allowlisted header fields, hashes, and caller-asserted request/time
  metadata. Its scope is strictly `local_retrieval_only`; it is not proof of HTTPS transport,
  provider identity, wire bytes, an authenticated archive, or a licensed snapshot pack.
- A small typed registry now constructs the eight required paths from SportsDataIO's public NBA
  OpenAPI schema with fixed date, season, team-ID, and game-count inputs. It does not prove that a
  Replay account enables any path or authorize a network request.
- An ignored, bounded, owned, mode-private `.sportsdataio.env` file can hold one
  `SPORTSDATAIO_API_KEY="..."` assignment. The default transport performs one
  certificate-verified fixed-host registered `GET`, forces HTTP debug output off, and has no
  redirect, retry, compression, unsafe-framing, or reflected-key acceptance. It returns only a
  `local_retrieval_only` raw capture. The injected transport remains a trusted test seam. No
  network call has been made. This does not prove Replay entitlement, provider identity,
  publication chronology, revision completeness, rights, data conformance, or model
  authorization; production and RL remain closed.
- Preserve one simple path: immutable raw snapshot pack -> cutoff-causal evidence bundle ->
  target-free model rows -> separately sealed resolutions.
- The provider-neutral snapshot, canonical evidence-bundle, and sealed T-60 row boundaries now
  exist locally. The offline preflight now binds snapshots, evidence, rows, Elo states, seasons,
  resolutions, rights, and exact artifact hashes end to end. A production connector still must
  derive every feature and final score from licensed raw bytes before readiness can become true.
- Separate per-entity source IDs from stable licensed rights scopes, and bind evidence to the full
  snapshot-metadata digest so live version and effective-time identity cannot be discarded.
- Treat `available_at` as knowledge time: retrieval for live captures and provider publication for
  attested archives. Permit future `effective_at` values without treating them as prior knowledge.
- Use T-60 as the primary supervised state and retain optional T-6h and T-15m states separately.
- Derive rest, back-to-back, recent schedule load, road-game load, rolling form, and schedule
  strength from licensed real history.
- Reset state by season and batch all same-date games before updating history.
- Express every feature as an oriented difference with an exact side swap.
- Freeze the 11 standard rest/load/travel/roster/lineup/team/player/schedule features; keep the two
  player-health-derived availability features local-only and outside standard Tinker exports.
- Compute Elo in-house from strictly prior outcomes under a frozen recipe or use a license-cleared
  pregame Elo source.
- The deterministic Elo verifier now replays a frozen recipe from sealed schedule rows and only
  applies results available by each cutoff. Preflight binds those rows to the exact training game,
  season, cutoff, tipoff, resolution, and supplied Elo state. The coverage seal structurally binds
  those rows to a claimed conformance report; an authenticated licensed connector and inventory
  must still prove schedule completeness and derive team/site identity correctly.
- A bounded provider-conformance validator now exact-compares every decoded revision envelope and
  raw-derived schedule fact with a claimed independently reviewed inventory. It verifies all
  declared T-6h/T-60/T-15 selections, required correction/deletion cases, inventory-relative
  coverage, replay and cohort agreement, and the cohort's exact snapshot-pack digest. It does not
  authenticate the reviewer, inventory, connector implementation, agreement, or timestamp; those
  external checks remain open.
- The evaluation verifier now joins separate target-free cohort, answer, forecast, and training-only
  calibration files and recomputes both Elo-relative gates. Preflight binds calibration rows to the
  exact training graph; it separately replays evaluation Elo and binds evaluation dates, baselines,
  and answers to sealed schedule rows, states, and scores. The frozen production policy requires at
  least 1,000 games and 20 calendar blocks in every season. Candidate-model provenance, external
  precommit/timestamp, and raw-provider derivation remain required.
- The production-schema tabular core and two fixed-path commands are implemented. Fitting consumes
  only sealed training rows and separate final-score resolutions, uses all 11 frozen features with
  training-only uncentered RMS scales, and writes a create-only model artifact. Answer-free
  prediction requires strictly later seasons, checks side-swap complement symmetry, and writes
  canonical forecasts plus a lock binding model, input rows, IDs, seasons, and output bytes.
- Fit that readable cross-entropy logistic correction to Elo on the licensed pack before paying for
  another fine-tune; no production model has been fitted yet.
- Train ForecastFM on the same realized winner and fixed candidate-token probability contract.
- Compare against raw Elo and an Elo recalibration fitted only on training data.
- Require positive Elo-relative log score and a positive one-sided 95% seven-day calendar-block
  bootstrap lower bound separately in every declared chronological evaluation season.
- Freeze the full ID/date/season/baseline cohort contract before scoring.
- Join candidate ID/probability pairs to that contract, penalize missing or selectively dropped
  forecasts, reject extra or duplicate IDs, and take scoring metadata only from the frozen cohort.
- Validate licensed evidence as oriented numeric records with source hashes, point-in-time capture,
  rights, and sensitivity lineage before it enters either model path.
- Align richer records to one exact rest, load, travel, roster, lineup, rolling-team,
  rolling-player, and schedule-strength feature schema.
- Refuse outcome-v2 SFT offline unless the full-data, upload-rights, artifact-hash,
  side-swap-pair, and cohort-coverage gates all pass; retain the final partial batch.
- Treat `full_outcome_v2_ready` as a summary, not authority: preflight also requires an empty
  missing-data list, wins over raw and training-only-recalibrated Elo, and at least two named
  untouched evaluation seasons.
- Keep the opened historical pass/fail booleans informational only; authorization comes from the
  freshly recomputed sealed candidate report, never those legacy manifest fields.
- Bind any readiness-true SFT run to the exact reviewed agreement bytes, rights lock, and sealed
  target-free row file; never authorize upload from manifest permission strings alone.
- The guarded outcome-v2 paid entrypoint consumes the exact passing preflight result, writes and
  re-verifies the immutable run lock, and only then imports the direct Tinker runtime. The runtime
  renders every selected batch before client creation and executes frozen binary cross-entropy on
  `logit(Elo) + logp(TEAM) - logp(OTHER)` without dropping a final pair-complete partial batch. A
  successful run writes and re-verifies a separate experiment seal binding the permanent state and
  sampler paths.
- Run post-SFT inference without answers: render and length-check everything before client creation,
  make four fixed-label logprob calls per game under one application attempt, durably journal starts
  and terminals, never recall an interrupted unit, and derive the generic forecast file directly
  from raw terminal scores. Disclose the pinned SDK's internal same-request retransmission window.
- After SFT, freeze a second answer-free cohort with IDs disjoint from the tabular gate and every
  season strictly later. Bind its feature rows, exact original/side-swapped prompts, pre-call
  generation lock, raw label-logprob records, derived forecasts, and explicit failures to the
  experiment sampler. Require the same raw- and recalibrated-Elo conjunction under a separately
  named post-SFT report.
- Treat a single lock over already assembled multi-season rows only as a retrospective answer-held
  holdout. The rolling path now requires an externally receipted multi-season plan before the
  earliest batch input, then seals each batch's terminal records inside one overlapping causal
  window and requires a second live receipt before the earliest T-60 cutoff. Season coverage seals
  now structurally bind exact replay rows to claimed provider-conformance reports before season
  inputs, and the multi-season aggregator rejects any duplicate, missing, or extra batch game while
  binding the schedule and answer-free forecasts. After outcomes, a separate scorer re-verifies the
  aggregate and seals the exact snapshot-pack and resolution bytes, cohort, answers, and forecasts.
  Elo update chronology uses the bound snapshot's `available_at`, not the resolution row's declared
  `resolved_at`; game dates use scheduled tipoff in `America/New_York`. The prospective plan and
  scoring seal bind the immutable run lock's evaluation-policy hash and original training-only
  calibration hash. The terminal gate rejects a weakened policy or substituted calibration,
  re-runs the generic multi-season gate, and checks its cohort, answer, forecast, and calibration
  hashes plus IDs and seasons against that seal. Its status is `structural_claim_only`, with both
  prospective-win and RL authorization `denied`.
  Schema v1 rejects cancellations or reschedules after the first season input. Production still
  needs opaque score payload parsing by a licensed connector, an authenticated licensed
  provider/reviewer, licensed raw bytes, real receipts, and remote execution attestation. No
  provider-backed production remote run has occurred, and future schedule changes need an
  amendment protocol. These local checks cannot authorize a prospective win claim or RL; the
  production rolling gate remains hard-closed.
- Keep production preflight hard-closed until reviewed connector and pre-event commitment verifiers
  replace the explicit fail-closed boundary. Self-authored hashes are not those external proofs.

The current repository has no licensed modern snapshot pack or production connector. Existing open
sources have date-only timestamps and lack the full travel, availability, expected-lineup, roster,
and rolling-player contract. Existing historical answers are contamination-prone. This milestone
does not claim that outcome v2 beats Elo; multiple untouched chronological seasons and a prospective
cohort are still required.

The checked-in historical recalibration is an opened diagnostic with its original frozen recipe.
The eventual untouched gate has a separately named and hashed intercept-plus-slope policy; the two
must not be presented as the same benchmark.

The first historical run failed the conjunction gate: pooled Elo-relative log score was positive,
but 2013 was inconclusive and 2015 was negative. The failure is preserved in
`data/processed/outcome_v2/manifest.json`; it must not be tuned away using those opened seasons.

An answer-free protocol for the rights-clean 2015–2022 FiveThirtyEight checking-our-work release
is frozen under `evaluation/outcome_v2_open_modern/` and was pushed before source access. One 2022
label was subsequently exposed during a schema check; the incident is preserved alongside the
protocol. Treat this lane as a protocol-frozen historical holdout, not an untouched test. It does
not substitute for licensed lineups, injuries, exact timestamps, or prospective evaluation.

## 6. Sequential evidence RL — gated

- Begin only after both the tabular prerequisite report and the distinct later post-SFT report
  clear their multi-season gates.
- Let the policy choose evidence sources, retrieval, updates, trust, and stopping decisions.
- Reward realized-outcome log score relative to Elo, minus predeclared tool cost and optional KL.
- Compare with static and budget-matched policies on the same exact chronological cohorts.
- Keep source rights, point-in-time availability, failure penalties, and prospective publication
  rules unchanged.

## 7. General forecasting domains

- Add versioned public time-series snapshots.
- Add original event questions with explicit resolution rules.
- Test binary, multiclass, and later continuous distributions.
- Maintain temporal and domain-held-out evaluation sets.

## 8. NBA domain pack

- Keep the provider-neutral connector contract; add a production connector only after its license
  and exact snapshot semantics pass review.
- Accept a buyer-owned licensed snapshot pack as the vendor-independent fallback.
- Keep restricted or buyer-licensed rows outside the redistributable core.
- Exclude player health and injury information from standard Tinker uploads.
- Compare numeric-only, ForecastFM-only, hybrid, and market-aware forecasts.
- Freeze model, prompt, decoding, and data hashes before prospective predictions.
- Commit an append-only forecast ledger before each game resolves.
- Require exact cohort coverage and paired comparisons during the 2026–27 season.

## 9. Interpretability

- Export the LoRA adapter and collect matched base/fine-tuned activations locally.
- Study features related to base-rate use, evidence updates, and overconfidence.
- Require causal ablation or steering results, not feature labels alone.
