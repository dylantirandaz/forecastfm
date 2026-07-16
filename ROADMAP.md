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

Local completion provides tamper evidence, not trusted time. The first real prospective run also
requires a protected remote or external timestamp receipt published before the earliest deadline.

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

- Derive rest, back-to-back, recent schedule load, road-game load, rolling form, and schedule
  strength from the pinned real history.
- Reset state by season and batch all same-date games before updating history.
- Express every feature as an oriented difference with an exact side swap.
- Fit a readable cross-entropy logistic correction to Elo before paying for another fine-tune.
- Train ForecastFM on the same realized winner and fixed candidate-token probability contract.
- Compare against raw Elo and an Elo recalibration fitted only on training data.
- Require positive Elo-relative log score and a positive one-sided 95% seven-day calendar-block
  bootstrap lower bound separately in every declared chronological evaluation season.
- Reject missing, extra, duplicate, relabeled, or selectively dropped forecasts.

The current source has date-only timestamps and no true travel, injury, expected-lineup, roster,
or player-level data. Existing historical answers are contamination-prone. This milestone does not
claim that outcome v2 beats Elo; a prospective cohort is still required for a truly untouched
result.

The first historical run failed the conjunction gate: pooled Elo-relative log score was positive,
but 2013 was inconclusive and 2015 was negative. The failure is preserved in
`data/processed/outcome_v2/manifest.json`; it must not be tuned away using those opened seasons.

## 6. Sequential evidence RL — gated

- Begin only after both the tabular and supervised ForecastFM paths clear the multi-season gate.
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

- Define a buyer-owned-data connector interface.
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
