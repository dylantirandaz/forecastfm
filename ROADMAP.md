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
- Prove the pipeline with one `Qwen/Qwen3.5-4B` LoRA step, then scale to 9B.
- Save the adapter, training metrics, configuration, and data hashes.
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

## 4. Proper-score optimization

- Give the model tools for inspecting point-in-time numerical evidence and base rates.
- Reward valid distributions with proper scoring rules.
- Penalize future evidence, malformed output, and incoherent probabilities.
- Keep the SFT model as a fixed baseline.

## 5. General forecasting domains

- Add versioned public time-series snapshots.
- Add original event questions with explicit resolution rules.
- Test binary, multiclass, and later continuous distributions.
- Maintain temporal and domain-held-out evaluation sets.

## 6. NBA domain pack

- Define a buyer-owned-data connector interface.
- Keep restricted or buyer-licensed rows outside the redistributable core.
- Exclude player health and injury information from standard Tinker uploads.
- Compare numeric-only, ForecastFM-only, hybrid, and market-aware forecasts.
- Freeze model, prompt, decoding, and data hashes before prospective predictions.
- Commit an append-only forecast ledger before each game resolves.
- Require exact cohort coverage and paired comparisons during the 2026–27 season.

## 7. Interpretability

- Export the LoRA adapter and collect matched base/fine-tuned activations locally.
- Study features related to base-rate use, evidence updates, and overconfidence.
- Require causal ablation or steering results, not feature labels alone.
