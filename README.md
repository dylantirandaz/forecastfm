# ForecastFM

ForecastFM is a small, typed foundation for building a domain-general probabilistic
forecaster. The core task is deliberately narrow:

```text
prior distribution + timestamped evidence -> posterior distribution
```

The repository currently provides:

- immutable forecast, evidence, and probability types;
- point-in-time leakage validation;
- Bayesian updates in log-likelihood space;
- proper scoring and calibration summaries;
- a pinned, real NBA Elo forecast dataset with chronological splits;
- strict JSONL serialization; and
- a vendor-neutral chat-data export boundary with conservative health-term screening;
- immutable training and experiment locks; and
- a hash-chained prospective forecast ledger with exact cohort coverage.

The runtime core has no third-party dependencies. Tinker-specific API calls will remain in a
thin integration module so the forecasting logic stays testable without a key or network.

## Setup

Python 3.12 is required.

```bash
uv sync --extra tinker
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

## Real NBA dataset

Download and transform the pinned real-data source:

```bash
uv run --extra tinker python examples/build_real_nba_dataset.py
```

This verifies the source SHA-256, selects one hash-balanced perspective per game, removes exact
prompt overlap across splits, and excludes game identity and postgame fields from every model
message. It writes separate target-free prompt files and answer files under `data/processed/`.
Raw records and generated JSONL files are ignored by Git; the trackable manifest records the
source, license, transformations, exclusions, split boundaries, row counts, and output hashes.

The source is FiveThirtyEight's historical NBA Elo dataset, used under
[CC BY 4.0](https://github.com/fivethirtyeight/data/blob/master/LICENSE). We filter and reformat
the data, and retain FiveThirtyEight's credit to Basketball-Reference.com for game information.
It covers NBA games through the 2015 season and contains retrospective pregame Elo backcasts,
so it is a real historical baseline—not a live 2026 forecast feed.

### Anti-cheating limits

The model sees only an anonymous question, a neutral-court Elo prior, and venue. It does not see
the teams, game ID, date, source URL, teacher probability during evaluation, or realized winner.
The build also quarantines four same-day games whose midnight cutoff is ambiguous, balances
home and away perspectives, requires unique model-facing prompts across splits, and refuses
one-hot training targets. The paid runner rejects stale prompt schemas and files whose hashes do
not match the current manifest.

This remains a formula-distillation benchmark: FiveThirtyEight's probability is almost exactly
a fixed 100-Elo venue adjustment of the supplied neutral prior. Historical Brier score and log
loss are therefore diagnostics, not evidence that the model learned general forecasting. A
credible forecasting claim requires forecasts committed before future games resolve.

The prospective protocol under `prospective/` freezes the code revision, prompt text and schema,
dataset files, tokenizer snapshot, training settings, decoding settings, and final Tinker sampler
path. Forecast and resolution batches are append-only and hash-chained. This is locally
tamper-evident, but a head hash must be pushed to a protected remote or external timestamp service
before tipoff to prove publication time. A local Git timestamp alone is not sufficient.

## First Tinker training step

Set the key in the same terminal that will start training, then verify it without printing it:

```bash
export TINKER_API_KEY="your-actual-key"
python -c 'import os; assert os.getenv("TINKER_API_KEY"); print("Tinker key is set")'
```

Build the real data and start the one-step smoke test:

```bash
uv run --extra tinker python examples/build_real_nba_dataset.py
uv run --extra tinker python examples/freeze_training_lock.py
uv run --extra tinker python examples/train_tinker_sft.py
uv run --extra tinker python examples/freeze_experiment.py
```

The training command makes a billable remote API call. Its readable constants live in
`run_config.py`; the safe default is one batch on `Qwen/Qwen3.5-4B`. The runner refuses code,
prompt, data, tokenizer, or settings that differ from the committed training lock. Tinker logs
and checkpoint metadata are written under the ignored `artifacts/` directory. The final command
creates a forecast-ready experiment lock from Tinker's permanent sampler path. Never place the API
key in a source file.

## Design rules

- All timestamps are timezone-aware.
- Evidence available after the forecast cutoff is rejected.
- Outcome labels and probability order must match exactly.
- Probability vectors must be finite, non-negative, and sum to one.
- Real data is split chronologically by season, never randomly by row.
- Exact model-facing prompts cannot cross split boundaries.
- Realized outcomes are evaluation labels, never one-hot SFT targets.
- Evaluation files separate model prompts from answer keys.
- Scoring a frozen cohort rejects missing, extra, duplicate, or relabeled forecasts.
- Prospective batches retain exact prompts, raw responses, and one request identity per game.
- Ledger validation rejects modified hashes, reordered records, late forecasts, and partial slates.
- A ledger head counts as time evidence only after independent publication before the deadline.
- Model targets contain sourced probabilities only; unsupported rationales are not fabricated.
- Training exports screen for known health-data language before anything reaches Tinker.
  This keyword screen is only a first pass; it does not establish policy or legal compliance.
- Player health and injury fields are not present in the accepted NBA input columns.

## Package map

- `models.py`: validated domain objects.
- `updating.py`: Bayesian probability updates.
- `scoring.py`: Brier score, log loss, and aggregate evaluation.
- `calibration.py`: reliability bins and expected calibration error.
- `nba_data.py`: pinned NBA download, leakage-safe transformation, and temporal splits.
- `serialization.py`: strict, readable JSONL input and output.
- `prompting.py`: the model prompt and strict prediction parser.
- `tinker_data.py`: screened SFT conversation export without SDK coupling.
- `run_config.py`: explicit model, tokenizer, training, and decoding settings.
- `run_lock.py`: immutable training and trained-sampler experiment locks.
- `ledger.py`: prospective cohort validation and append-only hash-chain verification.

## Next milestones

1. Measure base-model JSON validity and fidelity to the analytic Elo oracle.
2. Run the first small Tinker SFT and compare it on the identical anonymous cohort.
3. Publish the frozen experiment and first complete future-game cohort to a protected remote.
4. Commit forecasts before tipoff and score them only after appending sourced resolutions.
5. Add richer licensed or buyer-owned NBA inputs only after the baseline is sound.

See [ROADMAP.md](ROADMAP.md) for the acceptance criteria for each milestone.
