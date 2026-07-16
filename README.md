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
- leakage-safe historical NBA rolling features and a simple Elo-correction baseline;
- a rights-aware connector contract for licensed, point-in-time NBA evidence;
- a realized-winner outcome objective with side-swap augmentation;
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
home and away perspectives, and requires unique model-facing prompts across splits. The legacy
Elo-distillation adapter uses sourced probability targets. Outcome v1 instead places the realized
winner in a separate label field that is never included in the system or user message. Paid
runners reject stale schemas and files whose hashes do not match their manifests and locks.

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

Alternatively, put it in the ignored local `.env` file:

```text
TINKER_API_KEY="your-actual-key"
```

Build the real data and start the one-step smoke test:

```bash
uv run --extra tinker python examples/build_real_nba_dataset.py
uv run --extra tinker python examples/freeze_training_lock.py
uv run --extra tinker python -m examples.run_tinker_sft_local
uv run --extra tinker python examples/freeze_experiment.py
```

The training command makes a billable remote API call. Its readable constants live in
`run_config.py`; the safe default is one batch on `Qwen/Qwen3.5-4B`. The runner refuses code,
prompt, data, tokenizer, or settings that differ from the committed training lock. Tinker logs
and checkpoint metadata are written under the ignored `artifacts/` directory. The final command
creates a forecast-ready experiment lock from Tinker's permanent sampler path. Never place the API
key in a source file.

## Realized-winner outcome v1

Outcome v1 leaves the completed Elo adapter and its frozen artifacts unchanged. It carves a new
development period from the end of the pre-2010 training split, adds one side-swapped copy of every
fit and development game, and trains on what actually happened:

```text
team_wins     -> TEAM
opponent_wins -> OTHER
```

`OTHER` is intentional: the pinned Qwen tokenizer represents both labels as exactly one token,
while literal `OPPONENT` is two tokens. The runner verifies the token count and exact round trip
before any paid call. Cross-entropy has exactly one weighted position: the realized-winner token.
The old FiveThirtyEight forecast remains baseline metadata and never chooses the training label.

Build the ignored data plus its trackable manifest:

```bash
uv run python examples/build_outcome_dataset.py
```

After reviewing and committing the code and manifest, freeze and publish the new lock before
starting the billable 32-step canary:

```bash
uv run --extra tinker python examples/freeze_outcome_training_lock.py
git add prospective/outcome_v1/steps_32/training_lock.json
git commit -m "Freeze outcome v1 training"
git push origin main
uv run --extra tinker python -m examples.run_tinker_outcome_sft_local
```

After training completes, bind and publish the permanent sampler path before inference:

```bash
uv run --extra tinker python examples/freeze_outcome_experiment.py
git add prospective/outcome_v1/steps_32/experiment.json
git commit -m "Freeze outcome v1 32-step sampler"
git push origin main
```

Inference does not sample a decimal or JSON response. It scores the `TEAM` and `OTHER` tokens with
Tinker's prompt-log-probability API, renormalizes those two scores, and averages each forecast with
the complemented side-swapped forecast. The unnormalized valid-label mass is retained as a
diagnostic so renormalization cannot hide probability assigned to unrelated tokens.

### Frozen full development comparison

The outcome-v1 comparison uses all 2,612 original development games and their 2,612 deterministic
side swaps. Base and adapter each make four candidate-token calls per game, for 20,896 expected
logical calls in total. Only one model-game arm is active at a time; its four candidate calls all
finish before the next arm starts. Tinker's transport may still retransmit the same logical request
with the same session and sequence ID.

The lifecycle is intentionally staged and immutable:

```bash
# 1. Publish the tested protocol code.
git add README.md examples src tests
git commit -m "Add frozen outcome development evaluation"
git push origin main

# 2. Billable safety gate: four non-development calls per model, eight total.
uv run --extra tinker python -m examples.smoke_tinker_outcome_candidates

# 3. Freeze and publish prompts, manifest, scoring policy, and one attempt commitment.
uv run --extra tinker python examples/build_outcome_development_evaluation.py
git add evaluation/outcome_v1/steps_32
git commit -m "Freeze outcome development evaluation"
git push origin main

# 4. Billable: run or safely resume base-versus-adapter candidate scoring.
uv run --extra tinker python -m examples.run_tinker_outcome_development
git add evaluation/outcome_v1/steps_32/raw
git commit -m "Seal outcome development raw results"
git push origin main

# 5. Only after raw outputs are sealed and published, open answers and score.
uv run --extra tinker python -m examples.score_outcome_development
git add evaluation/outcome_v1/steps_32/scores.json
git commit -m "Score outcome development evaluation"
git push origin main
```

The runner never constructs or reads the answer path. Every arm is journaled before its provider
calls, application retries are disabled, interrupted arms become terminal failures, and failed
rows receive the precommitted worst-case realized-outcome probability rather than being removed.
After journaling any terminal arm failure, the runner stops for inspection. A later invocation
skips that failed arm and continues with the next unattempted arm; it never retries the failure.
Raw prompt tokens, label log-probabilities, valid-label mass, side-swap diagnostics, failures, and
the durable journal are sealed together. Scoring first proves those exact files are on
`origin/main`; only then does it hash and open the answer file.

An advisory process lock prevents two local runners from overlapping. The adapter has a permanent
Tinker sampler path, but Tinker does not expose a digest for the catalog base weights; a resume that
creates a new base sampling session therefore cannot cryptographically prove the upstream base
snapshot stayed identical. Tinker also supplies no signed provider-call receipt, so a local attempt
can be suppressed before its raw journal is published. These residual limitations are retained
with the evaluation report.

This blocks accidental and code-path leakage, but it cannot prove that a person or a separate
program never inspected plaintext answers already present on the local machine. The results are
historical development diagnostics and remain contamination-prone, not prospective evidence.

## Outcome v2: leakage-safe Elo correction

Outcome v2 uses the same pinned, real FiveThirtyEight history. It derives anonymous pregame
features from games on earlier dates only: venue-adjusted Elo log-odds, rest and back-to-back
status, games and road games in the previous seven days, and trailing-ten win rate, margin,
opponent Elo, and history length. Each value is an oriented team-minus-opponent difference.
History resets at the start of each season, and every game on a date is featurized before any
result from that date updates team history. The current game's result is only the training target.

The first model is deliberately small: dependency-free logistic regression adds a learned linear
correction to Elo's log-odds and minimizes ordinary winner cross-entropy:

```text
p(team wins) = sigmoid(logit(p_Elo) + dot(weights, pregame_features))
```

There is no free intercept, which preserves exact team/opponent side-swap symmetry. This tabular
model is a falsification baseline before another paid ForecastFM fine-tune, not a forecasting
claim by itself.

Advancement is a per-season conjunction. On every declared chronological evaluation season, the
model must have positive Elo-relative log score and a positive one-sided 95% lower bound from the
predeclared seven-day calendar-block bootstrap. A strong pooled result cannot conceal a losing
season.

Raw Elo and an Elo recalibration fitted on training data are both comparison baselines. Exact
cohort coverage is required; failures cannot be silently dropped.

Candidate forecasts carry only an opaque question ID and an interior probability; the scorer joins
season, date, outcome, and baseline from the frozen cohort. The pinned historical split is bound by
a full-cohort digest. A missing forecast, or malformed output represented as an explicit failure,
receives the predeclared worst-case realized probability of `1e-15` rather than being omitted or
silently clipped.

The source ends in 2015 and has date-only timestamps. It provides no true tipoff or publication
times, travel distance, injuries, expected lineups, rosters, or player-level metrics. Road-game
load is only a travel proxy. Existing historical answers are also contamination-prone, so this
evaluation cannot establish a prospective or truly untouched win over Elo. No such win is
currently claimed.

The first frozen historical diagnostic confirms why the gate is strict. The richer model's pooled
Elo-relative log score is positive (`+0.000884`), but only 2014 passes independently. The 2013
confidence bound crosses zero, and 2015 has negative mean improvement. The checked-in
`outcome_v2` manifest therefore marks both raw- and recalibrated-Elo gates false and RL not ready.

### When RL becomes useful

RL is gated on the tabular and supervised ForecastFM corrections first clearing the multi-season
Elo gate. Its intended job is sequential decision-making: choose which permitted evidence source
to inspect, whether to pay to retrieve it, how much to trust it, when to update, and when to stop.
The reward remains a proper realized-outcome log score relative to Elo, minus predeclared tool
costs and an optional KL penalty. RL does not replace the fixed chronological evaluation or make
missing point-in-time data safe to use.

## Legacy paired validation canary

The next gate is a frozen 64-game validation canary. Selection uses only the lexicographically
first 64 opaque validation IDs; it never opens answers. Each game has one original and one
deterministic side-swapped prompt, so the base and adapter each receive 128 prompts under identical
decoding settings.

The workflow is deliberately two-phase:

```bash
# Freeze and publish the answer-free call plan.
uv run python examples/build_validation_canary.py
git add evaluation/validation_canary
git commit -m "Freeze validation canary"
git push origin main

# Billable: generate exactly once and seal both raw model outputs.
uv run --extra tinker python -m examples.run_tinker_canary
git add evaluation/validation_canary/raw
git commit -m "Seal validation canary generations"
git push origin main

# Only now may historical answers be opened for secondary diagnostics.
uv run python examples/score_validation_canary.py
```

Primary metrics—strict JSON validity, prompt-derived Elo-oracle error, and side-swap consistency—
are answer-free. Missing or malformed rows remain in the denominator with fixed worst-case
penalties. Brier score, log loss, and teacher-target error are loaded only after raw outputs are
hash-sealed and published. Side swaps are never counted as extra historical games. The test split
is not used by this workflow.

## Design rules

- All timestamps are timezone-aware.
- Evidence available after the forecast cutoff is rejected.
- Outcome labels and probability order must match exactly.
- Probability vectors must be finite, non-negative, and sum to one.
- Real data is split chronologically by season, never randomly by row.
- Exact model-facing prompts cannot cross split boundaries.
- Realized outcomes are never model inputs. They are evaluation labels and, only for outcome v1,
  fixed-token cross-entropy targets.
- Evaluation files separate model prompts from answer keys.
- Scoring a frozen cohort penalizes missing or selectively dropped forecasts, rejects extra or
  duplicate IDs, and takes dates, outcomes, seasons, and baselines only from frozen metadata.
- Prospective batches retain exact prompts, raw responses, and one request identity per game.
- Ledger validation rejects modified hashes, reordered records, late forecasts, and partial slates.
- A ledger head counts as time evidence only after independent publication before the deadline.
- Legacy Elo targets contain sourced probabilities. Outcome-v1 labels contain sourced game
  results. Unsupported rationales are not fabricated.
- Training and inference exports screen for known health-data language before anything reaches
  Tinker.
  This keyword screen is only a first pass; it does not establish policy or legal compliance.
- Standard Tinker evidence conversion also rejects health-derived source lineage, even when its
  model-facing feature is an opaque numeric aggregate.

## Package map

- `models.py`: validated domain objects.
- `updating.py`: Bayesian probability updates.
- `scoring.py`: Brier score, log loss, and aggregate evaluation.
- `calibration.py`: reliability bins and expected calibration error.
- `nba_data.py`: pinned NBA download, leakage-safe transformation, and temporal splits.
- `nba_v2.py`: prior-date rolling NBA features with exact side-swap symmetry.
- `nba_evidence.py`: licensed-source rights, timing, lineage, and numeric evidence bundles.
- `nba_rich.py`: the predeclared richer NBA feature schema and exact side swaps.
- `elo_residual.py`: dependency-free cross-entropy correction to Elo log-odds.
- `outcome_v2_metrics.py`: strict per-season Elo-relative scores and block-bootstrap gate.
- `outcome_v2_preflight.py`: offline full-data, rights, hash, pair, and batch-coverage gate.
- `serialization.py`: strict, readable JSONL input and output.
- `prompting.py`: the model prompt and strict prediction parser.
- `tinker_data.py`: screened SFT conversation export without SDK coupling.
- `run_config.py`: explicit model, tokenizer, training, and decoding settings.
- `outcome.py`: fixed labels, outcome prompts, stable normalization, and symmetry averaging.
- `outcome_config.py`: readable outcome-v1 canary and scaling settings.
- `outcome_evaluation.py`: immutable full-cohort manifests, raw records, and seals.
- `outcome_metrics.py`: proper outcome scores, calibration, paired intervals, and difficulty bins.
- `publication.py`: exact local-versus-published Git gates for frozen evaluations.
- `run_lock.py`: immutable training and trained-sampler experiment locks.
- `ledger.py`: prospective cohort validation and an evidence-bound append-only hash chain.
- `canary.py`: frozen validation call plans, generation records, seals, and answer-free metrics.
- `canary_history.py`: answer-gated historical diagnostics for already sealed generations.

## Next milestones

1. Add licensed, point-in-time richer inputs to address the failed historical Elo gate.
2. Re-run a tabular falsification baseline on newly frozen chronological seasons.
3. Fine-tune ForecastFM on the same realized-winner objective only after that baseline is sound.
4. Require every declared season to clear the frozen Elo-relative log-score gate.
5. Attempt sequential evidence RL only after the supervised full-information model passes.

See [ROADMAP.md](ROADMAP.md) for the acceptance criteria for each milestone.
