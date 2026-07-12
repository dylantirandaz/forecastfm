# Prospective integrity protocol

This directory holds the files that turn a forecast into an auditable pre-event commitment.
All files here are tracked by Git. Forecast records must never be amended, rebased, squashed, or
force-pushed after publication.

The protocol has four layers:

1. `training_lock.json` freezes the committed code revision, exact prompt, dataset hashes,
   tokenizer revision, Tinker versions, training recipe, and decoding policy.
2. `experiment.json` is created only after training and binds the training lock to Tinker's final
   permanent `sampler_path`.
3. A cohort file freezes every game in a declared slate, its schedule snapshot, and each forecast
   deadline.
4. `ledger.jsonl` hash-chains one complete forecast batch and, later, one complete resolution batch
   for each cohort. Every raw model response is retained.

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

This repository currently has no remote. Until a ledger head is externally published, its status
is **locally tamper-evident, not externally timestamped**.

## Forecast rules

- Declare every game in the cohort before generating predictions.
- Use one model call and one retained raw response per game; never select among retries.
- Commit the complete cohort before its earliest deadline and before every scheduled tipoff.
- Append resolutions later; never edit forecasts to add outcomes.
- Verify against the externally published head when scoring.
- Report malformed output and missing coverage as failures rather than silently dropping rows.
