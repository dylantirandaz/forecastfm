# Open modern NBA evaluation lane

`protocol.json` freezes the split, features, baselines, scoring, and exclusions before the pinned
2015–2022 FiveThirtyEight answer file is downloaded. It was pushed before source access. One 2022
label was later exposed accidentally during a schema check, as recorded in `EXPOSURE.md`, so this
lane is a protocol-frozen historical holdout rather than a literally unopened test.

This lane is rights-clean and useful, but partial. It can extend the real diagnostic through the
2022 season and add lagged CC-BY RAPTOR features. It cannot supply exact tipoff/publication times,
historical lineup revisions, injuries, or a prospective claim. The full-data and RL gates remain
closed regardless of its result.

`source_seal.json` binds the exact source bytes, exposure record, chronological split, row counts,
and ordered IDs. The sealer writes labeled 2016–2020 development data plus label-free 2021–2022
inputs, then its CLI deletes the temporary all-answer source even when sealing fails. Downstream
loaders refuse any artifact whose committed hash, schema, row count, or ID order differs.

The validation experiment uses pregame source probabilities, game dates, team identities and
prior matchup schedule, and possession-weighted regular-season RAPTOR from the fully completed
prior season. Its feature code never reads an outcome field. One predeclared full residual
forecast with an L2 penalty of `0.01`, a fixed source-probability recalibration baseline, the
scoring rules, bootstrap, and side-swap gate must be committed and pushed before running. Both
models fit on 2016–2019; 2020 is used only for the advancement gate, never for candidate or
hyperparameter selection:

```bash
uv run python -m examples.run_open_modern_development
```

That command requires a clean Git tree and exclusively creates `validation_lock.json`; it cannot
overwrite or reopen a prior result. A failed validation gate leaves the historical holdout closed.
If the gate passes, subsequent 2021–2022 holdout inference uses the locked weights in one fixed
full-file pass without row-wise or other adaptive model updates.

## Frozen validation result

The first and only run of the published experiment at commit
`df07af48dcec52dd708606843b9729353d85cce1` produced:

| 2020 forecast | Log loss | Brier | 10-bin ECE |
| --- | ---: | ---: | ---: |
| Raw source probability | 0.636322 | 0.221621 | 0.061888 |
| Training-only recalibration | 0.628333 | 0.219106 | 0.046199 |
| Fixed full residual forecast | 0.624227 | 0.217125 | 0.030914 |

The full forecast's mean baseline-relative log score was `+0.012095` versus the raw source, with
a one-sided 95% calendar-block bootstrap lower bound of `+0.004730`. Versus the fixed
recalibration, its mean improvement was `+0.004106`, but the lower bound was `-0.002173`.
Therefore the gate failed only on statistical confidence versus recalibration. The immutable lock
is `validation_lock.json` (SHA-256
`1bb18b55305e561943ae294859ff7a8d633554282ed7309be588d2ff3fe1c7fd`), and no 2021–2022
predictions or scores were produced.
