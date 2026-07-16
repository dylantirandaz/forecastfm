# Open modern NBA evaluation lane

`protocol.json` freezes the split, features, baselines, scoring, and exclusions before the pinned
2015–2022 FiveThirtyEight answer file is downloaded. It was pushed before source access. One 2022
label was later exposed accidentally during a schema check, as recorded in `EXPOSURE.md`, so this
lane is a protocol-frozen historical holdout rather than a literally unopened test.

This lane is rights-clean and useful, but partial. It can extend the real diagnostic through the
2022 season and add lagged CC-BY RAPTOR features. It cannot supply exact tipoff/publication times,
historical lineup revisions, injuries, or a prospective claim. The full-data and RL gates remain
closed regardless of its result.
