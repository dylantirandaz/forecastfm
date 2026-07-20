# 2026-27 weekly evaluation protocol

Operating protocol for the rolling weekly runner `examples/run_prospective_weekly.py`, which
services the frozen candidate declared in
[`PREDECLARED_2026_27_CANDIDATE.md`](PREDECLARED_2026_27_CANDIDATE.md). The 2026-27 regular
season opens 2026-10-21 (evaluation label 2027).

## Cadence

Run weekly once the season is underway — Monday 06:15 UTC is a good default (Sunday slates
are complete and ESPN data has settled):

```cron
15 6 * * 1 cd ~/forecastfm && python examples/run_prospective_weekly.py refresh && python examples/run_prospective_weekly.py track --as-of "$(date -u +\%F)" >> data/processed/prospective_2026_27/weekly.log 2>&1
```

(`%` must be escaped in crontab entries.) `refresh` and `track` are separate invocations so a
fetch failure never masquerades as an evaluation failure. Before the season has ~50 games,
`track` exits 2 with an informational message; that is expected, not an error worth paging
on — either ignore exit 2 in the cron line or start the cron job in mid-November.

## Commands

- `refresh [--as-of YYYY-MM-DD] [--dry-run]` — re-fetches the ESPN season window
  2026-10-21 through the end date (default: today UTC) via `examples/fetch_espn_season.py`,
  which is resumable and skips cached scoreboards/summaries. It rewrites
  `data/raw/espn/espn_2025.csv` in place (the fetcher's fixed filename, which the pipeline
  reads for the 2026-27 label) and then copies it to a dated backup
  `data/raw/espn/espn_2025_backup_<utcdate>.csv` so the rolling 2026-27 file and the archived
  2025-26 use of the same filename never alias.
- `evaluate --as-of YYYY-MM-DD [--dry-run]` — runs the frozen prototype pipeline with the
  predeclared configuration: `--exclude-families team_form`, training seasons
  2022,2023,2024,2025,2026, evaluation season 2027, output
  `data/processed/prospective_2026_27/`. The runner injects the missing
  `SEASON_FILES[2027]` entry into `examples.run_private_prototype` at runtime (a new dict;
  the original is never mutated and the driver file is never edited — this is the sanctioned
  seam, documented in the runner's module docstring). Exits 2 with an informational message
  when the season CSV holds fewer than 50 games.
- `track --as-of YYYY-MM-DD [--dry-run]` — runs `evaluate`, then appends one canonical JSON
  line to `data/processed/prospective_2026_27/tracker.jsonl` with `as_of`, `games`, and, per
  variant (standard / health / projected) per season, the mean log loss plus the mean
  baseline-relative log score and one-sided 95% lower bound against both baselines (raw MOV
  Elo and the training-only recalibration). Re-tracking the same date rewrites that date's
  line instead of duplicating it, so reruns after a data correction are safe.

`--dry-run` on any command prints the plan as JSON with no network access and no writes;
use it to sanity-check the cron line.

## Tracker rows are informational — the gate evaluates at season end

Weekly tracker rows are **diagnostics only**. Per the predeclared freeze, the formal gate
evaluates once, at the end of the 2026-27 regular season, on the full exact cohort (at least
1,000 games and 20 calendar blocks), requiring a positive mean baseline-relative log score
and a positive one-sided 95% lower bound under the frozen 7-day calendar-block bootstrap
against **both** arms. Mid-season tracker values — including multi-week stretches where a
variant trails a baseline — have no decision authority and must not trigger any action other
than investigating data-collection bugs.

## No-tuning reminder

Nothing about the candidate may change in response to 2026-27 interim or final results: no
weights, features, recipes, thresholds, training seasons, or evaluation windows. A failing
interim (or final) result is preserved and reported, never tuned away. If the tracker reveals
a pipeline defect (bad join, stale CSV, missing snapshots), fix the defect and re-track the
affected dates — the idempotent tracker makes that safe — but the model and gate stay frozen.
