# NBA data-rights gate

No modern NBA feed is currently cleared for Tinker upload or commercial redistribution. Keep
every permission as `unknown` until a signed agreement says otherwise; the code fails closed.

## Recommended source path

1. Ask [SportsDataIO](https://sportsdata.io/contact-us) for a narrow Vault + Leagues agreement.
   Its published NBA coverage is the closest match for historical statistics, rosters,
   transactions, projected lineups, and injuries.
2. Ask [Sportradar](https://sportradar.com/contact/) for a custom official-data license if stronger
   provenance is worth the cost.
3. Do not treat a BALLDONTLIE subscription, an NBA.com download, or a public API key as permission
   to train through Tinker or sell a derived data pack.

Published terms are not enough for the intended use. The signed order form must expressly cover:

- retention of historical and live records and their revisions;
- feature engineering, supervised fine-tuning, RL, calibration, and evaluation;
- Thinking Machines/Tinker as a named third-party processor;
- commercial use and ownership of adapters, weights, forecasts, and aggregate evaluations;
- publication of metrics, scaling plots, audits, and failure analyses without raw licensed rows;
- survival of model and checkpoint rights after the subscription ends;
- injury and availability data, if processed locally; and
- point-in-time revision history, correction timestamps, backfill policy, and upstream-rights
  warranties.

## Point-in-time rule

A live snapshot is eligible for a prospective claim only when it was retrieved before the frozen
forecast deadline. A provider archive may support retrospective model development, but it is not
prospective proof. Before using an archive for a leakage-sensitive backtest, retain the exact
provider version identifier, publication timestamp, raw payload hash, and the vendor's written
description of revision and backfill behavior.

`nba_evidence.py` keeps raw rows outside model-facing files and binds each numeric feature to its
source hash, timestamps, rights decision, and sensitivity. Standard Tinker conversion requires
explicit third-party and Tinker permissions, rejects player-health lineage even after numeric
aggregation, and applies the lexical health screen as a second defense.
