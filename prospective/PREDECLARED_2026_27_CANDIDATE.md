# 2026-27 predeclared candidate freeze

Frozen 2026-07-20, before any 2026-27 data is collected. This document is the public
precommitment: the candidate below is evaluated on the 2026-27 regular season under the frozen
gate, and no weight, feature, recipe, or threshold may change after the first 2026-27 forecast
input exists. If the candidate fails, the failure is preserved, not tuned away.

## Candidate

- Model: no-intercept Elo-offset logistic regression (`elo_residual`, binary cross-entropy,
  L2 0.01, 2,000 full-batch steps, learning rate 0.05, training-only uncentered RMS scales).
- Features (12, home-minus-away differences): the standard 11 minus `rolling_team_net_rating`
  (dropped per the 2026-07-19 ablation diagnosis: basketball-wrong sign, redundant beside
  RAPM-based player value), plus `projected_rotation_value` (RAPM-weighted median expected
  minutes of the available rotation from the selected pre-T-60 injury snapshot; denominator
  keeps the full pool).
- Availability aggregates (local-only, reported but not required for the candidate):
  `unavailable_rotation_minutes` and `unavailable_rotation_value` priced by RAPM times median
  expected minutes over the last ten appearances (the repriced variant).
- Baseline: carryover margin-of-victory Elo (initial 1500, K 20, scale 400, home advantage 60,
  carryover 0.75, warmup from 2016-17).
- Comparator: training-only intercept-plus-slope logit recalibration fitted on the training
  seasons below (frozen policy: 2,000 steps, lr 0.05, init 0/1).
- Player ratings: per-season RAPM from the three strictly earlier seasons (frozen optimizer:
  ridge 0.01, lr 1.0, 120 epochs, batch 8,192, seed 20260718).
- Training seasons: 2021-22, 2022-23, 2023-24, 2024-25, 2025-26 (all data predating the
  evaluation season; using opened seasons for training is legitimate, evaluation must be
  pristine).
- Data contracts: injury snapshots strictly at or before T-60 with containment fallback;
  ESPN completed-only scoreboards as schedule ground truth; the 2026 NBA Cup final excluded
  by declaration when identified; pbp-derived games that fail structural validation are
  excluded with reasons disclosed, never repaired.

## Gate

- Evaluation season: 2026-27 regular season (label 2027), every joined game, exact cohort
  coverage with exclusions enumerated.
- Requirements: at least 1,000 games and at least 20 calendar blocks; positive mean
  baseline-relative log score AND positive one-sided 95% lower bound under the frozen 7-day
  calendar-block bootstrap (10,000 resamples, alpha 0.05, seed 20260716), separately against
  raw MOV Elo and the training-only recalibration. The season must pass both arms.
- A second consecutive season (2027-28) under the same conjunction is required for the
  production claim.
- Interim weekly diagnostics during the season are informational only; the gate evaluates at
  season end. The market benchmark arm (de-vigged ESPN moneyline) runs alongside as the
  declared ceiling reference, never as a feature.

## What is forbidden after this freeze

- Changing any candidate component because of 2026-27 interim or final results.
- Substituting a different model and presenting it as this candidate.
- Dropping or redefining games after seeing their outcomes.
- Reopening 2018-19 or any other season as a substitute evaluation to manufacture a pass.
