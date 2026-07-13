# Validation canary v2 result

Status: completed. The no-thinking renderer repaired the output protocol, but the adapter is
not an unqualified forecasting improvement.

The adapter tracks the prompt-derived Elo teacher more closely. It does not improve realized
outcome scoring on this 64-game diagnostic, and it introduces a large side-swap inconsistency.
The base model remains the safer checkpoint for the next prospective experiment.

## Frozen evidence

- Passing format-smoke commit: `e8c5039f8b98b3d21339c25ee442da6fd27b5e57`
- V2 protocol commit: `969de85a9c31502b0a462a52b829b6f88d50279d`
- Answer-blind cohort commit: `ed8194a372569cd61c30fec130ed34a25a9252bb`
- Sealed-generation commit: `b1c982352bd53a05f7422646588abe47b1c7f3f9`
- Score commit: `1ecfbbd2dcd9932096fb95b73273f2d045ff26ca`
- Canary manifest SHA-256:
  `a4e25639d59c0dd30dd73d625f1ba3ce2865404a2f75e642215e1a3b567a6a70`
- Generation seal SHA-256:
  `ada830dd30110b9e2b4d67c7439a55089f1ab80f78a3402f1b07cc9377820421`
- Score artifact SHA-256:
  `edc88a58242e0e6778b0d1018b36f937c68b08b9178eab7ae2eb02edab9fa707`
- Ordered cohort-ID SHA-256:
  `acefb83bb24ea238195a3c9b8499a1d61e17bfbd31ac4a9c16ce3636ad6d04f1`
- Paired prompt-file SHA-256:
  `3815edd18b060b393aa08092845b1012d5051259b3786843b8ccb30933fd0f7d`

The protocol and disjoint prompt-only cohort were committed and published before generation.
All raw outputs were then sealed, committed, and published before the scorer opened historical
answers. The scorer recorded and verified the authoritative GitHub revision.

## Output gate

| Arm | Logical calls | Completed | Clean renderer stop | Provider stop | Strict JSON |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 128/128 | 128/128 | 128/128 | 128/128 | 128/128 |
| Adapter | 128/128 | 128/128 | 128/128 | 128/128 | 128/128 |

This fixes canary v1's 0/128 format failure. The run made 256 logical SDK calls with one sample
per prompt and no project-level resampling. Tinker 0.22.7 may retransmit the same logical request
with the same session and sequence ID, so the number of underlying transport requests is not
independently observable.

## Main metrics

Lower is better for every error metric.

| Metric | Base | Adapter | Adapter - base | Relative change |
| --- | ---: | ---: | ---: | ---: |
| Prompt-derived Elo teacher MAE | 0.114282 | 0.090662 | -0.023621 | -20.67% |
| Historical Brier `(p-y)^2` | 0.189045 | 0.203241 | +0.014196 | +7.51% |
| Historical log loss | 0.558831 | 0.584936 | +0.026105 | +4.67% |
| Side-swap MAE | 0.028615 | 0.161690 | +0.133075 | +465.05% |
| Strict original validity | 100% | 100% | 0 pp | - |
| Strict side-swap validity | 100% | 100% | 0 pp | - |

The adapter-minus-base paired bootstrap intervals are exploratory, not preregistered. They use
200,000 game-level resamples of the 64 paired games, NumPy PCG64 seed `20260713`, and ordinary
2.5th/97.5th percentiles.

| Metric delta | Point estimate | Paired bootstrap 95% interval | Reading |
| --- | ---: | ---: | --- |
| Teacher MAE | -0.023621 | [-0.040355, -0.005938] | Improvement |
| Brier | +0.014196 | [-0.011495, +0.040241] | Inconclusive |
| Log loss | +0.026105 | [-0.031959, +0.085516] | Inconclusive |
| Side-swap MAE | +0.133075 | [+0.102719, +0.163385] | Regression |

Per-game adapter win/tie/loss counts were 44/1/19 on teacher MAE, 38/1/25 on Brier and log
loss, and 5/0/59 on side-swap error.

## Difficulty subsets

These are post-hoc descriptive subsets based only on the prompt-derived Elo teacher confidence
`max(q, 1-q)`: hard `[0.50, 0.60)`, medium `[0.60, 0.75)`, and easy `[0.75, 1.00]`.

| Subset | N | Teacher MAE base -> adapter | Brier base -> adapter | Log loss base -> adapter | Swap MAE base -> adapter |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hard | 18 | 0.125391 -> 0.147722 | 0.234846 -> 0.309899 | 0.662501 -> 0.827747 | 0.057431 -> 0.254726 |
| Medium | 25 | 0.116577 -> 0.086757 | 0.182718 -> 0.165806 | 0.546207 -> 0.507834 | 0.031905 -> 0.170071 |
| Easy | 21 | 0.102030 -> 0.046401 | 0.157319 -> 0.156384 | 0.484999 -> 0.468600 | 0.000000 -> 0.071969 |

The teacher-MAE gain comes from medium and easy cases. Hard cases become worse on all four
metrics. Side-swap consistency degrades in every subset.

## Largest realized-score failures

These are the five largest positive per-game adapter-minus-base Brier deltas. Probabilities and
the teacher target are for `team_wins`; every listed realized result was `opponent_wins`.

| Question ID | Base p -> adapter p | Teacher q | Brier delta | Swap error base -> adapter |
| --- | ---: | ---: | ---: | ---: |
| `nba-086346aa1fa7685c` | 0.358521 -> 0.587654 | 0.498465 | +0.216800 | 0.000000 -> 0.350988 |
| `nba-04affad47fabcab1` | 0.346422 -> 0.571429 | 0.485215 | +0.206523 | 0.000000 -> 0.304762 |
| `nba-051bbc42cc7c873b` | 0.354655 -> 0.571235 | 0.494252 | +0.200529 | 0.000000 -> 0.334568 |
| `nba-05c68b7fab776abf` | 0.619211 -> 0.763333 | 0.743043 | +0.199255 | 0.000000 -> 0.096666 |
| `nba-051cf7e45f8b5649` | 0.631669 -> 0.773333 | 0.753065 | +0.199039 | 0.000000 -> 0.068770 |

All five were upsets. In each case the adapter moved closer to the Elo teacher and farther from
the realized binary outcome. That is why teacher imitation can improve while Brier and log loss
do not.

## Scientific limits

This is a contamination-prone historical diagnostic, not pristine prospective evidence. The
v2 IDs are disjoint from v1, selection and generation were answer-blind, and neither model saw
the labels. However, the earlier v1 scoring process parsed the complete validation-answer file,
so v2 cannot honestly be called a never-opened holdout at the repository/process level. The NBA
test split remains untouched.

The sample is also small (`n=64` games), the difficulty and bootstrap analyses are post hoc, and
the raw Tinker responses are not provider-signed.

## Decision and next experiment

Do not promote this adapter as the forecasting foundation checkpoint. It is better at imitating
the Elo teacher but less invariant to equivalent framing, and it has no demonstrated improvement
on realized-outcome proper scores.

The next training run should:

1. Add original/side-swapped training pairs with complemented targets.
2. Evaluate symmetry during training instead of only after it.
3. Separate teacher-imitation metrics from realized-outcome Brier and log loss.
4. Predefine scaling checkpoints and easy/medium/hard thresholds before training.
5. Use a physically separate, never-opened prospective NBA season holdout. Keep the current test
   split sealed until that protocol is committed.
