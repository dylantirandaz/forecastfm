# Mantic-style RL run protocol (NBA outcome forecasting)

Status: predeclared 2026-07-20, before any paid run. Nothing in this document authorizes a
paid Tinker job by itself; every gate below must be checked off in order and the budget cap is
hard. Modeled on the Mantic/Thinking Machines recipe (RL fine-tuning with a strictly proper
reward on realized outcomes, research-phase context in the prompt), adapted to our evidence
pipeline and honesty rules.

## 1. Objective and success bars

Train a policy that outputs p(home team wins) for NBA games from an evidence-bundle prompt, by
policy-gradient RL on realized outcomes. Two predeclared success criteria, either of which
justifies continuation:

1. **Solo**: beats BOTH raw carryover MOV Elo and the training-only logit recalibration on the
   frozen conjunction (positive mean AND positive one-sided 95% LB per season, 7-day block
   bootstrap, 10k resamples, seed 20260716) on 2026-27 regular-season games (label 2027).
2. **Ensemble**: contributes positive weight and positive marginal score in the optimal mixture
   with the frozen tabular candidate and the de-vigged market baseline (Mantic Figure 7-9
   protocol: replaceability test on 2026-27 games).

Failure on both is preserved as a paid negative result and reported honestly. It is not
retried with a tweaked prompt.

## 2. Training data (realized outcomes, research phase already built)

- Questions: every regular-season game, 2021-22 through 2025-26 (labels 2022-2026), ~6,140
  games. Answer-free prompt construction (no score, no postgame field anywhere in the prompt).
- Prompt per game (both orientations, exact side-swap binding):
  - MOV Elo prior probability (carryover recipe, warmup from 2016-17);
  - the 11 standard features (home-minus-away differences): rest, back-to-back, games/road
    games last 7, travel miles/time zones, roster and lineup continuity proxies, rolling team
    net rating, rolling player value (causal RAPM or Kalman, whichever the diagnostics
    support), schedule strength.
- Fixed candidate labels TEAM and OTHER (frozen outcome-v2 contract); probability from
  candidate-token logprobs; Elo-offset head: final_logit = logit(p_elo) + logp(TEAM) -
  logp(OTHER), so a zero residual recovers Elo exactly.
- Splits are chronological by season; exact prompt overlap across splits is rejected.

### 2a. DECIDED — option B chosen 2026-07-22

~~Options A (default, strict) and B (owner amendment).~~ The owner chose **B**: aggregated
AND per-player injury-report content (names, statuses, reason categories from the official
public injury reports) is authorized in prompts for private-research RL runs, on the owner's
reading that publicly published league injury reports carry no additional privacy constraint
for local private research. Named health records still never enter this handoff or any
shared/published artifact, and prompts remain local training inputs only.

### 2b. Rights flag (must be acknowledged before launch)

Feature rows derive from shufinskiy (NBA endpoints) and ESPN JSON. Those numbers are
aggregates, not raw content, but upstream terms do not explicitly grant third-party ML
processing. Sending them to Tinker is a private-research judgment call the owner must
explicitly accept. FiveThirtyEight (CC BY) and Wikidata (CC0) components are clean.

## 3. Model and algorithm (Mantic recipe)

- Base: gpt-oss-120b via Tinker, LoRA. (Their finding: strong base + research context beats
  weak base + more training. Our 4B base measured 1.27 log loss raw — do not repeat it.)
- Reward: Brier score on the realized winner (bounded, lower-variance PG; their stability
  finding over log score). Report log loss at evaluation regardless.
- Reward computation (v1.1 amendment for the gpt-oss harmony format): the model's completion
  is `analysis` then `final` channel; the answer is the last TEAM/OTHER token. Per rollout,
  p(TEAM) = softmax(logprob of the sampled label token at its position, counterfactual
  logprob of the other label at the same position from one compute_logprobs scoring call on
  prompt + that rollout's completion prefix). This yields a true per-rollout probability (a
  single deterministic per-question probability would zero every GRPO advantage).
  Completions with no TEAM/OTHER token score reward 0 with no scoring call.
- v1.1 result (preserved): the recipe collapsed to a sharp classifier (~75 percent accuracy,
  p in {0,1}, log loss 3.45 in-sample) because Brier on softmax-over-two-tokens rewards p=1
  when right and near-deterministic groups equalize advantages.
- Reward computation (v2, probability-in-action-space): the completion contract changes to
  probability-as-text — the model states one decimal number in [0, 1] in the final channel
  (prompt template rl-prompt-v2, answer position `team_win_probability:`). Reward = 1 minus
  Brier on the PARSED number; unparseable or out-of-range numbers score 0. Calibration now
  lives in the action space and cannot collapse to argmax. No counterfactual scoring calls.
- Algorithm: policy gradient with GRPO-style advantage normalization, NO std-dev division,
  importance-sampling correction for sampler/trainer divergence, group size 8, batch 64.
- Scale: ~6,140 questions; 1 epoch; ~100 optimizer steps ≈ 51k rollouts.
- No training-time evaluation; checkpoint at 25/50/100 steps for the scaling plot.

## 4. Budget

- Hard cap: $500 total spend, enforced by a pre-launch cost check against the Tinker
  dashboard's then-current pricing for 120B LoRA sampling+training; if the projected run
  exceeds the cap, it does not launch.
- Kill-switch: abort and preserve partial state if spend exceeds the cap or training diverges
  (reward variance explosion, IS-correction ratio outside [0.5, 2.0] on >10% of rollouts).

## 5. Evaluation (post-run, frozen)

- Cohort: 2026-27 regular season, answer-free, both orientations, four candidate-logprob calls
  per game, raw terminal records retained, failures kept in the denominator at the frozen
  worst-case probability (1e-15).
- Bars as in section 1. The market benchmark arm runs alongside (declared ceiling).
- No training on, or early peeking at, 2026-27 outcomes before the season-end gate.

## 6. Prerequisites checklist before any paid call

1. [ ] T-15 and Kalman diagnostics read; player-rating input for prompts chosen (RAPM or
       Kalman) by those results.
2. [ ] Decision 2a recorded (default A unless the owner writes B).
3. [ ] Owner acknowledgment of 2b rights flag recorded.
4. [ ] Prompt renderer + candidate-logprob runtime verified locally on 16 games against the
       frozen outcome-v2 inference contract (no paid calls).
5. [ ] Training questions + evidence bundles sealed with hashes; chronological split manifest
       written; answer files separated.
6. [ ] Dashboard cost check passes the $500 cap.
7. [ ] User types GO on this exact protocol version (hash recorded in the run lock).
