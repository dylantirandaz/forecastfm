# Validation canary 1 result

Status: failed at the output protocol. This run does not compare forecasting skill.

## Frozen evidence

- Protocol commit: `6b6e54402352ebb1c8b2bc80e7efd4e93dee5bbd`
- Cohort commit: `d7188d9`
- Attempt-marker commit: `6603a82`
- Sealed-generation commit: `eaca1fa`
- Canary manifest SHA-256: `bb5396100c6867f4cca2ddbfffc600aef7109baeb8660055b3bc366a02d1c1f1`
- Generation seal SHA-256: `f9a1b2b522230aa5f218d1d337341b6ccd261999904d5a94e2dada3f5a154bbb`

The protocol, prompt-only cohort, attempt marker, and raw generations were committed and
published before the validation answers were opened.

## What happened

| Arm | Completed calls | 128-token outputs | Length stops | Malformed terminations | Strict JSON outputs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 128/128 | 128/128 | 128/128 | 128/128 | 0/128 |
| Adapter | 128/128 | 128/128 | 128/128 | 128/128 | 0/128 |

Every response spent the full output budget on a thinking trace. None contained an opening
JSON brace or reached a final answer. Both arms therefore received the preregistered
invalid-output penalties. The equal penalized scores are not evidence that the models have
equal forecasting skill.

## Diagnosis

The frozen inference renderer was `qwen3_5`. In the pinned `tinker-cookbook` version, that
renderer starts generation inside a `<think>` block. The 128-token limit was exhausted
before the model closed its reasoning and emitted JSON.

The same pinned package provides `qwen3_5_disable_thinking`, which starts with an empty
thinking block and asks the model to answer directly. This also matches the empty thinking
block that the renderer inserts before supervised assistant targets during training. The
evidence therefore points to an inference-format mismatch, not an accuracy result.

## Conservative next run

1. Keep this canary and its scores unchanged.
2. On one training example, make one base and one adapter format-smoke call with
   `qwen3_5_disable_thinking`. Continue only if both calls stop cleanly and return strict
   probability JSON within 128 tokens.
3. Freeze and publish a second protocol using the no-thinking renderer and a disjoint set of
   64 validation IDs. Do not reuse any of this run's 64 IDs.
4. If the smoke gate passes, run the same original/side-swapped pairing: 128 prompts per arm,
   256 validation calls total.
5. Seal and publish the raw generations before opening the new answers, then report schema
   validity, Elo-oracle error, side-swap consistency, teacher error, Brier score, log loss,
   and failure cases.

This requires two paid smoke calls. A passing smoke test would make a separate 256-call
validation run eligible for explicit user approval; a failing smoke test stops before that
expense.
