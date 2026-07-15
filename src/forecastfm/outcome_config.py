"""Readable settings for the first realized-outcome training canary."""

from forecastfm.outcome import OPPONENT_LABEL, TEAM_LABEL

OUTCOME_RENDERER_NAME = "qwen3_5_disable_thinking"

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
LORA_RANK = 16
MAX_LENGTH = 2_048

MAX_STEPS = 32
SCALING_STEPS = (32, 128, 512, 2_048)
SAVE_EVERY = 32
RUN_NAME = f"outcome_v1_steps_{MAX_STEPS}"


def outcome_training_settings() -> dict[str, object]:
    """Return every non-path setting used by outcome training v1."""
    return {
        "objective": "realized_winner_cross_entropy",
        "labels": [TEAM_LABEL, OPPONENT_LABEL],
        "renderer": OUTCOME_RENDERER_NAME,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "max_length": MAX_LENGTH,
        "max_steps": MAX_STEPS,
        "planned_scaling_steps": list(SCALING_STEPS),
        "num_epochs": 1,
        "save_every": SAVE_EVERY,
        "eval_every": 0,
        "infrequent_eval_every": 0,
        "submit_ahead": 0,
    }


def outcome_inference_settings() -> dict[str, object]:
    """Return the deterministic two-candidate probability policy."""
    return {
        "method": "renormalized_candidate_token_logprobs",
        "labels": [TEAM_LABEL, OPPONENT_LABEL],
        "logical_calls_per_orientation": 2,
        "orientations_per_forecast": 2,
        "logical_calls_per_symmetric_forecast": 4,
        "text_generation": False,
        "side_swap_averaging": True,
    }
