"""Explicit training and inference settings that prospective runs must freeze."""

from pathlib import Path

from forecastfm.integrity import file_sha256

BASE_MODEL = "Qwen/Qwen3.5-4B"
RENDERER_NAME = "qwen3_5"

TOKENIZER_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TOKENIZER_JSON_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
TOKENIZER_CONFIG_SHA256 = "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
CHAT_TEMPLATE_SHA256 = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"

TINKER_VERSION = "0.22.7"
TINKER_COOKBOOK_VERSION = "0.4.3"

BATCH_SIZE = 8
LEARNING_RATE = 2e-4
LORA_RANK = 16
MAX_STEPS = 1
MAX_LENGTH = 2_048

MAX_TOKENS = 128
TEMPERATURE = 0.0
TOP_K = -1
TOP_P = 1.0
SEED = 0


def require_tokenizer_snapshot() -> Path:
    """Return the pinned local tokenizer snapshot after checking its files."""
    path = (
        Path.home()
        / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots"
        / TOKENIZER_REVISION
    )
    expected_hashes = {
        "tokenizer.json": TOKENIZER_JSON_SHA256,
        "tokenizer_config.json": TOKENIZER_CONFIG_SHA256,
        "chat_template.jinja": CHAT_TEMPLATE_SHA256,
    }
    for name, expected_hash in expected_hashes.items():
        file_path = path / name
        if not file_path.is_file() or file_sha256(file_path) != expected_hash:
            raise RuntimeError(f"pinned tokenizer file is missing or changed: {file_path}")
    return path


def model_settings() -> dict[str, object]:
    """Return the exact model, renderer, and tokenizer references."""
    return {
        "provider": "Tinker",
        "base_model": BASE_MODEL,
        "server_weight_revision": None,
        "server_weight_revision_note": (
            "Tinker exposes a catalog model name, not a base-weight digest. "
            "The final sampler path is the authoritative trained policy identity."
        ),
        "renderer": RENDERER_NAME,
        "tokenizer_hugging_face_revision": TOKENIZER_REVISION,
        "tokenizer_json_sha256": TOKENIZER_JSON_SHA256,
        "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
    }


def training_settings() -> dict[str, object]:
    """Return every non-path setting used by the small Tinker run."""
    return {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "max_steps": MAX_STEPS,
        "max_length": MAX_LENGTH,
        "num_epochs": 1,
        "train_on": "last_assistant_message",
        "save_every": 0,
        "eval_every": 0,
        "infrequent_eval_every": 0,
        "submit_ahead": 0,
    }


def decoding_settings() -> dict[str, object]:
    """Return explicit single-attempt sampling settings for evaluation."""
    return {
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "seed": SEED,
        "num_samples": 1,
        "max_attempts": 1,
        "tools_enabled": False,
        "malformed_output_policy": "record_and_score_invalid",
    }


def package_versions() -> dict[str, object]:
    """Return pinned remote-training dependency versions."""
    return {
        "tinker": TINKER_VERSION,
        "tinker_cookbook": TINKER_COOKBOOK_VERSION,
    }
