"""Readable settings for the eventual licensed outcome-v2 runs."""

from forecastfm.elo_residual import EloResidualFitConfig
from forecastfm.nba_elo_replay import NbaEloRecipe
from forecastfm.nba_evaluation_gate import NbaEvaluationGatePolicy
from forecastfm.outcome import OPPONENT_LABEL, OUTCOME_INPUT_SCHEMA_VERSION, TEAM_LABEL

MANIFEST_SCHEMA_VERSION = 1
TRAINING_FILENAME = "nba_train_outcome.jsonl"
FEATURE_ROWS_FILENAME = "nba_train_feature_rows.jsonl"
SNAPSHOT_PACK_FILENAME = "nba_train_snapshots.jsonl"
EVIDENCE_BUNDLES_FILENAME = "nba_train_evidence.jsonl"
ELO_STATES_FILENAME = "nba_train_elo_states.jsonl"
ELO_REPLAY_FILENAME = "nba_train_elo_replay.jsonl"
SEASONS_FILENAME = "nba_train_seasons.json"
RESOLUTIONS_FILENAME = "nba_train_resolutions.jsonl"
RIGHTS_LOCK_FILENAME = "nba_rights_approval_lock.json"
EVALUATION_COHORT_FILENAME = "nba_untouched_evaluation_cohort.jsonl"
EVALUATION_FEATURE_ROWS_FILENAME = "nba_untouched_evaluation_feature_rows.jsonl"
EVALUATION_ANSWERS_FILENAME = "nba_untouched_evaluation_answers.jsonl"
EVALUATION_FORECASTS_FILENAME = "nba_untouched_evaluation_forecasts.jsonl"
EVALUATION_ELO_REPLAY_FILENAME = "nba_untouched_evaluation_elo_replay.jsonl"
EVALUATION_ELO_STATES_FILENAME = "nba_untouched_evaluation_elo_states.jsonl"
EVALUATION_RESOLUTIONS_FILENAME = "nba_untouched_evaluation_resolutions.jsonl"
RECALIBRATION_FILENAME = "nba_train_elo_recalibration.jsonl"
EVALUATION_REPORT_FILENAME = "nba_untouched_evaluation_gate.json"
RICH_BASELINE_MODEL_FILENAME = "nba_rich_baseline_model.json"
RICH_BASELINE_FORECAST_LOCK_FILENAME = "nba_rich_baseline_forecast_lock.json"
QUESTION_TEXT = "Will the listed team defeat its opponent in this NBA game?"
RESOLUTION_RULE = "Resolve to the team with the higher final score."

ELO_INITIAL_RATING = 1_500.0
ELO_K_FACTOR = 20.0
ELO_RATING_SCALE = 400.0
ELO_HOME_ADVANTAGE = 100.0

# Each batch holds seven complete original/side-swap pairs; a final partial batch is retained.
BATCH_SIZE = 14
DROP_LAST = False
LEARNING_RATE = 1e-4
LORA_RANK = 16
LORA_SEED = 0
TRAIN_MLP = True
TRAIN_ATTN = True
TRAIN_UNEMBED = True
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 0.0
MAX_LENGTH = 2_048
MAX_STEPS = 128
SHUFFLE_SEED = 0
OUTCOME_RENDERER_NAME = "qwen3_5_disable_thinking"
RUN_NAME = f"forecastfm-outcome-v2-sft-steps-{MAX_STEPS}"
FINAL_STATE_NAME = f"{RUN_NAME}-state"
FINAL_SAMPLER_NAME = f"{RUN_NAME}-sampler"
FINAL_CHECKPOINT_TTL_SECONDS: int | None = None
MINIMUM_EVALUATION_GAMES_PER_SEASON = 1_000
MINIMUM_EVALUATION_CALENDAR_BLOCKS_PER_SEASON = 20
RICH_BASELINE_STEPS = 1_000
RICH_BASELINE_LEARNING_RATE = 0.1
RICH_BASELINE_L2_PENALTY = 0.01


def outcome_v2_coverage_policy() -> dict[str, object]:
    """Return the machine-readable no-cherry-picking schedule policy."""
    return {
        "schema_version": 1,
        "league": "NBA",
        "season_types": ["regular"],
        "forecast_unit": "one_original_T-60_forecast_per_provider_source_game_id",
        "schedule_authority": "zero_gap_reviewed_provider_inventory",
        "season_commitment": "externally_receipted_before_first_season_feature_input",
        "batch_union": "exact_question_id_equality_with_committed_schedule_rows",
        "reschedule_policy": "schema_v1_rejects_changes_after_first_season_feature_input",
        "cancellation_policy": "schema_v1_rejects_changes_after_first_season_feature_input",
    }


def outcome_v2_elo_recipe() -> NbaEloRecipe:
    """Return the one frozen Elo recipe used by training and evaluation."""
    return NbaEloRecipe(
        initial_rating=ELO_INITIAL_RATING,
        k_factor=ELO_K_FACTOR,
        rating_scale=ELO_RATING_SCALE,
        home_advantage=ELO_HOME_ADVANTAGE,
    )


def outcome_v2_evaluation_policy() -> NbaEvaluationGatePolicy:
    """Return the frozen benchmark strength and recalibration recipe."""
    return NbaEvaluationGatePolicy(
        minimum_games_per_season=MINIMUM_EVALUATION_GAMES_PER_SEASON,
        minimum_calendar_blocks_per_season=(MINIMUM_EVALUATION_CALENDAR_BLOCKS_PER_SEASON),
        recalibration_gradient_steps=2_000,
        recalibration_learning_rate=0.05,
        recalibration_initial_intercept=0.0,
        recalibration_initial_slope=1.0,
    )


def outcome_v2_rich_baseline_fit_config() -> EloResidualFitConfig:
    """Return the frozen first-pass tabular baseline optimizer."""
    return EloResidualFitConfig(
        steps=RICH_BASELINE_STEPS,
        learning_rate=RICH_BASELINE_LEARNING_RATE,
        l2_penalty=RICH_BASELINE_L2_PENALTY,
    )


def outcome_v2_sft_settings() -> dict[str, object]:
    """Return the small, vendor-neutral settings checked before SFT."""
    return {
        "artifact": "licensed_point_in_time_outcome_v2",
        "input_schema_version": OUTCOME_INPUT_SCHEMA_VERSION,
        "batch_size": BATCH_SIZE,
        "drop_last": DROP_LAST,
        "training_filename": TRAINING_FILENAME,
        "feature_rows_filename": FEATURE_ROWS_FILENAME,
        "snapshot_pack_filename": SNAPSHOT_PACK_FILENAME,
        "evidence_bundles_filename": EVIDENCE_BUNDLES_FILENAME,
        "elo_states_filename": ELO_STATES_FILENAME,
        "elo_replay_filename": ELO_REPLAY_FILENAME,
        "seasons_filename": SEASONS_FILENAME,
        "resolutions_filename": RESOLUTIONS_FILENAME,
        "rights_lock_filename": RIGHTS_LOCK_FILENAME,
        "evaluation_cohort_filename": EVALUATION_COHORT_FILENAME,
        "evaluation_feature_rows_filename": EVALUATION_FEATURE_ROWS_FILENAME,
        "evaluation_answers_filename": EVALUATION_ANSWERS_FILENAME,
        "evaluation_forecasts_filename": EVALUATION_FORECASTS_FILENAME,
        "evaluation_elo_replay_filename": EVALUATION_ELO_REPLAY_FILENAME,
        "evaluation_elo_states_filename": EVALUATION_ELO_STATES_FILENAME,
        "evaluation_resolutions_filename": EVALUATION_RESOLUTIONS_FILENAME,
        "recalibration_filename": RECALIBRATION_FILENAME,
        "evaluation_report_filename": EVALUATION_REPORT_FILENAME,
        "rich_baseline_model_filename": RICH_BASELINE_MODEL_FILENAME,
        "rich_baseline_forecast_lock_filename": RICH_BASELINE_FORECAST_LOCK_FILENAME,
        "requires_full_outcome_v2_ready": True,
    }


def outcome_v2_training_settings() -> dict[str, object]:
    """Return every frozen setting for the first outcome-v2 SFT run."""
    return {
        "objective": "elo_offset_realized_winner_binary_cross_entropy",
        "loss_fn": "binary_cross_entropy_with_logits",
        "sdk_method": "forward_backward_custom_async",
        "final_logit": "logit(elo_team_probability)+logp(TEAM)-logp(OTHER)",
        "zero_residual_ablation": "raw_elo_probability",
        "full_vocabulary_cross_entropy": False,
        "labels": [TEAM_LABEL, OPPONENT_LABEL],
        "run_name": RUN_NAME,
        "renderer": OUTCOME_RENDERER_NAME,
        "batch_size": BATCH_SIZE,
        "drop_last": DROP_LAST,
        "learning_rate": LEARNING_RATE,
        "learning_rate_schedule": "constant",
        "lora_rank": LORA_RANK,
        "lora_seed": LORA_SEED,
        "train_mlp": TRAIN_MLP,
        "train_attn": TRAIN_ATTN,
        "train_unembed": TRAIN_UNEMBED,
        "adam_beta1": ADAM_BETA1,
        "adam_beta2": ADAM_BETA2,
        "adam_eps": ADAM_EPS,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "max_length": MAX_LENGTH,
        "max_steps": MAX_STEPS,
        "shuffle_seed": SHUFFLE_SEED,
        "checkpoint_policy": "final_state_and_sampler_only",
        "final_state_name": FINAL_STATE_NAME,
        "final_sampler_name": FINAL_SAMPLER_NAME,
        "final_checkpoint_ttl_seconds": FINAL_CHECKPOINT_TTL_SECONDS,
        "evaluation_during_training": False,
        "submit_ahead": 0,
        "resume": False,
    }


def outcome_v2_inference_settings() -> dict[str, object]:
    """Return the deterministic fixed-token probability contract."""
    return {
        "method": "elo_offset_candidate_token_logprobs",
        "api_method": "compute_logprobs_async",
        "final_logit": "logit(elo_team_probability)+logp(TEAM)-logp(OTHER)",
        "labels": [TEAM_LABEL, OPPONENT_LABEL],
        "generated_text_used": False,
        "sdk_internal_unused_generated_tokens_per_call": 1,
        "side_swap_averaging": True,
        "logical_calls_per_game": 4,
        "application_attempts_per_game": 1,
        "application_retries": 0,
        "sdk_retry_logic_enabled": False,
        "sdk_internal_retransmission_window_seconds": 300,
        "transport_retry_note": (
            "Tinker 0.22.7 may internally retransmit one logical request ID after connection or "
            "timeout errors and HTTP 408, 409, 429, or 5xx responses for up to five minutes."
        ),
        "missing_or_malformed_policy": "retain_and_score_as_failure",
    }
