"""Target-free outcome-v2 prompts derived from sealed NBA feature rows."""

from forecastfm.integrity import canonical_json
from forecastfm.nba_feature_rows import NbaRichFeatureRow
from forecastfm.outcome import (
    OPPONENT_OUTCOME,
    TEAM_OUTCOME,
)
from forecastfm.outcome_v2_config import QUESTION_TEXT, RESOLUTION_RULE
from forecastfm.prompting import ChatMessage

_FEATURE_CARD_PREFIX = "Pregame numeric feature: "

OUTCOME_V2_SYSTEM_PROMPT = """You are ForecastFM, a calibrated probabilistic forecaster.
Use only the Elo prior and pregame evidence supplied to you.
Your TEAM-versus-OTHER token log-odds are an adjustment added to the Elo log-odds.
Equal TEAM and OTHER scores leave the Elo forecast unchanged.
TEAM supports the listed team; OTHER supports its opponent.
Return exactly TEAM or OTHER, with no JSON, punctuation, or explanation."""


def build_outcome_v2_messages(row: NbaRichFeatureRow) -> tuple[ChatMessage, ...]:
    """Build the exact target-free classifier prompt for one sealed orientation."""
    user_content = canonical_json(
        {
            "evidence": [
                f"{_FEATURE_CARD_PREFIX}{canonical_json({name: value})}"
                for name, value in row.feature_items
            ],
            "outcomes": [TEAM_OUTCOME, OPPONENT_OUTCOME],
            "prior": {
                TEAM_OUTCOME: row.elo_team_win_probability,
                OPPONENT_OUTCOME: row.elo_opponent_win_probability,
            },
            "question": QUESTION_TEXT,
            "resolution_rule": RESOLUTION_RULE,
        }
    )
    return (
        ChatMessage(role="system", content=OUTCOME_V2_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    )
