"""Tests for the shared target-free outcome-v2 prompt contract."""

from datetime import UTC, datetime, timedelta

from forecastfm.integrity import canonical_json
from forecastfm.nba_data import SIDE_SWAP_SUFFIX
from forecastfm.nba_feature_rows import NbaRichFeatureRow
from forecastfm.nba_rich import NBA_RICH_FEATURE_NAMES, NbaRichFeatures
from forecastfm.outcome import OPPONENT_OUTCOME, TEAM_OUTCOME
from forecastfm.outcome_v2_config import QUESTION_TEXT, RESOLUTION_RULE
from forecastfm.outcome_v2_prompt import OUTCOME_V2_SYSTEM_PROMPT, build_outcome_v2_messages


def _row() -> NbaRichFeatureRow:
    cutoff = datetime(2026, 10, 21, 22, tzinfo=UTC)
    return NbaRichFeatureRow(
        question_id="nba-prompt-1",
        source_game_id="source-nba-prompt-1",
        team_id="Team",
        opponent_id="Opponent",
        site="neutral",
        season=2027,
        forecast_cutoff=cutoff,
        scheduled_tipoff=cutoff + timedelta(hours=1),
        elo_team_win_probability=0.61,
        elo_opponent_win_probability=0.39,
        elo_available_at=cutoff - timedelta(hours=1),
        elo_state_sha256="a" * 64,
        rich_features=NbaRichFeatures.from_vector(
            (1.0, -1.0, 2.0, -2.0, 300.0, -1.0, 0.2, -0.2, 4.0, -4.0, 25.0)
        ),
        evidence_bundle_sha256="b" * 64,
        input_available_at=cutoff - timedelta(minutes=30),
    )


def _expected_user_content(row: NbaRichFeatureRow) -> str:
    return canonical_json(
        {
            "evidence": [
                f"Pregame numeric feature: {canonical_json({name: value})}"
                for name, value in zip(
                    NBA_RICH_FEATURE_NAMES,
                    row.rich_features.vector,
                    strict=True,
                )
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


def test_builder_emits_the_exact_target_free_prompt() -> None:
    row = _row()

    messages = build_outcome_v2_messages(row)

    assert messages == (
        {"role": "system", "content": OUTCOME_V2_SYSTEM_PROMPT},
        {"role": "user", "content": _expected_user_content(row)},
    )
    assert "label" not in messages[1]["content"]
    assert "realized" not in messages[1]["content"]
    assert "adjustment added to the Elo log-odds" in messages[0]["content"]


def test_side_swap_prompt_uses_only_the_exact_swapped_row() -> None:
    original = _row()
    swapped = original.side_swap()

    messages = build_outcome_v2_messages(swapped)

    assert swapped.question_id == f"{original.question_id}{SIDE_SWAP_SUFFIX}"
    assert messages[1]["content"] == _expected_user_content(swapped)
    assert swapped.elo_team_win_probability == original.elo_opponent_win_probability
    assert swapped.rich_features.vector == tuple(-value for value in original.rich_features.vector)
