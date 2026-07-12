# Cohorts

Place one committed cohort JSON file here for each prospective slate. A cohort must identify the
experiment lock, schedule source and retained snapshot hash, inclusion rule, every expected game,
forecast deadline, scheduled tipoff, and allowed outcome labels.

The ledger append API requires exact cohort coverage: missing, extra, or duplicate games are
rejected.

A cohort file has this strict shape:

```json
{
  "cohort_id": "nba-2026-10-20",
  "experiment_sha256": "64 lowercase hex characters",
  "schedule_source": "retained official schedule URL or identifier",
  "schedule_snapshot_sha256": "64 lowercase hex characters",
  "schedule_retrieved": "2026-10-20T12:00:00Z",
  "inclusion_rule": "Every NBA game in the October 20 slate.",
  "games": [
    {
      "question_id": "nba-2026-10-20-example",
      "source_game_id": "official-source-id",
      "matchup": "Listed team vs opponent",
      "outcomes": ["listed_team_wins", "opponent_wins"],
      "forecast_deadline": "2026-10-20T22:30:00Z",
      "scheduled_tipoff": "2026-10-20T23:00:00Z"
    }
  ]
}
```
