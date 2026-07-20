"""Tests for the 2026-27 rolling weekly prospective runner."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NoReturn

import pytest
from examples import run_private_prototype
from examples import run_prospective_weekly as runner

from forecastfm.json_utils import require_object


def _driver_guard(message: str) -> Callable[[Sequence[str]], NoReturn]:
    def _fail(_argv: Sequence[str]) -> NoReturn:
        pytest.fail(message)

    return _fail


def _report() -> dict[str, object]:
    season = {
        "season": 2027,
        "game_count": 60,
        "calendar_block_count": 9,
        "model": {"mean_log_loss": 0.65, "mean_brier": 0.2},
        "baseline": {"mean_log_loss": 0.67, "mean_brier": 0.21},
        "mean_baseline_relative_log_score": 0.01,
        "lower_one_sided_95": -0.005,
        "passes": False,
    }
    gate = {
        "bootstrap_block_days": 7,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 20260716,
        "one_sided_alpha": 0.05,
        "seasons": [season],
        "game_count": 60,
        "pooled_baseline_relative_log_score": 0.01,
        "passes": False,
    }
    return {
        "evaluation_games": 60,
        "variants": {
            "standard_vs_raw_elo": gate,
            "standard_vs_recalibrated_elo": gate,
            "projected_vs_raw_elo": gate,
            "projected_vs_recalibrated_elo": gate,
        },
    }


def test_tracker_row_rewrite_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "tracker.jsonl"
    runner.append_tracker_row(path, {"as_of": "2026-11-01", "games": 60, "variants": {}})
    runner.append_tracker_row(path, {"as_of": "2026-11-08", "games": 100, "variants": {}})
    replacement: runner.JsonObject = {
        "as_of": "2026-11-01",
        "games": 61,
        "variants": {"standard": {}},
    }
    runner.append_tracker_row(path, replacement)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["as_of"] for row in rows] == ["2026-11-08", "2026-11-01"]
    assert rows[1] == json.loads(json.dumps(replacement, sort_keys=True))


def test_tracker_row_distils_variants_and_baselines() -> None:
    row = runner.tracker_row("2026-11-01", runner_report := _report())
    assert runner_report["evaluation_games"] == row["games"] == 60
    assert row["as_of"] == "2026-11-01"
    variants = require_object(row.get("variants"), "variants")
    assert set(variants) == {"standard", "projected"}
    standard = require_object(variants.get("standard"), "standard")
    season = require_object(standard.get("2027"), "2027")
    assert season["games"] == 60
    assert season["mean_log_loss"] == 0.65
    for arm in ("vs_raw_elo", "vs_recalibrated_elo"):
        record = require_object(season.get(arm), arm)
        assert record["mean_baseline_relative_log_score"] == 0.01
        assert record["lower_one_sided_95"] == -0.005
        assert record["passes"] is False


def test_season_files_patch_adds_2027_without_mutating() -> None:
    original = run_private_prototype.SEASON_FILES
    before = dict(original)
    assert runner.EVALUATION_SEASON not in original
    patched = runner.patched_season_files(original)
    assert patched is not original
    assert original == before
    assert patched[runner.EVALUATION_SEASON] == runner.ESPN_SEASON_CSV
    assert {season: path for season, path in patched.items() if season != 2027} == before


@pytest.mark.parametrize(
    "argv",
    [
        ["refresh", "--dry-run"],
        ["refresh", "--as-of", "2026-11-01", "--dry-run"],
        ["evaluate", "--as-of", "2026-11-01", "--dry-run"],
        ["track", "--as-of", "2026-11-01", "--dry-run"],
    ],
    ids=["refresh", "refresh-dated", "evaluate", "track"],
)
def test_dry_run_prints_plan_without_writes(
    argv: Sequence[str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "out"
    monkeypatch.setattr(runner, "ESPN_SEASON_CSV", tmp_path / "espn_2025.csv")
    monkeypatch.setattr(runner, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(
        run_private_prototype,
        "main",
        _driver_guard("driver must not run under --dry-run"),
    )
    assert runner.main(list(argv)) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["command"] == argv[0]
    assert list(tmp_path.rglob("*")) == []


def test_evaluate_exits_2_when_season_not_started(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    season_csv = tmp_path / "espn_2025.csv"
    season_csv.write_text("game_id\n" + "22500001\n" * 10, encoding="utf-8")
    monkeypatch.setattr(runner, "ESPN_SEASON_CSV", season_csv)
    monkeypatch.setattr(runner, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(
        run_private_prototype,
        "main",
        _driver_guard("driver must not run before the season starts"),
    )
    assert runner.main(["evaluate", "--as-of", "2026-11-01"]) == runner.EXIT_NOT_STARTED == 2
    assert "not meaningfully started" in capsys.readouterr().out
    assert list(tmp_path.rglob("*")) == [season_csv]
