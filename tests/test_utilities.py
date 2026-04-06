"""Tests for sample-data generation and CLI utility modes."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.sample.generate_sample import SyntheticSleepDataGenerator, generate_sample_data
from main import main


def test_generate_sample_data_writes_expected_files():
    output_dir = Path("tests_artifacts") / "sample-data"
    output_dir.mkdir(parents=True, exist_ok=True)
    written = generate_sample_data(output_dir=output_dir, days=5, seed=7)

    assert {"sleep_sessions", "sleep_stages", "heart_rate", "hrv", "spo2", "manifest"} == set(written)
    manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
    assert manifest["days"] == 5
    assert Path(written["sleep_sessions"]).exists()

    sessions = pd.read_csv(written["sleep_sessions"])
    stages = pd.read_csv(written["sleep_stages"])
    assert len(sessions) == 5
    assert not stages.empty
    assert {"light", "deep", "rem", "wake"}.issubset(set(stages["stage_type"]))


def test_cli_analyze_report_and_trend_modes():
    generator = SyntheticSleepDataGenerator(seed=11)
    frames = generator.generate(days=3)

    base_dir = Path("tests_artifacts") / "cli-utilities"
    base_dir.mkdir(parents=True, exist_ok=True)
    scores_input = base_dir / "scores-input.csv"
    analyzed_output = base_dir / "scores-output.csv"
    report_output = base_dir / "report.html"
    stages_input = base_dir / "stages.csv"

    sessions = frames["sleep_sessions"].rename(
        columns={
            "minutesAsleep": "total_sleep_minutes",
            "efficiency": "sleep_efficiency",
            "minutesToFallAsleep": "sleep_onset_latency",
            "awakeningsCount": "number_of_awakenings",
            "levels_summary_deep_minutes": "deep_minutes",
            "levels_summary_rem_minutes": "rem_minutes",
            "levels_summary_light_minutes": "light_minutes",
            "levels_summary_wake_minutes": "wake_minutes",
        }
    )
    sessions["deep_sleep_pct"] = sessions["deep_minutes"] / sessions["total_sleep_minutes"] * 100.0
    sessions["rem_sleep_pct"] = sessions["rem_minutes"] / sessions["total_sleep_minutes"] * 100.0
    sessions["awake_pct"] = sessions["wake_minutes"] / sessions["timeInBed"] * 100.0
    sessions["avg_hrv"] = frames["hrv"]["value_dailyRmssd"].values
    sessions["resting_hr"] = 58.0
    sessions["avg_spo2"] = frames["spo2"]["value_avg"].values

    sessions[
        [
            "dateOfSleep",
            "total_sleep_minutes",
            "deep_sleep_pct",
            "rem_sleep_pct",
            "awake_pct",
            "sleep_efficiency",
            "sleep_onset_latency",
            "number_of_awakenings",
            "avg_hrv",
            "resting_hr",
            "avg_spo2",
        ]
    ].to_csv(scores_input, index=False)
    frames["sleep_stages"].to_csv(stages_input, index=False)

    assert main(["analyze", "--input", str(scores_input), "--output", str(analyzed_output)]) == 0
    assert analyzed_output.exists()

    analyzed = pd.read_csv(analyzed_output)
    analyzed["score_date"] = analyzed["dateOfSleep"]
    analyzed.to_csv(analyzed_output, index=False)

    assert main(["report", "--scores", str(analyzed_output), "--stages", str(stages_input), "--output", str(report_output)]) == 0
    assert report_output.exists()
    assert "Sleep Report" in report_output.read_text(encoding="utf-8")

    assert main(["trend", "--input", str(analyzed_output)]) == 0
