"""Tests for the Plotly dashboard and Jinja2 report generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.visualization.dashboard import SleepDashboard
from src.visualization.report_generator import SleepReportGenerator


def _sample_sleep_scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "score_date": "2026-03-24",
                "final_score": 78.0,
                "duration_score": 82.0,
                "deep_score": 74.0,
                "rem_score": 80.0,
                "continuity_score": 76.0,
                "recovery_score": 79.0,
                "sleep_efficiency": 89.0,
                "total_sleep_minutes": 430,
            },
            {
                "score_date": "2026-03-25",
                "final_score": 85.0,
                "duration_score": 88.0,
                "deep_score": 81.0,
                "rem_score": 84.0,
                "continuity_score": 83.0,
                "recovery_score": 86.0,
                "sleep_efficiency": 91.0,
                "total_sleep_minutes": 455,
            },
            {
                "score_date": "2026-04-01",
                "final_score": 90.0,
                "duration_score": 92.0,
                "deep_score": 87.0,
                "rem_score": 89.0,
                "continuity_score": 88.0,
                "recovery_score": 91.0,
                "sleep_efficiency": 93.0,
                "total_sleep_minutes": 470,
            },
        ]
    )


def _sample_sleep_stages() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"stage_type": "light", "started_at": "2026-04-01T22:30:00", "seconds": 1800, "sequence_index": 0},
            {"stage_type": "deep", "started_at": "2026-04-01T23:00:00", "seconds": 3600, "sequence_index": 1},
            {"stage_type": "rem", "started_at": "2026-04-02T00:00:00", "seconds": 1800, "sequence_index": 2},
            {"stage_type": "wake", "started_at": "2026-04-02T00:30:00", "seconds": 600, "sequence_index": 3},
        ]
    )


def _sample_heart_rate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"timestamp": "2026-04-01T22:30:00", "heart_rate": 62},
            {"timestamp": "2026-04-01T23:00:00", "heart_rate": 58},
            {"timestamp": "2026-04-02T00:00:00", "heart_rate": 55},
            {"timestamp": "2026-04-02T00:30:00", "heart_rate": 64},
        ]
    )


def test_sleep_dashboard_builds_expected_plotly_figures():
    dashboard = SleepDashboard()
    scores = _sample_sleep_scores()
    stages = _sample_sleep_stages()
    heart_rate = _sample_heart_rate()

    figures = dashboard.create_dashboard_figures(
        sleep_scores=scores,
        sleep_stages=stages,
        heart_rate_data=heart_rate,
        score_components=scores.iloc[[0]],
    )

    assert set(figures) == {
        "sleep_score_trend",
        "sleep_hypnogram",
        "heart_rate_overlay",
        "score_radar_chart",
        "weekly_comparison_chart",
    }
    assert len(figures["sleep_score_trend"].data) == 2
    assert len(figures["sleep_hypnogram"].data) == 4
    assert any(trace.name == "Heart rate" for trace in figures["heart_rate_overlay"].data)
    assert figures["score_radar_chart"].data[0].type == "scatterpolar"
    assert all(trace.type == "bar" for trace in figures["weekly_comparison_chart"].data)


def test_report_generator_renders_html_and_writes_output():
    scores = _sample_sleep_scores()
    summary = scores.iloc[[0]].copy()
    summary["recommendations"] = [[{"title": "Keep bedtime consistent", "message": "Aim to sleep within the same 30-minute window each night."}]]

    output_path = Path("tests_artifacts") / "sleep-report.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = SleepReportGenerator().generate_html_report(
        user_name="Alex Example",
        summary=summary,
        sleep_scores=scores,
        sleep_stages=_sample_sleep_stages(),
        heart_rate_data=_sample_heart_rate(),
        report_date="2026-04-05",
        output_path=output_path,
    )

    assert "Sleep Report" in html
    assert "Alex Example" in html
    assert "Keep bedtime consistent" in html
    assert "plotly" in html.lower()
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == html
