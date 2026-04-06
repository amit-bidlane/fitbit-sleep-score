"""Plotly chart helpers for the Streamlit sleep dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.visualization.dashboard import SleepDashboard


def render_dashboard_charts(
    *,
    trend: dict[str, Any] | None = None,
    stages: list[dict[str, Any]] | None = None,
    score: dict[str, Any] | None = None,
    heart_rate_data: Any | None = None,
) -> dict[str, Any]:
    """Build and render the primary dashboard charts."""

    dashboard = SleepDashboard()
    score_frame = _scores_frame_from_trend(trend=trend, score=score)
    stage_frame = _to_frame(stages)
    heart_rate_frame = _to_frame(heart_rate_data)
    score_components = _to_frame(score)

    figures = dashboard.create_dashboard_figures(
        sleep_scores=score_frame,
        sleep_stages=stage_frame,
        heart_rate_data=heart_rate_frame,
        score_components=score_components,
    )

    top_left, top_right = st.columns(2)
    with top_left:
        st.plotly_chart(figures["sleep_score_trend"], use_container_width=True)
    with top_right:
        st.plotly_chart(figures["score_radar_chart"], use_container_width=True)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.plotly_chart(figures["sleep_hypnogram"], use_container_width=True)
    with bottom_right:
        st.plotly_chart(figures["heart_rate_overlay"], use_container_width=True)

    st.plotly_chart(figures["weekly_comparison_chart"], use_container_width=True)
    return figures


def render_recommendation_chart(recommendations: list[dict[str, Any]] | None) -> Any:
    """Render a simple category distribution chart for recommendations."""

    frame = _to_frame(recommendations)
    if frame.empty or "category" not in frame.columns:
        st.info("No recommendation chart data available.")
        return None

    counts = frame["category"].astype(str).value_counts().rename_axis("category").reset_index(name="count")
    figure = SleepDashboard()._empty_figure("Recommendations")
    figure.data = ()
    figure.add_bar(
        x=counts["category"],
        y=counts["count"],
        marker_color="#2a9d8f",
        hovertemplate="%{x}<br>Recommendations: %{y}<extra></extra>",
    )
    figure.update_layout(title="Recommendations by Category", xaxis_title="Category", yaxis_title="Count")
    st.plotly_chart(figure, use_container_width=True)
    return figure


def _scores_frame_from_trend(*, trend: dict[str, Any] | None, score: dict[str, Any] | None) -> pd.DataFrame:
    """Normalize score and trend payloads into a chart-friendly frame."""

    points = (trend or {}).get("points", [])
    if points:
        frame = pd.json_normalize(points, sep="_")
        if "overall_score" in frame.columns and "final_score" not in frame.columns:
            frame["final_score"] = frame["overall_score"]
        return frame
    return _to_frame(score)


def _to_frame(data: Any | None) -> pd.DataFrame:
    """Normalize supported inputs to a DataFrame."""

    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, dict):
        return pd.json_normalize([data], sep="_")
    if isinstance(data, list):
        return pd.json_normalize(data, sep="_")
    raise TypeError("Chart data must be a DataFrame, dict, list, or None.")
