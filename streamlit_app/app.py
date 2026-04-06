"""Main Streamlit dashboard for the Fitbit Sleep Score system."""

from __future__ import annotations

import os
from typing import Any

import httpx
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.components.api_client import SleepScoreAPIClient
from streamlit_app.components.charts import render_dashboard_charts
from streamlit_app.components.score_card import render_component_breakdown, render_score_cards
from streamlit_app.components.sidebar import render_sidebar

if "user_id" not in st.session_state:
    st.session_state["user_id"] = 1
def main() -> None:
    """Render the main dashboard application."""

    st.set_page_config(
        page_title="Fitbit Sleep Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Fitbit Sleep Score Dashboard")
    st.caption("Track nightly sleep quality, compare trends, and review score components from the FastAPI backend.")

    controls = render_sidebar(
        default_user_id=st.session_state.get("user_id"),
        default_api_url=st.session_state.get(
            "api_url",
            os.getenv("FITBIT_SLEEP_API_URL") or os.getenv("API_BASE_URL") or "http://localhost:8000",
        ),
    )
    st.session_state["api_url"] = controls["api_url"]

    if not controls["user_id"]:
        st.info("Enter a user ID in the sidebar to load dashboard data.")
        return

    client = SleepScoreAPIClient(
        base_url=controls["api_url"],
        user_id=controls["user_id"],
    )
    # ADD THIS BLOCK HERE
    if st.sidebar.button("Generate Demo Sleep Data"):

        demo_payload = {
            "date": "2026-04-06",
            "fitbit_log_id": 1,
            "start_time": "2026-04-05T23:00:00Z",
            "end_time": "2026-04-06T06:00:00Z",
            "time_in_bed": 420,
            "minutes_asleep": 400,
            "minutes_awake": 20,
            "minutes_after_wakeup": 5,
            "minutes_to_fall_asleep": 10,
            "awakenings_count": 2,
            "efficiency": 95,
            "is_main_sleep": True,
            "session_type": "stages",
            "stages": [
                {
                    "stage_type": "deep",
                    "started_at": "2026-04-05T23:00:00Z",
                    "ended_at": "2026-04-05T23:45:00Z",
                    "seconds": 2700,
                    "sequence_index": 0
            },
            {
                    "stage_type": "rem",
                    "started_at": "2026-04-06T02:00:00Z",
                    "ended_at": "2026-04-06T02:40:00Z",
                    "seconds": 2400,
                    "sequence_index": 1
            }
        ],
        "hrv": 60,
        "resting_hr": 58,
        "spo2": 98,
        "source_payload": {}
    }

        client.analyze_sleep(demo_payload)

        st.success("Demo sleep data generated!")

    if controls["analyze_requested"]:
        _submit_analysis(client, controls["analysis_payload"])

    dashboard_data = _load_dashboard_data(client, controls["selected_date"], controls["trend_days"])
    if dashboard_data is None:
        return

    score = dashboard_data["score"]
    trend = dashboard_data["trend"]
    weekly = dashboard_data["weekly"]
    stages = dashboard_data["stages"]
    recommendations = dashboard_data["recommendations"]
    best_night = dashboard_data["best_night"]
    worst_night = dashboard_data["worst_night"]

    score_col, summary_col = st.columns([1, 2])
    with score_col:
        _render_score_gauge(score)
    with summary_col:
        render_score_cards(
            score,
            weekly=weekly,
            best_night=best_night,
            worst_night=worst_night,
        )

    st.markdown("### Trend Charts")
    render_dashboard_charts(
        trend=trend,
        stages=stages,
        score=score,
    )

    detail_col, recommendation_col = st.columns([1, 1])
    with detail_col:
        st.markdown("### Score Breakdown")
        render_component_breakdown(score)

    with recommendation_col:
        st.markdown("### Recommendations")
        if recommendations:
            for item in recommendations:
                title = item.get("title", "Recommendation")
                message = item.get("message", "")
                category = str(item.get("category", "general")).replace("_", " ").title()
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.caption(category)
                    st.write(message)
        else:
            st.info("No recommendations available for the selected date.")


def _load_dashboard_data(
    client: SleepScoreAPIClient,
    selected_date: Any,
    trend_days: int,
) -> dict[str, Any] | None:
    """Load all dashboard data with user-friendly error handling."""

    try:
        with st.spinner("Loading dashboard data..."):
            return {
                "score": client.get_score(selected_date),
                "trend": client.get_trend(days=trend_days),
                "weekly": client.get_weekly_analytics(),
                "stages": client.get_stages(selected_date),
                "recommendations": client.get_recommendations(selected_date),
                "best_night": client.get_best_night(),
                "worst_night": client.get_worst_night(),
            }
    except ValueError as exc:
        st.error(str(exc))
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if status_code == 404:
            st.warning("No sleep data was found for the selected date yet.")
        elif status_code == 401:
            st.error("The backend rejected the request. Check the user ID in the sidebar.")
        else:
            st.error(f"API request failed with status {status_code}.")
    except httpx.HTTPError as exc:
        st.error(f"Could not reach the API at `{client.base_url}`: {exc}")
    return None


def _submit_analysis(client: SleepScoreAPIClient, payload: dict[str, Any]) -> None:
    """Submit nightly analysis data and surface the result."""

    try:
        with st.spinner("Submitting nightly analysis..."):
            result = client.analyze_sleep(payload)
        overall_score = (result.get("score") or {}).get("overall_score")
        if overall_score is not None:
            st.success(f"Night analyzed successfully. Sleep score: {float(overall_score):.1f}")
        else:
            st.success("Night analyzed successfully.")
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            body = exc.response.json()
            detail = body.get("detail", "")
        except Exception:
            detail = exc.response.text
        st.error(f"Analysis request failed ({exc.response.status_code}). {detail}".strip())
    except httpx.HTTPError as exc:
        st.error(f"Could not submit analysis: {exc}")


def _render_score_gauge(score: dict[str, Any] | None) -> None:
    """Render a Plotly gauge for the nightly sleep score."""

    if not score:
        st.info("No score data available yet.")
        return

    value = _safe_float(score.get("overall_score"))
    label = score.get("label") or "Nightly score"

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "/100"},
            title={"text": str(label)},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#264653"},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 60], "color": "#f4a261"},
                    {"range": [60, 75], "color": "#e9c46a"},
                    {"range": [75, 90], "color": "#2a9d8f"},
                    {"range": [90, 100], "color": "#264653"},
                ],
            },
        )
    )
    figure.update_layout(
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        height=320,
    )
    st.plotly_chart(figure, use_container_width=True)


def _safe_float(value: Any) -> float:
    """Convert a scalar-like value to float."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
