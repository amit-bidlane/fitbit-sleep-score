"""Streamlit dashboard page for nightly sleep summaries."""

from __future__ import annotations

from typing import Any

import httpx
import plotly.graph_objects as go
import streamlit as st

from streamlit_app.components.api_client import SleepScoreAPIClient
from streamlit_app.components.charts import render_dashboard_charts
from streamlit_app.components.score_card import render_component_breakdown, render_score_cards
from streamlit_app.components.sidebar import render_sidebar


st.set_page_config(page_title="Dashboard", layout="wide")


def main() -> None:
    """Render the dashboard page."""

    st.title("Dashboard")
    st.caption("A one-page view of the latest nightly score, KPIs, and supporting charts.")

    controls = render_sidebar(
        default_user_id=st.session_state.get("user_id"),
        default_api_url=st.session_state.get("api_url", "http://localhost:8000"),
    )
    st.session_state["api_url"] = controls["api_url"]

    if not controls["user_id"]:
        st.info("Enter a user ID in the sidebar to load dashboard data.")
        return

    client = SleepScoreAPIClient(base_url=controls["api_url"], user_id=controls["user_id"])
    if controls["analyze_requested"]:
        _submit_analysis(client, controls["analysis_payload"])

    data = _load_dashboard_data(client, controls["selected_date"], controls["trend_days"])
    if data is None:
        return

    score_col, metrics_col = st.columns([1, 2])
    with score_col:
        _render_score_gauge(data["score"])
    with metrics_col:
        render_score_cards(
            data["score"],
            weekly=data["weekly"],
            best_night=data["best_night"],
            worst_night=data["worst_night"],
        )

    st.markdown("### Charts")
    render_dashboard_charts(trend=data["trend"], stages=data["stages"], score=data["score"])

    left, right = st.columns(2)
    with left:
        st.markdown("### Score Breakdown")
        render_component_breakdown(data["score"])
    with right:
        st.markdown("### Recommendations")
        if data["recommendations"]:
            for item in data["recommendations"]:
                with st.container(border=True):
                    st.markdown(f"**{item.get('title', 'Recommendation')}**")
                    st.caption(str(item.get("category", "general")).replace("_", " ").title())
                    st.write(item.get("message", ""))
        else:
            st.info("No recommendations available for this date.")


def _load_dashboard_data(client: SleepScoreAPIClient, selected_date: Any, trend_days: int) -> dict[str, Any] | None:
    """Fetch dashboard data from the API."""

    try:
        with st.spinner("Loading dashboard..."):
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
        _show_api_error(exc)
    except httpx.HTTPError as exc:
        st.error(f"Could not reach the API: {exc}")
    return None


def _submit_analysis(client: SleepScoreAPIClient, payload: dict[str, Any]) -> None:
    """Submit nightly analysis input from the shared sidebar."""

    try:
        result = client.analyze_sleep(payload)
        score = (result.get("score") or {}).get("overall_score")
        if score is None:
            st.success("Night analyzed successfully.")
        else:
            st.success(f"Night analyzed successfully. Score: {float(score):.1f}")
    except httpx.HTTPStatusError as exc:
        _show_api_error(exc)
    except httpx.HTTPError as exc:
        st.error(f"Could not submit analysis: {exc}")


def _render_score_gauge(score: dict[str, Any] | None) -> None:
    """Render the dashboard gauge."""

    if not score:
        st.info("No score data available.")
        return

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score.get("overall_score") or 0.0),
            number={"suffix": "/100"},
            title={"text": str(score.get("label") or "Nightly score")},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#264653"},
                "steps": [
                    {"range": [0, 60], "color": "#f4a261"},
                    {"range": [60, 75], "color": "#e9c46a"},
                    {"range": [75, 90], "color": "#2a9d8f"},
                    {"range": [90, 100], "color": "#264653"},
                ],
            },
        )
    )
    figure.update_layout(template="plotly_white", height=320, margin={"l": 20, "r": 20, "t": 70, "b": 20})
    st.plotly_chart(figure, use_container_width=True)


def _show_api_error(exc: httpx.HTTPStatusError) -> None:
    """Display a concise API error message."""

    if exc.response.status_code == 404:
        st.warning("No sleep data was found for the selected date.")
        return
    try:
        detail = exc.response.json().get("detail", "")
    except Exception:
        detail = exc.response.text
    st.error(f"API request failed ({exc.response.status_code}). {detail}".strip())


main()
