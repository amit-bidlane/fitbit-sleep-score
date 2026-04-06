"""Streamlit page for long-window sleep score trends."""

from __future__ import annotations

import httpx
import pandas as pd
import streamlit as st

from streamlit_app.components.api_client import SleepScoreAPIClient
from streamlit_app.components.charts import render_dashboard_charts
from streamlit_app.components.sidebar import render_sidebar


st.set_page_config(page_title="Trends", layout="wide")


def main() -> None:
    """Render the trends page."""

    st.title("Trends")
    st.caption("Explore rolling score history and supporting analytics over a wider time range.")

    controls = render_sidebar(
        default_user_id=st.session_state.get("user_id"),
        default_api_url=st.session_state.get("api_url", "http://localhost:8000"),
        default_days=30,
    )
    st.session_state["api_url"] = controls["api_url"]

    if not controls["user_id"]:
        st.info("Enter a user ID in the sidebar to explore trend data.")
        return

    client = SleepScoreAPIClient(base_url=controls["api_url"], user_id=controls["user_id"])
    try:
        trend = client.get_trend(days=controls["trend_days"])
    except httpx.HTTPStatusError as exc:
        _show_api_error(exc)
        return
    except httpx.HTTPError as exc:
        st.error(f"Could not reach the API: {exc}")
        return

    points = trend.get("points", [])
    if not points:
        st.info("No trend points are available yet.")
        return

    frame = pd.DataFrame(points)
    latest = frame.iloc[-1]
    avg_score = pd.to_numeric(frame["overall_score"], errors="coerce").mean()
    max_score = pd.to_numeric(frame["overall_score"], errors="coerce").max()

    first, second, third = st.columns(3)
    with first:
        st.metric("Latest Score", f"{float(latest['overall_score']):.1f}")
    with second:
        st.metric("Average Score", f"{float(avg_score):.1f}")
    with third:
        st.metric("Best Score", f"{float(max_score):.1f}")

    render_dashboard_charts(trend=trend, score=points[-1], stages=None)
    st.markdown("### Trend Data")
    st.dataframe(frame, use_container_width=True, hide_index=True)


def _show_api_error(exc: httpx.HTTPStatusError) -> None:
    """Display API errors."""

    if exc.response.status_code == 404:
        st.warning("No trend data is available yet.")
        return
    try:
        detail = exc.response.json().get("detail", "")
    except Exception:
        detail = exc.response.text
    st.error(f"API request failed ({exc.response.status_code}). {detail}".strip())


main()
