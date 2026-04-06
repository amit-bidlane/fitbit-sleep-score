"""Streamlit page for nightly recommendation review."""

from __future__ import annotations

import httpx
import pandas as pd
import streamlit as st

from streamlit_app.components.api_client import SleepScoreAPIClient
from streamlit_app.components.charts import render_recommendation_chart
from streamlit_app.components.sidebar import render_sidebar


st.set_page_config(page_title="Recommendations", layout="wide")


def main() -> None:
    """Render the recommendations page."""

    st.title("Recommendations")
    st.caption("Review personalized sleep guidance returned for the selected night.")

    controls = render_sidebar(
        default_user_id=st.session_state.get("user_id"),
        default_api_url=st.session_state.get("api_url", "http://localhost:8000"),
    )
    st.session_state["api_url"] = controls["api_url"]

    if not controls["user_id"]:
        st.info("Enter a user ID in the sidebar to load recommendations.")
        return

    client = SleepScoreAPIClient(base_url=controls["api_url"], user_id=controls["user_id"])
    try:
        recommendations = client.get_recommendations(controls["selected_date"])
        score = client.get_score(controls["selected_date"])
    except httpx.HTTPStatusError as exc:
        _show_api_error(exc)
        return
    except httpx.HTTPError as exc:
        st.error(f"Could not reach the API: {exc}")
        return

    header_left, header_right = st.columns([1, 1])
    with header_left:
        st.metric("Night Score", f"{float(score.get('overall_score') or 0.0):.1f}", delta=score.get("label"))
    with header_right:
        st.metric("Recommendation Count", str(len(recommendations)))

    render_recommendation_chart(recommendations)
    if recommendations:
        st.markdown("### Recommendation Details")
        for item in recommendations:
            with st.container(border=True):
                st.markdown(f"**{item.get('title', 'Recommendation')}**")
                st.caption(str(item.get("category", "general")).replace("_", " ").title())
                st.write(item.get("message", ""))
                action_items = item.get("action_items") or []
                if action_items:
                    st.write("Actions: " + ", ".join(str(action) for action in action_items))
        st.markdown("### Raw Table")
        st.dataframe(pd.DataFrame(recommendations), use_container_width=True, hide_index=True)
    else:
        st.info("No recommendations available for this date.")


def _show_api_error(exc: httpx.HTTPStatusError) -> None:
    """Display API errors."""

    if exc.response.status_code == 404:
        st.warning("No recommendations were found for the selected date.")
        return
    try:
        detail = exc.response.json().get("detail", "")
    except Exception:
        detail = exc.response.text
    st.error(f"API request failed ({exc.response.status_code}). {detail}".strip())


main()
