"""Streamlit page for model metadata and backend availability."""

from __future__ import annotations

import httpx
import pandas as pd
import streamlit as st

from streamlit_app.components.api_client import SleepScoreAPIClient
from streamlit_app.components.sidebar import render_sidebar


st.set_page_config(page_title="Model Info", layout="wide")


def main() -> None:
    """Render the model info page."""

    st.title("Model Info")
    st.caption("Inspect the active model metadata currently exposed by the backend.")

    controls = render_sidebar(
        default_user_id=st.session_state.get("user_id"),
        default_api_url=st.session_state.get("api_url", "http://localhost:8000"),
    )
    st.session_state["api_url"] = controls["api_url"]

    if not controls["user_id"]:
        st.info("Enter a user ID in the sidebar to query model metadata.")
        return

    client = SleepScoreAPIClient(base_url=controls["api_url"], user_id=controls["user_id"])
    try:
        payload = client.get_active_models()
    except httpx.HTTPStatusError as exc:
        _show_api_error(exc)
        return
    except httpx.HTTPError as exc:
        st.error(f"Could not reach the API: {exc}")
        return

    models = payload.get("models", []) if isinstance(payload, dict) else []

    if models:
        st.success("Active model metadata was returned by the backend.")
        st.dataframe(pd.DataFrame(models), use_container_width=True, hide_index=True)
    else:
        st.info("No active models are currently marked active in the backend.")
        st.markdown(
            "\n".join(
                [
                    "- Endpoint: `/models/active`",
                    "- Source table: `model_versions`",
                    "- Condition: `is_active = true`",
                ]
            )
        )

    st.markdown("### Raw Payload")
    st.json(payload)


def _show_api_error(exc: httpx.HTTPStatusError) -> None:
    """Display API errors."""

    try:
        detail = exc.response.json().get("detail", "")
    except Exception:
        detail = exc.response.text
    st.error(f"API request failed ({exc.response.status_code}). {detail}".strip())


main()
