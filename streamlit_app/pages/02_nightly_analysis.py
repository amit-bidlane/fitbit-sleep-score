"""Streamlit page for submitting nightly sleep analysis payloads."""

from __future__ import annotations

from typing import Any

import httpx
import pandas as pd
import streamlit as st

from streamlit_app.components.api_client import SleepScoreAPIClient
from streamlit_app.components.score_card import render_component_breakdown
from streamlit_app.components.sidebar import render_analysis_inputs, render_sidebar


st.set_page_config(page_title="Nightly Analysis", layout="wide")


def main() -> None:
    """Render the nightly analysis page."""

    st.title("Nightly Analysis")
    st.caption("Submit one night of sleep inputs to generate a fresh score and persisted recommendations.")

    controls = render_sidebar(
        default_user_id=st.session_state.get("user_id"),
        default_api_url=st.session_state.get("api_url", "http://localhost:8000"),
    )
    st.session_state["api_url"] = controls["api_url"]

    if not controls["user_id"]:
        st.info("Enter a user ID in the sidebar before submitting analysis.")
        return

    client = SleepScoreAPIClient(base_url=controls["api_url"], user_id=controls["user_id"])

    input_col, result_col = st.columns([1, 1])
    with input_col:
        st.markdown("### Analysis Payload")
        payload = render_analysis_inputs(default_date=controls["selected_date"], key_prefix="page_analysis")
        payload = _render_stage_editor(payload)
        submit = st.button("Run Analysis", type="primary", use_container_width=True)

    with result_col:
        st.markdown("### Last Result")
        result = st.session_state.get("nightly_analysis_result")
        if result:
            _render_result(result)
        else:
            st.info("Submit a payload to see the analyzed score and returned stages.")

    if submit:
        _submit_analysis(client, payload)
        st.rerun()


def _render_stage_editor(payload: dict[str, Any]) -> dict[str, Any]:
    """Allow optional stage rows to be added with a simple editor."""

    st.markdown("### Optional Sleep Stages")
    default_stages = pd.DataFrame(
        [
            {"stage_type": "light", "started_at": f"{payload['date']}T22:00:00Z", "seconds": 1800},
            {"stage_type": "deep", "started_at": f"{payload['date']}T22:30:00Z", "seconds": 3600},
            {"stage_type": "rem", "started_at": f"{payload['date']}T23:30:00Z", "seconds": 1800},
        ]
    )
    edited = st.data_editor(
        default_stages,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "stage_type": st.column_config.SelectboxColumn(
                "Stage",
                options=["wake", "light", "deep", "rem", "asleep", "restless"],
                required=True,
            ),
            "started_at": st.column_config.TextColumn("Started At", required=True),
            "seconds": st.column_config.NumberColumn("Seconds", min_value=0, step=60, required=True),
        },
        key="analysis_stage_editor",
    )
    stages = []
    for index, row in edited.fillna("").iterrows():
        stage_type = str(row.get("stage_type", "")).strip().lower()
        started_at = str(row.get("started_at", "")).strip()
        seconds = row.get("seconds", "")
        if not stage_type or not started_at or seconds == "":
            continue
        stages.append(
            {
                "stage_type": stage_type,
                "started_at": started_at,
                "seconds": int(float(seconds)),
                "sequence_index": int(index),
            }
        )
    payload["stages"] = stages
    return payload


def _submit_analysis(client: SleepScoreAPIClient, payload: dict[str, Any]) -> None:
    """Submit the analysis payload and store the response."""

    try:
        with st.spinner("Submitting analysis..."):
            result = client.analyze_sleep(payload)
        st.session_state["nightly_analysis_result"] = result
        score = (result.get("score") or {}).get("overall_score")
        if score is None:
            st.success("Night analyzed successfully.")
        else:
            st.success(f"Night analyzed successfully. Score: {float(score):.1f}")
    except httpx.HTTPStatusError as exc:
        _show_api_error(exc)
    except httpx.HTTPError as exc:
        st.error(f"Could not submit analysis: {exc}")


def _render_result(result: dict[str, Any]) -> None:
    """Render the latest analysis result."""

    score = result.get("score") or {}
    render_component_breakdown(score)
    if result.get("stages"):
        st.markdown("### Returned Stages")
        st.dataframe(pd.DataFrame(result["stages"]), use_container_width=True, hide_index=True)


def _show_api_error(exc: httpx.HTTPStatusError) -> None:
    """Display API errors."""

    try:
        detail = exc.response.json().get("detail", "")
    except Exception:
        detail = exc.response.text
    st.error(f"API request failed ({exc.response.status_code}). {detail}".strip())


main()
