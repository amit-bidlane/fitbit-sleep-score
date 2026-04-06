"""Reusable Streamlit sidebar controls for the sleep score app."""

from __future__ import annotations

from datetime import date, time, timedelta
from typing import Any

import streamlit as st


DEFAULT_ANALYSIS_VALUES = {
    "time_in_bed": 480,
    "minutes_asleep": 450,
    "minutes_awake": 30,
    "minutes_after_wakeup": 5,
    "minutes_to_fall_asleep": 10,
    "awakenings_count": 1,
    "efficiency": 93.8,
    "hrv": 55.0,
    "resting_hr": 57.0,
    "spo2": 97.0,
}


def render_sidebar(
    *,
    default_user_id: int | str | None = None,
    default_api_url: str = "http://localhost:8000",
    default_days: int = 14,
) -> dict[str, Any]:
    """Render the shared sidebar and return the current selections."""

    today = date.today()
    with st.sidebar:
        st.markdown("## Sleep Controls")
        api_url = st.text_input("API base URL", value=default_api_url, help="Base URL for the FastAPI backend.")
        user_id = st.text_input(
            "User ID",
            value="" if default_user_id is None else str(default_user_id),
            help="Sent in the X-User-Id header for authenticated API requests.",
        )

        selected_date = st.date_input(
            "Sleep date",
            value=today,
            max_value=today,
            help="Choose the night you want to inspect.",
        )
        trend_days = st.slider(
            "Trend window",
            min_value=7,
            max_value=90,
            value=max(7, min(default_days, 90)),
            step=1,
            help="Controls how many days are requested for trend charts.",
        )

        st.markdown("## Quick Actions")
        refresh_requested = st.button("Refresh Data", use_container_width=True)
        analyze_requested = st.button("Analyze Night", use_container_width=True, type="primary")

        with st.expander("Nightly Analysis Input", expanded=False):
            analysis_payload = render_analysis_inputs(default_date=selected_date, key_prefix="sidebar_analysis")

        st.caption("Use the quick actions above to fetch dashboard data or submit a new analysis.")

    state = {
        "api_url": api_url.rstrip("/"),
        "user_id": user_id.strip(),
        "selected_date": selected_date,
        "trend_days": trend_days,
        "refresh_requested": refresh_requested,
        "analyze_requested": analyze_requested,
        "analysis_payload": analysis_payload,
    }
    if state["user_id"]:
        st.session_state["user_id"] = state["user_id"]
    return state


def render_analysis_inputs(*, default_date: date | None = None, key_prefix: str = "analysis") -> dict[str, Any]:
    """Render a compact nightly analysis form inside the sidebar."""

    sleep_date = default_date or date.today()
    start_date = sleep_date
    end_date = sleep_date + timedelta(days=1)

    start_col, end_col = st.columns(2)
    with start_col:
        start_time = st.time_input("Start time", value=_time_value(22, 0), key=f"{key_prefix}_start_time")
    with end_col:
        end_time = st.time_input("End time", value=_time_value(6, 0), key=f"{key_prefix}_end_time")

    numeric_left, numeric_right = st.columns(2)
    with numeric_left:
        time_in_bed = st.number_input("Time in bed", min_value=0, value=DEFAULT_ANALYSIS_VALUES["time_in_bed"], step=5, key=f"{key_prefix}_time_in_bed")
        minutes_asleep = st.number_input("Minutes asleep", min_value=0, value=DEFAULT_ANALYSIS_VALUES["minutes_asleep"], step=5, key=f"{key_prefix}_minutes_asleep")
        minutes_awake = st.number_input("Minutes awake", min_value=0, value=DEFAULT_ANALYSIS_VALUES["minutes_awake"], step=5, key=f"{key_prefix}_minutes_awake")
        latency = st.number_input(
            "Sleep latency",
            min_value=0,
            value=DEFAULT_ANALYSIS_VALUES["minutes_to_fall_asleep"],
            step=1,
            key=f"{key_prefix}_latency",
        )
        awakenings_count = st.number_input(
            "Awakenings",
            min_value=0,
            value=DEFAULT_ANALYSIS_VALUES["awakenings_count"],
            step=1,
            key=f"{key_prefix}_awakenings_count",
        )
    with numeric_right:
        minutes_after_wakeup = st.number_input(
            "After wakeup",
            min_value=0,
            value=DEFAULT_ANALYSIS_VALUES["minutes_after_wakeup"],
            step=1,
            key=f"{key_prefix}_minutes_after_wakeup",
        )
        efficiency = st.number_input(
            "Efficiency %",
            min_value=0.0,
            max_value=100.0,
            value=DEFAULT_ANALYSIS_VALUES["efficiency"],
            step=0.1,
            key=f"{key_prefix}_efficiency",
        )
        hrv = st.number_input("HRV", min_value=0.0, value=DEFAULT_ANALYSIS_VALUES["hrv"], step=0.1, key=f"{key_prefix}_hrv")
        resting_hr = st.number_input("Resting HR", min_value=0.0, value=DEFAULT_ANALYSIS_VALUES["resting_hr"], step=0.1, key=f"{key_prefix}_resting_hr")
        spo2 = st.number_input("SpO2", min_value=0.0, max_value=100.0, value=DEFAULT_ANALYSIS_VALUES["spo2"], step=0.1, key=f"{key_prefix}_spo2")

    return {
        "date": sleep_date.isoformat(),
        "start_time": f"{start_date.isoformat()}T{start_time.strftime('%H:%M:%S')}",
        "end_time": f"{end_date.isoformat()}T{end_time.strftime('%H:%M:%S')}",
        "time_in_bed": int(time_in_bed),
        "minutes_asleep": int(minutes_asleep),
        "minutes_awake": int(minutes_awake),
        "minutes_after_wakeup": int(minutes_after_wakeup),
        "minutes_to_fall_asleep": int(latency),
        "awakenings_count": int(awakenings_count),
        "efficiency": float(efficiency),
        "hrv": float(hrv),
        "resting_hr": float(resting_hr),
        "spo2": float(spo2),
        "stages": [],
    }


def _time_value(hour: int, minute: int):
    """Return a stable time object without importing datetime.time directly."""

    return time(hour=hour, minute=minute)
