"""Reusable KPI cards for Streamlit sleep dashboards."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def render_score_cards(
    score: dict[str, Any] | None,
    *,
    weekly: dict[str, Any] | None = None,
    best_night: dict[str, Any] | None = None,
    worst_night: dict[str, Any] | None = None,
) -> None:
    """Render the main dashboard metric cards."""

    current = score or {}
    metrics = [
        {
            "label": "Sleep Score",
            "value": _format_value(current.get("overall_score")),
            "delta": _delta_text(current.get("overall_score"), weekly.get("average_score") if weekly else None, suffix=" vs avg"),
            "help": current.get("label") or "Nightly overall sleep score.",
        },
        {
            "label": "Duration",
            "value": _format_duration(_feature_value(current, "total_sleep_minutes")),
            "delta": None,
            "help": "Estimated minutes asleep converted to hours and minutes.",
        },
        {
            "label": "Continuity",
            "value": _format_value(current.get("continuity_score")),
            "delta": _delta_text(current.get("continuity_score"), weekly.get("average_continuity_score") if weekly else None),
            "help": "How uninterrupted the night was.",
        },
        {
            "label": "Recovery",
            "value": _format_value(current.get("recovery_score")),
            "delta": _delta_text(current.get("recovery_score"), weekly.get("average_recovery_score") if weekly else None),
            "help": "Recovery quality inferred from the score model.",
        },
    ]

    cols = st.columns(len(metrics))
    for column, metric in zip(cols, metrics, strict=False):
        with column:
            st.metric(
                label=metric["label"],
                value=metric["value"],
                delta=metric["delta"],
                help=metric["help"],
                border=True,
            )

    _render_night_summaries(best_night=best_night, worst_night=worst_night)


def render_component_breakdown(score: dict[str, Any] | None) -> None:
    """Render a compact table of component scores."""

    if not score:
        st.info("No score details available yet.")
        return

    breakdown = pd.DataFrame(
        [
            {"Metric": "Overall", "Value": score.get("overall_score")},
            {"Metric": "Duration", "Value": score.get("duration_score")},
            {"Metric": "Sleep Efficiency", "Value": score.get("sleep_efficiency_score")},
            {"Metric": "Continuity", "Value": score.get("continuity_score")},
            {"Metric": "Recovery", "Value": score.get("recovery_score")},
        ]
    )
    st.dataframe(
        breakdown.assign(Value=lambda frame: frame["Value"].map(_format_value)),
        use_container_width=True,
        hide_index=True,
    )


def _render_night_summaries(
    *,
    best_night: dict[str, Any] | None = None,
    worst_night: dict[str, Any] | None = None,
) -> None:
    """Render best and worst night callouts."""

    best = best_night or {}
    worst = worst_night or {}
    left, right = st.columns(2)

    with left:
        st.markdown("#### Best Night")
        if best:
            st.metric(
                label=str(best.get("date", "Date")),
                value=_format_value(best.get("overall_score")),
                delta=best.get("label"),
                border=True,
            )
        else:
            st.info("Best night data unavailable.")

    with right:
        st.markdown("#### Worst Night")
        if worst:
            st.metric(
                label=str(worst.get("date", "Date")),
                value=_format_value(worst.get("overall_score")),
                delta=worst.get("label"),
                border=True,
            )
        else:
            st.info("Worst night data unavailable.")


def _feature_value(score: dict[str, Any], key: str) -> Any:
    """Return a nested feature snapshot value when present."""

    snapshot = score.get("feature_snapshot") or {}
    if isinstance(snapshot, dict):
        return snapshot.get(key)
    return None


def _format_duration(minutes: Any) -> str:
    """Format a duration in minutes as hours and minutes."""

    numeric = _to_float(minutes)
    if numeric is None:
        return "N/A"
    hours = int(numeric // 60)
    remainder = int(round(numeric % 60))
    return f"{hours}h {remainder}m"


def _delta_text(current: Any, baseline: Any, suffix: str = "") -> str | None:
    """Format a metric delta string when both values are present."""

    current_value = _to_float(current)
    baseline_value = _to_float(baseline)
    if current_value is None or baseline_value is None:
        return None
    delta = current_value - baseline_value
    return f"{delta:+.1f}{suffix}"


def _format_value(value: Any) -> str:
    """Format numeric dashboard values."""

    numeric = _to_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:.1f}"


def _to_float(value: Any) -> float | None:
    """Convert scalar-like values to float when possible."""

    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
