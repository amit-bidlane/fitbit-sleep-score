"""Reusable Plotly charts for Fitbit sleep analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


STAGE_ORDER = ["wake", "light", "deep", "rem", "asleep", "restless"]
STAGE_HEIGHTS = {
    "wake": 4,
    "light": 3,
    "deep": 2,
    "rem": 1,
    "asleep": 2.5,
    "restless": 3.5,
}
DEFAULT_COLORS = {
    "score": "#1f77b4",
    "heart_rate": "#d62728",
    "deep": "#264653",
    "light": "#2a9d8f",
    "rem": "#e9c46a",
    "wake": "#e76f51",
    "asleep": "#577590",
    "restless": "#f4a261",
    "background": "#f8f9fb",
    "grid": "#d8deea",
}


@dataclass(slots=True)
class SleepDashboard:
    """Build sleep dashboard visualizations from nightly Fitbit-style data."""

    colors: dict[str, str] = field(default_factory=lambda: DEFAULT_COLORS.copy())

    def create_sleep_score_trend(self, sleep_scores: Any) -> go.Figure:
        """Plot nightly sleep score trend with an adaptive rolling average."""

        frame = self._prepare_frame(sleep_scores)
        if frame.empty:
            return self._empty_figure("Sleep score trend")

        score_column = self._select_column(frame, ("final_score", "overall_score", "sleep_score", "score"))
        date_column = self._select_column(frame, ("score_date", "dateOfSleep", "session_date", "date"))
        frame = self._normalize_dates(frame, date_column).sort_values(date_column)
        scores = pd.to_numeric(frame[score_column], errors="coerce")
        rolling = scores.rolling(window=min(7, max(len(frame), 1)), min_periods=1).mean()

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=frame[date_column],
                y=scores,
                mode="lines+markers",
                name="Sleep score",
                line={"color": self.colors["score"], "width": 3},
                marker={"size": 8},
                hovertemplate="%{x|%b %d, %Y}<br>Score: %{y:.1f}<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=frame[date_column],
                y=rolling,
                mode="lines",
                name="Rolling average",
                line={"color": "#6c757d", "dash": "dash", "width": 2},
                hovertemplate="%{x|%b %d, %Y}<br>Avg: %{y:.1f}<extra></extra>",
            )
        )
        figure.update_layout(
            title="Sleep Score Trend",
            xaxis_title="Night",
            yaxis_title="Score",
            yaxis={"range": [0, 100]},
        )
        return self._apply_layout(figure)

    def create_sleep_hypnogram(self, sleep_stages: Any) -> go.Figure:
        """Render a nightly hypnogram from ordered sleep stages."""

        frame = self._prepare_frame(sleep_stages)
        if frame.empty:
            return self._empty_figure("Sleep hypnogram")

        frame = self._prepare_stage_frame(frame)
        figure = go.Figure()

        for stage_name, stage_frame in frame.groupby("stage_type", sort=False):
            x_values: list[pd.Timestamp | None] = []
            y_values: list[float | None] = []
            custom_data: list[tuple[str, float]] = []

            for row in stage_frame.itertuples():
                x_values.extend([row.started_at, row.ended_at, None])
                y_values.extend(
                    [
                        STAGE_HEIGHTS.get(stage_name, 0),
                        STAGE_HEIGHTS.get(stage_name, 0),
                        None,
                    ]
                )
                duration_minutes = max(row.seconds, 0) / 60.0
                custom_data.extend(
                    [
                        (stage_name.title(), duration_minutes),
                        (stage_name.title(), duration_minutes),
                        ("", 0.0),
                    ]
                )

            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines",
                    name=stage_name.title(),
                    line={"width": 18, "color": self.colors.get(stage_name, "#6c757d"), "shape": "hv"},
                    customdata=custom_data,
                    hovertemplate="%{customdata[0]}<br>Duration: %{customdata[1]:.1f} min<extra></extra>",
                )
            )

        stage_names = [name for name in STAGE_ORDER if name in frame["stage_type"].unique()]
        figure.update_layout(
            title="Sleep Hypnogram",
            xaxis_title="Time",
            yaxis={
                "title": "Stage",
                "tickmode": "array",
                "tickvals": [STAGE_HEIGHTS[name] for name in stage_names],
                "ticktext": [name.title() for name in stage_names],
                "autorange": "reversed",
            },
            showlegend=True,
        )
        return self._apply_layout(figure)

    def create_heart_rate_overlay(self, sleep_stages: Any, heart_rate_data: Any) -> go.Figure:
        """Overlay heart rate on top of a hypnogram using dual y-axes."""

        stage_frame = self._prepare_frame(sleep_stages)
        hr_frame = self._prepare_frame(heart_rate_data)
        if stage_frame.empty and hr_frame.empty:
            return self._empty_figure("Heart rate overlay")

        figure = make_subplots(specs=[[{"secondary_y": True}]])
        prepared_stages = self._prepare_stage_frame(stage_frame) if not stage_frame.empty else pd.DataFrame()

        if not prepared_stages.empty:
            for stage_name, group in prepared_stages.groupby("stage_type", sort=False):
                x_values: list[pd.Timestamp | None] = []
                y_values: list[float | None] = []
                for row in group.itertuples():
                    x_values.extend([row.started_at, row.ended_at, None])
                    y_values.extend([STAGE_HEIGHTS.get(stage_name, 0), STAGE_HEIGHTS.get(stage_name, 0), None])

                figure.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode="lines",
                        name=f"{stage_name.title()} stage",
                        line={"width": 14, "color": self.colors.get(stage_name, "#6c757d"), "shape": "hv"},
                        hovertemplate=f"{stage_name.title()}<extra></extra>",
                    ),
                    secondary_y=False,
                )

        if not hr_frame.empty:
            hr_time_column = self._select_column(hr_frame, ("timestamp", "dateTime", "time", "datetime"))
            hr_value_column = self._select_column(hr_frame, ("heart_rate", "value", "bpm", "beatsPerMinute"))
            hr_frame = self._normalize_dates(hr_frame, hr_time_column).sort_values(hr_time_column)
            hr_values = pd.to_numeric(hr_frame[hr_value_column], errors="coerce")
            figure.add_trace(
                go.Scatter(
                    x=hr_frame[hr_time_column],
                    y=hr_values,
                    mode="lines+markers",
                    name="Heart rate",
                    line={"color": self.colors["heart_rate"], "width": 3},
                    marker={"size": 5},
                    hovertemplate="%{x|%I:%M %p}<br>Heart rate: %{y:.0f} bpm<extra></extra>",
                ),
                secondary_y=True,
            )

        stage_names = [name for name in STAGE_ORDER if not prepared_stages.empty and name in prepared_stages["stage_type"].unique()]
        figure.update_layout(title="Heart Rate Overlay")
        figure.update_xaxes(title_text="Time")
        figure.update_yaxes(
            title_text="Sleep stage",
            tickmode="array",
            tickvals=[STAGE_HEIGHTS[name] for name in stage_names],
            ticktext=[name.title() for name in stage_names],
            autorange="reversed",
            secondary_y=False,
        )
        figure.update_yaxes(title_text="Heart rate (bpm)", secondary_y=True)
        return self._apply_layout(figure)

    def create_score_radar_chart(self, score_components: Any) -> go.Figure:
        """Plot score components on a radar chart."""

        frame = self._prepare_frame(score_components)
        if frame.empty:
            return self._empty_figure("Score radar chart")

        first_row = frame.iloc[0]
        component_map = {
            "Duration": self._safe_float(first_row.get("duration_score")),
            "Deep": self._safe_float(first_row.get("deep_score")),
            "REM": self._safe_float(first_row.get("rem_score")),
            "Continuity": self._safe_float(first_row.get("continuity_score")),
            "Recovery": self._safe_float(first_row.get("recovery_score")),
        }

        figure = go.Figure(
            data=[
                go.Scatterpolar(
                    r=list(component_map.values()) + [next(iter(component_map.values()))],
                    theta=list(component_map.keys()) + [next(iter(component_map.keys()))],
                    fill="toself",
                    name="Component score",
                    line={"color": self.colors["score"], "width": 3},
                    fillcolor="rgba(31, 119, 180, 0.25)",
                    hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
                )
            ]
        )
        figure.update_layout(
            title="Sleep Score Radar",
            polar={"radialaxis": {"visible": True, "range": [0, 100]}},
        )
        return self._apply_layout(figure)

    def create_weekly_comparison_chart(self, sleep_scores: Any) -> go.Figure:
        """Compare weekly average component scores."""

        frame = self._prepare_frame(sleep_scores)
        if frame.empty:
            return self._empty_figure("Weekly comparison chart")

        date_column = self._select_column(frame, ("score_date", "dateOfSleep", "session_date", "date"))
        frame = self._normalize_dates(frame, date_column)

        value_columns = {
            "Overall": self._select_optional_column(frame, ("final_score", "overall_score", "sleep_score", "score")),
            "Duration": self._select_optional_column(frame, ("duration_score",)),
            "Continuity": self._select_optional_column(frame, ("continuity_score",)),
            "Recovery": self._select_optional_column(frame, ("recovery_score",)),
        }
        selected = {label: column for label, column in value_columns.items() if column is not None}
        if not selected:
            return self._empty_figure("Weekly comparison chart")

        weekly = pd.DataFrame({"week_start": frame[date_column].dt.to_period("W-MON").dt.start_time})
        for label, column in selected.items():
            weekly[label] = pd.to_numeric(frame[column], errors="coerce")
        aggregated = weekly.groupby("week_start", as_index=False).mean(numeric_only=True)

        figure = go.Figure()
        for label in selected:
            figure.add_trace(
                go.Bar(
                    x=aggregated["week_start"],
                    y=aggregated[label],
                    name=label,
                    hovertemplate="%{x|Week of %b %d, %Y}<br>" + f"{label}: " + "%{y:.1f}<extra></extra>",
                )
            )

        figure.update_layout(
            title="Weekly Comparison",
            barmode="group",
            xaxis_title="Week",
            yaxis_title="Average score",
            yaxis={"range": [0, 100]},
        )
        return self._apply_layout(figure)

    def create_dashboard_figures(
        self,
        *,
        sleep_scores: Any,
        sleep_stages: Any | None = None,
        heart_rate_data: Any | None = None,
        score_components: Any | None = None,
    ) -> dict[str, go.Figure]:
        """Build the full set of dashboard figures."""

        score_input = score_components if score_components is not None else sleep_scores
        return {
            "sleep_score_trend": self.create_sleep_score_trend(sleep_scores),
            "sleep_hypnogram": self.create_sleep_hypnogram(sleep_stages),
            "heart_rate_overlay": self.create_heart_rate_overlay(sleep_stages, heart_rate_data),
            "score_radar_chart": self.create_score_radar_chart(score_input),
            "weekly_comparison_chart": self.create_weekly_comparison_chart(sleep_scores),
        }

    def _prepare_frame(self, data: Any | None) -> pd.DataFrame:
        """Normalize supported chart inputs to a DataFrame."""

        if data is None:
            return pd.DataFrame()
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, dict):
            return pd.json_normalize([data], sep="_")
        if isinstance(data, list):
            return pd.json_normalize(data, sep="_")
        raise TypeError("Visualization data must be a DataFrame, dict, list, or None.")

    def _prepare_stage_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize sleep stage inputs and ensure stage ordering."""

        stage_column = self._select_column(frame, ("stage_type", "level", "stage"))
        start_column = self._select_column(frame, ("started_at", "startTime", "dateTime", "start_time"))
        end_column = self._select_optional_column(frame, ("ended_at", "endTime", "end_time"))
        seconds_column = self._select_optional_column(frame, ("seconds", "duration", "duration_seconds"))

        normalized = frame.copy()
        normalized["stage_type"] = normalized[stage_column].astype(str).str.lower()
        normalized["started_at"] = pd.to_datetime(normalized[start_column], errors="coerce")

        if end_column is not None:
            normalized["ended_at"] = pd.to_datetime(normalized[end_column], errors="coerce")
        elif seconds_column is not None:
            seconds = pd.to_numeric(normalized[seconds_column], errors="coerce").fillna(0.0)
            normalized["ended_at"] = normalized["started_at"] + pd.to_timedelta(seconds, unit="s")
        else:
            raise ValueError("Sleep stage data must include either an end timestamp or duration seconds.")

        if seconds_column is not None:
            normalized["seconds"] = pd.to_numeric(normalized[seconds_column], errors="coerce").fillna(0.0)
        else:
            normalized["seconds"] = (
                normalized["ended_at"] - normalized["started_at"]
            ).dt.total_seconds().fillna(0.0)

        if "sequence_index" not in normalized.columns:
            normalized["sequence_index"] = range(len(normalized))

        return normalized.sort_values(["started_at", "sequence_index"]).reset_index(drop=True)

    def _normalize_dates(self, frame: pd.DataFrame, column: str) -> pd.DataFrame:
        """Convert a date-like column to pandas datetimes."""

        normalized = frame.copy()
        normalized[column] = pd.to_datetime(normalized[column], errors="coerce")
        return normalized.dropna(subset=[column])

    def _select_column(self, frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
        """Return the first matching required column."""

        column = self._select_optional_column(frame, candidates)
        if column is None:
            raise ValueError(f"Expected one of columns {candidates}, found {tuple(frame.columns)}.")
        return column

    def _select_optional_column(self, frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
        """Return the first matching optional column."""

        return next((column for column in candidates if column in frame.columns), None)

    def _apply_layout(self, figure: go.Figure) -> go.Figure:
        """Apply the shared dashboard theme."""

        figure.update_layout(
            template="plotly_white",
            paper_bgcolor=self.colors["background"],
            plot_bgcolor="white",
            font={"family": "Arial, sans-serif", "color": "#1f2933"},
            margin={"l": 50, "r": 30, "t": 70, "b": 50},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        figure.update_xaxes(showgrid=True, gridcolor=self.colors["grid"])
        figure.update_yaxes(showgrid=True, gridcolor=self.colors["grid"])
        return figure

    def _empty_figure(self, title: str) -> go.Figure:
        """Create a chart placeholder for empty inputs."""

        figure = go.Figure()
        figure.update_layout(title=title)
        figure.add_annotation(
            text="No data available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 16, "color": "#6c757d"},
        )
        return self._apply_layout(figure)

    def _safe_float(self, value: Any) -> float:
        """Convert scalar-like values to float."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


__all__ = ["SleepDashboard"]
