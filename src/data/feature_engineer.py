"""Feature engineering for nightly Fitbit sleep records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import numpy as np


@dataclass(slots=True)
class SleepFeatureEngineer:
    """Create nightly model features from Fitbit sleep and biometrics data."""

    continuity_awake_weight: float = 0.6
    continuity_awakenings_weight: float = 6.0
    continuity_latency_weight: float = 0.5

    def engineer(
        self,
        sleep_data: Any,
        hrv_data: pd.DataFrame | None = None,
        heart_rate_data: pd.DataFrame | None = None,
        spo2_data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build a feature set for each sleep record."""

        sleep_frame = self._to_frame(sleep_data)
        if sleep_frame.empty:
            return sleep_frame

        sleep_frame = sleep_frame.copy()
        if "dateOfSleep" in sleep_frame.columns:
            sleep_frame["dateOfSleep"] = pd.to_datetime(
                sleep_frame["dateOfSleep"], errors="coerce"
            ).dt.normalize()

        features = pd.DataFrame(index=sleep_frame.index)
        features["logId"] = sleep_frame.get("logId")
        features["dateOfSleep"] = sleep_frame.get("dateOfSleep")

        total_sleep_minutes = self._coalesce_numeric(
            sleep_frame,
            ("raw_minutesAsleep", "minutesAsleep", "totalMinutesAsleep"),
        )
        deep_minutes = self._coalesce_numeric(
            sleep_frame,
            (
                "raw_levels_summary_deep_minutes",
                "levels_summary_deep_minutes",
                "raw_deepSleepMinutes",
                "deepSleepMinutes",
            ),
        )
        rem_minutes = self._coalesce_numeric(
            sleep_frame,
            (
                "raw_levels_summary_rem_minutes",
                "levels_summary_rem_minutes",
                "raw_remSleepMinutes",
                "remSleepMinutes",
            ),
        )
        light_minutes = self._coalesce_numeric(
            sleep_frame,
            (
                "raw_levels_summary_light_minutes",
                "levels_summary_light_minutes",
                "raw_lightSleepMinutes",
                "lightSleepMinutes",
            ),
        )
        awake_minutes = self._coalesce_numeric(
            sleep_frame,
            (
                "raw_levels_summary_wake_minutes",
                "levels_summary_wake_minutes",
                "raw_minutesAwake",
                "minutesAwake",
            ),
        )
        time_in_bed = self._coalesce_numeric(
            sleep_frame,
            ("raw_timeInBed", "timeInBed", "raw_durationMinutes", "durationMinutes"),
        )

        features["total_sleep_minutes"] = total_sleep_minutes
        features["deep_sleep_pct"] = self._safe_percent(deep_minutes, total_sleep_minutes)
        features["rem_sleep_pct"] = self._safe_percent(rem_minutes, total_sleep_minutes)
        features["light_sleep_pct"] = self._safe_percent(light_minutes, total_sleep_minutes)
        features["awake_pct"] = self._safe_percent(awake_minutes, time_in_bed)
        features["sleep_efficiency"] = self._safe_percent(total_sleep_minutes, time_in_bed)
        features["sleep_onset_latency"] = self._coalesce_numeric(
            sleep_frame,
            ("raw_minutesToFallAsleep", "minutesToFallAsleep"),
            default=0.0,
        )
        features["number_of_awakenings"] = self._coalesce_numeric(
            sleep_frame,
            (
                "raw_awakeningsCount",
                "awakeningsCount",
                "raw_levels_summary_wake_count",
                "levels_summary_wake_count",
            ),
            default=0.0,
        )

        features["day_of_week"] = features["dateOfSleep"].dt.dayofweek
        features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)

        features["avg_hrv"] = self._merge_daily_metric(
            base_dates=features["dateOfSleep"],
            metric_frame=hrv_data,
            metric_columns=("value_dailyRmssd", "dailyRmssd", "rmssd"),
            date_columns=("dateTime", "dateOfSleep"),
        )
        features["resting_hr"] = self._merge_daily_metric(
            base_dates=features["dateOfSleep"],
            metric_frame=heart_rate_data,
            metric_columns=("restingHeartRate", "value_restingHeartRate"),
            date_columns=("dateTime", "dateOfSleep"),
        )
        features["avg_spo2"] = self._merge_daily_metric(
            base_dates=features["dateOfSleep"],
            metric_frame=spo2_data,
            metric_columns=("value_avg", "avg", "spo2", "average"),
            date_columns=("dateTime", "dateOfSleep"),
        )

        features["sleep_continuity_score"] = (
            100.0
            - (features["awake_pct"] * self.continuity_awake_weight)
            - (features["number_of_awakenings"] * self.continuity_awakenings_weight)
            - (features["sleep_onset_latency"] * self.continuity_latency_weight)
        ).clip(lower=0.0, upper=100.0)

        numeric_columns = [
            "total_sleep_minutes",
            "deep_sleep_pct",
            "rem_sleep_pct",
            "light_sleep_pct",
            "awake_pct",
            "sleep_efficiency",
            "sleep_onset_latency",
            "number_of_awakenings",
            "avg_hrv",
            "resting_hr",
            "avg_spo2",
            "sleep_continuity_score",
        ]
        for column in numeric_columns:
            features[column] = pd.to_numeric(features[column], errors="coerce")

        return features

    def _to_frame(self, data: Any) -> pd.DataFrame:
        """Normalize accepted inputs to DataFrame."""

        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, dict):
            if "sleep" in data:
                return pd.json_normalize(data["sleep"], sep="_")
            return pd.json_normalize([data], sep="_")
        if isinstance(data, list):
            return pd.json_normalize(data, sep="_")
        raise TypeError("sleep_data must be a DataFrame, dict, or list.")

    def _coalesce_numeric(
        self,
        frame: pd.DataFrame,
        columns: tuple[str, ...],
        default: float | None = None,
    ) -> pd.Series:
        """Return the first non-null numeric value from candidate columns."""

        result = pd.Series(np.nan, index=frame.index, dtype="float64")
        for column in columns:
            if column in frame.columns:
                values = pd.to_numeric(frame[column], errors="coerce")
                result = result.fillna(values)
        if default is not None:
            result = result.fillna(default)
        return result

    def _safe_percent(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Compute percentages while avoiding divide-by-zero issues."""

        denom = denominator.replace(0, pd.NA)
        return ((numerator / denom) * 100.0).fillna(0.0)

    def _merge_daily_metric(
        self,
        base_dates: pd.Series,
        metric_frame: pd.DataFrame | None,
        metric_columns: tuple[str, ...],
        date_columns: tuple[str, ...],
    ) -> pd.Series:
        """Align a daily metric frame to nightly sleep dates."""

        result = pd.Series(np.nan, index=base_dates.index, dtype="float64")
        if metric_frame is None or metric_frame.empty:
            return result

        metrics = metric_frame.copy()
        metric_date_column = next((column for column in date_columns if column in metrics.columns), None)
        metric_value_column = next((column for column in metric_columns if column in metrics.columns), None)
        if metric_date_column is None or metric_value_column is None:
            return result

        metrics[metric_date_column] = pd.to_datetime(metrics[metric_date_column], errors="coerce").dt.normalize()
        daily = (
            metrics.groupby(metric_date_column)[metric_value_column]
            .mean()
            .rename("metric_value")
            .reset_index()
        )

        merged = pd.DataFrame({"dateOfSleep": base_dates}).merge(
            daily,
            how="left",
            left_on="dateOfSleep",
            right_on=metric_date_column,
        )
        return pd.to_numeric(merged["metric_value"], errors="coerce")


def engineer_sleep_features(
    sleep_data: Any,
    hrv_data: pd.DataFrame | None = None,
    heart_rate_data: pd.DataFrame | None = None,
    spo2_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convenience wrapper for nightly sleep feature generation."""

    return SleepFeatureEngineer().engineer(
        sleep_data=sleep_data,
        hrv_data=hrv_data,
        heart_rate_data=heart_rate_data,
        spo2_data=spo2_data,
    )
