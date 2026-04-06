"""Preprocessing utilities for Fitbit sleep payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError as exc:  # pragma: no cover - depends on environment setup
    MinMaxScaler = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


@dataclass(slots=True)
class SleepPreprocessor:
    """Prepare Fitbit sleep payloads for feature engineering."""

    fill_method: str = "ffill"
    interpolation_method: str = "linear"
    outlier_multiplier: float = 1.5
    datetime_columns: tuple[str, ...] = (
        "dateOfSleep",
        "startTime",
        "endTime",
        "mainSleepStart",
        "mainSleepEnd",
    )
    protected_numeric_columns: tuple[str, ...] = ("logId",)
    scaler: Any = field(default=None)

    def preprocess(self, raw_sleep_payload: Any) -> pd.DataFrame:
        """Clean Fitbit sleep records into a model-ready DataFrame."""

        if MinMaxScaler is None:
            raise ImportError(
                "scikit-learn is required for MinMaxScaler support. Install requirements.txt first."
            ) from SKLEARN_IMPORT_ERROR
        if self.scaler is None:
            self.scaler = MinMaxScaler()

        frame = self._payload_to_frame(raw_sleep_payload)
        if frame.empty:
            return frame

        sort_columns = [
            column for column in ("dateOfSleep", "startTime", "logId") if column in frame.columns
        ]
        if sort_columns:
            frame = frame.sort_values(by=sort_columns, kind="stable").reset_index(drop=True)

        frame = self._parse_datetime_columns(frame)
        frame = self._derive_sleep_metrics(frame)
        frame = self._coerce_numeric_columns(frame)
        frame = self._fill_missing_values(frame)
        frame = self._remove_outliers_iqr(frame)
        frame = self._scale_numeric_features(frame)

        return frame.reset_index(drop=True)

    def _payload_to_frame(self, raw_sleep_payload: Any) -> pd.DataFrame:
        """Normalize raw Fitbit sleep JSON into a flat DataFrame."""

        if isinstance(raw_sleep_payload, pd.DataFrame):
            return raw_sleep_payload.copy()

        if isinstance(raw_sleep_payload, dict):
            if "sleep" in raw_sleep_payload:
                records = raw_sleep_payload.get("sleep", [])
            else:
                records = [raw_sleep_payload]
        elif isinstance(raw_sleep_payload, list):
            records = raw_sleep_payload
        else:
            raise TypeError("raw_sleep_payload must be a dict, list, or DataFrame.")

        if not records:
            return pd.DataFrame()

        flattened: list[dict[str, Any]] = []
        for record in records:
            row = pd.json_normalize(record, sep="_").to_dict(orient="records")[0]
            flattened.append(row)

        frame = pd.DataFrame(flattened)
        for nested_column in ("levels", "type"):
            if nested_column in frame.columns and frame[nested_column].apply(lambda value: isinstance(value, (dict, list))).any():
                frame = frame.drop(columns=[nested_column])
        return frame

    def _parse_datetime_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamp fields and keep them timezone-naive for modeling."""

        cleaned = frame.copy()
        for column in self.datetime_columns:
            if column in cleaned.columns:
                cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce", utc=True).dt.tz_localize(None)
        return cleaned

    def _derive_sleep_metrics(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Create consistent numeric sleep metrics from Fitbit fields."""

        enriched = frame.copy()

        if {"startTime", "endTime"}.issubset(enriched.columns):
            duration_minutes = (
                (enriched["endTime"] - enriched["startTime"]).dt.total_seconds() / 60.0
            )
            enriched["durationMinutes"] = enriched.get("durationMinutes", duration_minutes)
            enriched["durationMinutes"] = enriched["durationMinutes"].fillna(duration_minutes)

        if {"minutesAsleep", "timeInBed"}.issubset(enriched.columns):
            time_in_bed = enriched["timeInBed"].replace(0, pd.NA)
            efficiency = (enriched["minutesAsleep"] / time_in_bed) * 100.0
            enriched["sleepEfficiencyComputed"] = efficiency

        if {"minutesAwake", "timeInBed"}.issubset(enriched.columns):
            awake_ratio = enriched["minutesAwake"] / enriched["timeInBed"].replace(0, pd.NA)
            enriched["awakeRatio"] = awake_ratio

        return enriched

    def _coerce_numeric_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Convert object columns containing numbers into numeric dtype."""

        cleaned = frame.copy()
        excluded = set(self.datetime_columns)
        for column in cleaned.columns:
            if column in excluded:
                continue
            if cleaned[column].dtype == "object":
                converted = pd.to_numeric(cleaned[column], errors="ignore")
                cleaned[column] = converted
        return cleaned

    def _fill_missing_values(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply forward-fill and interpolation across numeric features."""

        cleaned = frame.copy()
        if self.fill_method == "ffill":
            cleaned = cleaned.ffill()
        else:
            raise ValueError("fill_method currently supports only 'ffill'.")

        numeric_columns = cleaned.select_dtypes(include=["number"]).columns.tolist()
        if numeric_columns:
            cleaned[numeric_columns] = cleaned[numeric_columns].interpolate(
                method=self.interpolation_method,
                limit_direction="both",
            )
            cleaned[numeric_columns] = cleaned[numeric_columns].ffill().bfill()

        non_numeric_columns = [column for column in cleaned.columns if column not in numeric_columns]
        if non_numeric_columns:
            cleaned[non_numeric_columns] = cleaned[non_numeric_columns].ffill().bfill()

        return cleaned

    def _remove_outliers_iqr(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Remove rows containing IQR-based outliers on numeric features."""

        numeric_columns = [
            column
            for column in frame.select_dtypes(include=["number"]).columns
            if column not in self.protected_numeric_columns
        ]
        if not numeric_columns:
            return frame.copy()

        mask = pd.Series(True, index=frame.index)
        for column in numeric_columns:
            q1 = frame[column].quantile(0.25)
            q3 = frame[column].quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            lower = q1 - (self.outlier_multiplier * iqr)
            upper = q3 + (self.outlier_multiplier * iqr)
            mask &= frame[column].between(lower, upper, inclusive="both")

        filtered = frame.loc[mask].copy()
        return filtered if not filtered.empty else frame.copy()

    def _scale_numeric_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric modeling features to a 0-1 range."""

        scaled = frame.copy()
        numeric_columns = [
            column
            for column in scaled.select_dtypes(include=["number"]).columns
            if column not in self.protected_numeric_columns
        ]
        if not numeric_columns:
            return scaled

        for column in numeric_columns:
            scaled[f"raw_{column}"] = scaled[column]

        scaled[numeric_columns] = self.scaler.fit_transform(scaled[numeric_columns])
        return scaled


def preprocess_sleep_data(raw_sleep_payload: Any) -> pd.DataFrame:
    """Convenience wrapper for preprocessing raw Fitbit sleep payloads."""

    return SleepPreprocessor().preprocess(raw_sleep_payload)
