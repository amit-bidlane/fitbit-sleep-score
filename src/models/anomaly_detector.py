"""Isolation Forest-based sleep anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
except ImportError as exc:  # pragma: no cover - depends on environment setup
    IsolationForest = None
    SKLEARN_IMPORT_ERROR = exc
else:
    SKLEARN_IMPORT_ERROR = None


@dataclass(slots=True)
class SleepAnomalyDetector:
    """Detect anomalous sleep records from nightly feature data."""

    contamination: float = 0.1
    random_state: int = 42
    default_feature_columns: tuple[str, ...] = (
        "total_sleep_minutes",
        "deep_sleep_pct",
        "rem_sleep_pct",
        "awake_pct",
        "sleep_efficiency",
        "sleep_onset_latency",
        "number_of_awakenings",
        "avg_hrv",
        "resting_hr",
        "avg_spo2",
        "sleep_continuity_score",
    )
    model: Any = field(default=None, init=False)

    def detect(
        self,
        sleep_features: Any,
        *,
        baseline_data: Any | None = None,
        feature_columns: list[str] | tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Return anomaly outputs for the provided sleep feature rows."""

        if IsolationForest is None:
            raise ImportError(
                "scikit-learn is required for IsolationForest support. Install requirements.txt first."
            ) from SKLEARN_IMPORT_ERROR

        frame = self._to_frame(sleep_features)
        if frame.empty:
            return frame.assign(
                anomaly_score=pd.Series(dtype="float64"),
                is_anomaly=pd.Series(dtype="bool"),
                anomaly_reason=pd.Series(dtype="object"),
            )

        training_frame = self._to_frame(baseline_data) if baseline_data is not None else frame.copy()
        model_columns = self._resolve_feature_columns(frame, training_frame, feature_columns)
        if not model_columns:
            raise ValueError("No numeric feature columns available for anomaly detection.")

        prepared_training = self._prepare_feature_frame(training_frame, model_columns)
        prepared_scoring = self._prepare_feature_frame(frame, model_columns, reference=prepared_training)

        if len(prepared_training) < 2:
            result = frame.copy()
            result["anomaly_score"] = 0.0
            result["is_anomaly"] = False
            result["anomaly_reason"] = "insufficient_baseline_data"
            return result

        contamination = min(max(self.contamination, 0.001), 0.5)
        self.model = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
        )
        self.model.fit(prepared_training)

        predictions = self.model.predict(prepared_scoring)
        decision = self.model.decision_function(prepared_scoring)
        anomaly_scores = self._normalize_anomaly_scores(-decision)

        result = frame.copy()
        result["anomaly_score"] = anomaly_scores.round(4)
        result["is_anomaly"] = predictions == -1
        result["anomaly_reason"] = self._build_anomaly_reasons(
            prepared_scoring,
            prepared_training,
            result["is_anomaly"],
        )
        return result

    def _to_frame(self, data: Any) -> pd.DataFrame:
        """Normalize supported inputs to a DataFrame."""

        if data is None:
            return pd.DataFrame()
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, dict):
            return pd.DataFrame([data])
        if isinstance(data, list):
            return pd.DataFrame(data)
        raise TypeError("sleep_features must be a DataFrame, dict, list, or None.")

    def _resolve_feature_columns(
        self,
        scoring_frame: pd.DataFrame,
        training_frame: pd.DataFrame,
        feature_columns: list[str] | tuple[str, ...] | None,
    ) -> list[str]:
        """Select usable numeric model columns."""

        candidate_columns = list(feature_columns or self.default_feature_columns)
        available_columns = [
            column
            for column in candidate_columns
            if column in scoring_frame.columns and column in training_frame.columns
        ]
        numeric_columns = [
            column
            for column in available_columns
            if pd.api.types.is_numeric_dtype(scoring_frame[column])
            or pd.api.types.is_numeric_dtype(training_frame[column])
        ]
        if numeric_columns:
            return numeric_columns

        fallback_columns = sorted(
            set(scoring_frame.select_dtypes(include=["number"]).columns).intersection(
                training_frame.select_dtypes(include=["number"]).columns
            )
        )
        return fallback_columns

    def _prepare_feature_frame(
        self,
        frame: pd.DataFrame,
        columns: list[str],
        *,
        reference: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Coerce, align, and impute numeric model inputs."""

        prepared = pd.DataFrame(index=frame.index)
        for column in columns:
            prepared[column] = pd.to_numeric(frame[column], errors="coerce")

        reference_frame = reference if reference is not None else prepared
        medians = reference_frame.median(numeric_only=True)
        prepared = prepared.fillna(medians)
        prepared = prepared.fillna(0.0)
        return prepared

    def _normalize_anomaly_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """Scale anomaly severity into a 0-1 range."""

        if raw_scores.size == 0:
            return raw_scores
        min_score = float(np.min(raw_scores))
        max_score = float(np.max(raw_scores))
        if np.isclose(min_score, max_score):
            return np.zeros_like(raw_scores, dtype=float)
        return (raw_scores - min_score) / (max_score - min_score)

    def _build_anomaly_reasons(
        self,
        scoring_frame: pd.DataFrame,
        training_frame: pd.DataFrame,
        anomaly_mask: pd.Series,
    ) -> pd.Series:
        """Create short, interpretable anomaly summaries."""

        medians = training_frame.median(numeric_only=True)
        spreads = (training_frame - medians).abs().median(numeric_only=True).replace(0, np.nan)
        reasons: list[str] = []

        for row_index, is_anomaly in anomaly_mask.items():
            if not is_anomaly:
                reasons.append("normal_sleep_pattern")
                continue

            row = scoring_frame.loc[row_index]
            deviations = ((row - medians) / spreads).replace([np.inf, -np.inf], np.nan).abs()
            strongest = deviations.sort_values(ascending=False).dropna().head(3)

            if strongest.empty:
                reasons.append("unusual_multivariate_sleep_pattern")
                continue

            parts: list[str] = []
            for column in strongest.index:
                direction = "high" if row[column] >= medians[column] else "low"
                parts.append(f"{direction} {column}")
            reasons.append("; ".join(parts))

        return pd.Series(reasons, index=scoring_frame.index, dtype="object")


def detect_sleep_anomalies(
    sleep_features: Any,
    *,
    baseline_data: Any | None = None,
    feature_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Convenience wrapper for sleep anomaly detection."""

    return SleepAnomalyDetector().detect(
        sleep_features=sleep_features,
        baseline_data=baseline_data,
        feature_columns=feature_columns,
    )
