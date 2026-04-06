"""Deterministic sleep score component calculator."""

from __future__ import annotations

from datetime import date
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SleepScoreCalculator:
    """Compute nightly sleep quality component scores and a final label."""

    duration_weight: float = 0.30
    deep_weight: float = 0.15
    rem_weight: float = 0.15
    continuity_weight: float = 0.20
    recovery_weight: float = 0.20

    def calculate(self, sleep_features: Any) -> pd.DataFrame:
        """Return component scores, final score, label, and recommendations."""

        frame = self._to_frame(sleep_features)
        if frame.empty:
            return frame.assign(
                duration_score=pd.Series(dtype="float64"),
                deep_score=pd.Series(dtype="float64"),
                rem_score=pd.Series(dtype="float64"),
                continuity_score=pd.Series(dtype="float64"),
                recovery_score=pd.Series(dtype="float64"),
                final_score=pd.Series(dtype="float64"),
                label=pd.Series(dtype="object"),
                recommendations=pd.Series(dtype="object"),
            )

        scored = frame.copy()
        total_sleep_minutes = self._get_numeric(scored, "total_sleep_minutes")
        deep_sleep_pct = self._get_numeric(scored, "deep_sleep_pct")
        rem_sleep_pct = self._get_numeric(scored, "rem_sleep_pct")
        sleep_efficiency = self._get_numeric(scored, "sleep_efficiency")
        awake_pct = self._get_numeric(scored, "awake_pct")
        latency = self._get_numeric(scored, "sleep_onset_latency")
        awakenings = self._get_numeric(scored, "number_of_awakenings")
        continuity_input = self._get_numeric(scored, "sleep_continuity_score")
        avg_hrv = self._get_numeric(scored, "avg_hrv")
        resting_hr = self._get_numeric(scored, "resting_hr")
        avg_spo2 = self._get_numeric(scored, "avg_spo2")

        scored["duration_score"] = self._duration_score(total_sleep_minutes)
        scored["deep_score"] = self._target_band_score(deep_sleep_pct, target=18.0, tolerance=8.0)
        scored["rem_score"] = self._target_band_score(rem_sleep_pct, target=22.0, tolerance=7.0)
        scored["continuity_score"] = self._continuity_score(
            continuity_input=continuity_input,
            sleep_efficiency=sleep_efficiency,
            awake_pct=awake_pct,
            sleep_onset_latency=latency,
            awakenings=awakenings,
        )
        scored["recovery_score"] = self._recovery_score(
            avg_hrv=avg_hrv,
            resting_hr=resting_hr,
            avg_spo2=avg_spo2,
            sleep_efficiency=sleep_efficiency,
        )

        final_score = (
            (scored["duration_score"] * self.duration_weight)
            + (scored["deep_score"] * self.deep_weight)
            + (scored["rem_score"] * self.rem_weight)
            + (scored["continuity_score"] * self.continuity_weight)
            + (scored["recovery_score"] * self.recovery_weight)
        ).clip(lower=0.0, upper=100.0)

        scored["final_score"] = final_score.round(2)
        scored["label"] = final_score.apply(self._label_for_score)
        scored["recommendations"] = self.generate_recommendations(scored)
        return scored

    def generate_recommendations(self, sleep_features: Any) -> pd.Series:
        """Generate rule-based sleep recommendations for each row."""

        frame = self._to_frame(sleep_features)
        recommendations: list[list[dict[str, Any]]] = []

        for _, row in frame.iterrows():
            row_recommendations: list[dict[str, Any]] = []

            deep_sleep_pct = self._safe_float(row.get("deep_sleep_pct"))
            rem_sleep_pct = self._safe_float(row.get("rem_sleep_pct"))
            awakenings = self._safe_float(row.get("number_of_awakenings"))
            avg_spo2 = self._safe_float(row.get("avg_spo2"))
            latency = self._safe_float(row.get("sleep_onset_latency"))
            total_sleep_minutes = self._safe_float(row.get("total_sleep_minutes"))

            if pd.notna(deep_sleep_pct) and deep_sleep_pct < 15.0:
                row_recommendations.append(
                    self._build_recommendation(
                        category="sleep_hygiene",
                        title="Reduce pre-bed alcohol",
                        message="Deep sleep was below 15%; avoid alcohol before bed to support restorative sleep.",
                        priority=2,
                        action_items=["Avoid alcohol in the hours before bedtime."],
                    )
                )

            if pd.notna(rem_sleep_pct) and rem_sleep_pct < 20.0:
                row_recommendations.append(
                    self._build_recommendation(
                        category="sleep_schedule",
                        title="Keep a consistent sleep schedule",
                        message="REM sleep was below 20%; maintain a consistent sleep schedule to support REM cycles.",
                        priority=2,
                        action_items=["Go to bed and wake up at the same time each day."],
                    )
                )

            if pd.notna(awakenings) and awakenings > 5.0:
                row_recommendations.append(
                    self._build_recommendation(
                        category="environment",
                        title="Review bedroom temperature",
                        message="Frequent awakenings were detected; adjust room temperature to reduce overnight disruptions.",
                        priority=2,
                        action_items=["Adjust room temperature for a cooler, more stable sleep environment."],
                    )
                )

            if pd.notna(avg_spo2) and avg_spo2 < 94.0:
                row_recommendations.append(
                    self._build_recommendation(
                        category="clinical_follow_up",
                        title="Discuss low overnight oxygen levels",
                        message="SpO2 was below 94%, which can be associated with possible sleep apnea.",
                        priority=3,
                        action_items=["Consider discussing possible sleep apnea with a clinician."],
                    )
                )

            if pd.notna(latency) and latency > 30.0:
                row_recommendations.append(
                    self._build_recommendation(
                        category="sleep_hygiene",
                        title="Add a wind-down routine",
                        message="Sleep latency exceeded 30 minutes; a wind-down routine may help you fall asleep more easily.",
                        priority=2,
                        action_items=["Use a 20-30 minute wind-down routine before bed."],
                    )
                )

            if pd.notna(total_sleep_minutes) and total_sleep_minutes < 360.0:
                row_recommendations.append(
                    self._build_recommendation(
                        category="sleep_schedule",
                        title="Increase total sleep time",
                        message="Sleep duration was under 6 hours; aim for 7-9 hours of sleep when possible.",
                        priority=3,
                        action_items=["Aim for 7-9 hours of sleep per night."],
                    )
                )

            recommendations.append(row_recommendations)

        return pd.Series(recommendations, index=frame.index, dtype="object")

    async def store_recommendations(
        self,
        session: Any,
        *,
        user_id: int,
        sleep_score_id: int | None,
        recommendation_date: date,
        sleep_features: Any,
    ) -> list[Any]:
        """Persist generated recommendations using the async CRUD layer."""

        from src.database.crud import create_recommendation

        frame = self._to_frame(sleep_features)
        generated = self.generate_recommendations(frame)
        created_records: list[Any] = []

        for recommendation_group in generated:
            for recommendation in recommendation_group:
                created_records.append(
                    await create_recommendation(
                        session,
                        user_id=user_id,
                        sleep_score_id=sleep_score_id,
                        recommendation_date=recommendation_date,
                        category=recommendation["category"],
                        title=recommendation["title"],
                        message=recommendation["message"],
                        priority=recommendation["priority"],
                        action_items=recommendation["action_items"],
                        is_active=True,
                    )
                )

        return created_records

    def _to_frame(self, data: Any) -> pd.DataFrame:
        """Normalize supported input types to a DataFrame."""

        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, dict):
            return pd.DataFrame([data])
        if isinstance(data, list):
            return pd.DataFrame(data)
        raise TypeError("sleep_features must be a DataFrame, dict, or list.")

    def _get_numeric(self, frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
        """Return a numeric series for the requested column."""

        if column not in frame.columns:
            return pd.Series(default, index=frame.index, dtype="float64")
        return pd.to_numeric(frame[column], errors="coerce")

    def _duration_score(self, total_sleep_minutes: pd.Series) -> pd.Series:
        """Score duration against a practical healthy range."""

        hours = total_sleep_minutes / 60.0
        score = np.where(
            hours.between(7.0, 9.0, inclusive="both"),
            100.0,
            100.0 - (np.abs(hours - 8.0) * 20.0),
        )
        return pd.Series(score, index=total_sleep_minutes.index, dtype="float64").clip(0.0, 100.0)

    def _target_band_score(self, values: pd.Series, *, target: float, tolerance: float) -> pd.Series:
        """Score how close a percentage is to a target band center."""

        distance = (values - target).abs()
        score = 100.0 - ((distance / tolerance) * 100.0)
        return pd.Series(score, index=values.index, dtype="float64").clip(0.0, 100.0).fillna(50.0)

    def _continuity_score(
        self,
        *,
        continuity_input: pd.Series,
        sleep_efficiency: pd.Series,
        awake_pct: pd.Series,
        sleep_onset_latency: pd.Series,
        awakenings: pd.Series,
    ) -> pd.Series:
        """Blend direct continuity signals into a 0-100 score."""

        derived = (
            (sleep_efficiency.fillna(85.0) * 0.5)
            + ((100.0 - awake_pct.fillna(10.0).clip(0.0, 100.0)) * 0.2)
            + ((100.0 - (sleep_onset_latency.fillna(15.0) * 2.0).clip(0.0, 100.0)) * 0.15)
            + ((100.0 - (awakenings.fillna(2.0) * 12.0).clip(0.0, 100.0)) * 0.15)
        )
        return continuity_input.fillna(derived).clip(0.0, 100.0)

    def _recovery_score(
        self,
        *,
        avg_hrv: pd.Series,
        resting_hr: pd.Series,
        avg_spo2: pd.Series,
        sleep_efficiency: pd.Series,
    ) -> pd.Series:
        """Score physiological recovery using simple bounded heuristics."""

        hrv_component = ((avg_hrv.fillna(40.0) - 20.0) / 40.0 * 100.0).clip(0.0, 100.0)
        resting_hr_component = (100.0 - ((resting_hr.fillna(65.0) - 50.0) * 4.0)).clip(0.0, 100.0)
        spo2_component = ((avg_spo2.fillna(96.0) - 90.0) / 10.0 * 100.0).clip(0.0, 100.0)
        efficiency_component = sleep_efficiency.fillna(85.0).clip(0.0, 100.0)

        recovery = (
            (hrv_component * 0.35)
            + (resting_hr_component * 0.25)
            + (spo2_component * 0.20)
            + (efficiency_component * 0.20)
        )
        return recovery.clip(0.0, 100.0)

    def _label_for_score(self, score: float) -> str:
        """Map a numeric score to a user-facing quality label."""

        if score < 60.0:
            return "Poor"
        if score < 75.0:
            return "Fair"
        if score < 90.0:
            return "Good"
        return "Excellent"

    def _build_recommendation(
        self,
        *,
        category: str,
        title: str,
        message: str,
        priority: int,
        action_items: list[str],
    ) -> dict[str, Any]:
        """Create a normalized recommendation payload."""

        return {
            "category": category,
            "title": title,
            "message": message,
            "priority": priority,
            "action_items": action_items,
        }

    def _safe_float(self, value: Any) -> float:
        """Convert a scalar-like value to float when possible."""

        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")


def calculate_sleep_score(sleep_features: Any) -> pd.DataFrame:
    """Convenience wrapper for nightly sleep score calculation."""

    return SleepScoreCalculator().calculate(sleep_features)
