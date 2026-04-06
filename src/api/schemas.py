"""Pydantic schemas for the Fitbit Sleep Score FastAPI backend."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    """Base API model with ORM-friendly configuration."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class MessageResponse(APIModel):
    """Simple status payload."""

    message: str


class UserRegistrationRequest(APIModel):
    """Payload for local user registration."""

    fitbit_user_id: str
    email: str | None = None
    full_name: str | None = None
    birth_date: date | None = None
    sex: str | None = None
    timezone: str = "UTC"


class UserResponse(APIModel):
    """Public user profile payload."""

    id: int
    fitbit_user_id: str
    email: str | None = None
    full_name: str | None = None
    birth_date: date | None = None
    sex: str | None = None
    timezone: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


class ModelVersionResponse(APIModel):
    """Public metadata for an active scoring model."""

    id: int
    model_name: str
    version: str
    framework: str
    artifact_uri: str | None = None
    training_metrics: dict[str, Any] | None = None
    feature_schema: dict[str, Any] | None = None
    notes: str | None = None
    is_active: bool
    trained_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class SleepStageInput(APIModel):
    """Incoming sleep stage interval for analysis."""

    stage_type: Literal["wake", "light", "deep", "rem", "asleep", "restless"]
    started_at: datetime
    ended_at: datetime | None = None
    seconds: int | None = Field(default=None, ge=0)
    sequence_index: int | None = Field(default=None, ge=0)


class SleepStageResponse(APIModel):
    """Returned sleep stage interval."""

    id: int | None = None
    stage_type: str
    started_at: datetime
    ended_at: datetime | None = None
    seconds: int
    sequence_index: int


class SleepAnalyzeRequest(APIModel):
    """Payload for nightly sleep analysis."""

    date: date
    fitbit_log_id: int | None = None
    start_time: datetime
    end_time: datetime
    time_in_bed: int = Field(ge=0)
    minutes_asleep: int = Field(ge=0)
    minutes_awake: int = Field(ge=0)
    minutes_after_wakeup: int | None = Field(default=None, ge=0)
    minutes_to_fall_asleep: int | None = Field(default=None, ge=0)
    awakenings_count: int | None = Field(default=None, ge=0)
    efficiency: float | None = Field(default=None, ge=0, le=100)
    is_main_sleep: bool = True
    session_type: str | None = "stages"
    stages: list[SleepStageInput] = Field(default_factory=list)
    hrv: float | None = None
    resting_hr: float | None = None
    spo2: float | None = None
    source_payload: dict[str, Any] | None = None


class RecommendationResponse(APIModel):
    """Recommendation payload returned by the API."""

    id: int | None = None
    category: str
    title: str
    message: str
    priority: int
    action_items: list[str] | None = None
    recommendation_date: date | None = None
    is_active: bool | None = None


class SleepScoreResponse(APIModel):
    """Nightly sleep score payload."""

    score_date: date
    overall_score: float
    confidence: float | None = None
    sleep_efficiency_score: float | None = None
    duration_score: float | None = None
    continuity_score: float | None = None
    recovery_score: float | None = None
    label: str | None = None
    feature_snapshot: dict[str, Any] | None = None
    recommendations: list[RecommendationResponse] = Field(default_factory=list)


class SleepAnalysisResponse(APIModel):
    """Composite response returned after analysis."""

    user_id: int
    sleep_session_id: int
    sleep_score_id: int
    score: SleepScoreResponse
    stages: list[SleepStageResponse] = Field(default_factory=list)


class TrendPoint(APIModel):
    """Single point in the sleep trend response."""

    date: date
    overall_score: float
    duration_score: float | None = None
    continuity_score: float | None = None
    recovery_score: float | None = None


class SleepTrendResponse(APIModel):
    """Sleep trend response."""

    points: list[TrendPoint]


class WeeklyAnalyticsResponse(APIModel):
    """Weekly aggregate analytics payload."""

    week_start: date
    week_end: date
    nights_tracked: int
    average_score: float
    average_duration_score: float | None = None
    average_continuity_score: float | None = None
    average_recovery_score: float | None = None


class NightSummaryResponse(APIModel):
    """Best/worst night summary payload."""

    date: date
    overall_score: float
    label: str | None = None
    duration_score: float | None = None
    continuity_score: float | None = None
    recovery_score: float | None = None


__all__ = [
    "MessageResponse",
    "ModelVersionResponse",
    "NightSummaryResponse",
    "RecommendationResponse",
    "SleepAnalysisResponse",
    "SleepAnalyzeRequest",
    "SleepScoreResponse",
    "SleepStageInput",
    "SleepStageResponse",
    "SleepTrendResponse",
    "TrendPoint",
    "UserRegistrationRequest",
    "UserResponse",
    "WeeklyAnalyticsResponse",
]
