"""SQLAlchemy ORM models for the Fitbit Sleep Score system."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

SLEEP_SESSION_TYPE = Enum(
    "stages",
    "classic",
    name="sleep_session_type",
    native_enum=False,
    validate_strings=True,
)
SLEEP_STAGE_TYPE = Enum(
    "wake",
    "light",
    "deep",
    "rem",
    "asleep",
    "restless",
    name="sleep_stage_type",
    native_enum=False,
    validate_strings=True,
)
RECOMMENDATION_CATEGORY = Enum(
    "sleep_schedule",
    "sleep_hygiene",
    "recovery",
    "stress",
    "activity",
    "environment",
    "clinical_follow_up",
    name="recommendation_category",
    native_enum=False,
    validate_strings=True,
)


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampMixin:
    """Reusable created/updated timestamp columns."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class User(TimestampMixin, Base):
    """Application user linked to Fitbit sleep data and model outputs."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("fitbit_user_id", name="uq_users_fitbit_user_id"),
        UniqueConstraint("email", name="uq_users_email"),
        Index("ix_users_timezone", "timezone"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fitbit_user_id: Mapped[str] = mapped_column(String(64), nullable=False)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    birth_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    sex: Mapped[str | None] = mapped_column(String(32), nullable=True)
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="UTC")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="1")

    sleep_sessions: Mapped[list["SleepSession"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    sleep_scores: Mapped[list["SleepScore"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    sleep_recommendations: Mapped[list["SleepRecommendation"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class SleepSession(TimestampMixin, Base):
    """A nightly sleep log imported from Fitbit for a specific user."""

    __tablename__ = "sleep_sessions"
    __table_args__ = (
        UniqueConstraint("fitbit_log_id", name="uq_sleep_sessions_fitbit_log_id"),
        Index("ix_sleep_sessions_user_date", "user_id", "session_date"),
        Index("ix_sleep_sessions_user_start_time", "user_id", "start_time"),
        Index("ix_sleep_sessions_main_sleep", "user_id", "is_main_sleep"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    fitbit_log_id: Mapped[int] = mapped_column(Integer, nullable=False)
    session_date: Mapped[date] = mapped_column(Date, nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    time_in_bed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    minutes_asleep: Mapped[int | None] = mapped_column(Integer, nullable=True)
    minutes_awake: Mapped[int | None] = mapped_column(Integer, nullable=True)
    minutes_after_wakeup: Mapped[int | None] = mapped_column(Integer, nullable=True)
    minutes_to_fall_asleep: Mapped[int | None] = mapped_column(Integer, nullable=True)
    awakenings_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    efficiency: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    is_main_sleep: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="1")
    session_type: Mapped[str | None] = mapped_column(SLEEP_SESSION_TYPE, nullable=True)
    source_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    user: Mapped["User"] = relationship(back_populates="sleep_sessions")
    sleep_stages: Mapped[list["SleepStage"]] = relationship(
        back_populates="sleep_session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="SleepStage.sequence_index",
    )
    sleep_score: Mapped["SleepScore | None"] = relationship(
        back_populates="sleep_session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )


class SleepStage(Base):
    """Fine-grained sleep stage interval belonging to a sleep session."""

    __tablename__ = "sleep_stages"
    __table_args__ = (
        Index("ix_sleep_stages_session_sequence", "sleep_session_id", "sequence_index"),
        Index("ix_sleep_stages_session_stage_type", "sleep_session_id", "stage_type"),
        Index("ix_sleep_stages_started_at", "started_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sleep_session_id: Mapped[int] = mapped_column(
        ForeignKey("sleep_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    stage_type: Mapped[str] = mapped_column(SLEEP_STAGE_TYPE, nullable=False)
    sequence_index: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    seconds: Mapped[int] = mapped_column(Integer, nullable=False)

    sleep_session: Mapped["SleepSession"] = relationship(back_populates="sleep_stages")


class ModelVersion(TimestampMixin, Base):
    """Versioned metadata for trained models used to generate sleep scores."""

    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint("version", name="uq_model_versions_version"),
        Index("ix_model_versions_model_name", "model_name"),
        Index("ix_model_versions_is_active", "is_active"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    framework: Mapped[str] = mapped_column(String(50), nullable=False, default="PyTorch", server_default="PyTorch")
    artifact_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    training_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    feature_schema: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="0")
    trained_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    sleep_scores: Mapped[list["SleepScore"]] = relationship(back_populates="model_version")


class SleepScore(Base):
    """Predicted or computed sleep quality score for a sleep session."""

    __tablename__ = "sleep_scores"
    __table_args__ = (
        UniqueConstraint("sleep_session_id", name="uq_sleep_scores_sleep_session_id"),
        Index("ix_sleep_scores_user_score_date", "user_id", "score_date"),
        Index("ix_sleep_scores_model_version", "model_version_id"),
        Index("ix_sleep_scores_overall_score", "overall_score"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    sleep_session_id: Mapped[int] = mapped_column(
        ForeignKey("sleep_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    score_date: Mapped[date] = mapped_column(Date, nullable=False)
    overall_score: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    sleep_efficiency_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    duration_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    continuity_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    recovery_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2), nullable=True)
    feature_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship(back_populates="sleep_scores")
    sleep_session: Mapped["SleepSession"] = relationship(back_populates="sleep_score")
    model_version: Mapped["ModelVersion | None"] = relationship(back_populates="sleep_scores")
    recommendations: Mapped[list["SleepRecommendation"]] = relationship(
        back_populates="sleep_score",
        passive_deletes=True,
    )


class SleepRecommendation(Base):
    """Actionable recommendation generated from a user's recent sleep profile."""

    __tablename__ = "sleep_recommendations"
    __table_args__ = (
        Index("ix_sleep_recommendations_user_date", "user_id", "recommendation_date"),
        Index("ix_sleep_recommendations_score_id", "sleep_score_id"),
        Index("ix_sleep_recommendations_is_active", "is_active"),
        Index("ix_sleep_recommendations_category", "category"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    sleep_score_id: Mapped[int | None] = mapped_column(
        ForeignKey("sleep_scores.id", ondelete="SET NULL"),
        nullable=True,
    )
    recommendation_date: Mapped[date] = mapped_column(Date, nullable=False)
    category: Mapped[str] = mapped_column(RECOMMENDATION_CATEGORY, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    action_items: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, server_default="1")
    delivered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship(back_populates="sleep_recommendations")
    sleep_score: Mapped["SleepScore | None"] = relationship(back_populates="recommendations")


__all__ = [
    "Base",
    "ModelVersion",
    "SleepRecommendation",
    "SleepScore",
    "SleepSession",
    "SleepStage",
    "User",
]
