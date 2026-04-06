"""FastAPI routes for the Fitbit Sleep Score backend."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas import (
    MessageResponse,
    ModelVersionResponse,
    NightSummaryResponse,
    RecommendationResponse,
    SleepAnalysisResponse,
    SleepAnalyzeRequest,
    SleepScoreResponse,
    SleepStageResponse,
    SleepTrendResponse,
    TrendPoint,
    UserRegistrationRequest,
    UserResponse,
    WeeklyAnalyticsResponse,
)
from src.data.feature_engineer import engineer_sleep_features
from src.database.crud import (
    CRUDIntegrityError,
    create_user,
    get_user_by_fitbit_user_id,
    get_user_by_id,
    list_model_versions,
    list_sleep_scores,
    list_sleep_sessions,
)
from src.database.db import get_db, init_db
from src.database.models import SleepRecommendation, SleepScore, SleepSession, SleepStage
from src.models.score_calculator import SleepScoreCalculator, calculate_sleep_score
from src.visualization.report_generator import SleepReportGenerator

logger = logging.getLogger("fitbit.api")
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
router = APIRouter()


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _user_local_today(current_user: Any) -> date:
    """Return today's date in the user's configured timezone."""

    timezone_name = getattr(current_user, "timezone", None) or "UTC"
    try:
        zone = ZoneInfo(str(timezone_name))
    except ZoneInfoNotFoundError:
        zone = timezone.utc
    return datetime.now(zone).date()


async def get_current_user(
    x_user_id: int | None = Header(default=None, alias="X-User-Id"),
    db: AsyncSession = Depends(get_db),
):
    """Resolve the current user from a lightweight header-based auth scheme."""

    if x_user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-User-Id header.",
        )

    user = await get_user_by_id(db, x_user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
        )
    return user


def _safe_float(value: Any) -> float | None:
    """Convert decimals and numeric-like values to float."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _calculate_stage_seconds(stage: SleepAnalyzeRequest, item_index: int) -> int:
    """Compute stage duration when not explicitly provided."""

    current = stage.stages[item_index]
    if current.seconds is not None:
        return current.seconds
    if current.ended_at is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Each stage requires either seconds or ended_at.",
        )
    return max(int((current.ended_at - current.started_at).total_seconds()), 0)


def _recommendation_to_schema(record: Any) -> RecommendationResponse:
    """Normalize recommendation records or dict payloads."""

    if isinstance(record, dict):
        return RecommendationResponse(
            id=record.get("id"),
            category=str(record.get("category", "sleep_hygiene")),
            title=str(record.get("title", "Recommendation")),
            message=str(record.get("message", "")),
            priority=int(record.get("priority", 1)),
            action_items=record.get("action_items"),
            recommendation_date=record.get("recommendation_date"),
            is_active=record.get("is_active"),
        )

    return RecommendationResponse(
        id=getattr(record, "id", None),
        category=record.category,
        title=record.title,
        message=record.message,
        priority=record.priority,
        action_items=record.action_items,
        recommendation_date=getattr(record, "recommendation_date", None),
        is_active=getattr(record, "is_active", None),
    )


def _score_to_schema(score: Any) -> SleepScoreResponse:
    """Serialize a sleep score ORM record into an API schema."""

    snapshot = dict(score.feature_snapshot or {})
    recommendations = [_recommendation_to_schema(item) for item in getattr(score, "recommendations", [])]
    label = snapshot.get("label")

    return SleepScoreResponse(
        score_date=score.score_date,
        overall_score=float(score.overall_score),
        confidence=_safe_float(score.confidence),
        sleep_efficiency_score=_safe_float(score.sleep_efficiency_score),
        duration_score=_safe_float(score.duration_score),
        continuity_score=_safe_float(score.continuity_score),
        recovery_score=_safe_float(score.recovery_score),
        label=str(label) if label is not None else None,
        feature_snapshot=snapshot or None,
        recommendations=recommendations,
    )


def _stage_to_schema(stage: Any) -> SleepStageResponse:
    """Serialize a sleep stage ORM record."""

    return SleepStageResponse(
        id=getattr(stage, "id", None),
        stage_type=stage.stage_type,
        started_at=stage.started_at,
        ended_at=stage.ended_at,
        seconds=int(stage.seconds),
        sequence_index=int(stage.sequence_index),
    )


def _scores_to_frame(scores: list[Any]) -> pd.DataFrame:
    """Convert ORM sleep scores into a dataframe for analytics/reporting."""

    rows: list[dict[str, Any]] = []
    for score in scores:
        snapshot = dict(score.feature_snapshot or {})
        rows.append(
            {
                "score_date": score.score_date,
                "overall_score": _safe_float(score.overall_score),
                "final_score": _safe_float(score.overall_score),
                "duration_score": _safe_float(score.duration_score),
                "continuity_score": _safe_float(score.continuity_score),
                "recovery_score": _safe_float(score.recovery_score),
                "sleep_efficiency_score": _safe_float(score.sleep_efficiency_score),
                "sleep_efficiency": snapshot.get("sleep_efficiency"),
                "total_sleep_minutes": snapshot.get("total_sleep_minutes"),
                "label": snapshot.get("label"),
                "recommendations": snapshot.get("recommendations", []),
            }
        )
    return pd.DataFrame(rows)


def _night_summary(score: Any) -> NightSummaryResponse:
    """Build a compact night summary payload."""

    snapshot = dict(score.feature_snapshot or {})
    label = snapshot.get("label")
    return NightSummaryResponse(
        date=score.score_date,
        overall_score=float(score.overall_score),
        label=str(label) if label is not None else None,
        duration_score=_safe_float(score.duration_score),
        continuity_score=_safe_float(score.continuity_score),
        recovery_score=_safe_float(score.recovery_score),
    )


@router.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def register_user(
    request: Request,
    payload: UserRegistrationRequest,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Register a user in the local application database."""

    existing = await get_user_by_fitbit_user_id(db, payload.fitbit_user_id)
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Fitbit user already registered.")

    try:
        user = await create_user(db, **payload.model_dump())
    except CRUDIntegrityError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    logger.info("Registered user id=%s fitbit_user_id=%s", user.id, user.fitbit_user_id)
    return UserResponse.model_validate(user)


@router.get("/auth/me", response_model=UserResponse)
@limiter.limit("60/minute")
async def get_me(
    request: Request,
    current_user=Depends(get_current_user),
) -> UserResponse:
    """Return the current authenticated user."""

    return UserResponse.model_validate(current_user)


@router.post("/sleep/analyze", response_model=SleepAnalysisResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("20/minute")
async def analyze_sleep(
    request: Request,
    payload: SleepAnalyzeRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> SleepAnalysisResponse:
    """Analyze one night of sleep, persist the score, and return the result."""

    if payload.end_time <= payload.start_time:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="end_time must be after start_time.")

    fitbit_log_id = payload.fitbit_log_id or int(time.time_ns() % 9_000_000_000_000)
    source_payload = payload.source_payload or payload.model_dump(mode="json")

    stage_payloads: list[dict[str, Any]] = []
    for index, stage in enumerate(payload.stages):
        seconds = stage.seconds
        if seconds is None:
            if stage.ended_at is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="Each stage requires either seconds or ended_at.",
                )
            seconds = max(int((stage.ended_at - stage.started_at).total_seconds()), 0)

        ended_at = stage.ended_at or (stage.started_at + timedelta(seconds=seconds))
        stage_payloads.append(
            {
                "stage_type": stage.stage_type,
                "sequence_index": stage.sequence_index if stage.sequence_index is not None else index,
                "started_at": stage.started_at,
                "ended_at": ended_at,
                "seconds": seconds,
            }
        )

    sleep_frame = pd.DataFrame(
        [
            {
                "logId": fitbit_log_id,
                "dateOfSleep": payload.date.isoformat(),
                "raw_minutesAsleep": payload.minutes_asleep,
                "raw_minutesAwake": payload.minutes_awake,
                "raw_timeInBed": payload.time_in_bed,
                "raw_minutesToFallAsleep": payload.minutes_to_fall_asleep,
                "raw_awakeningsCount": payload.awakenings_count,
                "raw_levels_summary_deep_minutes": sum(
                    stage["seconds"] / 60 for stage in stage_payloads if stage["stage_type"] == "deep"
                ),
                "raw_levels_summary_rem_minutes": sum(
                    stage["seconds"] / 60 for stage in stage_payloads if stage["stage_type"] == "rem"
                ),
                "raw_levels_summary_light_minutes": sum(
                    stage["seconds"] / 60 for stage in stage_payloads if stage["stage_type"] == "light"
                ),
                "raw_levels_summary_wake_minutes": sum(
                    stage["seconds"] / 60 for stage in stage_payloads if stage["stage_type"] == "wake"
                ),
            }
        ]
    )
    hrv_frame = pd.DataFrame([{"dateTime": payload.date.isoformat(), "value_dailyRmssd": payload.hrv}]) if payload.hrv is not None else None
    hr_frame = pd.DataFrame([{"dateTime": payload.date.isoformat(), "restingHeartRate": payload.resting_hr}]) if payload.resting_hr is not None else None
    spo2_frame = pd.DataFrame([{"dateTime": payload.date.isoformat(), "value_avg": payload.spo2}]) if payload.spo2 is not None else None

    feature_frame = engineer_sleep_features(
        sleep_data=sleep_frame,
        hrv_data=hrv_frame,
        heart_rate_data=hr_frame,
        spo2_data=spo2_frame,
    )
    scored_frame = calculate_sleep_score(feature_frame)
    score_row = scored_frame.iloc[0]
    calculator = SleepScoreCalculator()
    generated_recommendations = calculator.generate_recommendations(feature_frame).iloc[0]
    feature_snapshot = {
        key: value
        for key, value in score_row.to_dict().items()
        if key != "recommendations"
    }
    feature_snapshot["recommendations"] = generated_recommendations

    sleep_session = SleepSession(
        user_id=current_user.id,
        fitbit_log_id=fitbit_log_id,
        session_date=payload.date,
        start_time=payload.start_time,
        end_time=payload.end_time,
        duration_minutes=max(int((payload.end_time - payload.start_time).total_seconds() // 60), 0),
        time_in_bed=payload.time_in_bed,
        minutes_asleep=payload.minutes_asleep,
        minutes_awake=payload.minutes_awake,
        minutes_after_wakeup=payload.minutes_after_wakeup,
        minutes_to_fall_asleep=payload.minutes_to_fall_asleep,
        awakenings_count=payload.awakenings_count,
        efficiency=Decimal(str(payload.efficiency)) if payload.efficiency is not None else None,
        is_main_sleep=payload.is_main_sleep,
        session_type=payload.session_type,
        source_payload=source_payload,
    )

    try:
        async with db.begin():
            db.add(sleep_session)
            await db.flush()

            created_stages = [
                SleepStage(
                    sleep_session_id=sleep_session.id,
                    stage_type=stage["stage_type"],
                    sequence_index=stage["sequence_index"],
                    started_at=stage["started_at"],
                    ended_at=stage["ended_at"],
                    seconds=stage["seconds"],
                )
                for stage in stage_payloads
            ]
            if created_stages:
                db.add_all(created_stages)
                await db.flush()

            sleep_score = SleepScore(
                user_id=current_user.id,
                sleep_session_id=sleep_session.id,
                score_date=payload.date,
                overall_score=Decimal(str(round(float(score_row["final_score"]), 2))),
                confidence=None,
                sleep_efficiency_score=Decimal(str(round(float(score_row.get("sleep_efficiency", 0.0)), 2))),
                duration_score=Decimal(str(round(float(score_row["duration_score"]), 2))),
                continuity_score=Decimal(str(round(float(score_row["continuity_score"]), 2))),
                recovery_score=Decimal(str(round(float(score_row["recovery_score"]), 2))),
                feature_snapshot=feature_snapshot,
            )
            db.add(sleep_score)
            await db.flush()

            recommendation_records = [
                SleepRecommendation(
                    user_id=current_user.id,
                    sleep_score_id=sleep_score.id,
                    recommendation_date=payload.date,
                    category=recommendation["category"],
                    title=recommendation["title"],
                    message=recommendation["message"],
                    priority=recommendation["priority"],
                    action_items=recommendation["action_items"],
                    is_active=True,
                )
                for recommendation in generated_recommendations
            ]
            if recommendation_records:
                db.add_all(recommendation_records)
                await db.flush()
            sleep_score.recommendations = recommendation_records
    except (CRUDIntegrityError, IntegrityError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    logger.info(
        "Analyzed sleep for user id=%s date=%s score=%s",
        current_user.id,
        payload.date,
        sleep_score.overall_score,
    )
    return SleepAnalysisResponse(
        user_id=current_user.id,
        sleep_session_id=sleep_session.id,
        sleep_score_id=sleep_score.id,
        score=_score_to_schema(sleep_score),
        stages=[_stage_to_schema(stage) for stage in created_stages],
    )


@router.get("/sleep/score/{target_date}", response_model=SleepScoreResponse)
@limiter.limit("60/minute")
async def get_sleep_score(
    request: Request,
    target_date: date,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> SleepScoreResponse:
    """Return a user's sleep score for a specific date."""

    scores = await list_sleep_scores(db, user_id=current_user.id, score_date=target_date)
    if not scores:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sleep score not found for date.")
    return _score_to_schema(scores[-1])


@router.get("/sleep/trend", response_model=SleepTrendResponse)
@limiter.limit("60/minute")
async def get_sleep_trend(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
    days: int = Query(default=14, ge=1, le=365),
) -> SleepTrendResponse:
    """Return recent nightly sleep score trends for the current user."""

    scores = await list_sleep_scores(db, user_id=current_user.id)
    cutoff = _user_local_today(current_user) - timedelta(days=days - 1)
    filtered = [score for score in scores if score.score_date >= cutoff]
    points = [
        TrendPoint(
            date=score.score_date,
            overall_score=float(score.overall_score),
            duration_score=_safe_float(score.duration_score),
            continuity_score=_safe_float(score.continuity_score),
            recovery_score=_safe_float(score.recovery_score),
        )
        for score in filtered
    ]
    return SleepTrendResponse(points=points)


@router.get("/sleep/report/{target_date}", response_class=HTMLResponse)
@limiter.limit("30/minute")
async def get_sleep_report(
    request: Request,
    target_date: date,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> HTMLResponse:
    """Generate an HTML sleep report for the requested date."""

    scores = await list_sleep_scores(db, user_id=current_user.id)
    target_scores = [score for score in scores if score.score_date == target_date]
    if not target_scores:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sleep score not found for date.")

    sessions = await list_sleep_sessions(db, user_id=current_user.id, session_date=target_date, is_main_sleep=True)
    session = sessions[-1] if sessions else None
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sleep session not found for date.")

    score_frame = _scores_to_frame(scores)
    target_frame = _scores_to_frame([target_scores[-1]])
    stage_rows = [
        {
            "stage_type": stage.stage_type,
            "started_at": stage.started_at,
            "ended_at": stage.ended_at,
            "seconds": stage.seconds,
            "sequence_index": stage.sequence_index,
        }
        for stage in session.sleep_stages
    ]
    report_html = SleepReportGenerator().generate_html_report(
        user_name=current_user.full_name or current_user.fitbit_user_id,
        summary=target_frame,
        sleep_scores=score_frame,
        sleep_stages=stage_rows,
        recommendations=[_recommendation_to_schema(item).model_dump() for item in target_scores[-1].recommendations],
        report_date=_utc_now().date().isoformat(),
        title=f"Sleep Report for {target_date.isoformat()}",
    )
    return HTMLResponse(content=report_html)


@router.get("/sleep/stages/{target_date}", response_model=list[SleepStageResponse])
@limiter.limit("60/minute")
async def get_sleep_stages(
    request: Request,
    target_date: date,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> list[SleepStageResponse]:
    """Return sleep stage intervals for a specific night."""

    sessions = await list_sleep_sessions(db, user_id=current_user.id, session_date=target_date, is_main_sleep=True)
    if not sessions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sleep session not found for date.")
    return [_stage_to_schema(stage) for stage in sessions[-1].sleep_stages]


@router.get("/sleep/recommendations/{target_date}", response_model=list[RecommendationResponse])
@limiter.limit("60/minute")
async def get_sleep_recommendations(
    request: Request,
    target_date: date,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> list[RecommendationResponse]:
    """Return recommendations linked to a date's sleep score."""

    scores = await list_sleep_scores(db, user_id=current_user.id, score_date=target_date)
    if not scores:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sleep score not found for date.")
    return [_recommendation_to_schema(item) for item in scores[-1].recommendations]


@router.get("/analytics/weekly", response_model=WeeklyAnalyticsResponse)
@limiter.limit("60/minute")
async def get_weekly_analytics(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> WeeklyAnalyticsResponse:
    """Return rolling 7-day analytics for the current user."""

    scores = await list_sleep_scores(db, user_id=current_user.id)
    week_end = _user_local_today(current_user)
    cutoff = week_end - timedelta(days=6)
    weekly_scores = [score for score in scores if score.score_date >= cutoff]
    if not weekly_scores:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No sleep scores found for the last 7 days.")

    frame = _scores_to_frame(weekly_scores)
    return WeeklyAnalyticsResponse(
        week_start=cutoff,
        week_end=week_end,
        nights_tracked=len(weekly_scores),
        average_score=round(float(frame["overall_score"].mean()), 2),
        average_duration_score=round(float(frame["duration_score"].mean()), 2) if "duration_score" in frame else None,
        average_continuity_score=round(float(frame["continuity_score"].mean()), 2) if "continuity_score" in frame else None,
        average_recovery_score=round(float(frame["recovery_score"].mean()), 2) if "recovery_score" in frame else None,
    )


@router.get("/models/active", response_model=list[ModelVersionResponse])
@limiter.limit("30/minute")
async def get_active_models(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> list[ModelVersionResponse]:
    """Return active model metadata for the authenticated user."""

    _ = current_user
    models = await list_model_versions(db, is_active=True)
    return [ModelVersionResponse.model_validate(model) for model in models]


@router.get("/analytics/best-night", response_model=NightSummaryResponse)
@limiter.limit("60/minute")
async def get_best_night(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> NightSummaryResponse:
    """Return the highest-scoring night for the current user."""

    scores = await list_sleep_scores(db, user_id=current_user.id)
    if not scores:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No sleep scores available.")
    best = max(scores, key=lambda item: float(item.overall_score))
    return _night_summary(best)


@router.get("/analytics/worst-night", response_model=NightSummaryResponse)
@limiter.limit("60/minute")
async def get_worst_night(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
) -> NightSummaryResponse:
    """Return the lowest-scoring night for the current user."""

    scores = await list_sleep_scores(db, user_id=current_user.id)
    if not scores:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No sleep scores available.")
    worst = min(scores, key=lambda item: float(item.overall_score))
    return _night_summary(worst)


def create_app(*, run_startup: bool = True) -> FastAPI:
    """Create the FastAPI application with middleware, logging, and rate limiting."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if run_startup:
            await init_db()
        yield

    app = FastAPI(title="Fitbit Sleep Score API", version="1.0.0", lifespan=lifespan)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        started = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "%s %s status=%s duration_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    app.include_router(router)
    return app


app = create_app()


__all__ = ["app", "create_app", "router"]
