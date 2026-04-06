"""Async CRUD helpers for the Fitbit Sleep Score database models."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any, TypeVar

from sqlalchemy import Select, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute, selectinload

from src.database.models import (
    ModelVersion,
    SleepRecommendation,
    SleepScore,
    SleepSession,
    SleepStage,
    User,
)

ModelT = TypeVar("ModelT", User, SleepSession, SleepStage, SleepScore, SleepRecommendation, ModelVersion)


class CRUDIntegrityError(RuntimeError):
    """Raised when a write operation violates a database integrity constraint."""


def _extract_integrity_message(exc: IntegrityError) -> str:
    """Return a concise, user-facing message for an integrity failure."""

    original = getattr(exc, "orig", None)
    detail = str(original or exc).strip()
    return detail or "Database integrity constraint violated."


async def _commit_refresh(session: AsyncSession, instance: ModelT) -> ModelT:
    """Commit the current transaction, refreshing the given instance on success."""

    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise CRUDIntegrityError(_extract_integrity_message(exc)) from exc

    await session.refresh(instance)
    return instance


def _apply_updates(instance: ModelT, updates: dict[str, Any]) -> ModelT:
    """Apply partial update values to a SQLAlchemy model instance."""

    for field_name, value in updates.items():
        if hasattr(instance, field_name):
            setattr(instance, field_name, value)
    return instance


async def _get_by_id(
    session: AsyncSession,
    model: type[ModelT],
    record_id: int,
    *,
    options: Sequence[Any] = (),
) -> ModelT | None:
    """Fetch a single model instance by primary key."""

    statement = select(model).where(model.id == record_id).options(*options)
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def _list_records(
    session: AsyncSession,
    statement: Select[tuple[ModelT]],
) -> list[ModelT]:
    """Execute a query and return unique ORM instances."""

    result = await session.execute(statement)
    return list(result.scalars().unique().all())


async def _delete_instance(session: AsyncSession, instance: ModelT) -> None:
    """Delete an instance and persist the transaction safely."""

    await session.delete(instance)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise CRUDIntegrityError(_extract_integrity_message(exc)) from exc


def _ordered_statement(
    model: type[ModelT],
    *filters: Any,
    order_by: InstrumentedAttribute[Any] | None = None,
    options: Sequence[Any] = (),
) -> Select[tuple[ModelT]]:
    """Build a basic filtered select statement with optional eager loading."""

    statement = select(model).options(*options)
    if filters:
        statement = statement.where(*filters)
    if order_by is not None:
        statement = statement.order_by(order_by)
    return statement


USER_RELATIONSHIPS = (
    selectinload(User.sleep_sessions),
    selectinload(User.sleep_scores),
    selectinload(User.sleep_recommendations),
)
SLEEP_SESSION_RELATIONSHIPS = (
    selectinload(SleepSession.user),
    selectinload(SleepSession.sleep_stages),
    selectinload(SleepSession.sleep_score),
)
SLEEP_SCORE_RELATIONSHIPS = (
    selectinload(SleepScore.user),
    selectinload(SleepScore.sleep_session),
    selectinload(SleepScore.model_version),
    selectinload(SleepScore.recommendations),
)
RECOMMENDATION_RELATIONSHIPS = (
    selectinload(SleepRecommendation.user),
    selectinload(SleepRecommendation.sleep_score),
)


async def create_user(session: AsyncSession, **user_data: Any) -> User:
    """Create and persist a new user record."""

    user = User(**user_data)
    session.add(user)
    return await _commit_refresh(session, user)


async def get_user_by_id(session: AsyncSession, user_id: int) -> User | None:
    """Return a user by primary key, including related sleep data."""

    return await _get_by_id(session, User, user_id, options=USER_RELATIONSHIPS)


async def get_user_by_fitbit_user_id(
    session: AsyncSession,
    fitbit_user_id: str,
) -> User | None:
    """Return a user by Fitbit user identifier."""

    statement = (
        select(User)
        .where(User.fitbit_user_id == fitbit_user_id)
        .options(*USER_RELATIONSHIPS)
    )
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def list_users(session: AsyncSession, *, is_active: bool | None = None) -> list[User]:
    """List users, optionally filtered by active state."""

    filters: list[Any] = []
    if is_active is not None:
        filters.append(User.is_active == is_active)
    statement = _ordered_statement(User, *filters, order_by=User.id, options=USER_RELATIONSHIPS)
    return await _list_records(session, statement)


async def update_user(session: AsyncSession, user_id: int, **updates: Any) -> User | None:
    """Apply partial updates to a user."""

    user = await _get_by_id(session, User, user_id, options=USER_RELATIONSHIPS)
    if user is None:
        return None
    _apply_updates(user, updates)
    return await _commit_refresh(session, user)


async def delete_user(session: AsyncSession, user_id: int) -> bool:
    """Delete a user by id, returning whether a record was removed."""

    user = await _get_by_id(session, User, user_id)
    if user is None:
        return False
    await _delete_instance(session, user)
    return True


async def create_sleep_session(session: AsyncSession, **session_data: Any) -> SleepSession:
    """Create and persist a sleep session."""

    sleep_session = SleepSession(**session_data)
    session.add(sleep_session)
    return await _commit_refresh(session, sleep_session)


async def get_sleep_session_by_id(
    session: AsyncSession,
    sleep_session_id: int,
) -> SleepSession | None:
    """Return a sleep session by primary key."""

    return await _get_by_id(
        session,
        SleepSession,
        sleep_session_id,
        options=SLEEP_SESSION_RELATIONSHIPS,
    )


async def get_sleep_session_by_fitbit_log_id(
    session: AsyncSession,
    fitbit_log_id: int,
) -> SleepSession | None:
    """Return a sleep session by Fitbit log id."""

    statement = (
        select(SleepSession)
        .where(SleepSession.fitbit_log_id == fitbit_log_id)
        .options(*SLEEP_SESSION_RELATIONSHIPS)
    )
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def list_sleep_sessions(
    session: AsyncSession,
    *,
    user_id: int | None = None,
    session_date: date | None = None,
    is_main_sleep: bool | None = None,
) -> list[SleepSession]:
    """List sleep sessions with optional filtering."""

    filters: list[Any] = []
    if user_id is not None:
        filters.append(SleepSession.user_id == user_id)
    if session_date is not None:
        filters.append(SleepSession.session_date == session_date)
    if is_main_sleep is not None:
        filters.append(SleepSession.is_main_sleep == is_main_sleep)

    statement = _ordered_statement(
        SleepSession,
        *filters,
        order_by=SleepSession.start_time,
        options=SLEEP_SESSION_RELATIONSHIPS,
    )
    return await _list_records(session, statement)


async def update_sleep_session(
    session: AsyncSession,
    sleep_session_id: int,
    **updates: Any,
) -> SleepSession | None:
    """Apply partial updates to a sleep session."""

    sleep_session = await _get_by_id(
        session,
        SleepSession,
        sleep_session_id,
        options=SLEEP_SESSION_RELATIONSHIPS,
    )
    if sleep_session is None:
        return None
    _apply_updates(sleep_session, updates)
    return await _commit_refresh(session, sleep_session)


async def delete_sleep_session(session: AsyncSession, sleep_session_id: int) -> bool:
    """Delete a sleep session by id."""

    sleep_session = await _get_by_id(session, SleepSession, sleep_session_id)
    if sleep_session is None:
        return False
    await _delete_instance(session, sleep_session)
    return True


async def create_sleep_stage(session: AsyncSession, **stage_data: Any) -> SleepStage:
    """Create and persist a sleep stage interval."""

    sleep_stage = SleepStage(**stage_data)
    session.add(sleep_stage)
    return await _commit_refresh(session, sleep_stage)


async def get_sleep_stage_by_id(session: AsyncSession, sleep_stage_id: int) -> SleepStage | None:
    """Return a sleep stage by primary key."""

    return await _get_by_id(session, SleepStage, sleep_stage_id, options=(selectinload(SleepStage.sleep_session),))


async def list_sleep_stages(
    session: AsyncSession,
    *,
    sleep_session_id: int | None = None,
    stage_type: str | None = None,
) -> list[SleepStage]:
    """List sleep stage intervals with optional filtering."""

    filters: list[Any] = []
    if sleep_session_id is not None:
        filters.append(SleepStage.sleep_session_id == sleep_session_id)
    if stage_type is not None:
        filters.append(SleepStage.stage_type == stage_type)

    statement = _ordered_statement(
        SleepStage,
        *filters,
        order_by=SleepStage.sequence_index,
        options=(selectinload(SleepStage.sleep_session),),
    )
    return await _list_records(session, statement)


async def create_sleep_stages_bulk(
    session: AsyncSession,
    stages_data: Sequence[dict[str, Any]],
) -> list[SleepStage]:
    """Create and persist multiple sleep stages in one transaction."""

    stages = [SleepStage(**payload) for payload in stages_data]
    session.add_all(stages)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise CRUDIntegrityError(_extract_integrity_message(exc)) from exc

    for stage in stages:
        await session.refresh(stage)
    return stages


async def update_sleep_stage(
    session: AsyncSession,
    sleep_stage_id: int,
    **updates: Any,
) -> SleepStage | None:
    """Apply partial updates to a sleep stage."""

    sleep_stage = await _get_by_id(session, SleepStage, sleep_stage_id)
    if sleep_stage is None:
        return None
    _apply_updates(sleep_stage, updates)
    return await _commit_refresh(session, sleep_stage)


async def delete_sleep_stage(session: AsyncSession, sleep_stage_id: int) -> bool:
    """Delete a sleep stage by id."""

    sleep_stage = await _get_by_id(session, SleepStage, sleep_stage_id)
    if sleep_stage is None:
        return False
    await _delete_instance(session, sleep_stage)
    return True


async def delete_sleep_stages_for_session(session: AsyncSession, sleep_session_id: int) -> int:
    """Delete all sleep stages for a given session, returning row count."""

    statement = delete(SleepStage).where(SleepStage.sleep_session_id == sleep_session_id)
    try:
        result = await session.execute(statement)
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise CRUDIntegrityError(_extract_integrity_message(exc)) from exc
    return int(result.rowcount or 0)


async def create_sleep_score(session: AsyncSession, **score_data: Any) -> SleepScore:
    """Create and persist a sleep score."""

    sleep_score = SleepScore(**score_data)
    session.add(sleep_score)
    return await _commit_refresh(session, sleep_score)


async def get_sleep_score_by_id(session: AsyncSession, sleep_score_id: int) -> SleepScore | None:
    """Return a sleep score by primary key."""

    return await _get_by_id(session, SleepScore, sleep_score_id, options=SLEEP_SCORE_RELATIONSHIPS)


async def get_sleep_score_by_session_id(
    session: AsyncSession,
    sleep_session_id: int,
) -> SleepScore | None:
    """Return the score associated with a specific sleep session."""

    statement = (
        select(SleepScore)
        .where(SleepScore.sleep_session_id == sleep_session_id)
        .options(*SLEEP_SCORE_RELATIONSHIPS)
    )
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def list_sleep_scores(
    session: AsyncSession,
    *,
    user_id: int | None = None,
    score_date: date | None = None,
    model_version_id: int | None = None,
) -> list[SleepScore]:
    """List sleep scores with optional filtering."""

    filters: list[Any] = []
    if user_id is not None:
        filters.append(SleepScore.user_id == user_id)
    if score_date is not None:
        filters.append(SleepScore.score_date == score_date)
    if model_version_id is not None:
        filters.append(SleepScore.model_version_id == model_version_id)

    statement = _ordered_statement(
        SleepScore,
        *filters,
        order_by=SleepScore.score_date,
        options=SLEEP_SCORE_RELATIONSHIPS,
    )
    return await _list_records(session, statement)


async def update_sleep_score(
    session: AsyncSession,
    sleep_score_id: int,
    **updates: Any,
) -> SleepScore | None:
    """Apply partial updates to a sleep score."""

    sleep_score = await _get_by_id(session, SleepScore, sleep_score_id, options=SLEEP_SCORE_RELATIONSHIPS)
    if sleep_score is None:
        return None
    _apply_updates(sleep_score, updates)
    return await _commit_refresh(session, sleep_score)


async def delete_sleep_score(session: AsyncSession, sleep_score_id: int) -> bool:
    """Delete a sleep score by id."""

    sleep_score = await _get_by_id(session, SleepScore, sleep_score_id)
    if sleep_score is None:
        return False
    await _delete_instance(session, sleep_score)
    return True


async def create_recommendation(
    session: AsyncSession,
    **recommendation_data: Any,
) -> SleepRecommendation:
    """Create and persist a sleep recommendation."""

    recommendation = SleepRecommendation(**recommendation_data)
    session.add(recommendation)
    return await _commit_refresh(session, recommendation)


async def get_recommendation_by_id(
    session: AsyncSession,
    recommendation_id: int,
) -> SleepRecommendation | None:
    """Return a recommendation by primary key."""

    return await _get_by_id(
        session,
        SleepRecommendation,
        recommendation_id,
        options=RECOMMENDATION_RELATIONSHIPS,
    )


async def list_recommendations(
    session: AsyncSession,
    *,
    user_id: int | None = None,
    sleep_score_id: int | None = None,
    is_active: bool | None = None,
) -> list[SleepRecommendation]:
    """List recommendations with optional filtering."""

    filters: list[Any] = []
    if user_id is not None:
        filters.append(SleepRecommendation.user_id == user_id)
    if sleep_score_id is not None:
        filters.append(SleepRecommendation.sleep_score_id == sleep_score_id)
    if is_active is not None:
        filters.append(SleepRecommendation.is_active == is_active)

    statement = _ordered_statement(
        SleepRecommendation,
        *filters,
        order_by=SleepRecommendation.recommendation_date,
        options=RECOMMENDATION_RELATIONSHIPS,
    )
    return await _list_records(session, statement)


async def update_recommendation(
    session: AsyncSession,
    recommendation_id: int,
    **updates: Any,
) -> SleepRecommendation | None:
    """Apply partial updates to a recommendation."""

    recommendation = await _get_by_id(
        session,
        SleepRecommendation,
        recommendation_id,
        options=RECOMMENDATION_RELATIONSHIPS,
    )
    if recommendation is None:
        return None
    _apply_updates(recommendation, updates)
    return await _commit_refresh(session, recommendation)


async def delete_recommendation(session: AsyncSession, recommendation_id: int) -> bool:
    """Delete a recommendation by id."""

    recommendation = await _get_by_id(session, SleepRecommendation, recommendation_id)
    if recommendation is None:
        return False
    await _delete_instance(session, recommendation)
    return True


async def create_model_version(session: AsyncSession, **model_data: Any) -> ModelVersion:
    """Create and persist a model version."""

    model_version = ModelVersion(**model_data)
    session.add(model_version)
    return await _commit_refresh(session, model_version)


async def get_model_version_by_id(
    session: AsyncSession,
    model_version_id: int,
) -> ModelVersion | None:
    """Return a model version by primary key."""

    return await _get_by_id(
        session,
        ModelVersion,
        model_version_id,
        options=(selectinload(ModelVersion.sleep_scores),),
    )


async def get_model_version_by_version(
    session: AsyncSession,
    version: str,
) -> ModelVersion | None:
    """Return a model version by its semantic version string."""

    statement = (
        select(ModelVersion)
        .where(ModelVersion.version == version)
        .options(selectinload(ModelVersion.sleep_scores))
    )
    result = await session.execute(statement)
    return result.scalar_one_or_none()


async def list_model_versions(
    session: AsyncSession,
    *,
    model_name: str | None = None,
    is_active: bool | None = None,
) -> list[ModelVersion]:
    """List model versions with optional filtering."""

    filters: list[Any] = []
    if model_name is not None:
        filters.append(ModelVersion.model_name == model_name)
    if is_active is not None:
        filters.append(ModelVersion.is_active == is_active)

    statement = _ordered_statement(
        ModelVersion,
        *filters,
        order_by=ModelVersion.id,
        options=(selectinload(ModelVersion.sleep_scores),),
    )
    return await _list_records(session, statement)


async def update_model_version(
    session: AsyncSession,
    model_version_id: int,
    **updates: Any,
) -> ModelVersion | None:
    """Apply partial updates to a model version."""

    model_version = await _get_by_id(
        session,
        ModelVersion,
        model_version_id,
        options=(selectinload(ModelVersion.sleep_scores),),
    )
    if model_version is None:
        return None
    _apply_updates(model_version, updates)
    return await _commit_refresh(session, model_version)


async def delete_model_version(session: AsyncSession, model_version_id: int) -> bool:
    """Delete a model version by id."""

    model_version = await _get_by_id(session, ModelVersion, model_version_id)
    if model_version is None:
        return False
    await _delete_instance(session, model_version)
    return True


__all__ = [
    "CRUDIntegrityError",
    "create_model_version",
    "create_recommendation",
    "create_sleep_score",
    "create_sleep_session",
    "create_sleep_stage",
    "create_sleep_stages_bulk",
    "create_user",
    "delete_model_version",
    "delete_recommendation",
    "delete_sleep_score",
    "delete_sleep_session",
    "delete_sleep_stage",
    "delete_sleep_stages_for_session",
    "delete_user",
    "get_model_version_by_id",
    "get_model_version_by_version",
    "get_recommendation_by_id",
    "get_sleep_score_by_id",
    "get_sleep_score_by_session_id",
    "get_sleep_session_by_fitbit_log_id",
    "get_sleep_session_by_id",
    "get_sleep_stage_by_id",
    "get_user_by_fitbit_user_id",
    "get_user_by_id",
    "list_model_versions",
    "list_recommendations",
    "list_sleep_scores",
    "list_sleep_sessions",
    "list_sleep_stages",
    "list_users",
    "update_model_version",
    "update_recommendation",
    "update_sleep_score",
    "update_sleep_session",
    "update_sleep_stage",
    "update_user",
]
