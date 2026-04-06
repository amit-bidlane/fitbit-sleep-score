"""Async SQLAlchemy database configuration for the Fitbit Sleep Score app."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.database.models import Base

DEFAULT_DATABASE_URL = (
    "mysql+aiomysql://user:password@mysql:3306/sleep_score_db"
    "?charset=utf8mb4"
)


def build_database_url(raw_url: str | None = None) -> str:
    """Return a normalized MySQL async URL with supported connection options."""

    database_url = raw_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL
    url = make_url(database_url)

    if url.drivername != "mysql+aiomysql":
        raise ValueError("DATABASE_URL must use the mysql+aiomysql driver.")
    if url.host not in {None, "", "mysql"}:
        raise ValueError("DATABASE_URL host must be 'mysql' for Docker networking.")

    query = dict(url.query)
    query.setdefault("charset", "utf8mb4")
    query.pop("collation", None)

    normalized = URL.create(
        drivername=url.drivername,
        username=url.username,
        password=url.password,
        host=url.host or "mysql",
        port=url.port or 3306,
        database=url.database or "sleep_score_db",
        query=query,
    )
    return normalized.render_as_string(hide_password=False)


DATABASE_URL = build_database_url()

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session."""

    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    """Create all registered tables during application startup."""

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
