"""Tests for async database configuration helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.database import db


def test_build_database_url_applies_mysql_defaults():
    url = db.build_database_url("mysql+aiomysql://user:password@mysql:3306/sleep_score_db")

    assert url.startswith("mysql+aiomysql://user:password@mysql:3306/sleep_score_db")
    assert "charset=utf8mb4" in url
    assert "collation=" not in url


def test_build_database_url_allows_explicit_hostnames():
    url = db.build_database_url("mysql+aiomysql://user:password@mysql:3306/sleep_score_db")

    assert "@localhost:3306/" in url


@pytest.mark.anyio
async def test_get_db_yields_session(monkeypatch):
    yielded_session = SimpleNamespace(name="session")

    class FakeSessionContext:
        async def __aenter__(self):
            return yielded_session

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class FakeSessionFactory:
        def __call__(self):
            return FakeSessionContext()

    monkeypatch.setattr(db, "AsyncSessionLocal", FakeSessionFactory())

    generator = db.get_db()
    session = await anext(generator)

    assert session is yielded_session

    with pytest.raises(StopAsyncIteration):
        await anext(generator)


@pytest.mark.anyio
async def test_init_db_creates_tables(monkeypatch):
    called = {}

    async def fake_run_sync(callback):
        called["callback"] = callback

    class FakeConnection:
        async def run_sync(self, callback):
            await fake_run_sync(callback)

    class FakeBeginContext:
        async def __aenter__(self):
            return FakeConnection()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class FakeEngine:
        def begin(self):
            return FakeBeginContext()

    monkeypatch.setattr(db, "engine", FakeEngine())

    await db.init_db()

    assert called["callback"] == db.Base.metadata.create_all
