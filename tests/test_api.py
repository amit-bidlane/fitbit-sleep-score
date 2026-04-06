"""Tests for the FastAPI backend routes."""

from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi.testclient import TestClient

from src.api.routes import create_app
from src.database.db import get_db


def _user() -> SimpleNamespace:
    now = datetime(2026, 4, 5, 8, 0, tzinfo=timezone.utc)
    return SimpleNamespace(
        id=7,
        fitbit_user_id="fitbit-007",
        email="user@example.com",
        full_name="Taylor User",
        birth_date=None,
        sex=None,
        timezone="UTC",
        is_active=True,
        created_at=now,
        updated_at=now,
        sleep_sessions=[],
        sleep_scores=[],
        sleep_recommendations=[],
    )


def _score(target_date: date, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        id=33,
        score_date=target_date,
        overall_score=score,
        confidence=None,
        sleep_efficiency_score=91.0,
        duration_score=87.0,
        continuity_score=84.0,
        recovery_score=89.0,
        feature_snapshot={
            "label": "Good",
            "total_sleep_minutes": 450,
            "sleep_efficiency": 91.0,
            "recommendations": [],
        },
        recommendations=[],
    )


def _model_version() -> SimpleNamespace:
    now = datetime(2026, 4, 5, 8, 0, tzinfo=timezone.utc)
    return SimpleNamespace(
        id=5,
        model_name="sleep-score-ensemble",
        version="1.2.0",
        framework="PyTorch",
        artifact_uri="checkpoints/sleep_classifier.pt",
        training_metrics={"accuracy": 0.91},
        feature_schema={"fields": ["total_sleep_minutes"]},
        notes="Primary production model",
        is_active=True,
        trained_at=now,
        created_at=now,
        updated_at=now,
        sleep_scores=[],
    )


def _session(target_date: date) -> SimpleNamespace:
    return SimpleNamespace(
        id=21,
        fitbit_log_id=9001,
        session_date=target_date,
        sleep_stages=[
            SimpleNamespace(
                id=1,
                stage_type="light",
                started_at=datetime(2026, 4, 5, 22, 0, tzinfo=timezone.utc),
                ended_at=datetime(2026, 4, 5, 22, 30, tzinfo=timezone.utc),
                seconds=1800,
                sequence_index=0,
            )
        ],
    )


async def _fake_db():
    yield object()


class _FakeTransaction:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        self.session.begin_called = True
        return self.session

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakeDBSession:
    def __init__(self):
        self.begin_called = False
        self._id_counter = 20
        self._objects = []

    def begin(self):
        return _FakeTransaction(self)

    def add(self, obj):
        self._objects.append(obj)

    def add_all(self, objects):
        self._objects.extend(objects)

    async def flush(self):
        for obj in self._objects:
            if getattr(obj, "id", None) is None:
                self._id_counter += 1
                obj.id = self._id_counter


async def _fake_analysis_db():
    yield _FakeDBSession()


def test_auth_routes(monkeypatch):
    app = create_app(run_startup=False)
    app.dependency_overrides[get_db] = _fake_db
    client = TestClient(app)
    user = _user()

    async def fake_get_user_by_fitbit_user_id(db, fitbit_user_id):
        return None

    async def fake_create_user(db, **user_data):
        return user

    async def fake_get_user_by_id(db, user_id):
        return user if user_id == 7 else None

    monkeypatch.setattr("src.api.routes.get_user_by_fitbit_user_id", fake_get_user_by_fitbit_user_id)
    monkeypatch.setattr("src.api.routes.create_user", fake_create_user)
    monkeypatch.setattr("src.api.routes.get_user_by_id", fake_get_user_by_id)

    register_response = client.post(
        "/auth/register",
        json={"fitbit_user_id": "fitbit-007", "email": "user@example.com", "full_name": "Taylor User", "timezone": "UTC"},
    )
    me_response = client.get("/auth/me", headers={"X-User-Id": "7"})

    assert register_response.status_code == 201
    assert register_response.json()["fitbit_user_id"] == "fitbit-007"
    assert me_response.status_code == 200
    assert me_response.json()["id"] == 7


def test_sleep_read_routes(monkeypatch):
    app = create_app(run_startup=False)
    app.dependency_overrides[get_db] = _fake_db
    client = TestClient(app)
    user = _user()
    scores = [_score(date(2026, 4, 4), 82.0), _score(date(2026, 4, 5), 91.0)]
    sessions = [_session(date(2026, 4, 5))]

    async def fake_get_user_by_id(db, user_id):
        return user

    async def fake_list_sleep_scores(db, **kwargs):
        score_date = kwargs.get("score_date")
        if score_date is None:
            return scores
        return [score for score in scores if score.score_date == score_date]

    async def fake_list_sleep_sessions(db, **kwargs):
        session_date = kwargs.get("session_date")
        if session_date is None:
            return sessions
        return [session for session in sessions if session.session_date == session_date]

    monkeypatch.setattr("src.api.routes.get_user_by_id", fake_get_user_by_id)
    monkeypatch.setattr("src.api.routes.list_sleep_scores", fake_list_sleep_scores)
    monkeypatch.setattr("src.api.routes.list_sleep_sessions", fake_list_sleep_sessions)

    headers = {"X-User-Id": "7"}
    score_response = client.get("/sleep/score/2026-04-05", headers=headers)
    trend_response = client.get("/sleep/trend?days=10", headers=headers)
    stages_response = client.get("/sleep/stages/2026-04-05", headers=headers)
    best_response = client.get("/analytics/best-night", headers=headers)
    weekly_response = client.get("/analytics/weekly", headers=headers)

    assert score_response.status_code == 200
    assert score_response.json()["overall_score"] == 91.0
    assert trend_response.status_code == 200
    assert len(trend_response.json()["points"]) == 2
    assert stages_response.status_code == 200
    assert stages_response.json()[0]["stage_type"] == "light"
    assert best_response.status_code == 200
    assert best_response.json()["overall_score"] == 91.0
    assert weekly_response.status_code == 200
    assert weekly_response.json()["nights_tracked"] >= 1


def test_model_routes(monkeypatch):
    app = create_app(run_startup=False)
    app.dependency_overrides[get_db] = _fake_db
    client = TestClient(app)
    user = _user()
    active_models = [_model_version()]

    async def fake_get_user_by_id(db, user_id):
        return user

    async def fake_list_model_versions(db, **kwargs):
        assert kwargs == {"is_active": True}
        return active_models

    monkeypatch.setattr("src.api.routes.get_user_by_id", fake_get_user_by_id)
    monkeypatch.setattr("src.api.routes.list_model_versions", fake_list_model_versions)

    response = client.get("/models/active", headers={"X-User-Id": "7"})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["version"] == "1.2.0"
    assert payload[0]["is_active"] is True


def test_sleep_read_routes_use_user_timezone(monkeypatch):
    app = create_app(run_startup=False)
    app.dependency_overrides[get_db] = _fake_db
    client = TestClient(app)
    user = _user()
    user.timezone = "America/Los_Angeles"
    scores = [_score(date(2026, 4, 4), 82.0), _score(date(2026, 4, 5), 91.0)]

    async def fake_get_user_by_id(db, user_id):
        return user

    async def fake_list_sleep_scores(db, **kwargs):
        return scores

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            moment = datetime(2026, 4, 6, 0, 30, tzinfo=timezone.utc)
            if tz is None:
                return moment.replace(tzinfo=None)
            return moment.astimezone(tz)

    monkeypatch.setattr("src.api.routes.get_user_by_id", fake_get_user_by_id)
    monkeypatch.setattr("src.api.routes.list_sleep_scores", fake_list_sleep_scores)
    monkeypatch.setattr("src.api.routes.datetime", FixedDateTime)

    trend_response = client.get("/sleep/trend?days=1", headers={"X-User-Id": "7"})
    weekly_response = client.get("/analytics/weekly", headers={"X-User-Id": "7"})

    assert trend_response.status_code == 200
    assert [point["date"] for point in trend_response.json()["points"]] == ["2026-04-05"]
    assert weekly_response.status_code == 200
    assert weekly_response.json()["week_end"] == "2026-04-05"


def test_sleep_analyze_route(monkeypatch):
    app = create_app(run_startup=False)
    app.dependency_overrides[get_db] = _fake_analysis_db
    client = TestClient(app)
    user = _user()

    async def fake_get_user_by_id(db, user_id):
        return user

    def fake_engineer_sleep_features(*, sleep_data, hrv_data=None, heart_rate_data=None, spo2_data=None):
        return pd.DataFrame(
            [
                {
                    "total_sleep_minutes": 450,
                    "sleep_efficiency": 93.8,
                    "duration_score": 86.5,
                    "continuity_score": 84.2,
                    "recovery_score": 88.1,
                }
            ]
        )

    def fake_calculate_sleep_score(frame):
        return pd.DataFrame(
            [
                {
                    "total_sleep_minutes": 450,
                    "sleep_efficiency": 93.8,
                    "duration_score": 86.5,
                    "continuity_score": 84.2,
                    "recovery_score": 88.1,
                    "final_score": 88.4,
                    "label": "Good",
                    "recommendations": [],
                }
            ]
        )

    monkeypatch.setattr("src.api.routes.get_user_by_id", fake_get_user_by_id)
    monkeypatch.setattr("src.api.routes.engineer_sleep_features", fake_engineer_sleep_features)
    monkeypatch.setattr("src.api.routes.calculate_sleep_score", fake_calculate_sleep_score)

    response = client.post(
        "/sleep/analyze",
        headers={"X-User-Id": "7"},
        json={
            "date": "2026-04-05",
            "start_time": "2026-04-05T22:00:00Z",
            "end_time": "2026-04-06T05:30:00Z",
            "time_in_bed": 480,
            "minutes_asleep": 450,
            "minutes_awake": 30,
            "minutes_to_fall_asleep": 10,
            "awakenings_count": 1,
            "efficiency": 93.8,
            "hrv": 55.0,
            "resting_hr": 57.0,
            "spo2": 97.0,
            "stages": [
                {"stage_type": "light", "started_at": "2026-04-05T22:00:00Z", "seconds": 1800},
                {"stage_type": "deep", "started_at": "2026-04-05T22:30:00Z", "seconds": 3600},
            ],
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["sleep_session_id"] >= 21
    assert payload["score"]["overall_score"] == 88.4
    assert len(payload["stages"]) == 2


def test_sleep_analyze_invalid_stage_does_not_start_transaction(monkeypatch):
    fake_db = _FakeDBSession()

    async def fake_get_user_by_id(db, user_id):
        return _user()

    async def _fake_db_with_invalid_stage():
        yield fake_db

    app = create_app(run_startup=False)
    app.dependency_overrides[get_db] = _fake_db_with_invalid_stage
    client = TestClient(app)

    monkeypatch.setattr("src.api.routes.get_user_by_id", fake_get_user_by_id)

    response = client.post(
        "/sleep/analyze",
        headers={"X-User-Id": "7"},
        json={
            "date": "2026-04-05",
            "start_time": "2026-04-05T22:00:00Z",
            "end_time": "2026-04-06T05:30:00Z",
            "time_in_bed": 480,
            "minutes_asleep": 450,
            "minutes_awake": 30,
            "stages": [
                {"stage_type": "light", "started_at": "2026-04-05T22:00:00Z"}
            ],
        },
    )

    assert response.status_code == 422
    assert fake_db.begin_called is False
