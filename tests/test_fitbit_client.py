"""Tests for the Fitbit Web API client."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest

from src.data.fitbit_client import FitbitClient, FitbitOAuthConfig


class FakeResponse:
    """Minimal response double for requests.Response."""

    def __init__(self, status_code: int, payload, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.ok = 200 <= status_code < 300
        self.content = b"" if payload == {} else b"payload"

    def json(self):
        return self._payload


class FakeSession:
    """Simple scripted session for deterministic client tests."""

    def __init__(self, request_responses=None, post_responses=None):
        self.request_responses = list(request_responses or [])
        self.post_responses = list(post_responses or [])
        self.request_calls = []
        self.post_calls = []

    def request(self, **kwargs):
        self.request_calls.append(kwargs)
        if not self.request_responses:
            raise AssertionError("Unexpected request() call")
        return self.request_responses.pop(0)

    def post(self, url, data, headers, timeout):
        self.post_calls.append(
            {
                "url": url,
                "data": data,
                "headers": headers,
                "timeout": timeout,
            }
        )
        if not self.post_responses:
            raise AssertionError("Unexpected post() call")
        return self.post_responses.pop(0)


def build_config(**overrides) -> FitbitOAuthConfig:
    """Create a baseline test config."""

    base = dict(
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri="http://localhost/callback",
        access_token="access-token",
        refresh_token="refresh-token",
        token_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    base.update(overrides)
    return FitbitOAuthConfig(**base)


def test_build_authorization_url_includes_scopes_and_pkce():
    client = FitbitClient(build_config())
    _, code_challenge = client.generate_pkce_pair()
    url = client.build_authorization_url(state="state-123", code_challenge=code_challenge)

    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    assert parsed.scheme == "https"
    assert params["client_id"] == ["client-id"]
    assert params["state"] == ["state-123"]
    assert params["scope"] == ["sleep heartrate oxygen_saturation respiratory_rate"]
    assert params["code_challenge_method"] == ["S256"]


def test_fetch_sleep_logs_normalizes_nested_payload():
    session = FakeSession(
        request_responses=[
            FakeResponse(
                200,
                {
                    "sleep": [
                        {
                            "logId": 111,
                            "dateOfSleep": "2026-04-01",
                            "startTime": "2026-03-31T23:15:00.000",
                            "endTime": "2026-04-01T07:10:00.000",
                            "minutesAsleep": 440,
                            "levels": {
                                "data": [
                                    {
                                        "dateTime": "2026-03-31T23:15:00.000",
                                        "level": "wake",
                                        "seconds": 300,
                                    }
                                ],
                                "shortData": [
                                    {
                                        "dateTime": "2026-04-01T02:00:00.000",
                                        "level": "wake",
                                        "seconds": 60,
                                    }
                                ],
                            },
                        }
                    ],
                    "summary": {
                        "totalMinutesAsleep": 440,
                        "totalSleepRecords": 1,
                        "totalTimeInBed": 475,
                    },
                },
            )
        ]
    )
    client = FitbitClient(build_config(), session=session)

    frames = client.fetch_sleep_logs("2026-04-01")

    assert list(frames.logs["logId"]) == [111]
    assert list(frames.levels["level"]) == ["wake"]
    assert list(frames.short_levels["seconds"]) == [60]
    assert int(frames.summary.iloc[0]["totalMinutesAsleep"]) == 440
    assert "levels" not in frames.logs.columns


def test_refresh_on_401_then_retry_succeeds():
    session = FakeSession(
        request_responses=[
            FakeResponse(401, {"errors": [{"message": "expired token"}]}),
            FakeResponse(
                200,
                {
                    "spo2": [
                        {
                            "dateTime": "2026-04-01",
                            "value": {"avg": 97.2, "min": 95.0, "max": 99.0},
                        }
                    ]
                },
            ),
        ],
        post_responses=[
            FakeResponse(
                200,
                {
                    "access_token": "fresh-token",
                    "refresh_token": "new-refresh-token",
                    "expires_in": 3600,
                    "scope": "sleep heartrate oxygen_saturation respiratory_rate",
                },
            )
        ],
    )
    client = FitbitClient(
        build_config(access_token="expired-token"),
        session=session,
    )

    frame = client.fetch_spo2("2026-04-01")

    assert client.config.access_token == "fresh-token"
    assert client.config.refresh_token == "new-refresh-token"
    assert session.post_calls[0]["data"]["grant_type"] == "refresh_token"
    assert float(frame.iloc[0]["value_avg"]) == 97.2


def test_rate_limit_response_waits_and_retries():
    waited = []
    session = FakeSession(
        request_responses=[
            FakeResponse(429, {"errors": [{"message": "slow down"}]}, headers={"fitbit-rate-limit-reset": "2"}),
            FakeResponse(
                200,
                {
                    "hrv": [
                        {
                            "dateTime": "2026-04-01",
                            "value": {"dailyRmssd": 52.1, "deepRmssd": 57.8},
                        }
                    ]
                },
            ),
        ]
    )
    client = FitbitClient(build_config(), session=session, sleep_fn=waited.append)

    frame = client.fetch_hrv("2026-04-01")

    assert waited == [2]
    assert float(frame.iloc[0]["value_dailyRmssd"]) == 52.1


def test_breathing_rate_rejects_intervals_over_30_days():
    client = FitbitClient(build_config(), session=FakeSession())

    with pytest.raises(ValueError, match="maximum interval of 30 days"):
        client.fetch_breathing_rate("2026-01-01", "2026-02-15")
