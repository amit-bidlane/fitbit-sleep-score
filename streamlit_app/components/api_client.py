"""Streamlit API client for the Fitbit Sleep Score backend."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

import httpx
import streamlit as st


class SleepScoreAPIClient:
    """Small Streamlit-friendly client for the FastAPI sleep score service."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        user_id: int | str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self.base_url = self._resolve_base_url(base_url)
        self.user_id = self._resolve_user_id(user_id)
        self.timeout = timeout

    def get_score(self, target_date: date | datetime | str) -> dict[str, Any]:
        """Return the scored sleep summary for a given date."""

        return self._cached_get(
            self.base_url,
            f"/sleep/score/{self._format_date(target_date)}",
            self._headers(),
            None,
            self.timeout,
        )

    def get_trend(self, *, days: int = 14) -> dict[str, Any]:
        """Return recent score trend data."""

        return self._cached_get(
            self.base_url,
            "/sleep/trend",
            self._headers(),
            {"days": days},
            self.timeout,
        )

    def get_stages(self, target_date: date | datetime | str) -> list[dict[str, Any]]:
        """Return sleep stage intervals for a night."""

        return self._cached_get(
            self.base_url,
            f"/sleep/stages/{self._format_date(target_date)}",
            self._headers(),
            None,
            self.timeout,
        )

    def get_recommendations(self, target_date: date | datetime | str) -> list[dict[str, Any]]:
        """Return recommendations for a given date."""

        return self._cached_get(
            self.base_url,
            f"/sleep/recommendations/{self._format_date(target_date)}",
            self._headers(),
            None,
            self.timeout,
        )

    def get_weekly_analytics(self) -> dict[str, Any]:
        """Return rolling weekly analytics."""

        return self._cached_get(
            self.base_url,
            "/analytics/weekly",
            self._headers(),
            None,
            self.timeout,
        )

    def get_best_night(self) -> dict[str, Any]:
        """Return the user's best night summary."""

        return self._cached_get(
            self.base_url,
            "/analytics/best-night",
            self._headers(),
            None,
            self.timeout,
        )

    def get_worst_night(self) -> dict[str, Any]:
        """Return the user's worst night summary."""

        return self._cached_get(
            self.base_url,
            "/analytics/worst-night",
            self._headers(),
            None,
            self.timeout,
        )

    def analyze_sleep(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Submit one night's sleep data for analysis."""

        normalized_payload = dict(payload)
        response = self._request(
            "POST",
            "/sleep/analyze",
            headers=self._headers(),
            json=normalized_payload,
        )
        self._clear_cached_reads()
        return response

    def get_active_models(self) -> dict[str, Any]:
        """Return active model metadata when the backend exposes it."""

        payload = self._cached_get(
            self.base_url,
            "/models/active",
            self._headers(),
            None,
            self.timeout,
        )
        if isinstance(payload, list):
            return {"models": payload}
        return payload

    def _request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str],
        params: Mapping[str, Any] | None = None,
        json: Mapping[str, Any] | None = None,
    ) -> Any:
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            response = client.request(
                method,
                path,
                headers=headers,
                params=dict(params) if params else None,
                json=dict(json) if json else None,
            )
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()

    def _headers(self) -> dict[str, str]:
        user_id = self._require_user_id()
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-User-Id": str(user_id),
        }

    def _require_user_id(self) -> int | str:
        if self.user_id in (None, ""):
            raise ValueError(
                "SleepScoreAPIClient requires a user_id. Pass one explicitly, "
                "set streamlit session_state['user_id'], or define FITBIT_SLEEP_USER_ID."
            )
        return self.user_id

    @staticmethod
    def _resolve_base_url(base_url: str | None) -> str:
        if base_url:
            return base_url.rstrip("/")

        secrets_value = SleepScoreAPIClient._read_secret(("api_base_url",), ("api", "base_url"))
        env_value = os.getenv("FITBIT_SLEEP_API_URL") or os.getenv("API_BASE_URL")
        resolved = secrets_value or env_value or "http://localhost:8000"
        return str(resolved).rstrip("/")

    @staticmethod
    def _resolve_user_id(user_id: int | str | None) -> int | str | None:
        if user_id not in (None, ""):
            return user_id

        session_user_id = st.session_state.get("user_id")
        if session_user_id not in (None, ""):
            return session_user_id

        secrets_value = SleepScoreAPIClient._read_secret(("user_id",), ("api", "user_id"))
        env_value = os.getenv("FITBIT_SLEEP_USER_ID") or os.getenv("API_USER_ID")
        return secrets_value or env_value

    @staticmethod
    def _read_secret(*paths: tuple[str, ...]) -> Any | None:
        for path in paths:
            current: Any = st.secrets
            try:
                for key in path:
                    current = current[key]
            except Exception:
                continue
            if current not in (None, ""):
                return current
        return None

    @staticmethod
    def _format_date(value: date | datetime | str) -> str:
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return str(value)

    @staticmethod
    @st.cache_data(show_spinner=False, ttl=300)
    def _cached_get(
        base_url: str,
        path: str,
        headers: dict[str, str],
        params: Mapping[str, Any] | None,
        timeout: float,
    ) -> Any:
        with httpx.Client(base_url=base_url, timeout=timeout) as client:
            response = client.get(
                path,
                headers=headers,
                params=dict(params) if params else None,
            )
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _clear_cached_reads() -> None:
        SleepScoreAPIClient._cached_get.clear()
