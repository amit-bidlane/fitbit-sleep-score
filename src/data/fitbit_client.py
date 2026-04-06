"""Production-ready Fitbit Web API client for sleep-focused analytics."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests
import yaml

LOGGER = logging.getLogger(__name__)

DEFAULT_SCOPES = (
    "sleep",
    "heartrate",
    "oxygen_saturation",
    "respiratory_rate",
)


class FitbitClientError(RuntimeError):
    """Base exception for Fitbit client failures."""


class FitbitAuthError(FitbitClientError):
    """Raised for OAuth authentication or token lifecycle errors."""


class FitbitRateLimitError(FitbitClientError):
    """Raised when Fitbit API rate limits prevent progress."""


class FitbitAPIError(FitbitClientError):
    """Raised when the Fitbit API returns an error response."""


@dataclass(slots=True)
class RateLimitState:
    """Latest observed Fitbit rate-limit headers."""

    limit: int | None = None
    remaining: int | None = None
    reset_after_seconds: int | None = None
    observed_at: datetime | None = None


@dataclass(slots=True)
class FitbitOAuthConfig:
    """Configuration required to authenticate against the Fitbit Web API."""

    client_id: str
    client_secret: str
    redirect_uri: str
    access_token: str | None = None
    refresh_token: str | None = None
    token_expires_at: datetime | None = None
    scopes: tuple[str, ...] = DEFAULT_SCOPES
    user_id: str = "-"
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    backoff_factor: float = 1.5
    token_expiry_buffer_seconds: int = 60
    auto_refresh: bool = True
    max_rate_limit_wait_seconds: int = 120
    api_base_url: str = "https://api.fitbit.com"
    authorize_url: str = "https://www.fitbit.com/oauth2/authorize"
    token_url: str = "https://api.fitbit.com/oauth2/token"
    accept_language: str = "en_US"
    accept_locale: str = "en_US"
    token_cache_path: Path | None = None

    @classmethod
    def from_env(cls) -> "FitbitOAuthConfig":
        """Build configuration from environment variables."""

        client_id = os.getenv("FITBIT_CLIENT_ID")
        client_secret = os.getenv("FITBIT_CLIENT_SECRET")
        redirect_uri = os.getenv("FITBIT_REDIRECT_URI")
        if not client_id or not client_secret or not redirect_uri:
            raise FitbitAuthError(
                "FITBIT_CLIENT_ID, FITBIT_CLIENT_SECRET, and FITBIT_REDIRECT_URI must be set."
            )

        scopes = _parse_scopes(os.getenv("FITBIT_SCOPES")) or DEFAULT_SCOPES
        expires_at = _parse_datetime_like(os.getenv("FITBIT_TOKEN_EXPIRES_AT"))
        token_cache_path = os.getenv("FITBIT_TOKEN_CACHE_PATH")

        return cls(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            access_token=os.getenv("FITBIT_ACCESS_TOKEN"),
            refresh_token=os.getenv("FITBIT_REFRESH_TOKEN"),
            token_expires_at=expires_at,
            scopes=scopes,
            user_id=os.getenv("FITBIT_USER_ID", "-"),
            timeout_seconds=float(os.getenv("FITBIT_TIMEOUT_SECONDS", "30")),
            retry_attempts=int(os.getenv("FITBIT_RETRY_ATTEMPTS", "3")),
            backoff_factor=float(os.getenv("FITBIT_BACKOFF_FACTOR", "1.5")),
            token_expiry_buffer_seconds=int(
                os.getenv("FITBIT_TOKEN_EXPIRY_BUFFER_SECONDS", "60")
            ),
            auto_refresh=os.getenv("FITBIT_AUTO_REFRESH", "true").lower() == "true",
            max_rate_limit_wait_seconds=int(
                os.getenv("FITBIT_MAX_RATE_LIMIT_WAIT_SECONDS", "120")
            ),
            accept_language=os.getenv("FITBIT_ACCEPT_LANGUAGE", "en_US"),
            accept_locale=os.getenv("FITBIT_ACCEPT_LOCALE", "en_US"),
            token_cache_path=Path(token_cache_path) if token_cache_path else None,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FitbitOAuthConfig":
        """Build configuration from a YAML file."""

        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        fitbit = raw.get("fitbit", raw)

        try:
            client_id = fitbit["client_id"]
            client_secret = fitbit["client_secret"]
            redirect_uri = fitbit["redirect_uri"]
        except KeyError as exc:
            raise FitbitAuthError(f"Missing required Fitbit config key: {exc.args[0]}") from exc

        token_cache_path = fitbit.get("token_cache_path")
        if token_cache_path:
            token_cache_path = (config_path.parent / token_cache_path).resolve()

        return cls(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            access_token=fitbit.get("access_token"),
            refresh_token=fitbit.get("refresh_token"),
            token_expires_at=_parse_datetime_like(fitbit.get("token_expires_at")),
            scopes=_parse_scopes(fitbit.get("scopes")) or DEFAULT_SCOPES,
            user_id=fitbit.get("user_id", "-"),
            timeout_seconds=float(fitbit.get("timeout_seconds", 30)),
            retry_attempts=int(fitbit.get("retry_attempts", 3)),
            backoff_factor=float(fitbit.get("backoff_factor", 1.5)),
            token_expiry_buffer_seconds=int(
                fitbit.get("token_expiry_buffer_seconds", 60)
            ),
            auto_refresh=bool(fitbit.get("auto_refresh", True)),
            max_rate_limit_wait_seconds=int(
                fitbit.get("max_rate_limit_wait_seconds", 120)
            ),
            api_base_url=fitbit.get("api_base_url", "https://api.fitbit.com"),
            authorize_url=fitbit.get(
                "authorize_url", "https://www.fitbit.com/oauth2/authorize"
            ),
            token_url=fitbit.get("token_url", "https://api.fitbit.com/oauth2/token"),
            accept_language=fitbit.get("accept_language", "en_US"),
            accept_locale=fitbit.get("accept_locale", "en_US"),
            token_cache_path=token_cache_path,
        )


@dataclass(slots=True)
class SleepLogFrames:
    """Structured DataFrames returned from the sleep log endpoint."""

    logs: pd.DataFrame = field(default_factory=pd.DataFrame)
    levels: pd.DataFrame = field(default_factory=pd.DataFrame)
    short_levels: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(slots=True)
class HeartRateFrames:
    """Structured DataFrames returned from the heart-rate endpoint."""

    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    zones: pd.DataFrame = field(default_factory=pd.DataFrame)
    custom_zones: pd.DataFrame = field(default_factory=pd.DataFrame)
    intraday: pd.DataFrame = field(default_factory=pd.DataFrame)


class FitbitClient:
    """OAuth-aware Fitbit Web API client with retries and DataFrame normalization."""

    def __init__(
        self,
        config: FitbitOAuthConfig,
        session: requests.Session | None = None,
        sleep_fn: Any = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.sleep_fn = sleep_fn
        self.logger = logger or LOGGER
        self.rate_limit = RateLimitState()
        self._load_cached_tokens()

    @staticmethod
    def generate_pkce_pair() -> tuple[str, str]:
        """Generate a PKCE verifier/challenge pair for the authorization flow."""

        verifier = secrets.token_urlsafe(72)[:96]
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("utf-8")).digest()
        ).rstrip(b"=")
        return verifier, challenge.decode("utf-8")

    def build_authorization_url(
        self,
        state: str,
        scopes: tuple[str, ...] | list[str] | None = None,
        code_challenge: str | None = None,
        code_challenge_method: str = "S256",
    ) -> str:
        """Build the Fitbit OAuth authorization URL."""

        requested_scopes = tuple(scopes) if scopes else self.config.scopes
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(requested_scopes),
            "state": state,
        }
        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = code_challenge_method
        return f"{self.config.authorize_url}?{urlencode(params)}"

    def exchange_code_for_token(
        self,
        code: str,
        code_verifier: str | None = None,
    ) -> dict[str, Any]:
        """Exchange an OAuth authorization code for access and refresh tokens."""

        form_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.config.redirect_uri,
        }
        if code_verifier:
            form_data["code_verifier"] = code_verifier
        return self._token_request(form_data)

    def refresh_access_token(self, force: bool = False) -> dict[str, Any]:
        """Refresh the Fitbit access token if it is expired or near expiry."""

        if not self.config.refresh_token:
            raise FitbitAuthError("No refresh token is configured for Fitbit OAuth.")

        if not force and not self._token_needs_refresh():
            return self.export_token_state()

        return self._token_request(
            {
                "grant_type": "refresh_token",
                "refresh_token": self.config.refresh_token,
            }
        )

    def export_token_state(self) -> dict[str, Any]:
        """Return the current token state in a serializable format."""

        return {
            "access_token": self.config.access_token,
            "refresh_token": self.config.refresh_token,
            "token_expires_at": self.config.token_expires_at.isoformat()
            if self.config.token_expires_at
            else None,
            "scopes": list(self.config.scopes),
        }

    def fetch_sleep_logs(self, sleep_date: str | date | datetime) -> SleepLogFrames:
        """Fetch Fitbit sleep logs for a single date."""

        sleep_date_str = _coerce_date(sleep_date).isoformat()
        payload = self._request(
            "GET",
            f"/1.2/user/{self.config.user_id}/sleep/date/{sleep_date_str}.json",
        )

        raw_logs = payload.get("sleep", [])
        log_rows: list[dict[str, Any]] = []
        levels_rows: list[dict[str, Any]] = []
        short_levels_rows: list[dict[str, Any]] = []

        for raw_log in raw_logs:
            flat_log = {key: value for key, value in raw_log.items() if key != "levels"}
            log_rows.append(flat_log)

            levels = raw_log.get("levels", {})
            for item in levels.get("data", []):
                levels_rows.append(
                    {
                        "logId": raw_log.get("logId"),
                        "dateOfSleep": raw_log.get("dateOfSleep"),
                        "level": item.get("level"),
                        "seconds": item.get("seconds"),
                        "dateTime": item.get("dateTime"),
                    }
                )

            for item in levels.get("shortData", []):
                short_levels_rows.append(
                    {
                        "logId": raw_log.get("logId"),
                        "dateOfSleep": raw_log.get("dateOfSleep"),
                        "level": item.get("level"),
                        "seconds": item.get("seconds"),
                        "dateTime": item.get("dateTime"),
                    }
                )

        logs_df = _json_to_dataframe(log_rows)
        levels_df = _json_to_dataframe(levels_rows)
        short_levels_df = _json_to_dataframe(short_levels_rows)
        summary_df = _json_to_dataframe([payload["summary"]]) if "summary" in payload else pd.DataFrame()
        meta_df = _json_to_dataframe([payload["meta"]]) if "meta" in payload else pd.DataFrame()

        _parse_datetime_columns(logs_df, ["startTime", "endTime"])
        _parse_datetime_columns(levels_df, ["dateTime"])
        _parse_datetime_columns(short_levels_df, ["dateTime"])
        _parse_datetime_columns(logs_df, ["dateOfSleep"], utc=False)
        _parse_datetime_columns(levels_df, ["dateOfSleep"], utc=False)
        _parse_datetime_columns(short_levels_df, ["dateOfSleep"], utc=False)

        return SleepLogFrames(
            logs=logs_df,
            levels=levels_df,
            short_levels=short_levels_df,
            summary=summary_df,
            meta=meta_df,
        )

    def fetch_heart_rate_time_series(
        self,
        start_date: str | date | datetime,
        end_date: str | date | datetime | None = None,
        detail_level: str = "1min",
        start_time: str | None = None,
        end_time: str | None = None,
        timezone_name: str | None = None,
    ) -> HeartRateFrames:
        """Fetch Fitbit heart rate summary and intraday time series data."""

        if detail_level not in {"1sec", "1min"}:
            raise ValueError("detail_level must be either '1sec' or '1min'.")

        start_date_obj = _coerce_date(start_date)
        end_date_obj = _coerce_date(end_date or start_date_obj)

        endpoint = (
            f"/1/user/{self.config.user_id}/activities/heart/date/"
            f"{start_date_obj.isoformat()}/{end_date_obj.isoformat()}/{detail_level}.json"
        )
        if start_time and end_time:
            endpoint = endpoint.replace(
                ".json", f"/time/{start_time}/{end_time}.json"
            )

        payload = self._request(
            "GET",
            endpoint,
            params={"timezone": timezone_name} if timezone_name else None,
        )

        raw_summary = payload.get("activities-heart", [])
        summary_rows: list[dict[str, Any]] = []
        zone_rows: list[dict[str, Any]] = []
        custom_zone_rows: list[dict[str, Any]] = []

        for item in raw_summary:
            value = item.get("value", {})
            summary_rows.append(
                {
                    "dateTime": item.get("dateTime"),
                    "restingHeartRate": value.get("restingHeartRate"),
                }
            )
            for zone in value.get("heartRateZones", []):
                zone_rows.append({"dateTime": item.get("dateTime"), **zone})
            for zone in value.get("customHeartRateZones", []):
                custom_zone_rows.append({"dateTime": item.get("dateTime"), **zone})

        intraday_section = payload.get("activities-heart-intraday", {})
        intraday_rows: list[dict[str, Any]] = []
        anchor_date = summary_rows[0]["dateTime"] if summary_rows else start_date_obj.isoformat()
        for point in intraday_section.get("dataset", []):
            timestamp = f"{anchor_date}T{point['time']}" if point.get("time") else None
            intraday_rows.append(
                {
                    "date": anchor_date,
                    "time": point.get("time"),
                    "value": point.get("value"),
                    "datasetInterval": intraday_section.get("datasetInterval"),
                    "datasetType": intraday_section.get("datasetType"),
                    "timestamp": timestamp,
                }
            )

        summary_df = _json_to_dataframe(summary_rows)
        zones_df = _json_to_dataframe(zone_rows)
        custom_zones_df = _json_to_dataframe(custom_zone_rows)
        intraday_df = _json_to_dataframe(intraday_rows)

        _parse_datetime_columns(summary_df, ["dateTime"], utc=False)
        _parse_datetime_columns(zones_df, ["dateTime"], utc=False)
        _parse_datetime_columns(custom_zones_df, ["dateTime"], utc=False)
        _parse_datetime_columns(intraday_df, ["timestamp"], utc=False)

        return HeartRateFrames(
            summary=summary_df,
            zones=zones_df,
            custom_zones=custom_zones_df,
            intraday=intraday_df,
        )

    def fetch_spo2(
        self,
        start_date: str | date | datetime,
        end_date: str | date | datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch Fitbit SpO2 summaries for a date or interval."""

        start_date_obj, end_date_obj = _normalize_range(start_date, end_date)
        payload = self._request(
            "GET",
            f"/1/user/{self.config.user_id}/spo2/date/"
            f"{start_date_obj.isoformat()}/{end_date_obj.isoformat()}.json",
        )

        if isinstance(payload, dict):
            records = payload.get("spo2", [])
        else:
            records = payload

        frame = _json_to_dataframe(records)
        _parse_datetime_columns(frame, ["dateTime"], utc=False)
        return frame

    def fetch_hrv(
        self,
        start_date: str | date | datetime,
        end_date: str | date | datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch Fitbit HRV summaries for up to 30 days."""

        start_date_obj, end_date_obj = _normalize_range(start_date, end_date, max_days=30)
        payload = self._request(
            "GET",
            f"/1/user/{self.config.user_id}/hrv/date/"
            f"{start_date_obj.isoformat()}/{end_date_obj.isoformat()}.json",
        )
        frame = _json_to_dataframe(payload.get("hrv", []))
        _parse_datetime_columns(frame, ["dateTime"], utc=False)
        return frame

    def fetch_breathing_rate(
        self,
        start_date: str | date | datetime,
        end_date: str | date | datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch Fitbit nightly breathing-rate summaries for up to 30 days."""

        start_date_obj, end_date_obj = _normalize_range(start_date, end_date, max_days=30)
        payload = self._request(
            "GET",
            f"/1/user/{self.config.user_id}/br/date/"
            f"{start_date_obj.isoformat()}/{end_date_obj.isoformat()}.json",
        )
        frame = _json_to_dataframe(payload.get("br", []))
        _parse_datetime_columns(frame, ["dateTime"], utc=False)
        return frame

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """Execute a Fitbit API request with refresh, retry, and error handling."""

        self._ensure_access_token()
        url = endpoint if endpoint.startswith("http") else f"{self.config.api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        attempts = self.config.retry_attempts + 1
        auth_retry_used = False

        for attempt in range(1, attempts + 1):
            request_headers = self._request_headers()
            if headers:
                request_headers.update(headers)

            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    headers=request_headers,
                    timeout=self.config.timeout_seconds,
                )
            except requests.RequestException as exc:
                if attempt < attempts:
                    self.sleep_fn(self._backoff_delay(attempt))
                    continue
                raise FitbitClientError(f"Fitbit request failed: {exc}") from exc

            self._update_rate_limit_state(response.headers)

            if response.status_code == 401 and not auth_retry_used and self.config.auto_refresh:
                auth_retry_used = True
                self.refresh_access_token(force=True)
                continue

            if response.status_code == 429:
                wait_time = self._get_retry_delay(response.headers)
                if attempt < attempts and wait_time <= self.config.max_rate_limit_wait_seconds:
                    self.logger.warning("Fitbit rate limit hit; retrying in %s seconds.", wait_time)
                    self.sleep_fn(wait_time)
                    continue
                raise FitbitRateLimitError(
                    f"Fitbit API rate limit reached; retry after {wait_time} seconds."
                )

            if 500 <= response.status_code < 600 and attempt < attempts:
                self.sleep_fn(self._backoff_delay(attempt))
                continue

            if response.ok:
                if response.status_code == 204 or not getattr(response, "content", b""):
                    return {}
                try:
                    return response.json()
                except ValueError as exc:
                    raise FitbitAPIError("Fitbit API returned invalid JSON.") from exc

            raise self._build_api_error(response)

        raise FitbitClientError("Fitbit request attempts were exhausted.")

    def _token_request(self, form_data: dict[str, str]) -> dict[str, Any]:
        """Perform a Fitbit token endpoint request and persist the result."""

        auth_token = base64.b64encode(
            f"{self.config.client_id}:{self.config.client_secret}".encode("utf-8")
        ).decode("utf-8")

        headers = {
            "Authorization": f"Basic {auth_token}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        try:
            response = self.session.post(
                self.config.token_url,
                data=form_data,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise FitbitAuthError(f"Fitbit token request failed: {exc}") from exc

        if not response.ok:
            raise self._build_api_error(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise FitbitAuthError("Fitbit token endpoint returned invalid JSON.") from exc

        self.config.access_token = payload.get("access_token")
        self.config.refresh_token = payload.get("refresh_token", self.config.refresh_token)
        self.config.scopes = _parse_scopes(payload.get("scope")) or self.config.scopes

        expires_in = payload.get("expires_in")
        if expires_in is not None:
            self.config.token_expires_at = datetime.now(timezone.utc) + timedelta(
                seconds=int(expires_in)
            )

        self._persist_tokens()
        return payload

    def _request_headers(self) -> dict[str, str]:
        """Build the standard Fitbit request headers."""

        if not self.config.access_token:
            raise FitbitAuthError("Fitbit access token is missing.")

        return {
            "Authorization": f"Bearer {self.config.access_token}",
            "Accept": "application/json",
            "Accept-Language": self.config.accept_language,
            "Accept-Locale": self.config.accept_locale,
        }

    def _ensure_access_token(self) -> None:
        """Refresh the access token if needed before making a request."""

        if self.config.access_token and not self._token_needs_refresh():
            return

        if self.config.auto_refresh and self.config.refresh_token:
            self.refresh_access_token(force=True)
            return

        raise FitbitAuthError(
            "An access token is required. Exchange an authorization code or configure a refresh token."
        )

    def _token_needs_refresh(self) -> bool:
        """Return True when the token is expired or close to expiry."""

        if not self.config.access_token:
            return True
        if self.config.token_expires_at is None:
            return False

        now = datetime.now(timezone.utc)
        expires_at = _ensure_timezone(self.config.token_expires_at)
        return expires_at <= now + timedelta(
            seconds=self.config.token_expiry_buffer_seconds
        )

    def _update_rate_limit_state(self, headers: dict[str, Any]) -> None:
        """Capture Fitbit's latest rate-limit headers."""

        self.rate_limit = RateLimitState(
            limit=_int_or_none(headers.get("fitbit-rate-limit-limit")),
            remaining=_int_or_none(headers.get("fitbit-rate-limit-remaining")),
            reset_after_seconds=_int_or_none(headers.get("fitbit-rate-limit-reset")),
            observed_at=datetime.now(timezone.utc),
        )

    def _get_retry_delay(self, headers: dict[str, Any]) -> int:
        """Calculate retry delay from Fitbit rate-limit headers."""

        retry_after = _int_or_none(headers.get("Retry-After"))
        reset_after = _int_or_none(headers.get("fitbit-rate-limit-reset"))
        wait_time = retry_after or reset_after or 1
        return max(1, min(wait_time, self.config.max_rate_limit_wait_seconds))

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff delay for transient failures."""

        delay = self.config.backoff_factor * (2 ** (attempt - 1))
        return min(delay, float(self.config.max_rate_limit_wait_seconds))

    def _build_api_error(self, response: requests.Response) -> FitbitClientError:
        """Translate an HTTP error response into a specific client exception."""

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        message = f"Fitbit API returned {response.status_code}."
        errors = payload.get("errors", []) if isinstance(payload, dict) else []
        if errors:
            formatted = "; ".join(
                filter(
                    None,
                    (
                        " | ".join(
                            part
                            for part in (
                                str(item.get("errorType", "")).strip(),
                                str(item.get("fieldName", "")).strip(),
                                str(item.get("message", "")).strip(),
                            )
                            if part
                        )
                        for item in errors
                    ),
                )
            )
            if formatted:
                message = f"{message} {formatted}"

        if response.status_code == 401:
            return FitbitAuthError(message)
        if response.status_code == 429:
            return FitbitRateLimitError(message)
        return FitbitAPIError(message)

    def _load_cached_tokens(self) -> None:
        """Load persisted tokens from disk when a cache path is configured."""

        cache_path = self.config.token_cache_path
        if not cache_path or not cache_path.exists():
            return

        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning("Unable to load Fitbit token cache: %s", exc)
            return

        self.config.access_token = self.config.access_token or payload.get("access_token")
        self.config.refresh_token = self.config.refresh_token or payload.get("refresh_token")
        if self.config.token_expires_at is None:
            self.config.token_expires_at = _parse_datetime_like(payload.get("token_expires_at"))

    def _persist_tokens(self) -> None:
        """Persist token state to disk when configured."""

        cache_path = self.config.token_cache_path
        if not cache_path:
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(self.export_token_state(), indent=2),
            encoding="utf-8",
        )


def _parse_scopes(raw_scopes: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize scopes from env, config, or token payloads."""

    if raw_scopes is None:
        return ()
    if isinstance(raw_scopes, (list, tuple)):
        return tuple(str(scope).strip() for scope in raw_scopes if str(scope).strip())
    normalized = raw_scopes.replace(",", " ")
    return tuple(scope for scope in normalized.split() if scope)


def _json_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Safely normalize a list of dictionaries into a DataFrame."""

    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records, sep="_")


def _parse_datetime_columns(
    frame: pd.DataFrame,
    columns: list[str],
    utc: bool = True,
) -> None:
    """Parse known datetime columns in-place when present."""

    if frame.empty:
        return

    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=utc)


def _normalize_range(
    start_date: str | date | datetime,
    end_date: str | date | datetime | None = None,
    max_days: int | None = None,
) -> tuple[date, date]:
    """Normalize and validate a date range."""

    start_date_obj = _coerce_date(start_date)
    end_date_obj = _coerce_date(end_date or start_date_obj)

    if end_date_obj < start_date_obj:
        raise ValueError("end_date must be on or after start_date.")

    total_days = (end_date_obj - start_date_obj).days + 1
    if max_days is not None and total_days > max_days:
        raise ValueError(f"Fitbit only supports a maximum interval of {max_days} days.")

    return start_date_obj, end_date_obj


def _coerce_date(value: str | date | datetime) -> date:
    """Convert common date-like inputs into a date."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _parse_datetime_like(value: Any) -> datetime | None:
    """Parse string or timestamp values into timezone-aware datetimes."""

    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return _ensure_timezone(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    return _ensure_timezone(datetime.fromisoformat(str(value).replace("Z", "+00:00")))


def _ensure_timezone(value: datetime) -> datetime:
    """Ensure datetimes are timezone-aware."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _int_or_none(value: Any) -> int | None:
    """Convert header values to ints when possible."""

    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
