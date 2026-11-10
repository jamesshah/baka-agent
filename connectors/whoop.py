"""Whoop: sleep, recovery, and workout webhooks (v2 model) and OAuth linking."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

import httpx

from connectors.base import (
    ConnectorError,
    ExternalEvent,
    OAuthConnector,
    PendingEvent,
)
from mcp_client.token_store import StoredOAuthTokens

if TYPE_CHECKING:
    from mcp_client.token_store import FileTokenStore

logger = logging.getLogger(__name__)

WHOOP = "whoop"
AUTHORIZE_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
API_BASE = "https://api.prod.whoop.com/developer"
SCOPES = "offline read:recovery read:sleep read:workout read:cycles read:profile"

_SIGNATURE_HEADER = "x-whoop-signature"
_TIMESTAMP_HEADER = "x-whoop-signature-timestamp"
# Whoop retries failed deliveries for about an hour; allow that plus slack.
_SIGNATURE_MAX_AGE_S = 2 * 60 * 60
_STATE_TTL_S = 15 * 60
_MS_PER_HOUR = 3_600_000

EVENT_TYPES = [
    "recovery.updated",
    "recovery.deleted",
    "sleep.updated",
    "sleep.deleted",
    "workout.updated",
    "workout.deleted",
]

METRIC_SCHEMA = {
    # recovery.updated
    "recovery_score": "number",
    "hrv_ms": "number",
    "resting_hr": "number",
    "spo2": "number",
    "skin_temp_c": "number",
    "calibrating": "bool",
    # sleep.updated (recovery.updated also carries sleep_hours/sleep_performance)
    "sleep_hours": "number",
    "in_bed_hours": "number",
    "sleep_performance": "number",
    "sleep_efficiency": "number",
    "sleep_consistency": "number",
    "sleep_debt_hours": "number",
    "respiratory_rate": "number",
    "disturbances": "number",
    "nap": "bool",
    # workout.updated
    "sport": "string",
    "strain": "number",
    "avg_hr": "number",
    "max_hr": "number",
    "kilojoules": "number",
    "distance_km": "number",
    "duration_min": "number",
}


class WhoopAccounts:
    """Maps Whoop user ids (sent on every webhook) to iMessage phones."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def _write(self, data: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self._path)

    def phone_for(self, user_id: str) -> str | None:
        with self._lock:
            return self._read().get(str(user_id))

    def user_for(self, phone: str) -> str | None:
        with self._lock:
            for user_id, linked_phone in self._read().items():
                if linked_phone == phone:
                    return user_id
        return None

    def link(self, user_id: str, phone: str) -> None:
        with self._lock:
            data = {k: v for k, v in self._read().items() if v != phone}
            data[str(user_id)] = phone
            self._write(data)

    def unlink_phone(self, phone: str) -> None:
        with self._lock:
            data = self._read()
            kept = {k: v for k, v in data.items() if v != phone}
            if kept != data:
                self._write(kept)


class WhoopConnector(OAuthConnector):
    name = WHOOP
    display_name = "Whoop"

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: str,
        redirect_url: str,
        token_store: FileTokenStore,
        owner_phone: str = "",
        http: httpx.Client | None = None,
        clock: Any = time.time,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._redirect_url = redirect_url
        self._tokens = token_store
        self._owner_phone = owner_phone
        self._http = http or httpx.Client(timeout=15.0)
        self._clock = clock
        # Outside the token dir: FileTokenStore treats every */whoop/*.json as tokens.
        self._accounts = WhoopAccounts(token_store.root / "whoop_accounts.json")
        self._states: dict[str, tuple[str, float]] = {}
        self._states_lock = threading.Lock()
        self._refresh_lock = threading.Lock()

    # --- contract -------------------------------------------------------------

    def is_configured(self) -> bool:
        return bool(self._client_id and self._client_secret and self._redirect_url)

    def metric_schema(self) -> dict[str, str]:
        return dict(METRIC_SCHEMA)

    def event_types(self) -> list[str]:
        return list(EVENT_TYPES)

    def verify(self, headers: Mapping[str, str], raw_body: bytes) -> bool:
        signature = headers.get(_SIGNATURE_HEADER) or ""
        timestamp = headers.get(_TIMESTAMP_HEADER) or ""
        if not signature or not timestamp or not self._client_secret:
            return False
        try:
            sent_ms = int(timestamp)
        except ValueError:
            return False
        if abs(self._clock() - sent_ms / 1000.0) > _SIGNATURE_MAX_AGE_S:
            logger.warning("whoop webhook rejected: stale signature timestamp")
            return False
        expected = sign_payload(self._client_secret, timestamp, raw_body)
        return hmac.compare_digest(signature.strip(), expected)

    def parse(self, body: Any) -> list[PendingEvent]:
        if not isinstance(body, dict):
            return []
        event_type = str(body.get("type") or "").strip()
        object_id = str(body.get("id") or "").strip()
        user_id = str(body.get("user_id") or "").strip()
        if event_type not in EVENT_TYPES or not object_id or not user_id:
            logger.info("ignoring whoop webhook: %s", body)
            return []
        phone = self._phone_for_user(user_id)
        if not phone:
            logger.warning("whoop webhook for unlinked user %s ignored", user_id)
            return []
        trace_id = str(body.get("trace_id") or "").strip() or object_id
        return [
            PendingEvent(
                source=WHOOP,
                type=event_type,
                dedupe_key=f"whoop:{event_type}:{object_id}:{trace_id}",
                principal_phone=phone,
                payload={"id": object_id, "user_id": user_id, "trace_id": trace_id},
            )
        ]

    def enrich(self, pending: PendingEvent) -> list[ExternalEvent]:
        object_id = str(pending.payload.get("id") or "")
        kind, _, action = pending.type.partition(".")
        if action == "deleted":
            return [
                ExternalEvent(
                    source=WHOOP,
                    type=pending.type,
                    occurred_at=datetime.now(timezone.utc),
                    dedupe_key=pending.dedupe_key,
                    principal_phone=pending.principal_phone,
                    summary=f"Whoop {kind} deleted",
                    raw={"id": object_id},
                )
            ]
        phone = pending.principal_phone
        if kind == "sleep":
            sleep = self._get(phone, f"/v2/activity/sleep/{object_id}")
            return self._sleep_event(pending, sleep)
        if kind == "recovery":
            # v2 recovery webhooks carry the id of the sleep the recovery belongs to.
            sleep = self._get(phone, f"/v2/activity/sleep/{object_id}")
            if sleep is None or not sleep.get("cycle_id"):
                return []
            recovery = self._get(phone, f"/v2/cycle/{sleep['cycle_id']}/recovery")
            return self._recovery_event(pending, recovery, sleep)
        if kind == "workout":
            workout = self._get(phone, f"/v2/activity/workout/{object_id}")
            return self._workout_event(pending, workout)
        return []

    def status(self, phone: str = "") -> dict[str, Any]:
        result: dict[str, Any] = {
            "configured": self.is_configured(),
            "webhook_path": f"/webhooks/connectors/{WHOOP}",
        }
        if phone:
            tokens = self._tokens.load(WHOOP, phone)
            result["linked"] = tokens is not None and bool(tokens.refresh_token or tokens.access_token)
            result["whoop_user_id"] = self._accounts.user_for(phone)
        return result

    # --- OAuth ----------------------------------------------------------------

    def authorization_url(self, phone: str) -> str:
        state = secrets.token_urlsafe(16)
        now = self._clock()
        with self._states_lock:
            self._states = {
                key: value for key, value in self._states.items()
                if now - value[1] < _STATE_TTL_S
            }
            self._states[state] = (phone, now)
        query = urlencode({
            "response_type": "code",
            "client_id": self._client_id,
            "redirect_uri": self._redirect_url,
            "scope": SCOPES,
            "state": state,
        })
        return f"{AUTHORIZE_URL}?{query}"

    def complete_oauth(self, code: str, state: str) -> str:
        with self._states_lock:
            entry = self._states.pop(state, None)
        if entry is None or self._clock() - entry[1] > _STATE_TTL_S:
            raise ConnectorError("Whoop link expired or was not started here; ask for a new link")
        phone = entry[0]
        token = self._token_request({
            "grant_type": "authorization_code",
            "code": code,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "redirect_uri": self._redirect_url,
        })
        self._save_tokens(phone, token)
        profile = self._get(phone, "/v2/user/profile/basic")
        user_id = str((profile or {}).get("user_id") or "").strip()
        if not user_id:
            raise ConnectorError("Whoop profile did not include a user id")
        self._accounts.link(user_id, phone)
        logger.info("linked whoop user %s to %s", user_id, phone)
        return phone

    def unlink(self, phone: str) -> dict[str, Any]:
        revoked = False
        if self._tokens.load(WHOOP, phone) is not None:
            try:
                response = self._request(phone, "DELETE", "/v2/user/access")
                revoked = response is not None
            except Exception:  # noqa: BLE001 — still delete local credentials
                logger.warning("whoop revoke failed for %s", phone, exc_info=True)
        deleted = self._tokens.delete(WHOOP, phone)
        self._accounts.unlink_phone(phone)
        return {
            "status": "unlinked" if deleted else "not_linked",
            "revoked": revoked,
        }

    # --- HTTP -----------------------------------------------------------------

    def _phone_for_user(self, user_id: str) -> str | None:
        phone = self._accounts.phone_for(user_id)
        if phone:
            return phone
        # Account map lost but the single owner is linked: attribute to them.
        if self._owner_phone and self._tokens.load(WHOOP, self._owner_phone):
            return self._owner_phone
        return None

    def _token_request(self, form: dict[str, str]) -> dict[str, Any]:
        response = self._http.post(TOKEN_URL, data=form)
        if response.status_code >= 400:
            raise ConnectorError(
                f"Whoop token request failed ({response.status_code}): {response.text[:200]}"
            )
        return response.json()

    def _save_tokens(self, phone: str, token: dict[str, Any]) -> StoredOAuthTokens:
        previous = self._tokens.load(WHOOP, phone)
        expires_in = float(token.get("expires_in") or 0)
        stored = StoredOAuthTokens(
            phone=phone,
            client_id=self._client_id,
            access_token=str(token.get("access_token") or ""),
            refresh_token=str(
                token.get("refresh_token")
                or (previous.refresh_token if previous else "")
            ),
            expires_at=self._clock() + expires_in if expires_in else 0.0,
            scope=str(token.get("scope") or SCOPES),
        )
        self._tokens.save(WHOOP, stored)
        return stored

    def _access_token(self, phone: str, *, force_refresh: bool = False) -> str:
        with self._refresh_lock:
            tokens = self._tokens.load(WHOOP, phone)
            if tokens is None:
                raise ConnectorError(f"Whoop is not linked for {phone}")
            fresh = tokens.access_token and (
                tokens.expires_at <= 0 or self._clock() < tokens.expires_at - 60
            )
            if not force_refresh and fresh:
                return tokens.access_token
            if not tokens.refresh_token:
                raise ConnectorError("Whoop access expired and no refresh token is stored")
            refreshed = self._save_tokens(phone, self._token_request({
                "grant_type": "refresh_token",
                "refresh_token": tokens.refresh_token,
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "scope": "offline",
            }))
            return refreshed.access_token

    def _request(self, phone: str, method: str, path: str) -> httpx.Response | None:
        """Authorized request with one refresh-and-retry on 401; None on 404."""
        url = f"{API_BASE}{path}"
        for attempt in range(2):
            token = self._access_token(phone, force_refresh=attempt > 0)
            response = self._http.request(
                method, url, headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 401 and attempt == 0:
                continue
            if response.status_code == 404:
                return None
            if response.status_code >= 400:
                raise ConnectorError(
                    f"Whoop {method} {path} failed ({response.status_code})"
                )
            return response
        return None

    def _get(self, phone: str, path: str) -> dict[str, Any] | None:
        response = self._request(phone, "GET", path)
        if response is None:
            return None
        data = response.json()
        return data if isinstance(data, dict) else None

    # --- normalization --------------------------------------------------------

    def _sleep_event(
        self, pending: PendingEvent, sleep: dict[str, Any] | None
    ) -> list[ExternalEvent]:
        if sleep is None or sleep.get("score_state") == "PENDING_SCORE":
            return []
        metrics = sleep_metrics(sleep)
        hours = metrics.get("sleep_hours")
        if metrics.get("nap"):
            summary = f"Nap of {hours} h" if hours is not None else "Nap logged"
        elif hours is None:
            summary = "Sleep logged (not scored)"
        else:
            details = [f"performance {_pct(metrics.get('sleep_performance'))}"]
            if metrics.get("sleep_debt_hours"):
                details.append(f"sleep debt {metrics['sleep_debt_hours']} h")
            summary = f"Slept {hours} h ({', '.join(details)})"
        return [self._event(pending, "sleep", sleep, metrics, summary, sleep.get("end"))]

    def _recovery_event(
        self,
        pending: PendingEvent,
        recovery: dict[str, Any] | None,
        sleep: dict[str, Any],
    ) -> list[ExternalEvent]:
        if recovery is None or recovery.get("score_state") == "PENDING_SCORE":
            return []
        metrics = recovery_metrics(recovery)
        sleep_values = sleep_metrics(sleep)
        for key in ("sleep_hours", "sleep_performance"):
            if key in sleep_values:
                metrics[key] = sleep_values[key]
        score = metrics.get("recovery_score")
        if score is None:
            summary = "Recovery not scored"
        else:
            details = []
            if metrics.get("hrv_ms") is not None:
                details.append(f"HRV {metrics['hrv_ms']} ms")
            if metrics.get("resting_hr") is not None:
                details.append(f"RHR {metrics['resting_hr']} bpm")
            summary = f"Recovery {_pct(score)}"
            if details:
                summary += f" ({', '.join(details)})"
            if metrics.get("calibrating"):
                summary += ", still calibrating"
        dedupe_id = str(pending.payload.get("id") or sleep.get("id") or "")
        return [
            self._event(
                pending, "recovery", recovery, metrics, summary,
                sleep.get("end"), dedupe_id=dedupe_id,
            )
        ]

    def _workout_event(
        self, pending: PendingEvent, workout: dict[str, Any] | None
    ) -> list[ExternalEvent]:
        if workout is None or workout.get("score_state") == "PENDING_SCORE":
            return []
        metrics = workout_metrics(workout)
        sport = str(metrics.get("sport") or "workout").replace("_", " ").title()
        details = []
        if metrics.get("strain") is not None:
            details.append(f"strain {metrics['strain']}")
        if metrics.get("avg_hr") is not None:
            details.append(f"avg HR {metrics['avg_hr']}")
        if metrics.get("duration_min") is not None:
            details.append(f"{metrics['duration_min']} min")
        summary = f"{sport}: {', '.join(details)}" if details else sport
        return [self._event(pending, "workout", workout, metrics, summary, workout.get("end"))]

    def _event(
        self,
        pending: PendingEvent,
        kind: str,
        record: dict[str, Any],
        metrics: dict[str, Any],
        summary: str,
        occurred: Any,
        *,
        dedupe_id: str | None = None,
    ) -> ExternalEvent:
        record_id = dedupe_id or str(record.get("id") or pending.payload.get("id"))
        version = str(record.get("updated_at") or pending.payload.get("trace_id") or "")
        return ExternalEvent(
            source=WHOOP,
            type=pending.type,
            occurred_at=_parse_time(occurred) or datetime.now(timezone.utc),
            dedupe_key=f"whoop:{kind}:{record_id}:{version}",
            principal_phone=pending.principal_phone,
            metrics=metrics,
            summary=summary,
            raw=record,
        )


def sign_payload(client_secret: str, timestamp: str, raw_body: bytes) -> str:
    digest = hmac.new(
        client_secret.encode("utf-8"),
        timestamp.encode("utf-8") + raw_body,
        hashlib.sha256,
    ).digest()
    return base64.b64encode(digest).decode("ascii")


def recovery_metrics(recovery: dict[str, Any]) -> dict[str, Any]:
    score = recovery.get("score") or {}
    metrics: dict[str, Any] = {}
    _put(metrics, "recovery_score", score.get("recovery_score"), 0)
    _put(metrics, "hrv_ms", score.get("hrv_rmssd_milli"), 0)
    _put(metrics, "resting_hr", score.get("resting_heart_rate"), 0)
    _put(metrics, "spo2", score.get("spo2_percentage"), 1)
    _put(metrics, "skin_temp_c", score.get("skin_temp_celsius"), 1)
    if "user_calibrating" in score:
        metrics["calibrating"] = bool(score.get("user_calibrating"))
    return metrics


def sleep_metrics(sleep: dict[str, Any]) -> dict[str, Any]:
    score = sleep.get("score") or {}
    stages = score.get("stage_summary") or {}
    needed = score.get("sleep_needed") or {}
    metrics: dict[str, Any] = {"nap": bool(sleep.get("nap"))}
    asleep_ms = sum(
        float(stages.get(key) or 0)
        for key in (
            "total_light_sleep_time_milli",
            "total_slow_wave_sleep_time_milli",
            "total_rem_sleep_time_milli",
        )
    )
    if stages:
        metrics["sleep_hours"] = round(asleep_ms / _MS_PER_HOUR, 1)
        _put(metrics, "in_bed_hours", _hours(stages.get("total_in_bed_time_milli")), 1)
        _put(metrics, "disturbances", stages.get("disturbance_count"), 0)
    _put(metrics, "sleep_performance", score.get("sleep_performance_percentage"), 0)
    _put(metrics, "sleep_efficiency", score.get("sleep_efficiency_percentage"), 0)
    _put(metrics, "sleep_consistency", score.get("sleep_consistency_percentage"), 0)
    _put(metrics, "respiratory_rate", score.get("respiratory_rate"), 1)
    _put(metrics, "sleep_debt_hours", _hours(needed.get("need_from_sleep_debt_milli")), 1)
    return metrics


def workout_metrics(workout: dict[str, Any]) -> dict[str, Any]:
    score = workout.get("score") or {}
    metrics: dict[str, Any] = {}
    sport = workout.get("sport_name")
    if sport:
        metrics["sport"] = str(sport).lower()
    _put(metrics, "strain", score.get("strain"), 1)
    _put(metrics, "avg_hr", score.get("average_heart_rate"), 0)
    _put(metrics, "max_hr", score.get("max_heart_rate"), 0)
    _put(metrics, "kilojoules", score.get("kilojoule"), 0)
    distance = score.get("distance_meter")
    _put(metrics, "distance_km", distance / 1000 if distance else None, 2)
    start, end = _parse_time(workout.get("start")), _parse_time(workout.get("end"))
    if start and end:
        metrics["duration_min"] = round((end - start).total_seconds() / 60)
    return metrics


def _put(metrics: dict[str, Any], key: str, value: Any, digits: int) -> None:
    if value is None:
        return
    try:
        number = round(float(value), digits)
    except (TypeError, ValueError):
        return
    metrics[key] = int(number) if digits == 0 else number


def _hours(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value) / _MS_PER_HOUR
    except (TypeError, ValueError):
        return None


def _pct(value: Any) -> str:
    return f"{value}%" if value is not None else "n/a"


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
