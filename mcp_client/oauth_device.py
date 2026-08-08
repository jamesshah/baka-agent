"""OAuth 2.0 device authorization grant for remote MCP servers (e.g. SnapTrade)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from mcp_client.token_store import FileTokenStore, StoredOAuthTokens

logger = logging.getLogger(__name__)

DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"


@dataclass(frozen=True)
class OAuthEndpoints:
    resource: str
    authorization_server: str
    registration_endpoint: str
    device_authorization_endpoint: str
    token_endpoint: str
    revocation_endpoint: str | None = None


@dataclass(frozen=True)
class DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: int
    interval: int


@dataclass
class DeviceLinkSession:
    """In-flight device authorization for one phone ↔ server."""

    server: str
    phone: str
    resource_url: str
    endpoints: OAuthEndpoints
    client_id: str
    device: DeviceAuthorization
    scopes: list[str]
    started_at: float


class OAuthDeviceError(RuntimeError):
    """Device / token OAuth failure."""


class OAuthDeviceClient:
    """Discover AS metadata, DCR, device auth, poll, refresh, revoke."""

    def __init__(
        self,
        store: FileTokenStore,
        *,
        client_name: str = "baka-imessage-agent",
        timeout: float = 30.0,
    ) -> None:
        self._store = store
        self._client_name = client_name
        self._timeout = timeout
        self._endpoints_cache: dict[str, OAuthEndpoints] = {}

    async def discover(self, resource_url: str) -> OAuthEndpoints:
        cached = self._endpoints_cache.get(resource_url)
        if cached is not None:
            return cached

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            prm = await _fetch_protected_resource(client, resource_url)
            as_base = _authorization_server_base(prm, resource_url)
            metadata = await _fetch_as_metadata(client, as_base)

        endpoints = OAuthEndpoints(
            resource=str(prm.get("resource") or resource_url),
            authorization_server=as_base,
            registration_endpoint=_require_url(
                metadata, "registration_endpoint", as_base, "/oauth/register/"
            ),
            device_authorization_endpoint=_require_url(
                metadata,
                "device_authorization_endpoint",
                as_base,
                "/oauth/device_authorization/",
            ),
            token_endpoint=_require_url(
                metadata, "token_endpoint", as_base, "/oauth/token/"
            ),
            revocation_endpoint=_optional_url(
                metadata, "revocation_endpoint", as_base, "/oauth/revoke_token/"
            ),
        )
        self._endpoints_cache[resource_url] = endpoints
        return endpoints

    async def ensure_client_id(
        self,
        endpoints: OAuthEndpoints,
        *,
        server: str,
        phone: str,
        scopes: list[str],
    ) -> str:
        existing = self._store.load(server, phone)
        if existing and existing.client_id:
            return existing.client_id

        body = {
            "client_name": self._client_name,
            "grant_types": [DEVICE_GRANT, "refresh_token"],
            "token_endpoint_auth_method": "none",
            "scope": " ".join(scopes) if scopes else "read",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                endpoints.registration_endpoint,
                json=body,
                headers={"Content-Type": "application/json"},
            )
        if response.status_code >= 400:
            raise OAuthDeviceError(
                f"Dynamic client registration failed ({response.status_code}): "
                f"{response.text}"
            )
        data = response.json()
        client_id = data.get("client_id")
        if not client_id:
            raise OAuthDeviceError("Registration response missing client_id")
        return str(client_id)

    async def begin_device_authorization(
        self,
        *,
        server: str,
        phone: str,
        resource_url: str,
        scopes: list[str],
    ) -> DeviceLinkSession:
        endpoints = await self.discover(resource_url)
        client_id = await self.ensure_client_id(
            endpoints, server=server, phone=phone, scopes=scopes
        )
        scope = " ".join(scopes) if scopes else "read"
        form = {"client_id": client_id, "scope": scope}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                endpoints.device_authorization_endpoint,
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if response.status_code >= 400:
            raise OAuthDeviceError(
                f"Device authorization failed ({response.status_code}): {response.text}"
            )
        data = response.json()
        device = DeviceAuthorization(
            device_code=str(data["device_code"]),
            user_code=str(data.get("user_code") or ""),
            verification_uri=str(data.get("verification_uri") or ""),
            verification_uri_complete=str(
                data.get("verification_uri_complete")
                or data.get("verification_uri")
                or ""
            ),
            expires_in=int(data.get("expires_in") or 1800),
            interval=max(1, int(data.get("interval") or 5)),
        )
        if not device.verification_uri_complete:
            raise OAuthDeviceError("Device authorization missing verification URI")
        return DeviceLinkSession(
            server=server,
            phone=phone,
            resource_url=resource_url,
            endpoints=endpoints,
            client_id=client_id,
            device=device,
            scopes=scopes,
            started_at=time.time(),
        )

    async def poll_until_tokens(self, session: DeviceLinkSession) -> StoredOAuthTokens:
        deadline = session.started_at + session.device.expires_in
        interval = session.device.interval
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            while time.time() < deadline:
                form = {
                    "grant_type": DEVICE_GRANT,
                    "device_code": session.device.device_code,
                    "client_id": session.client_id,
                }
                response = await client.post(
                    session.endpoints.token_endpoint,
                    data=form,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                if response.status_code < 400:
                    tokens = _tokens_from_response(
                        response.json(),
                        phone=session.phone,
                        client_id=session.client_id,
                        scope=" ".join(session.scopes) if session.scopes else "read",
                    )
                    self._store.save(session.server, tokens)
                    return tokens

                try:
                    err = response.json()
                except Exception:  # noqa: BLE001
                    err = {}
                error = str(err.get("error") or "")
                if error == "authorization_pending":
                    await _sleep(interval)
                    continue
                if error == "slow_down":
                    interval += 5
                    await _sleep(interval)
                    continue
                if error == "expired_token":
                    raise OAuthDeviceError("Device code expired — start link again")
                raise OAuthDeviceError(
                    f"Token poll failed ({response.status_code}): {response.text}"
                )
        raise OAuthDeviceError("Timed out waiting for SnapTrade authorization")

    async def refresh_tokens(
        self,
        *,
        server: str,
        phone: str,
        resource_url: str,
        tokens: StoredOAuthTokens | None = None,
    ) -> StoredOAuthTokens:
        current = tokens or self._store.load(server, phone)
        if current is None or not current.refresh_token:
            raise OAuthDeviceError("No refresh token — re-link required")
        endpoints = await self.discover(resource_url)
        form = {
            "grant_type": "refresh_token",
            "refresh_token": current.refresh_token,
            "client_id": current.client_id,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                endpoints.token_endpoint,
                data=form,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if response.status_code >= 400:
            raise OAuthDeviceError(
                f"Refresh failed ({response.status_code}): {response.text}"
            )
        refreshed = _tokens_from_response(
            response.json(),
            phone=phone,
            client_id=current.client_id,
            scope=current.scope,
            fallback_refresh=current.refresh_token,
        )
        self._store.save(server, refreshed)
        return refreshed

    async def ensure_fresh_tokens(
        self,
        *,
        server: str,
        phone: str,
        resource_url: str,
    ) -> StoredOAuthTokens:
        current = self._store.load(server, phone)
        if current is None or not current.access_token:
            raise OAuthDeviceError("Not linked")
        if current.access_valid():
            return current
        return await self.refresh_tokens(
            server=server,
            phone=phone,
            resource_url=resource_url,
            tokens=current,
        )

    async def revoke(
        self,
        *,
        server: str,
        phone: str,
        resource_url: str,
    ) -> None:
        current = self._store.load(server, phone)
        if current is None:
            return
        try:
            endpoints = await self.discover(resource_url)
            revoke_url = endpoints.revocation_endpoint
            if revoke_url and (current.refresh_token or current.access_token):
                token = current.refresh_token or current.access_token
                form = {"token": token, "client_id": current.client_id}
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    await client.post(
                        revoke_url,
                        data=form,
                        headers={
                            "Content-Type": "application/x-www-form-urlencoded"
                        },
                    )
        except Exception:  # noqa: BLE001 — best-effort revoke
            logger.exception("OAuth revoke failed for %s / %s", server, phone)
        finally:
            self._store.delete(server, phone)


class BearerTokenAuth(httpx.Auth):
    """httpx Auth that injects Bearer tokens and refreshes once on 401."""

    requires_response_body = True

    def __init__(
        self,
        *,
        oauth: OAuthDeviceClient,
        server: str,
        phone: str,
        resource_url: str,
    ) -> None:
        self._oauth = oauth
        self._server = server
        self._phone = phone
        self._resource_url = resource_url

    async def async_auth_flow(self, request: httpx.Request):  # type: ignore[override]
        tokens = await self._oauth.ensure_fresh_tokens(
            server=self._server,
            phone=self._phone,
            resource_url=self._resource_url,
        )
        request.headers["Authorization"] = f"Bearer {tokens.access_token}"
        response = yield request
        if response.status_code != 401:
            return
        await response.aread()
        try:
            refreshed = await self._oauth.refresh_tokens(
                server=self._server,
                phone=self._phone,
                resource_url=self._resource_url,
            )
        except OAuthDeviceError:
            logger.exception("Bearer refresh after 401 failed for %s", self._server)
            return
        request.headers["Authorization"] = f"Bearer {refreshed.access_token}"
        yield request


async def _sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def _tokens_from_response(
    data: dict[str, Any],
    *,
    phone: str,
    client_id: str,
    scope: str,
    fallback_refresh: str = "",
) -> StoredOAuthTokens:
    access = data.get("access_token")
    if not access:
        raise OAuthDeviceError("Token response missing access_token")
    expires_in = float(data.get("expires_in") or 0)
    expires_at = time.time() + expires_in if expires_in else 0.0
    refresh = str(data.get("refresh_token") or fallback_refresh or "")
    return StoredOAuthTokens(
        phone=phone,
        client_id=client_id,
        access_token=str(access),
        refresh_token=refresh,
        expires_at=expires_at,
        scope=str(data.get("scope") or scope),
        updated_at=time.time(),
    )


async def _fetch_protected_resource(
    client: httpx.AsyncClient, resource_url: str
) -> dict[str, Any]:
    parsed = urlparse(resource_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/") or "/mcp"
    candidates = [
        urljoin(base + "/", f".well-known/oauth-protected-resource{path}"),
        urljoin(base + "/", ".well-known/oauth-protected-resource"),
    ]
    for url in candidates:
        try:
            response = await client.get(url)
            if response.status_code < 400:
                data = response.json()
                if isinstance(data, dict):
                    return data
        except Exception:  # noqa: BLE001
            logger.debug("PRM fetch failed for %s", url, exc_info=True)
    # SnapTrade-compatible fallback
    return {
        "resource": resource_url,
        "authorization_servers": ["https://api.snaptrade.com/"],
        "scopes_supported": ["read"],
    }


async def _fetch_as_metadata(
    client: httpx.AsyncClient, as_base: str
) -> dict[str, Any]:
    base = as_base if as_base.endswith("/") else as_base + "/"
    candidates = [
        urljoin(base, ".well-known/oauth-authorization-server/mcp"),
        urljoin(base, ".well-known/oauth-authorization-server"),
        urljoin(base, ".well-known/openid-configuration"),
    ]
    for url in candidates:
        try:
            response = await client.get(url)
            if response.status_code < 400:
                data = response.json()
                if isinstance(data, dict):
                    return data
        except Exception:  # noqa: BLE001
            logger.debug("AS metadata fetch failed for %s", url, exc_info=True)
    # SnapTrade-compatible fallback endpoints
    return {
        "registration_endpoint": urljoin(base, "oauth/register/"),
        "device_authorization_endpoint": urljoin(base, "oauth/device_authorization/"),
        "token_endpoint": urljoin(base, "oauth/token/"),
        "revocation_endpoint": urljoin(base, "oauth/revoke_token/"),
    }


def _authorization_server_base(prm: dict[str, Any], resource_url: str) -> str:
    servers = prm.get("authorization_servers") or []
    if isinstance(servers, list) and servers:
        return str(servers[0]).rstrip("/") + "/"
    parsed = urlparse(resource_url)
    return f"{parsed.scheme}://{parsed.netloc}/"


def _require_url(
    metadata: dict[str, Any],
    key: str,
    as_base: str,
    fallback_path: str,
) -> str:
    value = metadata.get(key)
    if value:
        return str(value)
    return urljoin(as_base if as_base.endswith("/") else as_base + "/", fallback_path.lstrip("/"))


def _optional_url(
    metadata: dict[str, Any],
    key: str,
    as_base: str,
    fallback_path: str,
) -> str | None:
    value = metadata.get(key)
    if value:
        return str(value)
    if fallback_path:
        return urljoin(
            as_base if as_base.endswith("/") else as_base + "/",
            fallback_path.lstrip("/"),
        )
    return None
