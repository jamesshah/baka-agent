"""File-backed OAuth token storage for MCP servers."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SAFE_PHONE_RE = re.compile(r"[^0-9+]+")


@dataclass
class StoredOAuthTokens:
    """Persisted OAuth credentials for one phone ↔ MCP server link."""

    phone: str
    client_id: str
    access_token: str
    refresh_token: str = ""
    expires_at: float = 0.0
    scope: str = "read"
    updated_at: float = 0.0

    def access_valid(self, *, skew_seconds: float = 60.0) -> bool:
        if not self.access_token:
            return False
        if self.expires_at <= 0:
            return True
        return time.time() < (self.expires_at - skew_seconds)


class FileTokenStore:
    """Store tokens at ``<data_dir>/<server>/<phone>.json``."""

    def __init__(self, data_dir: str | Path) -> None:
        self._root = Path(data_dir)

    def path_for(self, server: str, phone: str) -> Path:
        safe_server = _safe_segment(server)
        safe_phone = _safe_phone(phone)
        return self._root / safe_server / f"{safe_phone}.json"

    def load(self, server: str, phone: str) -> StoredOAuthTokens | None:
        path = self.path_for(server, phone)
        if not path.is_file():
            return None
        try:
            with path.open(encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to read OAuth tokens from %s", path)
            return None
        if not isinstance(raw, dict):
            return None
        try:
            return StoredOAuthTokens(
                phone=str(raw.get("phone") or phone),
                client_id=str(raw.get("client_id") or ""),
                access_token=str(raw.get("access_token") or ""),
                refresh_token=str(raw.get("refresh_token") or ""),
                expires_at=float(raw.get("expires_at") or 0),
                scope=str(raw.get("scope") or "read"),
                updated_at=float(raw.get("updated_at") or 0),
            )
        except (TypeError, ValueError):
            logger.exception("Invalid OAuth token file %s", path)
            return None

    def save(self, server: str, tokens: StoredOAuthTokens) -> Path:
        path = self.path_for(server, tokens.phone)
        path.parent.mkdir(parents=True, exist_ok=True)
        tokens.updated_at = time.time()
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(asdict(tokens), fh, indent=2)
            fh.write("\n")
        tmp.replace(path)
        logger.info("Saved OAuth tokens for %s / %s", server, tokens.phone)
        return path

    def delete(self, server: str, phone: str) -> bool:
        path = self.path_for(server, phone)
        if not path.is_file():
            return False
        path.unlink()
        logger.info("Deleted OAuth tokens for %s / %s", server, phone)
        return True

    def find_for_server(self, server: str, preferred_phone: str = "") -> StoredOAuthTokens | None:
        """Load tokens for preferred phone, else the sole file under the server dir."""
        if preferred_phone:
            found = self.load(server, preferred_phone)
            if found is not None:
                return found

        server_dir = self._root / _safe_segment(server)
        if not server_dir.is_dir():
            return None
        files = sorted(server_dir.glob("*.json"))
        if len(files) == 1:
            phone = files[0].stem
            # Files use sanitized names; prefer reading phone from payload.
            return self.load(server, phone) or _load_path(files[0])
        if preferred_phone:
            return None
        if not files:
            return None
        # Multiple files without a preferred owner — load none to avoid wrong account.
        logger.warning(
            "Multiple OAuth token files for MCP server '%s'; set MCP_OAUTH_OWNER_NUMBER",
            server,
        )
        return None


def _safe_segment(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_")
    return cleaned or "server"


def _safe_phone(phone: str) -> str:
    cleaned = _SAFE_PHONE_RE.sub("", phone.strip())
    return cleaned or "unknown"


def _load_path(path: Path) -> StoredOAuthTokens | None:
    try:
        with path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            return None
        return StoredOAuthTokens(
            phone=str(raw.get("phone") or path.stem),
            client_id=str(raw.get("client_id") or ""),
            access_token=str(raw.get("access_token") or ""),
            refresh_token=str(raw.get("refresh_token") or ""),
            expires_at=float(raw.get("expires_at") or 0),
            scope=str(raw.get("scope") or "read"),
            updated_at=float(raw.get("updated_at") or 0),
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
