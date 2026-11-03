"""Connector contract: turn a provider's webhooks or polls into normalized events.

Connectors are plumbing only. They verify, fetch, and normalize; whether an
event matters is decided by the rule engine and triage, never here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

AuthKind = Literal["oauth_code", "shared_secret"]
DeliveryMode = Literal["push", "poll"]


class ConnectorError(RuntimeError):
    """A connector could not fetch or normalize provider data."""


@dataclass(frozen=True)
class PendingEvent:
    """What the webhook route stores before enrichment (cheap to build)."""

    source: str
    type: str
    dedupe_key: str
    principal_phone: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExternalEvent:
    """A provider event in the shape rules, triage, and tools understand."""

    source: str
    type: str
    occurred_at: datetime
    dedupe_key: str
    principal_phone: str
    metrics: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class Connector(ABC):
    """One external provider (Whoop, Plaid, ...)."""

    name: str
    display_name: str = ""
    auth: AuthKind
    delivery: DeliveryMode = "push"

    @property
    def title(self) -> str:
        return self.display_name or self.name.title()

    def is_configured(self) -> bool:
        """True when the settings needed to receive and fetch data are set."""
        return True

    @abstractmethod
    def verify(self, headers: Mapping[str, str], raw_body: bytes) -> bool:
        """Check the provider signature. ``headers`` keys are lowercase."""

    @abstractmethod
    def parse(self, body: Any) -> list[PendingEvent]:
        """Turn a verified webhook body into pending events (no network)."""

    @abstractmethod
    def enrich(self, pending: PendingEvent) -> list[ExternalEvent]:
        """Fetch full records and normalize them; may return several events."""

    @abstractmethod
    def metric_schema(self) -> dict[str, str]:
        """Metric name -> type (``number``, ``bool``, ``string``) for rules."""

    def event_types(self) -> list[str]:
        """Event types this connector emits, for rule authors."""
        return []

    def status(self, phone: str = "") -> dict[str, Any]:
        """Health / link details for ``/health`` and status tools."""
        return {"configured": self.is_configured()}


class OAuthConnector(Connector):
    """A connector linked per user through the OAuth authorization-code flow."""

    auth: AuthKind = "oauth_code"

    @abstractmethod
    def authorization_url(self, phone: str) -> str:
        """Start a link for ``phone`` and return the consent URL to text them."""

    @abstractmethod
    def complete_oauth(self, code: str, state: str) -> str:
        """Exchange the callback code; return the phone that was linked."""

    @abstractmethod
    def unlink(self, phone: str) -> dict[str, Any]:
        """Revoke access (best effort) and delete stored credentials."""


class PollingConnector(Connector):
    """A source without webhooks; the processor calls ``poll`` on a schedule."""

    delivery: DeliveryMode = "poll"
    poll_interval_seconds: float = 900.0

    def verify(self, headers: Mapping[str, str], raw_body: bytes) -> bool:
        return False

    def parse(self, body: Any) -> list[PendingEvent]:
        return []

    def enrich(self, pending: PendingEvent) -> list[ExternalEvent]:
        return []

    @abstractmethod
    def poll(self, since: datetime | None) -> list[ExternalEvent]:
        """Return events newer than ``since`` (None on first run)."""
