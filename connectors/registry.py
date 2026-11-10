"""Enabled connectors, looked up by name from the generic webhook route."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from connectors.base import Connector, OAuthConnector, PollingConnector

if TYPE_CHECKING:
    from config import Settings
    from mcp_client.token_store import FileTokenStore

logger = logging.getLogger(__name__)


class ConnectorRegistry:
    def __init__(self, connectors: list[Connector] | None = None) -> None:
        self._connectors: dict[str, Connector] = {}
        for connector in connectors or []:
            self.register(connector)

    def register(self, connector: Connector) -> None:
        self._connectors[connector.name] = connector

    def get(self, name: str) -> Connector | None:
        return self._connectors.get(name)

    def names(self) -> list[str]:
        return sorted(self._connectors)

    def all(self) -> list[Connector]:
        return [self._connectors[name] for name in self.names()]

    def oauth(self) -> list[OAuthConnector]:
        return [c for c in self.all() if isinstance(c, OAuthConnector)]

    def polling(self) -> list[PollingConnector]:
        return [c for c in self.all() if isinstance(c, PollingConnector)]

    def metric_names(self, source: str) -> set[str] | None:
        """Known metric names for ``source``; None if the source is unknown."""
        connector = self.get(source)
        if connector is None:
            return None
        return set(connector.metric_schema())

    def __len__(self) -> int:
        return len(self._connectors)


def build_connector_registry(
    settings: Settings, token_store: FileTokenStore
) -> ConnectorRegistry:
    """Instantiate every connector whose settings are present."""
    from connectors.whoop import WhoopConnector

    registry = ConnectorRegistry()
    if settings.whoop_client_id and settings.whoop_client_secret:
        registry.register(
            WhoopConnector(
                client_id=settings.whoop_client_id,
                client_secret=settings.whoop_client_secret,
                redirect_url=settings.resolved_whoop_redirect_url,
                token_store=token_store,
                owner_phone=settings.resolved_oauth_owner_number,
            )
        )
    else:
        logger.info("Whoop connector disabled (WHOOP_CLIENT_ID/SECRET unset)")
    return registry
