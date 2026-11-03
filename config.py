from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sendblue_api_key: str = ""
    sendblue_api_secret: str = ""
    sendblue_from_number: str = ""
    sendblue_global_webhook_secret: str = ""
    # Public URL for POST /webhooks/receive (tunnel host). Used to auto-register
    # the Sendblue receive webhook on startup when missing.
    sendblue_webhook_url: str = ""

    llama_base_url: str = "http://127.0.0.1:8080/v1"
    llama_model: str = "local"

    allowed_numbers: str = ""

    max_history_messages: int = 40
    max_agent_iterations: int = 5
    max_executor_iterations: int = 8

    # Durable local conversation and semantic memory.
    database_path: str = ".data/baka.db"
    memory_enabled: bool = True
    memory_consolidation_enabled: bool = True
    memory_summary_every_turns: int = 20
    memory_max_chars: int = 3000
    skill_max_chars: int = 5000
    skills_dir: str = "skills"

    # Local llama.cpp embedding sidecar.
    embedding_enabled: bool = True
    embedding_base_url: str = "http://127.0.0.1:8081/v1"
    embedding_model: str = "local-embedding"
    embedding_dimensions: int = 768
    embedding_timeout_seconds: float = 10.0
    memory_lexical_weight: float = 0.45
    memory_vector_weight: float = 0.55
    memory_minimum_vector_score: float = 0.3

    # Cursor-compatible MCP config (mcpServers in a JSON file).
    mcp_config_path: str = "mcp.json"
    mcp_enabled: bool = True
    mcp_oauth_data_dir: str = ".data/mcp-oauth"
    mcp_oauth_owner_number: str = ""

    # Public origin of this server (tunnel host), e.g. https://xxxx.trycloudflare.com.
    # Connector webhooks arrive at <public_base_url>/webhooks/connectors/<name>.
    public_base_url: str = ""

    # Whoop connector (https://developer.whoop.com). Set the app's webhook URL
    # to <public_base_url>/webhooks/connectors/whoop with model version v2.
    whoop_client_id: str = ""
    whoop_client_secret: str = ""
    # Defaults to <public_base_url>/oauth/whoop/callback; must match the dashboard.
    whoop_redirect_url: str = ""

    # Connector event triage.
    events_enabled: bool = True
    event_rules_path: str = "events/rules/defaults.yaml"
    # Local window when non-urgent "notify" events wait for the digest, e.g. 22-7.
    event_quiet_hours: str = "22-7"
    event_max_proactive_per_day: int = 3
    # Local time for the daily digest (HH:MM); empty disables it.
    event_digest_time: str = "08:30"
    # IANA timezone for quiet hours and the digest; empty uses the system zone.
    event_timezone: str = ""
    recent_events_hours: int = 24

    @property
    def resolved_whoop_redirect_url(self) -> str:
        if self.whoop_redirect_url.strip():
            return self.whoop_redirect_url.strip()
        base = self.public_base_url.strip().rstrip("/")
        return f"{base}/oauth/whoop/callback" if base else ""

    @property
    def allowed_numbers_set(self) -> set[str]:
        if not self.allowed_numbers.strip():
            return set()
        return {n.strip() for n in self.allowed_numbers.split(",") if n.strip()}

    @property
    def resolved_oauth_owner_number(self) -> str:
        """Phone that owns persisted MCP OAuth tokens."""
        if self.mcp_oauth_owner_number.strip():
            return self.mcp_oauth_owner_number.strip()
        allowed = self.allowed_numbers_set
        if len(allowed) == 1:
            return next(iter(allowed))
        return ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
