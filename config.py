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

    llama_base_url: str = "http://127.0.0.1:8080/v1"
    llama_model: str = "local"

    allowed_numbers: str = ""

    max_history_messages: int = 40
    max_agent_iterations: int = 5
    system_prompt: str = (
        "You are a helpful local iMessage assistant. "
        "Keep replies concise and conversational. "
        "SnapTrade portfolio tools are read-only and require a one-time link: "
        "if the user asks about balances/positions/orders and SnapTrade is not "
        "linked, call link_snaptrade and send them the verification URL. "
        "Brokerage connection links from SnapTrade expire in about 5 minutes."
    )

    # Cursor-compatible MCP config (mcpServers in a JSON file).
    mcp_config_path: str = "mcp.json"
    mcp_enabled: bool = True
    mcp_oauth_data_dir: str = ".data/mcp-oauth"
    mcp_oauth_owner_number: str = ""

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
