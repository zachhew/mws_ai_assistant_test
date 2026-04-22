from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="ai-model-selection-assistant", alias="APP_NAME")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    mws_models_url: str = Field(
        default="https://mws.ru/docs/cloud-platform/gpt/general/gpt-models.html",
        alias="MWS_MODELS_URL",
    )
    mws_pricing_url: str = Field(
        default="https://mws.ru/docs/cloud-platform/gpt/general/pricing.html",
        alias="MWS_PRICING_URL",
    )
    catalog_cache_ttl_seconds: int = Field(
        default=900,
        alias="CATALOG_CACHE_TTL_SECONDS",
    )

    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    google_genai_use_vertexai: bool = Field(
        default=False,
        alias="GOOGLE_GENAI_USE_VERTEXAI",
    )
    adk_agent_model: str = Field(default="gemini-2.0-flash", alias="ADK_AGENT_MODEL")

    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    openrouter_model: str = Field(
        default="openai/gpt-4o-mini",
        alias="OPENROUTER_MODEL",
    )

    default_session_ttl_minutes: int = Field(
        default=60,
        alias="DEFAULT_SESSION_TTL_MINUTES",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()