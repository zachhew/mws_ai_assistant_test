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

    app_name: str = Field(default="mws-ai-assistant-test", alias="APP_NAME")
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

    default_session_ttl_minutes: int = Field(
        default=60,
        alias="DEFAULT_SESSION_TTL_MINUTES",
    )

    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    openrouter_model: str = Field(
        default="openai/gpt-4o-mini",
        alias="OPENROUTER_MODEL",
    )

    adk_litellm_model: str = Field(
        default="openrouter/openai/gpt-4o-mini",
        alias="ADK_LITELLM_MODEL",
    )
    or_api_key: str = Field(default="", alias="OR_API_KEY")
    or_site_url: str = Field(default="http://localhost:8000", alias="OR_SITE_URL")
    or_app_name: str = Field(default="mws-ai-assistant-test", alias="OR_APP_NAME")

    openrouter_http_referer: str = Field(
        default="http://localhost:8000",
        alias="OPENROUTER_HTTP_REFERER",
    )
    openrouter_app_title: str = Field(
        default="mws-ai-assistant-test",
        alias="OPENROUTER_APP_TITLE",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
