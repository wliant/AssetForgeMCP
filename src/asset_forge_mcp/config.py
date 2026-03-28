"""Configuration loaded from environment variables and .env file."""

from __future__ import annotations

import functools

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr
    openai_base_url: str = "https://api.openai.com/v1"
    openai_image_model: str = "gpt-image-1"
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8080
    log_level: str = "INFO"

    # S3 / MinIO settings
    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: SecretStr = SecretStr("minioadmin")
    s3_bucket: str = "asset-forge"
    s3_region: str = "us-east-1"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton. Call clear_settings() in tests."""
    return Settings()  # type: ignore[call-arg]


def clear_settings() -> None:
    """Clear the cached settings (for testing)."""
    get_settings.cache_clear()
