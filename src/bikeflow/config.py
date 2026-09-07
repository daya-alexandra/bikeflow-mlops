"""Application configuration loaded from BIKEFLOW_* environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the API scaffold."""

    model_config = SettingsConfigDict(env_prefix="BIKEFLOW_", env_file=".env")

    log_level: str = "INFO"
    stub_prediction: float = Field(default=42.0, ge=0)


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
