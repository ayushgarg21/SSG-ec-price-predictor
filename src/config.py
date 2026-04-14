"""Centralised configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://ec_user:ec_password@localhost:5432/ec_prices"

    # URA API
    ura_access_key: str = ""
    ura_base_url: str = "https://www.ura.gov.sg/uraDataService"
    ura_max_retries: int = 3
    ura_retry_backoff: float = 1.0

    # Model
    model_path: str = "./artifacts/model.joblib"

    # API
    rate_limit_rpm: int = 60
    cache_ttl_seconds: int = 300
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
