"""Central, typed configuration.

Every tunable lives here so runs are reproducible and secrets stay out of code. Any
field can be overridden with an environment variable prefixed ``ZEPHYRUS_`` or via a
local ``.env`` file (see ``.env.example``). Access it through :func:`get_settings`,
which returns a cached singleton.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings, populated from env / ``.env`` with safe defaults."""

    model_config = SettingsConfigDict(
        env_prefix="ZEPHYRUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- External APIs (all free; no key needed for the endpoints we call) ---
    octopus_base_url: str = "https://api.octopus.energy/v1"
    carbon_base_url: str = "https://api.carbonintensity.org.uk"
    elexon_base_url: str = "https://data.elexon.co.uk/bmrs/api/v1"
    open_meteo_base_url: str = "https://api.open-meteo.com/v1"

    # Weather location for demand/solar features (default: central London).
    weather_lat: float = 51.5072
    weather_lon: float = -0.1276

    # Octopus Agile: product is auto-discovered unless pinned; region is a GSP group
    # letter (A-P). 'C' = London. See docs/architecture.md for the full region map.
    octopus_agile_product: str | None = None
    gsp_group: str = "C"

    # Carbon Intensity: national unless a region id (1-17) is set. 13 = London.
    carbon_region_id: int | None = None

    # --- HTTP behaviour ---
    http_timeout_s: float = 20.0
    http_retries: int = 3

    # --- Optimiser defaults ---
    carbon_weight_gbp_per_kg: float = 0.0  # £ per kg CO2; 0.0 = pure cost minimisation

    # --- API serving (M6) ---
    api_key: str | None = None  # when set, mutating endpoints require this X-API-Key
    rate_limit_per_min: int = 120  # per-client request cap (0 disables)

    # --- Paths / ops ---
    data_dir: Path = PROJECT_ROOT / "data"
    mlflow_tracking_uri: str = "file:./mlruns"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance (import-safe singleton)."""
    return Settings()
