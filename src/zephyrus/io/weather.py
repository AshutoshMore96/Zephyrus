"""Open-Meteo weather client — exogenous features for demand & solar (project #1, M1).

Free, no API key. Open-Meteo returns *hourly* values; homes settle on half-hours, so we
upsample each hour to its two half-hourly slots (linear for smooth fields like
temperature, forward-fill for fluxes). Everything is tz-aware UTC to align with prices
and carbon on the same grid.

Docs: https://open-meteo.com/en/docs
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ..config import get_settings
from ..logging import get_logger
from ..schemas import WeatherSlot
from .base import BaseHTTPClient

logger = get_logger(__name__)

_HOURLY_VARS = "temperature_2m,shortwave_radiation,cloud_cover"


class OpenMeteoClient(BaseHTTPClient):
    """Fetch half-hourly weather features for a lat/lon."""

    def __init__(self, base_url: str | None = None) -> None:
        super().__init__(base_url or get_settings().open_meteo_base_url)

    def get_forecast(
        self,
        start: datetime | None = None,
        hours: int = 48,
        lat: float | None = None,
        lon: float | None = None,
    ) -> list[WeatherSlot]:
        """Return half-hourly weather slots covering ``hours`` from ``start`` (UTC)."""
        settings = get_settings()
        start = (
            (start or datetime.now(UTC)).astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        )
        end = start + timedelta(hours=hours)
        params = {
            "latitude": lat if lat is not None else settings.weather_lat,
            "longitude": lon if lon is not None else settings.weather_lon,
            "hourly": _HOURLY_VARS,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "timezone": "UTC",
        }
        data = self.get_json("/forecast", params=params)
        slots = _to_half_hourly(data.get("hourly", {}))
        window = [s for s in slots if start <= s.valid_from < end]
        logger.info("Fetched %d half-hourly weather slots", len(window))
        return window


def _to_half_hourly(hourly: dict[str, list]) -> list[WeatherSlot]:
    """Expand hourly Open-Meteo arrays into half-hourly :class:`WeatherSlot` objects."""
    times = hourly.get("time", [])
    temp = hourly.get("temperature_2m", [])
    swr = hourly.get("shortwave_radiation", [])
    cloud = hourly.get("cloud_cover", [])
    n = len(times)
    slots: list[WeatherSlot] = []
    for i in range(n):
        t0 = _parse(times[i])
        t_now = float(temp[i]) if i < len(temp) and temp[i] is not None else 0.0
        # Linear interpolation of temperature to the :30 slot; fluxes held flat.
        t_next = float(temp[i + 1]) if i + 1 < len(temp) and temp[i + 1] is not None else t_now
        for half, ts in enumerate((t0, t0 + timedelta(minutes=30))):
            slots.append(
                WeatherSlot(
                    valid_from=ts,
                    temperature_c=round(t_now + (t_next - t_now) * (0.5 * half), 3),
                    shortwave_radiation_w_m2=(
                        float(swr[i]) if i < len(swr) and swr[i] is not None else 0.0
                    ),
                    cloud_cover_pct=(
                        float(cloud[i]) if i < len(cloud) and cloud[i] is not None else 0.0
                    ),
                )
            )
    return slots


def _parse(stamp: str) -> datetime:
    """Parse an Open-Meteo ISO timestamp (``2026-01-01T00:00``) as UTC."""
    dt = datetime.fromisoformat(stamp)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
