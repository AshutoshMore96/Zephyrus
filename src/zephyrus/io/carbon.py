"""National Grid Carbon Intensity API client — 48-hour half-hourly forecast.

Free, no key. National and regional (GB DNO region) endpoints are supported.
Docs: https://carbon-intensity.github.io/api-definitions/
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..config import get_settings
from ..logging import get_logger
from ..schemas import CarbonSlot
from .base import BaseHTTPClient

logger = get_logger(__name__)


class CarbonIntensityClient(BaseHTTPClient):
    def __init__(self, base_url: str | None = None) -> None:
        super().__init__(base_url or get_settings().carbon_base_url)

    def get_forecast_48h(
        self, start: datetime | None = None, region_id: int | None = None
    ) -> list[CarbonSlot]:
        """Half-hourly forecast for the 48 hours from ``start`` (default: now, UTC).

        Uses the national endpoint unless a ``region_id`` (1-17) is supplied here or in
        settings.
        """
        start = start or datetime.now(UTC)
        stamp = _iso_minutes(start)
        region_id = region_id if region_id is not None else get_settings().carbon_region_id

        if region_id is None:
            data = self.get_json(f"/intensity/{stamp}/fw48h")
            rows = data.get("data", [])
            slots = [_to_slot(r) for r in rows]
        else:
            data = self.get_json(f"/regional/intensity/{stamp}/fw48h/regionid/{region_id}")
            rows = _unwrap_regional(data)
            slots = [_to_slot(r) for r in rows]

        logger.info("Fetched %d carbon slots (region=%s)", len(slots), region_id or "national")
        return slots


def _unwrap_regional(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Regional payloads nest the slot list one level deeper; handle both shapes."""
    payload = data.get("data")
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if isinstance(payload, dict):
        return payload.get("data", [])
    return []


def _to_slot(row: dict[str, Any]) -> CarbonSlot:
    intensity = row.get("intensity", {}) or {}
    value = intensity.get("forecast")
    if value is None:
        value = intensity.get("actual")
    return CarbonSlot(
        valid_from=row["from"],
        valid_to=row["to"],
        intensity_g_per_kwh=float(value) if value is not None else 0.0,
        index=intensity.get("index"),
    )


def _iso_minutes(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")
