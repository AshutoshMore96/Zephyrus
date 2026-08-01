"""Elexon BMRS (Insights) client — imbalance prices & generation mix (project #4, M4).

All BMRS Insights endpoints are free and need no key. System prices arrive in £/MWh and
are converted to the project's pence/kWh convention (£/MWh ÷ 10 = p/kWh). FUELHH gives
half-hourly generation by fuel type — exogenous features for the price model.

Docs: https://bmrs.elexon.co.uk/api-documentation
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..config import get_settings
from ..logging import get_logger
from ..schemas import FuelGenerationSlot, SystemPrice
from .base import BaseHTTPClient

logger = get_logger(__name__)


def _mwh_price_to_p_per_kwh(price_gbp_per_mwh: float) -> float:
    """£/MWh -> pence/kWh: £/MWh × 100 p/£ ÷ 1000 kWh/MWh = £/MWh ÷ 10."""
    return price_gbp_per_mwh / 10.0


class ElexonClient(BaseHTTPClient):
    def __init__(self, base_url: str | None = None) -> None:
        super().__init__(base_url or get_settings().elexon_base_url)

    def get_system_prices(self, settlement_date: datetime) -> list[SystemPrice]:
        """System (imbalance) price + NIV per settlement period for a day."""
        stamp = (
            settlement_date.astimezone(UTC).strftime("%Y-%m-%d")
            if settlement_date.tzinfo
            else (settlement_date.strftime("%Y-%m-%d"))
        )
        data = self.get_json(f"/balancing/settlement/system-prices/{stamp}")
        rows = data.get("data", []) if isinstance(data, dict) else data
        prices = [p for p in (_to_system_price(r) for r in rows) if p is not None]
        prices.sort(key=lambda p: p.valid_from)
        logger.info("Fetched %d system-price slots for %s", len(prices), stamp)
        return prices

    def get_generation_by_fuel_type(
        self, start: datetime, end: datetime
    ) -> list[FuelGenerationSlot]:
        """Half-hourly generation outturn by fuel type (FUELHH) between ``start`` and ``end``."""
        params = {"publishDateTimeFrom": _iso(start), "publishDateTimeTo": _iso(end)}
        data = self.get_json("/datasets/FUELHH", params=params)
        rows = data.get("data", []) if isinstance(data, dict) else data
        slots = [s for s in (_to_generation_slot(r) for r in rows) if s is not None]
        slots.sort(key=lambda s: (s.valid_from, s.fuel_type))
        logger.info("Fetched %d FUELHH generation rows", len(slots))
        return slots


def _to_system_price(row: dict[str, Any]) -> SystemPrice | None:
    price = row.get("systemSellPrice", row.get("price", row.get("systemBuyPrice")))
    start = row.get("startTime") or row.get("settlementPeriodStartTime")
    period = row.get("settlementPeriod")
    if price is None or start is None or period is None:
        return None
    return SystemPrice(
        valid_from=start,
        settlement_period=int(period),
        price_p_per_kwh=round(_mwh_price_to_p_per_kwh(float(price)), 4),
        net_imbalance_volume_mwh=_opt_float(row.get("netImbalanceVolume")),
    )


def _to_generation_slot(row: dict[str, Any]) -> FuelGenerationSlot | None:
    start = row.get("startTime") or row.get("publishDateTime")
    fuel = row.get("fuelType")
    gen = row.get("generation")
    if start is None or fuel is None or gen is None:
        return None
    return FuelGenerationSlot(valid_from=start, fuel_type=str(fuel), generation_mw=float(gen))


def _opt_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
