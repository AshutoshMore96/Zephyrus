"""Octopus Energy API client — live Agile half-hourly *import* prices.

The product/tariff price endpoints are public and need no authentication (consumption
endpoints, not used here, would need an account key). Product codes roll over over time
(e.g. ``AGILE-24-04-03``) so we discover the current one rather than hard-coding it.

Docs: https://developer.octopus.energy/rest/
"""

from __future__ import annotations

from datetime import UTC, datetime

from ..config import get_settings
from ..logging import get_logger
from ..schemas import HalfHourlyPrice
from .base import BaseHTTPClient

logger = get_logger(__name__)


class OctopusClient(BaseHTTPClient):
    def __init__(self, base_url: str | None = None) -> None:
        super().__init__(base_url or get_settings().octopus_base_url)

    def find_agile_import_product(self) -> str:
        """Return the code of the latest single-register Agile import product.

        Falls back to the configured ``octopus_agile_product`` if one is pinned.
        """
        configured = get_settings().octopus_agile_product
        if configured:
            return configured
        data = self.get_json("/products/")
        agile = [
            p
            for p in data.get("results", [])
            if str(p.get("code", "")).startswith("AGILE") and p.get("direction") == "IMPORT"
        ]
        if not agile:
            raise RuntimeError("No Agile import product found via /products/")
        latest = sorted(agile, key=lambda p: p["code"])[-1]  # date-suffixed -> newest
        logger.info("Discovered Agile product: %s", latest["code"])
        return str(latest["code"])

    def get_agile_prices(
        self,
        period_from: datetime,
        period_to: datetime,
        product_code: str | None = None,
        gsp_group: str | None = None,
    ) -> list[HalfHourlyPrice]:
        """Fetch half-hourly Agile unit rates for a GSP region and time window."""
        settings = get_settings()
        product_code = product_code or self.find_agile_import_product()
        region = (gsp_group or settings.gsp_group).upper()
        tariff_code = f"E-1R-{product_code}-{region}"
        path = (
            f"/products/{product_code}/electricity-tariffs/" f"{tariff_code}/standard-unit-rates/"
        )
        params: dict[str, object] = {
            "period_from": _iso_seconds(period_from),
            "period_to": _iso_seconds(period_to),
            "page_size": 1500,
        }
        prices: list[HalfHourlyPrice] = []
        data = self.get_json(path, params=params)
        while True:
            for row in data.get("results", []):
                prices.append(
                    HalfHourlyPrice(
                        valid_from=row["valid_from"],
                        valid_to=row["valid_to"],
                        price_p_per_kwh=row["value_inc_vat"],
                    )
                )
            nxt = data.get("next")
            if not nxt:
                break
            data = self.get_json(nxt.replace(self.base_url, ""))  # 'next' is absolute
        prices.sort(key=lambda p: p.valid_from)
        logger.info("Fetched %d Agile price slots for region %s", len(prices), region)
        return prices


def _iso_seconds(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
