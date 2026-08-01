"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from zephyrus.schemas import CarbonSlot, HalfHourlyPrice


@pytest.fixture
def price_carbon_series():
    """Factory for aligned half-hourly price/carbon series.

    Prices are cheap overnight (first 16 slots), spike in the evening peak (slots 34-42)
    and sit mid-range otherwise — a realistic Agile shape the optimiser can exploit.
    """

    def _build(n: int = 48, cheap: float = 8.0, peak: float = 35.0, mid: float = 18.0):
        t0 = datetime(2026, 1, 1, tzinfo=UTC)
        prices, carbon = [], []
        for i in range(n):
            start = t0 + timedelta(minutes=30 * i)
            end = start + timedelta(minutes=30)
            price = cheap if i < 16 else (peak if 34 <= i <= 42 else mid)
            prices.append(HalfHourlyPrice(valid_from=start, valid_to=end, price_p_per_kwh=price))
            carbon.append(CarbonSlot(valid_from=start, valid_to=end, intensity_g_per_kwh=200.0))
        return prices, carbon

    return _build
