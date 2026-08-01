"""Elexon BMRS client tests — upstream mocked with respx (offline)."""

from __future__ import annotations

from datetime import UTC, datetime

import respx
from httpx import Response

from zephyrus.io.elexon import ElexonClient


@respx.mock
def test_system_prices_converted_to_p_per_kwh():
    payload = {
        "data": [
            {
                "settlementPeriod": 1,
                "startTime": "2026-01-01T00:00:00Z",
                "systemSellPrice": 50.0,
                "netImbalanceVolume": -120.0,
            },
            {
                "settlementPeriod": 2,
                "startTime": "2026-01-01T00:30:00Z",
                "systemSellPrice": 80.0,
                "netImbalanceVolume": 60.0,
            },
        ]
    }
    respx.get(url__regex=r".*/system-prices/.*").mock(return_value=Response(200, json=payload))

    prices = ElexonClient().get_system_prices(datetime(2026, 1, 1, tzinfo=UTC))
    assert len(prices) == 2
    assert prices[0].price_p_per_kwh == 5.0  # £50/MWh -> 5 p/kWh
    assert prices[0].settlement_period == 1
    assert prices[0].net_imbalance_volume_mwh == -120.0


@respx.mock
def test_generation_by_fuel_type_parsed():
    payload = {
        "data": [
            {"startTime": "2026-01-01T00:00:00Z", "fuelType": "WIND", "generation": 8000.0},
            {"startTime": "2026-01-01T00:00:00Z", "fuelType": "CCGT", "generation": 12000.0},
        ]
    }
    respx.get(url__regex=r".*/datasets/FUELHH.*").mock(return_value=Response(200, json=payload))

    gen = ElexonClient().get_generation_by_fuel_type(
        datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)
    )
    assert {g.fuel_type for g in gen} == {"WIND", "CCGT"}
    assert gen[0].generation_mw in (8000.0, 12000.0)
