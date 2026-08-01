from __future__ import annotations

from datetime import UTC, datetime

import respx
from httpx import Response

from zephyrus.io.octopus import OctopusClient


@respx.mock
def test_agile_price_parsing():
    client = OctopusClient()
    rates = {
        "next": None,
        "results": [
            {
                "value_inc_vat": 12.5,
                "valid_from": "2026-01-01T00:00:00Z",
                "valid_to": "2026-01-01T00:30:00Z",
            },
            {
                "value_inc_vat": 9.1,
                "valid_from": "2026-01-01T00:30:00Z",
                "valid_to": "2026-01-01T01:00:00Z",
            },
        ],
    }
    respx.get(url__regex=r".*/standard-unit-rates/.*").mock(return_value=Response(200, json=rates))

    prices = client.get_agile_prices(
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 1, 1, tzinfo=UTC),
        product_code="AGILE-TEST",
        gsp_group="C",
    )

    assert len(prices) == 2
    assert prices[0].price_gbp_per_kwh == 0.125  # 12.5 p -> £0.125
    assert prices[0].valid_from.hour == 0
