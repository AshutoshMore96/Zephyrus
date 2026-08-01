from __future__ import annotations

from datetime import UTC, datetime

import respx
from httpx import Response

from zephyrus.config import get_settings
from zephyrus.io.carbon import CarbonIntensityClient


@respx.mock
def test_national_forecast_parsing():
    base = get_settings().carbon_base_url
    payload = {
        "data": [
            {
                "from": "2026-01-01T00:00Z",
                "to": "2026-01-01T00:30Z",
                "intensity": {"forecast": 123, "index": "moderate"},
            },
            {
                "from": "2026-01-01T00:30Z",
                "to": "2026-01-01T01:00Z",
                "intensity": {"forecast": 90, "index": "low"},
            },
        ]
    }
    respx.get(url__regex=r".*/intensity/.*/fw48h").mock(return_value=Response(200, json=payload))

    slots = CarbonIntensityClient(base).get_forecast_48h(datetime(2026, 1, 1, tzinfo=UTC))

    assert len(slots) == 2
    assert slots[0].intensity_g_per_kwh == 123
    assert slots[1].index == "low"
