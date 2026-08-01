"""Open-Meteo client tests — upstream mocked with respx (offline)."""

from __future__ import annotations

from datetime import UTC, datetime

import respx
from httpx import Response

from zephyrus.io.weather import OpenMeteoClient


@respx.mock
def test_hourly_expanded_to_half_hourly_and_interpolated():
    payload = {
        "hourly": {
            "time": ["2026-01-01T00:00", "2026-01-01T01:00", "2026-01-01T02:00"],
            "temperature_2m": [10.0, 12.0, 14.0],
            "shortwave_radiation": [0.0, 5.0, 20.0],
            "cloud_cover": [80, 60, 40],
        }
    }
    respx.get(url__regex=r".*/forecast.*").mock(return_value=Response(200, json=payload))

    slots = OpenMeteoClient().get_forecast(
        start=datetime(2026, 1, 1, tzinfo=UTC), hours=2, lat=51.5, lon=-0.1
    )

    assert len(slots) == 4  # 2 hours -> 4 half-hourly slots
    assert slots[0].valid_from == datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    assert slots[1].valid_from == datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
    # :30 temperature is the linear midpoint between consecutive hours (10 -> 12).
    assert slots[1].temperature_c == 11.0
    assert slots[0].shortwave_radiation_w_m2 == 0.0
    assert slots[0].cloud_cover_pct == 80.0
