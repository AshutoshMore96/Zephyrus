"""API smoke tests — skipped unless the ``api`` extra (FastAPI) is installed.

Upstreams (Octopus, Carbon Intensity) are mocked with respx so the suite stays offline.
These assert the forecaster is wired into /optimise and that /forecast is served.
"""

from __future__ import annotations

import importlib.util

import pytest

pytest.importorskip("fastapi", reason="requires the 'api' extra")

import respx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from httpx import Response  # noqa: E402

from zephyrus.api.main import app  # noqa: E402

_HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None


def _mock_upstreams() -> None:
    rates = {
        "next": None,
        "results": [
            {
                "value_inc_vat": 10.0 + i,
                "valid_from": f"2026-01-01T{i // 2:02d}:{'30' if i % 2 else '00'}:00Z",
                "valid_to": f"2026-01-01T{(i + 1) // 2:02d}:{'00' if i % 2 else '30'}:00Z",
            }
            for i in range(4)
        ],
    }
    carbon = {
        "data": [
            {
                "from": f"2026-01-01T{i // 2:02d}:{'30' if i % 2 else '00'}:00Z",
                "to": f"2026-01-01T{(i + 1) // 2:02d}:{'00' if i % 2 else '30'}:00Z",
                "intensity": {"forecast": 200, "index": "moderate"},
            }
            for i in range(4)
        ]
    }
    respx.get(url__regex=r".*/standard-unit-rates/.*").mock(return_value=Response(200, json=rates))
    respx.get(url__regex=r".*/intensity/.*").mock(return_value=Response(200, json=carbon))


def test_health() -> None:
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_forecast_endpoint_returns_probabilistic_slots() -> None:
    with TestClient(app) as client:
        resp = client.post("/forecast", json={"hours": 12, "history_weeks": 4})
    assert resp.status_code == 200
    slots = resp.json()
    assert len(slots) == 24  # 12h * 2 half-hours
    for slot in slots:
        assert slot["lower_kwh"] <= slot["load_kwh"] <= slot["upper_kwh"]


@respx.mock
def test_optimise_uses_forecast_driven_load() -> None:
    _mock_upstreams()
    with TestClient(app) as client:
        resp = client.post("/optimise", json={"hours": 2, "battery": {"capacity_kwh": 5}})
    assert resp.status_code == 200
    body = resp.json()
    assert "cost_saving_gbp" in body
    assert len(body["slots"]) == 4
    # Load fed to the optimiser came from the forecaster (non-zero household demand).
    assert any(slot["load_kwh"] > 0 for slot in body["slots"])


@respx.mock
def test_vpp_endpoint_aggregates_assets() -> None:
    _mock_upstreams()
    payload = {
        "hours": 2,
        "assets": [{"capacity_kwh": 5}, {"capacity_kwh": 8}],
        "network_headroom_kw": 20.0,
        "peak_bid_gbp_per_kw": 5.0,
    }
    with TestClient(app) as client:
        resp = client.post("/vpp", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_assets"] == 2
    assert "peak_reduction_kw" in body


def test_api_key_enforced_when_configured(monkeypatch) -> None:
    from zephyrus.config import get_settings

    monkeypatch.setenv("ZEPHYRUS_API_KEY", "s3cret")
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            unauth = client.post("/forecast", json={"hours": 2})
            assert unauth.status_code == 401
            ok = client.post("/forecast", json={"hours": 2}, headers={"X-API-Key": "s3cret"})
            assert ok.status_code == 200
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
