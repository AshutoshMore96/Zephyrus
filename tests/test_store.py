"""Parquet snapshot store + model<->frame round-trips (M1)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from zephyrus.io.store import (
    SnapshotStore,
    frame_to_models,
    load_result,
    models_to_frame,
    save_result,
)
from zephyrus.optimise.milp import optimise_schedule
from zephyrus.schemas import BatterySpec, CarbonSlot, HalfHourlyPrice


def _prices(n: int = 4) -> list[HalfHourlyPrice]:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    return [
        HalfHourlyPrice(
            valid_from=t0 + timedelta(minutes=30 * i),
            valid_to=t0 + timedelta(minutes=30 * (i + 1)),
            price_p_per_kwh=10.0 + i,
        )
        for i in range(n)
    ]


def test_models_frame_roundtrip():
    prices = _prices()
    frame = models_to_frame(prices)
    assert frame.index.name == "valid_from"
    assert "price_p_per_kwh" in frame.columns
    restored = frame_to_models(frame, HalfHourlyPrice)
    assert [p.price_p_per_kwh for p in restored] == [p.price_p_per_kwh for p in prices]


def test_save_and_load_parquet(tmp_path):
    store = SnapshotStore(base_dir=tmp_path)
    frame = models_to_frame(_prices())
    assert not store.exists("agile")
    store.save("agile", frame)
    assert store.exists("agile")
    loaded = store.load("agile")
    pd.testing.assert_frame_equal(loaded, frame)


def test_cached_or_build_uses_cache_within_ttl(tmp_path):
    store = SnapshotStore(base_dir=tmp_path)
    calls = {"n": 0}

    def builder() -> pd.DataFrame:
        calls["n"] += 1
        return models_to_frame(_prices())

    store.cached_or_build("agile", builder, ttl_seconds=3600)
    store.cached_or_build("agile", builder, ttl_seconds=3600)  # second call hits cache
    assert calls["n"] == 1


def test_save_and_load_result_roundtrip(tmp_path):
    prices = _prices(6)
    carbon = [
        CarbonSlot(valid_from=p.valid_from, valid_to=p.valid_to, intensity_g_per_kwh=200.0)
        for p in prices
    ]
    result = optimise_schedule(prices, carbon, BatterySpec(capacity_kwh=5), load_kwh=[0.3] * 6)

    save_result("plan", result, base_dir=tmp_path)
    loaded = load_result("plan", base_dir=tmp_path)
    assert loaded.cost_gbp == result.cost_gbp
    assert len(loaded.slots) == len(result.slots)
    assert loaded.cost_saving_gbp == result.cost_saving_gbp


def test_cached_or_build_rebuilds_when_stale(tmp_path):
    store = SnapshotStore(base_dir=tmp_path)
    calls = {"n": 0}

    def builder() -> pd.DataFrame:
        calls["n"] += 1
        return models_to_frame(_prices())

    store.cached_or_build("agile", builder, ttl_seconds=0)  # ttl 0 -> always stale
    store.cached_or_build("agile", builder, ttl_seconds=0)
    assert calls["n"] == 2
