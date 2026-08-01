"""Price forecaster + arbitrage backtest (M4). Offline; numpy-ridge fallback."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from zephyrus.forecast.price import (
    PriceForecaster,
    backtest_arbitrage,
    persistence_price_forecast,
    realised_cost_gbp,
    synthetic_price_history,
    to_price_models,
)
from zephyrus.optimise.milp import optimise_schedule
from zephyrus.schemas import (
    ArbitrageBacktestReport,
    BatterySpec,
    HalfHourlyPrice,
    PriceForecastSlot,
)

MONDAY = datetime(2025, 1, 6, tzinfo=UTC)


@pytest.fixture
def prices():
    return synthetic_price_history(MONDAY, days=42)


def test_persistence_baseline_repeats_input():
    rows = [HalfHourlyPrice(valid_from=MONDAY, valid_to=MONDAY, price_p_per_kwh=10.0)]
    assert persistence_price_forecast(rows) == rows


def test_price_forecaster_produces_monotone_band(prices):
    slots = PriceForecaster().fit(prices).predict(48)
    assert len(slots) == 48
    assert all(isinstance(s, PriceForecastSlot) for s in slots)
    for s in slots:
        assert s.lower_p_per_kwh <= s.price_p_per_kwh <= s.upper_p_per_kwh


def test_predict_prices_adapts_to_optimiser_input(prices):
    models = PriceForecaster().fit(prices).predict_prices(24)
    assert len(models) == 24
    assert all(isinstance(m, HalfHourlyPrice) for m in models)
    # p/kWh -> £/kWh helper stays consistent with the median.
    assert models[0].price_gbp_per_kwh == pytest.approx(models[0].price_p_per_kwh / 100)


def test_to_price_models_sets_half_hour_windows(prices):
    slots = PriceForecaster().fit(prices).predict(3)
    models = to_price_models(slots)
    assert (models[0].valid_to - models[0].valid_from).total_seconds() == 1800


def test_realised_cost_settles_schedule_at_actual_prices(price_carbon_series):
    prices, carbon = price_carbon_series()
    n = len(prices)
    plan = optimise_schedule(prices, carbon, BatterySpec(capacity_kwh=5), load_kwh=[0.3] * n)
    # Settling at the same prices reproduces the plan's own cost.
    assert realised_cost_gbp(plan, prices) == pytest.approx(plan.cost_gbp, abs=1e-3)
    # Settling at a uniformly higher price costs at least as much (net import >= 0).
    dearer = [
        HalfHourlyPrice(
            valid_from=p.valid_from, valid_to=p.valid_to, price_p_per_kwh=p.price_p_per_kwh + 10
        )
        for p in prices
    ]
    assert realised_cost_gbp(plan, dearer) >= realised_cost_gbp(plan, prices) - 1e-6


def test_realised_cost_length_mismatch_raises(price_carbon_series):
    prices, carbon = price_carbon_series()
    plan = optimise_schedule(prices, carbon, BatterySpec(), load_kwh=[0.3] * len(prices))
    with pytest.raises(ValueError):
        realised_cost_gbp(plan, prices[:-1])


def test_arbitrage_backtest_beats_persistence(prices):
    report = backtest_arbitrage(prices, horizon=48, n_days=4)
    assert isinstance(report, ArbitrageBacktestReport)
    # A real forecast should capture some oracle P&L and not trail persistence.
    assert report.perfect_foresight_pnl_gbp > 0
    assert report.model_pnl_gbp >= report.persistence_pnl_gbp - 1e-6
    assert 0.0 <= report.capture_rate <= 1.0
    assert 0.0 <= report.hit_rate <= 1.0
