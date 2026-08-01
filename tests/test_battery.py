from __future__ import annotations

from zephyrus.optimise.battery import baseline_cost_and_carbon, usable_bounds_kwh
from zephyrus.schemas import BatterySpec


def test_usable_bounds():
    lo, hi = usable_bounds_kwh(BatterySpec(capacity_kwh=10, soc_min_frac=0.1, soc_max_frac=0.9))
    assert (lo, hi) == (1.0, 9.0)


def test_baseline_imports_net_load(price_carbon_series):
    prices, carbon = price_carbon_series(n=4)
    cost, carbon_kg = baseline_cost_and_carbon(prices, carbon, [1.0] * 4, [0.0] * 4)
    assert cost > 0
    assert carbon_kg > 0


def test_baseline_spills_surplus_solar(price_carbon_series):
    prices, carbon = price_carbon_series(n=4)
    cost, _ = baseline_cost_and_carbon(prices, carbon, [0.5] * 4, [2.0] * 4)
    assert cost == 0.0  # solar exceeds load every slot -> no import cost
