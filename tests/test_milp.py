"""The MILP tests double as the executable spec for correct optimiser behaviour."""

from __future__ import annotations

import pytest

from zephyrus.forecast.demand import synthetic_household_load
from zephyrus.optimise.milp import optimise_schedule
from zephyrus.schemas import BatterySpec


def test_optimiser_never_worse_than_baseline(price_carbon_series):
    prices, carbon = price_carbon_series()
    load = synthetic_household_load(len(prices))
    result = optimise_schedule(
        prices,
        carbon,
        BatterySpec(capacity_kwh=5, max_charge_kw=3, max_discharge_kw=3),
        load_kwh=load,
    )
    assert result.baseline_cost_gbp > 0
    assert result.cost_saving_gbp >= -1e-6  # arbitrage can only help vs doing nothing


def test_soc_stays_within_physical_bounds(price_carbon_series):
    prices, carbon = price_carbon_series()
    load = synthetic_household_load(len(prices))
    result = optimise_schedule(prices, carbon, BatterySpec(capacity_kwh=5), load_kwh=load)
    for slot in result.slots:
        assert -1e-6 <= slot.soc_kwh <= 5 + 1e-6


def test_no_simultaneous_charge_and_discharge(price_carbon_series):
    prices, carbon = price_carbon_series()
    load = synthetic_household_load(len(prices))
    result = optimise_schedule(prices, carbon, BatterySpec(), load_kwh=load)
    for slot in result.slots:
        assert not (slot.charge_kw > 1e-6 and slot.discharge_kw > 1e-6)


def test_misaligned_series_raises(price_carbon_series):
    prices, carbon = price_carbon_series()
    with pytest.raises(ValueError):
        optimise_schedule(prices, carbon[:-1], BatterySpec())
