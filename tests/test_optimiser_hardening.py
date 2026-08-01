"""M2 optimiser-hardening spec: export tariff, EV mode, degradation, sensitivity."""

from __future__ import annotations

import pytest

from zephyrus.optimise.milp import optimise_schedule
from zephyrus.optimise.sensitivity import sweep_savings
from zephyrus.schemas import BatterySpec, EVConstraints


def _throughput(result) -> float:
    return sum(s.charge_kw + s.discharge_kw for s in result.slots)


# -- Solar + Outgoing export tariff -----------------------------------------------------
def test_export_tariff_credits_surplus_solar(price_carbon_series):
    prices, carbon = price_carbon_series()
    n = len(prices)
    solar = [2.0] * n  # generous solar, no load -> big surplus
    battery = BatterySpec(capacity_kwh=5, allow_export=True)
    export_price = [0.15] * n  # flat 15p Outgoing tariff (£/kWh)

    result = optimise_schedule(
        prices,
        carbon,
        battery,
        load_kwh=[0.0] * n,
        solar_kwh=solar,
        export_prices_gbp=export_price,
    )
    # Exporting surplus earns money -> negative net cost is possible, savings >= 0.
    assert any(s.grid_export_kwh > 0 for s in result.slots)
    assert result.cost_saving_gbp >= -1e-6


def test_no_export_when_disallowed(price_carbon_series):
    prices, carbon = price_carbon_series()
    n = len(prices)
    result = optimise_schedule(
        prices, carbon, BatterySpec(allow_export=False), load_kwh=[0.0] * n, solar_kwh=[2.0] * n
    )
    assert all(s.grid_export_kwh == 0 for s in result.slots)


# -- EV mode ----------------------------------------------------------------------------
def test_ev_unplugged_slots_are_idle(price_carbon_series):
    prices, carbon = price_carbon_series()
    n = len(prices)
    availability = [i >= 8 for i in range(n)]  # unplugged for the first 8 slots
    ev = EVConstraints(availability=availability)
    result = optimise_schedule(prices, carbon, BatterySpec(), load_kwh=[0.3] * n, ev=ev)
    for i, slot in enumerate(result.slots):
        if not availability[i]:
            assert slot.charge_kw == 0 and slot.discharge_kw == 0


def test_ev_departure_soc_target_is_met(price_carbon_series):
    prices, carbon = price_carbon_series()
    n = len(prices)
    battery = BatterySpec(capacity_kwh=5, max_charge_kw=3, initial_soc_frac=0.2)
    ev = EVConstraints(departure_index=16, departure_soc_frac=0.9)
    result = optimise_schedule(prices, carbon, battery, load_kwh=[0.1] * n, ev=ev)
    # SoC at the start of the departure slot (end of slot 15) must meet the target.
    assert result.slots[15].soc_kwh >= 0.9 * 5 - 1e-4


def test_ev_infeasible_target_raises(price_carbon_series):
    prices, carbon = price_carbon_series()
    n = len(prices)
    # Full charge demanded almost immediately with limited power + a short plug-in window.
    battery = BatterySpec(capacity_kwh=10, max_charge_kw=3, initial_soc_frac=0.1)
    ev = EVConstraints(
        availability=[i < 2 for i in range(n)], departure_index=2, departure_soc_frac=1.0
    )
    with pytest.raises(ValueError):
        optimise_schedule(prices, carbon, battery, load_kwh=[0.0] * n, ev=ev)


# -- Degradation cost -------------------------------------------------------------------
def test_degradation_penalty_reduces_cycling(price_carbon_series):
    prices, carbon = price_carbon_series()
    n = len(prices)
    load = [0.3] * n
    cheap = optimise_schedule(prices, carbon, BatterySpec(capacity_kwh=5), load_kwh=load)
    worn = optimise_schedule(
        prices,
        carbon,
        BatterySpec(capacity_kwh=5),
        load_kwh=load,
        degradation_gbp_per_kwh=0.20,
    )
    assert _throughput(worn) <= _throughput(cheap) + 1e-6
    assert _throughput(worn) < _throughput(cheap)  # a real wear cost throttles arbitrage


# -- Sensitivity sweep ------------------------------------------------------------------
def test_sweep_savings_grid_and_monotonicity(price_carbon_series):
    prices, carbon = price_carbon_series()
    load = [0.3] * len(prices)
    frame = sweep_savings(prices, carbon, capacities_kwh=[2.0, 5.0], powers_kw=[3.0], load_kwh=load)
    assert len(frame) == 2
    assert {"capacity_kwh", "power_kw", "cost_saving_gbp", "cost_saving_per_kwh"} <= set(
        frame.columns
    )
    # More capacity cannot reduce absolute savings (same power, richer feasible set).
    by_cap = frame.set_index("capacity_kwh")["cost_saving_gbp"]
    assert by_cap.loc[5.0] >= by_cap.loc[2.0] - 1e-6
