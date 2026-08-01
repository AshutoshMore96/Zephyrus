"""VPP co-optimisation + aggregation (M5)."""

from __future__ import annotations

import pytest

from zephyrus.optimise.vpp import aggregate, optimise_portfolio
from zephyrus.schemas import BatterySpec, PortfolioResult


def _peak_import_kw(result) -> float:
    return max(s.grid_import_kwh for s in result.slots) / 0.5


def test_aggregate_rolls_up_savings(price_carbon_series):
    from zephyrus.optimise.milp import optimise_schedule

    prices, carbon = price_carbon_series()
    load = [0.3] * len(prices)
    results = [
        optimise_schedule(prices, carbon, BatterySpec(capacity_kwh=c), load_kwh=load)
        for c in (5, 8)
    ]
    kpi = aggregate(results)
    assert kpi["assets"] == 2
    assert kpi["cost_saving_gbp"] == pytest.approx(
        round(sum(r.cost_saving_gbp for r in results), 2)
    )


def test_aggregate_handles_empty():
    assert aggregate([])["assets"] == 0


def test_portfolio_optimises_heterogeneous_assets(price_carbon_series):
    prices, carbon = price_carbon_series()
    load = [0.3] * len(prices)
    assets = [BatterySpec(capacity_kwh=5), BatterySpec(capacity_kwh=8, max_charge_kw=5)]
    result = optimise_portfolio(prices, carbon, assets, load_kwh=load)
    assert isinstance(result, PortfolioResult)
    assert result.n_assets == 2
    assert len(result.per_asset) == 2
    assert result.cost_saving_gbp >= -1e-6


def test_network_headroom_cuts_coincident_peak(price_carbon_series):
    prices, carbon = price_carbon_series()
    load = [0.3] * len(prices)
    assets = [BatterySpec(capacity_kwh=6, max_charge_kw=4) for _ in range(3)]

    unconstrained = optimise_portfolio(prices, carbon, assets, load_kwh=load)
    headroom = unconstrained.peak_import_kw * 0.6  # below the naive coincident peak

    constrained = optimise_portfolio(
        prices, carbon, assets, load_kwh=load, network_headroom_kw=headroom
    )
    # Aggregate import never exceeds the headroom, and the coincident peak really drops.
    for t in range(len(prices)):
        agg_kw = sum(a.slots[t].grid_import_kwh for a in constrained.per_asset) / 0.5
        assert agg_kw <= headroom + 1e-4
    assert constrained.peak_import_kw <= unconstrained.peak_import_kw + 1e-6
    assert constrained.peak_import_kw < unconstrained.peak_import_kw


def test_balancing_bid_values_peak_reduction(price_carbon_series):
    prices, carbon = price_carbon_series()
    load = [0.3] * len(prices)
    assets = [BatterySpec(capacity_kwh=6, max_charge_kw=4) for _ in range(3)]
    unconstrained = optimise_portfolio(prices, carbon, assets, load_kwh=load)
    headroom = unconstrained.peak_import_kw * 0.6

    result = optimise_portfolio(
        prices,
        carbon,
        assets,
        load_kwh=load,
        network_headroom_kw=headroom,
        peak_bid_gbp_per_kw=5.0,
    )
    assert result.peak_reduction_kw > 0  # the VPP shaves the naive coincident peak
    assert result.bid_revenue_gbp == pytest.approx(result.peak_reduction_kw * 5.0, abs=1e-4)
    assert result.total_value_gbp == pytest.approx(
        result.cost_saving_gbp + result.bid_revenue_gbp, abs=1e-4
    )


def test_portfolio_requires_assets(price_carbon_series):
    prices, carbon = price_carbon_series()
    with pytest.raises(ValueError):
        optimise_portfolio(prices, carbon, [])
