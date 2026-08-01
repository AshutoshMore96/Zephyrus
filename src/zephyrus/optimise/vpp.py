"""Aggregate many single-asset schedules into a portfolio / virtual power plant (M5).

Two entry points:

* :func:`aggregate` — cheap KPI roll-up over already-optimised assets.
* :func:`optimise_portfolio` — the real VPP step: co-optimise N heterogeneous assets in
  one MILP under a shared **network-headroom** constraint (aggregate grid import per slot
  ≤ a cap), which spreads charging and cuts the *coincident* peak that independent
  optimisation would create. A simple balancing-market bid + settlement values the
  delivered turn-down against a peak signal.

This is the step that turns "a battery app" into "a VPP" (maps onto Axle / Kraken Flex /
Kaluza).
"""

from __future__ import annotations

import pulp

from ..logging import get_logger
from ..schemas import (
    BatterySpec,
    CarbonSlot,
    HalfHourlyPrice,
    OptimisationResult,
    PortfolioResult,
)
from .milp import SLOT_HOURS, _build_asset, _extract_result, _solve

logger = get_logger(__name__)


def aggregate(results: list[OptimisationResult]) -> dict[str, float | int]:
    """Return portfolio KPIs across many already-optimised assets."""
    if not results:
        return {"assets": 0, "cost_saving_gbp": 0.0, "carbon_saving_kg": 0.0}
    return {
        "assets": len(results),
        "cost_saving_gbp": round(sum(r.cost_saving_gbp for r in results), 2),
        "carbon_saving_kg": round(sum(r.carbon_saving_kg for r in results), 2),
    }


def optimise_portfolio(
    prices: list[HalfHourlyPrice],
    carbon: list[CarbonSlot],
    assets: list[BatterySpec],
    load_kwh: list[float] | None = None,
    solar_kwh: list[float] | None = None,
    carbon_weight_gbp_per_kg: float | None = None,
    network_headroom_kw: float | None = None,
    peak_bid_gbp_per_kw: float = 0.0,
) -> PortfolioResult:
    """Co-optimise ``assets`` in one MILP; aggregate cost, carbon, peak and flexibility.

    Args:
        assets: heterogeneous battery/EV specs sharing the same site/feeder.
        load_kwh / solar_kwh: per-slot load/solar applied to *every* asset (the same
            household shape); pass zeros-by-default behaviour by omitting them.
        network_headroom_kw: cap on *aggregate* grid import power across all assets in any
            slot (kW). Binds the co-optimisation, cutting the coincident peak.
        peak_bid_gbp_per_kw: £/kW paid for the coincident-peak reduction the VPP delivers
            (a stand-in for a balancing/DFS settlement).
    """
    if not assets:
        raise ValueError("optimise_portfolio needs at least one asset")
    n = len(prices)
    problem = pulp.LpProblem("zephyrus_portfolio", pulp.LpMinimize)

    built = [
        _build_asset(
            problem,
            prices,
            carbon,
            asset,
            load_kwh=load_kwh,
            solar_kwh=solar_kwh,
            carbon_weight_gbp_per_kg=carbon_weight_gbp_per_kg,
            prefix=f"a{i}_",
        )
        for i, asset in enumerate(assets)
    ]

    # Shared network-headroom constraint: aggregate import power per slot ≤ cap.
    if network_headroom_kw is not None:
        headroom_kwh = network_headroom_kw * SLOT_HOURS
        for t in range(n):
            problem += (
                pulp.lpSum(a.grid_import[t] for a in built) <= headroom_kwh,
                f"headroom_t{t}",
            )

    problem += pulp.lpSum(a.obj_expr for a in built)
    _solve(problem)
    per_asset = [_extract_result(a, prices, carbon) for a in built]
    peak = _coincident_peak_kw(per_asset)

    # Reference peak = the coincident charging peak a *naive* (headroom-free) cost
    # optimisation would create; the managed VPP is valued on shaving that down. Solving
    # the unconstrained portfolio once gives an honest, non-negative reduction.
    if network_headroom_kw is not None:
        reference = optimise_portfolio(
            prices,
            carbon,
            assets,
            load_kwh=load_kwh,
            solar_kwh=solar_kwh,
            carbon_weight_gbp_per_kg=carbon_weight_gbp_per_kg,
        )
        baseline_peak = reference.peak_import_kw
    else:
        baseline_peak = peak

    flexibility_kwh = sum(
        s.charge_kw * SLOT_HOURS + s.discharge_kw * SLOT_HOURS for a in per_asset for s in a.slots
    )
    peak_reduction = max(baseline_peak - peak, 0.0)
    return PortfolioResult(
        n_assets=len(per_asset),
        cost_saving_gbp=round(sum(a.cost_saving_gbp for a in per_asset), 4),
        carbon_saving_kg=round(sum(a.carbon_saving_kg for a in per_asset), 4),
        peak_import_kw=round(peak, 4),
        baseline_peak_import_kw=round(baseline_peak, 4),
        flexibility_kwh=round(flexibility_kwh, 4),
        bid_revenue_gbp=round(peak_reduction * peak_bid_gbp_per_kw, 4),
        per_asset=per_asset,
    )


def _coincident_peak_kw(per_asset: list[OptimisationResult]) -> float:
    """Max over slots of the aggregate (simultaneous) grid import power across assets."""
    n = len(per_asset[0].slots)
    return max(
        (sum(a.slots[t].grid_import_kwh for a in per_asset) / SLOT_HOURS for t in range(n)),
        default=0.0,
    )
