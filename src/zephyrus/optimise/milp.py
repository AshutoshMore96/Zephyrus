"""Battery/EV scheduling as a Mixed-Integer Linear Program (project #2 — core).

Given half-hourly import prices and carbon intensity (plus optional household load and
solar), choose charge/discharge to minimise a blended cost + carbon objective, subject
to state-of-charge and power limits. A binary per slot forbids simultaneous charge and
discharge, making this a genuine MILP.

Solved with CBC, which ships inside the ``pulp`` wheel — no paid solver required.

A single asset is built by :func:`_build_asset`, which the VPP layer (M5) reuses to
co-optimise many assets under a shared network-headroom constraint.

The forecasting components feed this function:
  * project #1 (demand)  -> ``load_kwh``
  * project #4 (price)   -> ``prices``  (forecast rather than actual, when backtesting)
"""

from __future__ import annotations

from dataclasses import dataclass

import pulp

from ..config import get_settings
from ..logging import get_logger
from ..schemas import (
    BatterySpec,
    CarbonSlot,
    EVConstraints,
    HalfHourlyPrice,
    OptimisationResult,
    ScheduleSlot,
)
from .battery import baseline_cost_and_carbon, usable_bounds_kwh

logger = get_logger(__name__)

SLOT_HOURS = 0.5  # half-hourly settlement periods


@dataclass
class _AssetVars:
    """PuLP variables + expressions for one asset added to a (possibly shared) problem."""

    n: int
    battery: BatterySpec
    charge: dict[int, pulp.LpVariable]
    discharge: dict[int, pulp.LpVariable]
    grid_import: dict[int, pulp.LpVariable]
    grid_export: dict[int, pulp.LpVariable]
    soc: dict[int, pulp.LpVariable]
    load: list[float]
    solar: list[float]
    export_price: list[float]
    cost_expr: pulp.LpAffineExpression  # £ energy bill (import - export credit)
    obj_expr: pulp.LpAffineExpression  # full objective: cost + carbon + degradation


def optimise_schedule(
    prices: list[HalfHourlyPrice],
    carbon: list[CarbonSlot],
    battery: BatterySpec,
    load_kwh: list[float] | None = None,
    solar_kwh: list[float] | None = None,
    carbon_weight_gbp_per_kg: float | None = None,
    export_prices_gbp: list[float] | None = None,
    degradation_gbp_per_kwh: float = 0.0,
    ev: EVConstraints | None = None,
) -> OptimisationResult:
    """Return the cost/carbon-optimal battery schedule and the achieved savings.

    Args:
        prices: aligned half-hourly import prices.
        carbon: aligned half-hourly carbon intensity (same length as ``prices``).
        battery: asset limits.
        load_kwh: household demand per slot (kWh). Defaults to zeros.
        solar_kwh: on-site generation per slot (kWh). Defaults to zeros.
        carbon_weight_gbp_per_kg: £ value per kg CO2 in the objective. Defaults to the
            configured value (0.0 = pure cost minimisation).
        export_prices_gbp: Outgoing/export unit price per slot (£/kWh). Defaults to the
            import price. Only used when ``battery.allow_export``.
        degradation_gbp_per_kwh: £ cost per kWh of battery throughput — a wear term.
        ev: optional EV smart-charging constraints (plug-in windows + departure SoC).
    """
    problem = pulp.LpProblem("zephyrus_schedule", pulp.LpMinimize)
    asset = _build_asset(
        problem,
        prices,
        carbon,
        battery,
        load_kwh=load_kwh,
        solar_kwh=solar_kwh,
        carbon_weight_gbp_per_kg=carbon_weight_gbp_per_kg,
        export_prices_gbp=export_prices_gbp,
        degradation_gbp_per_kwh=degradation_gbp_per_kwh,
        ev=ev,
    )
    problem += asset.obj_expr
    _solve(problem)
    return _extract_result(asset, prices, carbon)


def _build_asset(
    problem: pulp.LpProblem,
    prices: list[HalfHourlyPrice],
    carbon: list[CarbonSlot],
    battery: BatterySpec,
    load_kwh: list[float] | None = None,
    solar_kwh: list[float] | None = None,
    carbon_weight_gbp_per_kg: float | None = None,
    export_prices_gbp: list[float] | None = None,
    degradation_gbp_per_kwh: float = 0.0,
    ev: EVConstraints | None = None,
    prefix: str = "",
) -> _AssetVars:
    """Add one asset's variables + constraints to ``problem`` and return its expressions."""
    n = len(prices)
    if len(carbon) != n:
        raise ValueError("prices and carbon must be aligned and the same length")
    load = load_kwh if load_kwh is not None else [0.0] * n
    solar = solar_kwh if solar_kwh is not None else [0.0] * n
    export_price = (
        export_prices_gbp
        if export_prices_gbp is not None
        else [p.price_gbp_per_kwh for p in prices]
    )
    if len(export_price) != n:
        raise ValueError("export_prices_gbp must be aligned and the same length as prices")
    cw = (
        carbon_weight_gbp_per_kg
        if carbon_weight_gbp_per_kg is not None
        else get_settings().carbon_weight_gbp_per_kg
    )
    available = _availability_mask(ev, n)
    soc_min, soc_max = usable_bounds_kwh(battery)
    eta = battery.one_way_efficiency
    big_m = max(battery.max_charge_kw, battery.max_discharge_kw)

    def name(base: str) -> str:
        return f"{prefix}{base}"

    charge = pulp.LpVariable.dicts(name("charge_kw"), range(n), 0, battery.max_charge_kw)
    discharge = pulp.LpVariable.dicts(name("discharge_kw"), range(n), 0, battery.max_discharge_kw)
    grid_import = pulp.LpVariable.dicts(name("grid_import_kwh"), range(n), 0)
    grid_export = pulp.LpVariable.dicts(name("grid_export_kwh"), range(n), 0)
    curtail = pulp.LpVariable.dicts(name("curtail_kwh"), range(n), 0)
    soc = pulp.LpVariable.dicts(name("soc_kwh"), range(n + 1), soc_min, soc_max)
    charging = pulp.LpVariable.dicts(name("is_charging"), range(n), cat="Binary")

    problem += soc[0] == battery.initial_soc_frac * battery.capacity_kwh

    for t in range(n):
        ch_kwh = charge[t] * SLOT_HOURS
        dis_kwh = discharge[t] * SLOT_HOURS
        problem += soc[t + 1] == soc[t] + eta * ch_kwh - dis_kwh / eta
        problem += (
            grid_import[t] + solar[t] + dis_kwh == load[t] + ch_kwh + grid_export[t] + curtail[t]
        )
        problem += grid_export[t] <= solar[t] + dis_kwh
        problem += charge[t] <= big_m * charging[t]
        problem += discharge[t] <= big_m * (1 - charging[t])
        if not battery.allow_export:
            problem += grid_export[t] == 0
        if not available[t]:  # EV unplugged: battery idle, state carried.
            problem += charge[t] == 0
            problem += discharge[t] == 0

    _add_ev_departure_target(problem, soc, battery, ev)

    cost_expr = pulp.lpSum(
        grid_import[t] * prices[t].price_gbp_per_kwh - grid_export[t] * export_price[t]
        for t in range(n)
    )
    obj_expr = cost_expr + pulp.lpSum(
        cw * grid_import[t] * carbon[t].intensity_g_per_kwh / 1000.0
        + degradation_gbp_per_kwh * (charge[t] + discharge[t]) * SLOT_HOURS
        for t in range(n)
    )
    return _AssetVars(
        n=n,
        battery=battery,
        charge=charge,
        discharge=discharge,
        grid_import=grid_import,
        grid_export=grid_export,
        soc=soc,
        load=load,
        solar=solar,
        export_price=export_price,
        cost_expr=cost_expr,
        obj_expr=obj_expr,
    )


def _solve(problem: pulp.LpProblem) -> None:
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    logger.info("MILP status: %s", pulp.LpStatus[status])
    if pulp.LpStatus[status] != "Optimal":
        raise ValueError(
            f"MILP not solved to optimality ({pulp.LpStatus[status]}); check EV departure "
            "target vs plug-in windows, network headroom and power limits"
        )


def _extract_result(
    asset: _AssetVars, prices: list[HalfHourlyPrice], carbon: list[CarbonSlot]
) -> OptimisationResult:
    """Read a solved asset's variables into a typed :class:`OptimisationResult`."""
    slots: list[ScheduleSlot] = []
    cost = 0.0
    carbon_kg = 0.0
    for t in range(asset.n):
        gi = asset.grid_import[t].value() or 0.0
        ge = asset.grid_export[t].value() or 0.0
        cost += gi * prices[t].price_gbp_per_kwh - ge * asset.export_price[t]
        carbon_kg += gi * carbon[t].intensity_g_per_kwh / 1000.0
        slots.append(
            ScheduleSlot(
                valid_from=prices[t].valid_from,
                price_gbp_per_kwh=prices[t].price_gbp_per_kwh,
                carbon_g_per_kwh=carbon[t].intensity_g_per_kwh,
                load_kwh=asset.load[t],
                solar_kwh=asset.solar[t],
                charge_kw=round(asset.charge[t].value() or 0.0, 4),
                discharge_kw=round(asset.discharge[t].value() or 0.0, 4),
                grid_import_kwh=round(gi, 4),
                grid_export_kwh=round(ge, 4),
                soc_kwh=round(asset.soc[t + 1].value() or 0.0, 4),
            )
        )

    base_cost, base_carbon = baseline_cost_and_carbon(
        prices, carbon, asset.load, asset.solar, asset.export_price, asset.battery.allow_export
    )
    return OptimisationResult(
        slots=slots,
        cost_gbp=round(cost, 4),
        carbon_kg=round(carbon_kg, 4),
        baseline_cost_gbp=round(base_cost, 4),
        baseline_carbon_kg=round(base_carbon, 4),
    )


def _availability_mask(ev: EVConstraints | None, n: int) -> list[bool]:
    """Per-slot plug-in mask (all-available unless an EV availability list is given)."""
    if ev is None or ev.availability is None:
        return [True] * n
    if len(ev.availability) != n:
        raise ValueError("ev.availability must be the same length as prices")
    return list(ev.availability)


def _add_ev_departure_target(
    problem: pulp.LpProblem,
    soc: dict[int, pulp.LpVariable],
    battery: BatterySpec,
    ev: EVConstraints | None,
) -> None:
    """Require SoC >= target by the departure slot, so the EV leaves charged enough."""
    if ev is None or ev.departure_index is None or ev.departure_soc_frac is None:
        return
    target_kwh = ev.departure_soc_frac * battery.capacity_kwh
    problem += soc[ev.departure_index] >= target_kwh
