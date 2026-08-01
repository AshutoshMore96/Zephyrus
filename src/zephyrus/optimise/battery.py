"""Battery helpers and the naive baseline controller.

The baseline represents 'dumb' operation (meet net load from the grid, no arbitrage).
The optimiser is measured against it to quantify the £ and kg-CO2 saved.
"""

from __future__ import annotations

from ..schemas import BatterySpec, CarbonSlot, HalfHourlyPrice


def usable_bounds_kwh(spec: BatterySpec) -> tuple[float, float]:
    """Return (min, max) usable state-of-charge in kWh."""
    return spec.soc_min_frac * spec.capacity_kwh, spec.soc_max_frac * spec.capacity_kwh


def baseline_cost_and_carbon(
    prices: list[HalfHourlyPrice],
    carbon: list[CarbonSlot],
    load_kwh: list[float],
    solar_kwh: list[float],
    export_prices_gbp: list[float] | None = None,
    allow_export: bool = False,
) -> tuple[float, float]:
    """Cost (£) and carbon (kg) with no battery optimisation.

    Net demand ``max(load - solar, 0)`` is imported slot-by-slot at that slot's price
    and carbon intensity. Surplus solar ``max(solar - load, 0)`` is spilled unless
    ``allow_export`` — then it is credited at the Outgoing (export) price, so savings are
    measured against a fair "solar home without a battery" baseline.
    """
    cost = 0.0
    carbon_kg = 0.0
    for i, (price, slot, load, solar) in enumerate(
        zip(prices, carbon, load_kwh, solar_kwh, strict=True)
    ):
        net = max(load - solar, 0.0)
        surplus = max(solar - load, 0.0)
        cost += net * price.price_gbp_per_kwh
        carbon_kg += net * slot.intensity_g_per_kwh / 1000.0
        if allow_export and surplus > 0.0:
            export_price = (
                export_prices_gbp[i] if export_prices_gbp is not None else price.price_gbp_per_kwh
            )
            cost -= surplus * export_price
    return cost, carbon_kg
