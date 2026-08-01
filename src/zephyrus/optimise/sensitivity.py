"""Savings sensitivity to battery sizing (project #2, M2).

Sweeps battery capacity and power, re-solving the MILP for each, and returns a tidy
frame of savings — the data behind the "how big a battery is worth it?" question. A thin,
importable core so a notebook or the dashboard can chart it without duplicating logic.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ..logging import get_logger
from ..schemas import BatterySpec, CarbonSlot, HalfHourlyPrice
from .milp import optimise_schedule

logger = get_logger(__name__)


def sweep_savings(
    prices: list[HalfHourlyPrice],
    carbon: list[CarbonSlot],
    capacities_kwh: Sequence[float],
    powers_kw: Sequence[float],
    load_kwh: list[float] | None = None,
    solar_kwh: list[float] | None = None,
    carbon_weight_gbp_per_kg: float | None = None,
) -> pd.DataFrame:
    """Return a tidy frame of savings over the ``capacity × power`` grid.

    Columns: ``capacity_kwh``, ``power_kw``, ``cost_saving_gbp``, ``carbon_saving_kg``,
    ``cost_saving_per_kwh`` (£ saved per kWh of capacity — the marginal-value signal).
    """
    rows: list[dict[str, float]] = []
    for capacity in capacities_kwh:
        for power in powers_kw:
            battery = BatterySpec(
                capacity_kwh=capacity, max_charge_kw=power, max_discharge_kw=power
            )
            result = optimise_schedule(
                prices,
                carbon,
                battery,
                load_kwh=load_kwh,
                solar_kwh=solar_kwh,
                carbon_weight_gbp_per_kg=carbon_weight_gbp_per_kg,
            )
            rows.append(
                {
                    "capacity_kwh": float(capacity),
                    "power_kw": float(power),
                    "cost_saving_gbp": result.cost_saving_gbp,
                    "carbon_saving_kg": result.carbon_saving_kg,
                    "cost_saving_per_kwh": (
                        round(result.cost_saving_gbp / capacity, 4) if capacity else 0.0
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    logger.info("Sensitivity sweep: %d configurations", len(frame))
    return frame
