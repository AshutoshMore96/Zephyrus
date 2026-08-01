"""End-to-end demo: fetch live data, optimise, and print a slot-by-slot table.

    python scripts/run_demo.py

Prints the optimised schedule and the headline £ / kg-CO2 savings. Zero config and no
API keys required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from zephyrus.forecast.demand import synthetic_household_load
from zephyrus.io.carbon import CarbonIntensityClient
from zephyrus.io.octopus import OctopusClient
from zephyrus.optimise.milp import optimise_schedule
from zephyrus.schemas import BatterySpec
from zephyrus.utils import align_price_carbon


def main() -> None:
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    prices = OctopusClient().get_agile_prices(start, start + timedelta(hours=24), gsp_group="C")
    carbon = CarbonIntensityClient().get_forecast_48h(start)
    prices, carbon = align_price_carbon(prices, carbon)
    if not prices:
        print("No overlapping price/carbon data yet — try again shortly.")
        return

    load = synthetic_household_load(len(prices))
    result = optimise_schedule(prices, carbon, BatterySpec(), load_kwh=load)

    frame = pd.DataFrame([slot.model_dump() for slot in result.slots])
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(frame[["valid_from", "price_gbp_per_kwh", "carbon_g_per_kwh",
                     "charge_kw", "discharge_kw", "grid_import_kwh", "soc_kwh"]].to_string(index=False))
    print(
        f"\nOptimised £{result.cost_gbp:.2f} vs baseline £{result.baseline_cost_gbp:.2f}  "
        f"-> saved £{result.cost_saving_gbp:.2f} and {result.carbon_saving_kg:.2f} kg CO2."
    )


if __name__ == "__main__":
    main()
