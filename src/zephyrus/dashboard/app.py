"""Streamlit dashboard: live prices, carbon and the optimised battery plan.

Run::

    streamlit run src/zephyrus/dashboard/app.py

This is the recruiter-facing demo — the headline metrics show £ and kg CO2 saved.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from zephyrus.forecast.demand import demand_forecast_slots
from zephyrus.io.carbon import CarbonIntensityClient
from zephyrus.io.octopus import OctopusClient
from zephyrus.optimise.milp import optimise_schedule
from zephyrus.optimise.sensitivity import sweep_savings
from zephyrus.optimise.vpp import optimise_portfolio
from zephyrus.schemas import BatterySpec, EVConstraints
from zephyrus.utils import align_price_carbon

st.set_page_config(page_title="Zephyrus", layout="wide")
st.title("⚡ Zephyrus — flexibility optimiser")
st.caption("Live Octopus Agile prices + National Grid carbon intensity -> optimal battery plan")

with st.sidebar:
    st.header("Asset & region")
    region = st.selectbox("GSP region", list("ABCDEFGHJKLMNP"), index=2)
    capacity = st.slider("Battery capacity (kWh)", 2.0, 20.0, 5.0, 0.5)
    power = st.slider("Max charge / discharge (kW)", 1.0, 7.0, 3.0, 0.5)
    hours = st.slider("Horizon (hours)", 12, 48, 24, 6)
    carbon_weight = st.slider("Carbon value (£/kg CO2)", 0.0, 0.20, 0.0, 0.01)

    st.header("Battery wear & EV mode")
    degradation = st.slider("Degradation cost (£/kWh throughput)", 0.0, 0.20, 0.0, 0.01)
    ev_mode = st.checkbox("EV smart-charging (plug-in window + departure target)")
    if ev_mode:
        plug_from, plug_to = st.slider(
            "Plugged-in window (hours from now)", 0, hours, (0, hours), 1
        )
        departure_hour = st.slider("Departure (hours from now)", 1, hours, hours, 1)
        departure_soc = st.slider("Departure SoC (%)", 0, 100, 80, 5)

    st.header("Virtual power plant")
    fleet_size = st.slider("Fleet size (assets)", 1, 50, 10, 1)
    headroom = st.slider("Network headroom (kW, 0 = none)", 0.0, 200.0, 0.0, 5.0)

start = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
prices = OctopusClient().get_agile_prices(start, start + timedelta(hours=hours), gsp_group=region)
carbon = CarbonIntensityClient().get_forecast_48h(start)
prices, carbon = align_price_carbon(prices, carbon)

if not prices:
    st.warning("No overlapping price/carbon data right now — try again shortly.")
    st.stop()

battery = BatterySpec(capacity_kwh=capacity, max_charge_kw=power, max_discharge_kw=power)
forecast_slots = demand_forecast_slots(len(prices), start)
load = [slot.load_kwh for slot in forecast_slots]

ev = None
if ev_mode:
    n = len(prices)
    availability = [plug_from <= i * 0.5 < plug_to for i in range(n)]
    ev = EVConstraints(
        availability=availability,
        departure_index=min(int(departure_hour * 2), n),
        departure_soc_frac=departure_soc / 100.0,
    )

try:
    result = optimise_schedule(
        prices,
        carbon,
        battery,
        load_kwh=load,
        carbon_weight_gbp_per_kg=carbon_weight,
        degradation_gbp_per_kwh=degradation,
        ev=ev,
    )
except ValueError as exc:
    st.error(f"Infeasible EV plan — relax the plug-in window or departure target.\n\n{exc}")
    st.stop()

throughput_kwh = sum((s.charge_kw + s.discharge_kw) * 0.5 for s in result.slots)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Cost saving vs baseline", f"£{result.cost_saving_gbp:.2f}")
col2.metric("Carbon saving vs baseline", f"{result.carbon_saving_kg:.2f} kg CO2")
col3.metric("Battery throughput", f"{throughput_kwh:.1f} kWh")
if ev_mode:
    reached = result.slots[min(int(departure_hour * 2), len(result.slots)) - 1].soc_kwh
    col4.metric("SoC at departure", f"{reached:.1f} / {departure_soc / 100 * capacity:.1f} kWh")
else:
    col4.metric("Baseline cost", f"£{result.baseline_cost_gbp:.2f}")

frame = pd.DataFrame([slot.model_dump() for slot in result.slots]).set_index("valid_from")
st.subheader("Demand forecast (median + conformal band)")
fc_frame = pd.DataFrame([slot.model_dump() for slot in forecast_slots]).set_index("valid_from")
st.line_chart(fc_frame[["lower_kwh", "load_kwh", "upper_kwh"]])
st.subheader("Price & carbon signals")
st.line_chart(frame[["price_gbp_per_kwh"]])
st.line_chart(frame[["carbon_g_per_kwh"]])
st.subheader("Battery actions & state of charge")
st.bar_chart(frame[["charge_kw", "discharge_kw"]])
st.area_chart(frame[["soc_kwh"]])

st.header("🔌 Virtual power plant")
portfolio = optimise_portfolio(
    prices,
    carbon,
    [BatterySpec(capacity_kwh=capacity, max_charge_kw=power, max_discharge_kw=power)] * fleet_size,
    load_kwh=load,
    carbon_weight_gbp_per_kg=carbon_weight,
    network_headroom_kw=headroom or None,
    peak_bid_gbp_per_kw=5.0,
)
p1, p2, p3, p4 = st.columns(4)
p1.metric("Fleet cost saving", f"£{portfolio.cost_saving_gbp:.2f}")
p2.metric("Coincident peak", f"{portfolio.peak_import_kw:.1f} kW")
p3.metric("Peak shaved vs naive", f"{portfolio.peak_reduction_kw:.1f} kW")
p4.metric("Balancing bid value", f"£{portfolio.bid_revenue_gbp:.2f}")

st.header("📈 Scenario compare — savings vs battery size")
sweep = sweep_savings(
    prices, carbon, capacities_kwh=[2, 4, 6, 8, 10, 14, 20], powers_kw=[power], load_kwh=load
)
fig = px.line(
    sweep,
    x="capacity_kwh",
    y="cost_saving_gbp",
    markers=True,
    labels={"capacity_kwh": "Battery capacity (kWh)", "cost_saving_gbp": "Cost saving (£)"},
)
st.plotly_chart(fig, use_container_width=True)
