"""Zephyrus command line — a thin, typed wrapper around the library.

Examples::

    zephyrus prices --hours 24 --region C
    zephyrus demo   --hours 24 --region C --capacity 5
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import typer

from .forecast.demand import (
    DemandForecaster,
    backtest_demand,
    demand_forecast_slots,
    synthetic_household_load,
    synthetic_metered_history,
)
from .io.carbon import CarbonIntensityClient
from .io.octopus import OctopusClient
from .io.store import SnapshotStore, models_to_frame
from .io.weather import OpenMeteoClient
from .optimise.milp import optimise_schedule
from .quality import CARBON_RANGE, PRICE_RANGE_P, check_halfhourly_frame
from .schemas import BatterySpec, HalfHourlyPrice
from .utils import align_price_carbon

app = typer.Typer(add_completion=False, help="Forecast-driven flexibility optimiser.")


def _now_utc_hh() -> datetime:
    return datetime.now(UTC).replace(minute=0, second=0, microsecond=0)


@app.command()
def prices(hours: int = 24, region: str = "C") -> None:
    """Print live Octopus Agile prices for a GSP region."""
    start = _now_utc_hh()
    rows = OctopusClient().get_agile_prices(start, start + timedelta(hours=hours), gsp_group=region)
    for row in rows:
        typer.echo(f"{row.valid_from:%Y-%m-%d %H:%M}  {row.price_p_per_kwh:6.2f} p/kWh")


@app.command()
def demo(
    hours: int = 24,
    region: str = "C",
    capacity: float = 5.0,
    price_forecast: bool = False,
) -> None:
    """Run the full pipeline on live data and print the savings.

    With ``--price-forecast`` the optimiser decides against *forecast* prices (M4) and the
    plan is then settled at *actual* prices — the honest day-ahead arbitrage view, with
    the gap to the actual-optimal plan reported as the 'forecast tax'.
    """
    start = _now_utc_hh()
    prices_ = OctopusClient().get_agile_prices(
        start, start + timedelta(hours=hours), gsp_group=region
    )
    carbon = CarbonIntensityClient().get_forecast_48h(start)
    prices_, carbon = align_price_carbon(prices_, carbon)
    if not prices_:
        typer.secho("No overlapping price/carbon data yet — try again shortly.", fg="yellow")
        raise typer.Exit(code=1)

    load = synthetic_household_load(len(prices_))
    battery = BatterySpec(capacity_kwh=capacity)
    if price_forecast:
        _demo_price_forecast(prices_, carbon, battery, load)
        return

    result = optimise_schedule(prices_, carbon, battery, load_kwh=load)
    typer.secho(
        f"Optimised £{result.cost_gbp:.2f} vs baseline £{result.baseline_cost_gbp:.2f}  "
        f"-> saved £{result.cost_saving_gbp:.2f} "
        f"and {result.carbon_saving_kg:.2f} kg CO2 over {hours}h.",
        fg="green",
    )


def _demo_price_forecast(prices_, carbon, battery, load) -> None:
    """Optimise against forecast prices, settle at actual — the day-ahead arbitrage view."""
    from .forecast.price import (
        PriceForecaster,
        realised_cost_gbp,
        synthetic_price_history,
    )

    start = prices_[0].valid_from
    history = synthetic_price_history(start - timedelta(weeks=6), days=42)
    forecast_slots = PriceForecaster().fit(history).predict(len(prices_))
    decision_prices = [
        HalfHourlyPrice(
            valid_from=p.valid_from, valid_to=p.valid_to, price_p_per_kwh=f.price_p_per_kwh
        )
        for p, f in zip(prices_, forecast_slots, strict=True)
    ]

    plan_fc = optimise_schedule(decision_prices, carbon, battery, load_kwh=load)
    plan_oracle = optimise_schedule(prices_, carbon, battery, load_kwh=load)
    realised = realised_cost_gbp(plan_fc, prices_)
    forecast_tax = round(realised - plan_oracle.cost_gbp, 4)

    typer.secho(f"Forecast-driven plan settled at actual prices: £{realised:.2f}", fg="cyan")
    typer.secho(
        f"vs actual-optimal £{plan_oracle.cost_gbp:.2f} "
        f"(do-nothing baseline £{plan_oracle.baseline_cost_gbp:.2f})  "
        f"-> forecast tax £{forecast_tax:.2f}.",
        fg="green",
    )


@app.command()
def forecast(hours: int = 24, region: str = "C", capacity: float = 5.0, weeks: int = 6) -> None:
    """Forecast demand (M3), feed it into the optimiser, and print the savings.

    Trains :class:`DemandForecaster` on a synthetic metered history (stand-in for real
    smart-meter data until M1) and uses its median forecast as the optimiser's load —
    the end-to-end 'forecast-driven optimisation' path.
    """
    start = _now_utc_hh()
    prices_ = OctopusClient().get_agile_prices(
        start, start + timedelta(hours=hours), gsp_group=region
    )
    carbon = CarbonIntensityClient().get_forecast_48h(start)
    prices_, carbon = align_price_carbon(prices_, carbon)
    if not prices_:
        typer.secho("No overlapping price/carbon data yet — try again shortly.", fg="yellow")
        raise typer.Exit(code=1)

    history = synthetic_metered_history(start - timedelta(weeks=weeks), days=weeks * 7)
    forecaster = DemandForecaster().fit(history)
    slots = forecaster.predict(len(prices_))
    load = [slot.load_kwh for slot in slots]

    typer.secho(f"Demand forecast ({forecaster.backend} backend), first 6 slots:", fg="cyan")
    for slot in slots[:6]:
        typer.echo(
            f"  {slot.valid_from:%Y-%m-%d %H:%M}  "
            f"{slot.load_kwh:5.3f} kWh  [{slot.lower_kwh:5.3f}, {slot.upper_kwh:5.3f}]"
        )

    result = optimise_schedule(prices_, carbon, BatterySpec(capacity_kwh=capacity), load_kwh=load)
    typer.secho(
        f"Optimised £{result.cost_gbp:.2f} vs baseline £{result.baseline_cost_gbp:.2f}  "
        f"-> saved £{result.cost_saving_gbp:.2f} "
        f"and {result.carbon_saving_kg:.2f} kg CO2 over {hours}h.",
        fg="green",
    )


@app.command()
def backtest(weeks: int = 6, horizon: int = 48, folds: int = 4, mlflow: bool = False) -> None:
    """Backtest the demand forecaster vs seasonal-naive (MAPE + pinball); optionally MLflow."""
    history = synthetic_metered_history(_now_utc_hh() - timedelta(weeks=weeks), days=weeks * 7)
    report = backtest_demand(history, horizon=horizon, n_folds=folds, log_mlflow=mlflow)
    color = "green" if report.beats_baseline else "red"
    typer.secho(
        f"MAPE: model {report.model_mape:.2f}% vs naive {report.baseline_mape:.2f}% "
        f"({report.mape_improvement_pct:+.1f}%)",
        fg=color,
    )
    typer.echo(
        f"Pinball: model {report.model_pinball:.4f} vs naive {report.baseline_pinball:.4f}  |  "
        f"coverage {report.coverage:.0%} (nominal {report.nominal_coverage:.0%})  "
        f"over {report.n_folds} folds [{report.model_backend}]"
    )
    if mlflow:
        typer.secho("Logged run to MLflow.", fg="cyan")


@app.command()
def vpp(
    hours: int = 24,
    region: str = "C",
    assets: int = 5,
    capacity: float = 5.0,
    headroom_kw: float = 0.0,
) -> None:
    """Co-optimise a fleet of batteries as a virtual power plant (M5)."""
    from .optimise.vpp import optimise_portfolio

    start = _now_utc_hh()
    prices_ = OctopusClient().get_agile_prices(
        start, start + timedelta(hours=hours), gsp_group=region
    )
    carbon = CarbonIntensityClient().get_forecast_48h(start)
    prices_, carbon = align_price_carbon(prices_, carbon)
    if not prices_:
        typer.secho("No overlapping price/carbon data yet — try again shortly.", fg="yellow")
        raise typer.Exit(code=1)

    load = [s.load_kwh for s in demand_forecast_slots(len(prices_), start)]
    specs = [BatterySpec(capacity_kwh=capacity) for _ in range(assets)]
    result = optimise_portfolio(
        prices_,
        carbon,
        specs,
        load_kwh=load,
        network_headroom_kw=headroom_kw or None,
        peak_bid_gbp_per_kw=5.0,
    )
    typer.secho(
        f"VPP of {result.n_assets} assets: saved £{result.cost_saving_gbp:.2f}, "
        f"{result.carbon_saving_kg:.2f} kg CO2; shifted {result.flexibility_kwh:.1f} kWh.",
        fg="green",
    )
    typer.echo(
        f"Coincident peak {result.peak_import_kw:.1f} kW "
        f"(-{result.peak_reduction_kw:.1f} kW vs naive), bid £{result.bid_revenue_gbp:.2f}."
    )


@app.command()
def arbitrage(weeks: int = 6, horizon: int = 48, days: int = 5) -> None:
    """Backtest a day-ahead price-arbitrage strategy vs persistence (M4)."""
    from .forecast.price import backtest_arbitrage, synthetic_price_history

    history = synthetic_price_history(_now_utc_hh() - timedelta(weeks=weeks), days=weeks * 7)
    report = backtest_arbitrage(history, horizon=horizon, n_days=days)
    color = "green" if report.pnl_uplift_gbp >= 0 else "red"
    typer.secho(
        f"P&L: model £{report.model_pnl_gbp:.2f} vs persistence £{report.persistence_pnl_gbp:.2f} "
        f"(uplift £{report.pnl_uplift_gbp:.2f})",
        fg=color,
    )
    typer.echo(
        f"Capture {report.capture_rate:.0%} of oracle £{report.perfect_foresight_pnl_gbp:.2f}  |  "
        f"Sharpe {report.model_sharpe:.2f}, hit-rate {report.hit_rate:.0%} "
        f"over {report.n_days} days"
    )


@app.command()
def snapshot(hours: int = 48, region: str = "C", weather: bool = True) -> None:
    """Fetch live data, run data-quality checks, and cache it to parquet (M1)."""
    start = _now_utc_hh()
    store = SnapshotStore()

    prices = OctopusClient().get_agile_prices(
        start, start + timedelta(hours=hours), gsp_group=region
    )
    carbon = CarbonIntensityClient().get_forecast_48h(start)
    price_frame = models_to_frame(prices)
    carbon_frame = models_to_frame(carbon)
    check_halfhourly_frame(price_frame, ranges={"price_p_per_kwh": PRICE_RANGE_P})
    check_halfhourly_frame(carbon_frame, ranges={"intensity_g_per_kwh": CARBON_RANGE})
    store.save("agile_prices", price_frame)
    store.save("carbon_forecast", carbon_frame)
    typer.secho(f"Saved {len(prices)} price + {len(carbon)} carbon slots (validated).", fg="green")

    if weather:
        weather_slots = OpenMeteoClient().get_forecast(start, hours=hours)
        store.save("weather", models_to_frame(weather_slots))
        typer.secho(f"Saved {len(weather_slots)} weather slots.", fg="green")

    typer.echo(f"Snapshots in {store.base_dir}")


if __name__ == "__main__":
    app()
