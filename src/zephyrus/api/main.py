"""FastAPI service exposing the forecaster and optimiser over live data.

Run::

    uvicorn zephyrus.api.main:app --reload

Endpoints:
  * ``GET  /health``   — liveness probe.
  * ``POST /optimise`` — battery schedule + savings for one asset (forecast-driven load).
  * ``POST /forecast`` — probabilistic demand forecast (median + conformal band).

The ``/vpp`` endpoint and API-key auth are added in later milestones (M5/M6).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from ..forecast.demand import demand_forecast_slots
from ..io.carbon import CarbonIntensityClient
from ..io.octopus import OctopusClient
from ..io.store import save_result
from ..optimise.milp import optimise_schedule
from ..optimise.vpp import optimise_portfolio
from ..schemas import BatterySpec, LoadForecastSlot, OptimisationResult, PortfolioResult
from ..utils import align_price_carbon
from .security import rate_limited, require_api_key

_GUARDS = [Depends(require_api_key), Depends(rate_limited)]

app = FastAPI(
    title="Zephyrus",
    version="0.1.0",
    summary="Forecast-driven flexibility optimiser for home batteries & EVs.",
)


class OptimiseRequest(BaseModel):
    battery: BatterySpec = BatterySpec()
    gsp_group: str | None = None
    hours: int = 24
    carbon_weight_gbp_per_kg: float | None = None
    persist_as: str | None = None  # if set, persist the schedule under this name


class ForecastRequest(BaseModel):
    hours: int = 24
    history_weeks: int = 6


class VppRequest(BaseModel):
    assets: list[BatterySpec]
    gsp_group: str | None = None
    hours: int = 24
    network_headroom_kw: float | None = None
    peak_bid_gbp_per_kw: float = 0.0


def _now_hh() -> datetime:
    return datetime.now(UTC).replace(minute=0, second=0, microsecond=0)


def _fetch_aligned(hours: int, gsp_group: str | None):
    """Fetch + align live Agile prices and carbon intensity, raising HTTP errors."""
    start = _now_hh()
    try:
        prices = OctopusClient().get_agile_prices(
            start, start + timedelta(hours=hours), gsp_group=gsp_group
        )
        carbon = CarbonIntensityClient().get_forecast_48h(start)
    except Exception as exc:  # noqa: BLE001 - surface upstream failures as 502
        raise HTTPException(status_code=502, detail=f"Upstream data error: {exc}") from exc
    prices, carbon = align_price_carbon(prices, carbon)
    if not prices:
        raise HTTPException(status_code=404, detail="No overlapping price/carbon slots")
    return start, prices, carbon


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/optimise", response_model=OptimisationResult, dependencies=_GUARDS)
def optimise(req: OptimiseRequest) -> OptimisationResult:
    """Optimise a battery against live Agile prices + carbon, using the demand forecast."""
    start, prices, carbon = _fetch_aligned(req.hours, req.gsp_group)
    load = [s.load_kwh for s in demand_forecast_slots(len(prices), start)]
    result = optimise_schedule(
        prices,
        carbon,
        req.battery,
        load_kwh=load,
        carbon_weight_gbp_per_kg=req.carbon_weight_gbp_per_kg,
    )
    if req.persist_as:
        save_result(req.persist_as, result)
    return result


@app.post("/forecast", response_model=list[LoadForecastSlot], dependencies=_GUARDS)
def forecast(req: ForecastRequest) -> list[LoadForecastSlot]:
    """Return a probabilistic half-hourly demand forecast (median + conformal band)."""
    return demand_forecast_slots(req.hours * 2, _now_hh(), history_weeks=req.history_weeks)


@app.post("/vpp", response_model=PortfolioResult, dependencies=_GUARDS)
def vpp(req: VppRequest) -> PortfolioResult:
    """Co-optimise a portfolio of assets and aggregate the flexibility (virtual power plant)."""
    if not req.assets:
        raise HTTPException(status_code=422, detail="Provide at least one asset")
    start, prices, carbon = _fetch_aligned(req.hours, req.gsp_group)
    load = [s.load_kwh for s in demand_forecast_slots(len(prices), start)]
    return optimise_portfolio(
        prices,
        carbon,
        req.assets,
        load_kwh=load,
        network_headroom_kw=req.network_headroom_kw,
        peak_bid_gbp_per_kw=req.peak_bid_gbp_per_kw,
    )
