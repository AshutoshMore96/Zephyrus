"""Typed data contracts shared across ingestion, modelling and optimisation.

Keeping these in one place means every module speaks the same units: price is always
pence/kWh incl. VAT (with a £ helper), carbon is always gCO2/kWh, energy is always kWh,
and power is always kW.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, computed_field


class HalfHourlyPrice(BaseModel):
    """A single half-hourly unit rate (e.g. Octopus Agile import)."""

    valid_from: datetime
    valid_to: datetime
    price_p_per_kwh: float = Field(..., description="Unit rate, pence/kWh incl. VAT")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def price_gbp_per_kwh(self) -> float:
        return self.price_p_per_kwh / 100.0


class CarbonSlot(BaseModel):
    """A half-hourly carbon-intensity slot from the National Grid API."""

    valid_from: datetime
    valid_to: datetime
    intensity_g_per_kwh: float = Field(..., description="Forecast gCO2/kWh")
    index: str | None = None  # 'very low' | 'low' | 'moderate' | 'high' | 'very high'


class LoadForecastSlot(BaseModel):
    """A probabilistic half-hourly household-load forecast (project #1).

    ``load_kwh`` is the median (P50) point forecast — the value fed to the optimiser's
    ``load_kwh`` — while ``lower_kwh``/``upper_kwh`` are the conformal prediction
    interval. Bounds are non-crossing: ``lower_kwh <= load_kwh <= upper_kwh``.
    """

    valid_from: datetime
    load_kwh: float = Field(..., ge=0.0, description="Median (P50) load forecast, kWh")
    lower_kwh: float = Field(..., ge=0.0, description="Lower conformal bound, kWh")
    upper_kwh: float = Field(..., ge=0.0, description="Upper conformal bound, kWh")


class PriceForecastSlot(BaseModel):
    """A probabilistic half-hourly price forecast (project #4).

    ``price_p_per_kwh`` is the median; the band is the conformal interval. Prices may be
    negative (Agile/imbalance can go below zero), so no non-negativity is imposed.
    """

    valid_from: datetime
    price_p_per_kwh: float = Field(..., description="Median forecast unit rate, pence/kWh")
    lower_p_per_kwh: float = Field(..., description="Lower conformal bound, pence/kWh")
    upper_p_per_kwh: float = Field(..., description="Upper conformal bound, pence/kWh")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def price_gbp_per_kwh(self) -> float:
        return self.price_p_per_kwh / 100.0


class SystemPrice(BaseModel):
    """A half-hourly system (imbalance) price from Elexon BMRS (project #4)."""

    valid_from: datetime
    settlement_period: int = Field(..., ge=1, le=50)
    price_p_per_kwh: float = Field(..., description="System price, pence/kWh (from £/MWh)")
    net_imbalance_volume_mwh: float | None = Field(default=None, description="NIV, MWh")


class FuelGenerationSlot(BaseModel):
    """Half-hourly generation outturn for one fuel type (FUELHH) — a price feature."""

    valid_from: datetime
    fuel_type: str
    generation_mw: float


class ArbitrageBacktestReport(BaseModel):
    """Battery-arbitrage backtest: forecast-driven strategy vs persistence, at actuals."""

    n_days: int
    model_pnl_gbp: float
    persistence_pnl_gbp: float
    perfect_foresight_pnl_gbp: float
    model_sharpe: float
    hit_rate: float = Field(..., ge=0.0, le=1.0)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pnl_uplift_gbp(self) -> float:
        """Extra P&L from forecasting vs naive persistence."""
        return round(self.model_pnl_gbp - self.persistence_pnl_gbp, 4)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def capture_rate(self) -> float:
        """Fraction of the perfect-foresight (oracle) P&L the strategy captured."""
        if self.perfect_foresight_pnl_gbp <= 0:
            return 0.0
        return round(self.model_pnl_gbp / self.perfect_foresight_pnl_gbp, 4)


class DemandBacktestReport(BaseModel):
    """Rolling-origin backtest scores for the demand forecaster vs seasonal-naive.

    MAPE is a percentage (lower is better); pinball is the mean quantile loss over the
    forecaster's quantiles (kWh). ``coverage`` is the empirical fraction of actuals that
    fell inside the conformal interval — it should sit near the interval's nominal level.
    """

    horizon: int
    n_folds: int
    model_backend: str
    quantiles: list[float]
    nominal_coverage: float
    model_mape: float
    baseline_mape: float
    model_pinball: float
    baseline_pinball: float
    coverage: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mape_improvement_pct(self) -> float:
        """Relative MAPE reduction vs the seasonal-naive baseline (%)."""
        if self.baseline_mape == 0:
            return 0.0
        return 100.0 * (self.baseline_mape - self.model_mape) / self.baseline_mape

    @computed_field  # type: ignore[prop-decorator]
    @property
    def beats_baseline(self) -> bool:
        return self.model_mape <= self.baseline_mape


class DriftReport(BaseModel):
    """Result of a drift / performance-degradation check (M7 monitoring)."""

    metric: str
    value: float
    threshold: float
    detail: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def alert(self) -> bool:
        """True when the observed value breaches the threshold (drift/regression)."""
        return self.value > self.threshold


class ModelVersion(BaseModel):
    """A registered model version and the metrics it was scored on (M7 registry)."""

    name: str
    version: int
    primary_metric: str
    metrics: dict[str, float]
    params: dict[str, str] = Field(default_factory=dict)
    created_at: datetime


class WeatherSlot(BaseModel):
    """A half-hourly weather observation/forecast — exogenous features for the models.

    Temperature drives heating/cooling demand; shortwave radiation drives PV output.
    Sourced from Open-Meteo (free, no key) and resampled to the half-hourly grid.
    """

    valid_from: datetime
    temperature_c: float = Field(..., description="Air temperature, °C")
    shortwave_radiation_w_m2: float = Field(
        default=0.0, ge=0.0, description="Global horizontal irradiance, W/m²"
    )
    cloud_cover_pct: float = Field(default=0.0, ge=0.0, le=100.0, description="Cloud cover, %")


class BatterySpec(BaseModel):
    """Physical + operational limits of a home battery or EV."""

    capacity_kwh: float = 5.0
    max_charge_kw: float = 3.0
    max_discharge_kw: float = 3.0
    round_trip_efficiency: float = Field(default=0.90, gt=0.0, le=1.0)
    soc_min_frac: float = Field(default=0.10, ge=0.0, le=1.0)
    soc_max_frac: float = Field(default=1.00, ge=0.0, le=1.0)
    initial_soc_frac: float = Field(default=0.50, ge=0.0, le=1.0)
    allow_export: bool = False  # True models an Outgoing/export tariff

    @property
    def one_way_efficiency(self) -> float:
        """Efficiency applied on each leg (charge and discharge)."""
        return self.round_trip_efficiency**0.5


class EVConstraints(BaseModel):
    """Smart-charging constraints for an EV (project #2, M2).

    ``availability`` masks the slots where the vehicle is plugged in and can (dis)charge
    (``True`` = plugged in); when unplugged the battery is idle and its state is carried.
    ``departure_soc_frac`` is the minimum state-of-charge required by ``departure_index``
    (the slot the driver leaves), so the car is charged in time regardless of price.
    """

    availability: list[bool] | None = None
    departure_index: int | None = Field(default=None, ge=0)
    departure_soc_frac: float | None = Field(default=None, ge=0.0, le=1.0)


class ScheduleSlot(BaseModel):
    """The optimiser's decision + resulting state for one half-hour."""

    valid_from: datetime
    price_gbp_per_kwh: float
    carbon_g_per_kwh: float
    load_kwh: float
    solar_kwh: float
    charge_kw: float
    discharge_kw: float
    grid_import_kwh: float
    grid_export_kwh: float
    soc_kwh: float


class OptimisationResult(BaseModel):
    """Full schedule plus cost/carbon achieved vs the naive baseline."""

    slots: list[ScheduleSlot]
    cost_gbp: float
    carbon_kg: float
    baseline_cost_gbp: float
    baseline_carbon_kg: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cost_saving_gbp(self) -> float:
        return self.baseline_cost_gbp - self.cost_gbp

    @computed_field  # type: ignore[prop-decorator]
    @property
    def carbon_saving_kg(self) -> float:
        return self.baseline_carbon_kg - self.carbon_kg


class PortfolioResult(BaseModel):
    """Aggregated KPIs for a co-optimised portfolio of assets (a virtual power plant).

    ``peak_import_kw`` is the coincident (simultaneous) grid import across all assets —
    the number a network operator cares about — and ``flexibility_kwh`` is the total
    energy shifted. ``bid_revenue_gbp`` values the delivered turn-down against a balancing
    signal (a stand-in for DFS / balancing-market settlement).
    """

    n_assets: int
    cost_saving_gbp: float
    carbon_saving_kg: float
    peak_import_kw: float
    baseline_peak_import_kw: float
    flexibility_kwh: float
    bid_revenue_gbp: float = 0.0
    per_asset: list[OptimisationResult] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def peak_reduction_kw(self) -> float:
        """Coincident-peak import shaved vs the unoptimised portfolio."""
        return round(self.baseline_peak_import_kw - self.peak_import_kw, 4)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_value_gbp(self) -> float:
        return round(self.cost_saving_gbp + self.bid_revenue_gbp, 4)
