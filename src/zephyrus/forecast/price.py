"""Imbalance / wholesale price forecasting + arbitrage backtest (project #4).

* ``persistence_price_forecast`` — the transparent baseline (tomorrow = latest prices).
* :class:`PriceForecaster` — the shared conformal engine applied to prices (spike-aware:
  the upper conformal quantile widens around volatile periods). Prices may go negative,
  so the point path is *not* clipped.
* :func:`backtest_arbitrage` — values a day-ahead battery strategy: optimise against the
  *forecast*, settle at *actual* prices, and compare P&L / Sharpe / hit-rate to a
  persistence-driven strategy and to a perfect-foresight oracle.

The optimiser consumes forecast prices in place of actuals when backtesting a strategy;
:func:`to_price_models` adapts a forecast into the ``HalfHourlyPrice`` list it expects.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd

from ..logging import get_logger
from ..schemas import (
    ArbitrageBacktestReport,
    BatterySpec,
    CarbonSlot,
    HalfHourlyPrice,
    OptimisationResult,
    PriceForecastSlot,
)
from ._core import DEFAULT_LAGS, DEFAULT_QUANTILES, SLOT, SLOTS_PER_DAY, ConformalForecaster
from .demand import seasonal_naive_forecast

logger = get_logger(__name__)


def persistence_price_forecast(prices: list[HalfHourlyPrice]) -> list[HalfHourlyPrice]:
    """Trivial baseline forecast: repeat the most recent known prices."""
    return list(prices)


class PriceForecaster(ConformalForecaster):
    """Probabilistic half-hourly price forecaster (prices may be negative — no clipping)."""

    def __init__(
        self,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        lags: Sequence[int] = DEFAULT_LAGS,
        backend: str = "auto",
        calibration_frac: float = 0.2,
        random_state: int = 42,
    ) -> None:
        super().__init__(
            quantiles=quantiles,
            lags=lags,
            backend=backend,
            calibration_frac=calibration_frac,
            random_state=random_state,
            clip_min=None,
        )

    def predict(self, horizon: int = SLOTS_PER_DAY) -> list[PriceForecastSlot]:
        """Forecast ``horizon`` half-hours ahead as probabilistic price slots (p/kWh)."""
        times, preds = self.predict_quantiles(horizon)
        median, lower, upper = preds[0.5], preds[self.quantiles[0]], preds[self.quantiles[-1]]
        return [
            PriceForecastSlot(
                valid_from=times[h].to_pydatetime(),
                price_p_per_kwh=round(float(median[h]), 4),
                lower_p_per_kwh=round(float(min(lower[h], median[h])), 4),
                upper_p_per_kwh=round(float(max(upper[h], median[h])), 4),
            )
            for h in range(horizon)
        ]

    def predict_prices(self, horizon: int = SLOTS_PER_DAY) -> list[HalfHourlyPrice]:
        """Median forecast as ``HalfHourlyPrice`` objects — drop-in for the optimiser."""
        return to_price_models(self.predict(horizon))


def to_price_models(slots: list[PriceForecastSlot]) -> list[HalfHourlyPrice]:
    """Adapt price-forecast slots into the ``HalfHourlyPrice`` list the optimiser expects."""
    return [
        HalfHourlyPrice(
            valid_from=s.valid_from,
            valid_to=s.valid_from + SLOT,
            price_p_per_kwh=s.price_p_per_kwh,
        )
        for s in slots
    ]


def synthetic_price_history(
    start: datetime,
    days: int = 42,
    seed: int = 5,
) -> pd.Series:
    """Weekly-seasonal, spiky half-hourly price series (p/kWh) for demos/backtests.

    Cheap overnight, an evening peak, weekends flatter — with occasional deterministic
    spikes. Structure that seasonal-naive (repeat yesterday) misses across weekend
    boundaries, so a calendar/lag model can beat it.
    """
    from ._core import ensure_utc

    base_weekday = np.full(SLOTS_PER_DAY, 12.0)
    base_weekday[:14] = 7.0  # cheap overnight
    base_weekday[34:42] = 30.0  # evening peak
    base_weekend = np.full(SLOTS_PER_DAY, 11.0)
    base_weekend[:16] = 6.5
    base_weekend[36:44] = 22.0  # later, lower weekend peak

    start = ensure_utc(start)
    index = pd.date_range(start=start, periods=days * SLOTS_PER_DAY, freq=SLOT)
    rng = np.random.default_rng(seed)
    values = np.empty(len(index))
    for i, ts in enumerate(index):
        profile = base_weekend if ts.dayofweek >= 5 else base_weekday
        values[i] = profile[i % SLOTS_PER_DAY]
    values = values + rng.normal(0, 0.8, len(index))
    # A few evening spikes on weekdays to exercise the spike-aware upper quantile.
    for day in range(days):
        if index[day * SLOTS_PER_DAY].dayofweek < 5 and day % 3 == 0:
            values[day * SLOTS_PER_DAY + 37] += 40.0
    return pd.Series(values.round(3), index=index, name="price_p_per_kwh")


def _settle_cost_gbp(
    grid_import_kwh: np.ndarray, grid_export_kwh: np.ndarray, actual_p_per_kwh: np.ndarray
) -> float:
    """£ cost of a fixed import/export schedule settled at actual prices."""
    gbp = actual_p_per_kwh / 100.0
    return float(np.sum(grid_import_kwh * gbp - grid_export_kwh * gbp))


def realised_cost_gbp(result: OptimisationResult, actual_prices: list[HalfHourlyPrice]) -> float:
    """£ a schedule actually costs when settled at ``actual_prices``.

    A schedule chosen on *forecast* prices still gets billed at *actual* prices — this is
    the honest day-ahead realisation, and the gap to the actual-optimal plan is the
    "forecast tax".
    """
    gi = np.array([s.grid_import_kwh for s in result.slots])
    ge = np.array([s.grid_export_kwh for s in result.slots])
    actual_p = np.array([p.price_p_per_kwh for p in actual_prices])
    if not (len(gi) == len(actual_p)):
        raise ValueError("schedule and actual_prices must be the same length")
    return round(_settle_cost_gbp(gi, ge, actual_p), 4)


def backtest_arbitrage(
    price_series: pd.Series,
    battery: BatterySpec | None = None,
    horizon: int = SLOTS_PER_DAY,
    n_days: int = 5,
    backend: str = "auto",
) -> ArbitrageBacktestReport:
    """Value a day-ahead arbitrage strategy: optimise on forecast, settle on actuals.

    Each day the battery (export-enabled, no load) is scheduled against three price views
    — the :class:`PriceForecaster`, persistence, and perfect foresight — then every
    schedule is settled at that day's *actual* prices. Daily P&L is savings vs doing
    nothing (which is £0 with no load), i.e. arbitrage profit.
    """
    from ..optimise.milp import optimise_schedule  # local import avoids a cycle

    battery = battery or BatterySpec(
        capacity_kwh=5, max_charge_kw=3, max_discharge_kw=3, allow_export=True
    )
    series = ConformalForecaster._validate_series(price_series)
    values = series.to_numpy(dtype=float)

    origins = [len(values) - horizon - k * horizon for k in range(n_days)][::-1]
    origins = [o for o in origins if o - 2 * 7 * SLOTS_PER_DAY > 0]
    if not origins:
        raise ValueError("not enough history for the requested horizon/days")

    model_daily: list[float] = []
    persistence_daily: list[float] = []
    oracle_daily: list[float] = []

    for origin in origins:
        train = series.iloc[:origin]
        actual = values[origin : origin + horizon]
        actual_times = series.index[origin : origin + horizon]

        fc = PriceForecaster(backend=backend).fit(train)
        forecast_prices = fc.predict_prices(horizon)
        persistence = seasonal_naive_forecast(train.to_numpy().tolist(), horizon)
        persistence_prices = _prices_from_values(persistence, actual_times)
        actual_prices = _prices_from_values(actual.tolist(), actual_times)
        carbon = _flat_carbon(actual_times)

        model_daily.append(
            _strategy_pnl(optimise_schedule, forecast_prices, actual, carbon, battery)
        )
        persistence_daily.append(
            _strategy_pnl(optimise_schedule, persistence_prices, actual, carbon, battery)
        )
        oracle_daily.append(
            _strategy_pnl(optimise_schedule, actual_prices, actual, carbon, battery)
        )

    model = np.asarray(model_daily)
    persistence_arr = np.asarray(persistence_daily)
    sharpe = float(model.mean() / model.std()) if model.std() > 1e-9 else 0.0
    hit_rate = float(np.mean(model >= persistence_arr - 1e-9))

    report = ArbitrageBacktestReport(
        n_days=len(origins),
        model_pnl_gbp=round(float(model.sum()), 4),
        persistence_pnl_gbp=round(float(persistence_arr.sum()), 4),
        perfect_foresight_pnl_gbp=round(float(np.sum(oracle_daily)), 4),
        model_sharpe=round(sharpe, 4),
        hit_rate=round(hit_rate, 4),
    )
    logger.info(
        "Arbitrage backtest: model £%.2f vs persistence £%.2f (capture %.0f%%, hit-rate %.0f%%)",
        report.model_pnl_gbp,
        report.persistence_pnl_gbp,
        report.capture_rate * 100,
        report.hit_rate * 100,
    )
    return report


def _strategy_pnl(optimise_schedule, decision_prices, actual, carbon, battery) -> float:
    """P&L (£) of a schedule optimised on ``decision_prices`` but settled at ``actual``."""
    result = optimise_schedule(decision_prices, carbon, battery, load_kwh=[0.0] * len(actual))
    gi = np.array([s.grid_import_kwh for s in result.slots])
    ge = np.array([s.grid_export_kwh for s in result.slots])
    # Baseline (no battery, no load) costs £0; P&L is the negative of the settled cost.
    return -_settle_cost_gbp(gi, ge, np.asarray(actual))


def _prices_from_values(values: Sequence[float], times: pd.DatetimeIndex) -> list[HalfHourlyPrice]:
    return [
        HalfHourlyPrice(
            valid_from=t.to_pydatetime(),
            valid_to=(t + SLOT).to_pydatetime(),
            price_p_per_kwh=float(v),
        )
        for t, v in zip(times, values, strict=True)
    ]


def _flat_carbon(times: pd.DatetimeIndex) -> list[CarbonSlot]:
    return [
        CarbonSlot(
            valid_from=t.to_pydatetime(),
            valid_to=(t + SLOT).to_pydatetime(),
            intensity_g_per_kwh=200.0,
        )
        for t in times
    ]
