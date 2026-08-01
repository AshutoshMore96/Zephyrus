"""Demand forecasting (project #1) — the upstream signal for the optimiser.

Two layers ship here:

* Transparent **baselines** — ``seasonal_naive_forecast`` (repeat yesterday) and a
  ``synthetic_household_load`` profile — so the whole pipeline runs end-to-end today.
* A probabilistic :class:`DemandForecaster` (ROADMAP M3) built on the shared
  :class:`~zephyrus.forecast._core.ConformalForecaster`: calendar + lag features fed to a
  LightGBM (or numpy-ridge fallback) point model, wrapped in split-conformal intervals.
  :func:`backtest_demand` scores it (MAPE + pinball) vs the baseline and logs to MLflow;
  :func:`forecast_load_kwh` / :func:`demand_forecast_slots` feed the median straight into
  the optimiser's ``load_kwh`` — no downstream change.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from ..config import get_settings
from ..logging import get_logger
from ..schemas import DemandBacktestReport, LoadForecastSlot
from ._core import (
    DEFAULT_LAGS,
    DEFAULT_QUANTILES,
    SLOT,
    SLOTS_PER_DAY,
    SLOTS_PER_WEEK,
    ConformalForecaster,
    ensure_utc,
    mape,
    pinball_loss,
)

__all__ = [
    "seasonal_naive_forecast",
    "synthetic_household_load",
    "synthetic_metered_history",
    "DemandForecaster",
    "backtest_demand",
    "demand_forecast_slots",
    "forecast_load_kwh",
    "mape",
    "pinball_loss",
]

logger = get_logger(__name__)


# --------------------------------------------------------------------------------------
# Baselines (kept stable — the optimiser, API, CLI and demos import these directly)
# --------------------------------------------------------------------------------------
def seasonal_naive_forecast(
    history_kwh: list[float], horizon: int, period: int = 48
) -> list[float]:
    """Predict each future slot as the same slot one period (day) earlier."""
    if len(history_kwh) < period:
        mean = float(np.mean(history_kwh)) if history_kwh else 0.0
        return [mean] * horizon
    tail = history_kwh[-period:]
    return [tail[t % period] for t in range(horizon)]


def synthetic_household_load(n_slots: int, daily_kwh: float = 9.0, seed: int = 7) -> list[float]:
    """A realistic-ish half-hourly UK home load: overnight base + AM/PM peaks.

    Deterministic given ``seed``; used for demos/tests until real or forecast load is
    wired in. Scaled so one day integrates to roughly ``daily_kwh``.
    """
    base = np.full(48, 0.10)
    for slot in range(14, 18):  # ~07:00-09:00 morning peak
        base[slot] += 0.25
    for slot in range(34, 42):  # ~17:00-21:00 evening peak
        base[slot] += 0.45
    base *= daily_kwh / base.sum()

    rng = np.random.default_rng(seed)
    days = int(np.ceil(n_slots / 48))
    profile = np.tile(base, days)[:n_slots] + rng.normal(0, 0.01, n_slots)
    return np.clip(profile, 0.02, None).round(4).tolist()


def synthetic_metered_history(
    start: datetime,
    days: int = 42,
    daily_kwh: float = 9.0,
    weekend_uplift: float = 1.30,
    seed: int = 11,
) -> pd.Series:
    """A weekly-seasonal half-hourly load series, indexed by tz-aware timestamps.

    Stand-in for real smart-meter history (arrives with M1's data layer) so the
    forecaster and backtest have *learnable* structure the seasonal-naive baseline cannot
    capture: weekends are uplifted and shifted later, so "repeat yesterday" is
    systematically wrong across weekday/weekend boundaries.
    """
    weekday = np.full(SLOTS_PER_DAY, 0.10)
    for slot in range(14, 18):  # morning peak
        weekday[slot] += 0.25
    for slot in range(34, 42):  # evening peak
        weekday[slot] += 0.45
    weekday *= daily_kwh / weekday.sum()

    weekend = np.roll(weekday, 2) * weekend_uplift
    weekend[20:34] += 0.03 * weekend_uplift  # midday plateau

    start = ensure_utc(start)
    index = pd.date_range(start=start, periods=days * SLOTS_PER_DAY, freq=SLOT)
    rng = np.random.default_rng(seed)
    values = np.empty(len(index))
    for i, ts in enumerate(index):
        day_profile = weekend if ts.dayofweek >= 5 else weekday
        values[i] = day_profile[i % SLOTS_PER_DAY]
    values = np.clip(values + rng.normal(0, 0.012, len(index)), 0.02, None)
    return pd.Series(values.round(4), index=index, name="load_kwh")


# --------------------------------------------------------------------------------------
# Forecaster
# --------------------------------------------------------------------------------------
class DemandForecaster(ConformalForecaster):
    """Probabilistic half-hourly demand forecaster (loads clipped non-negative)."""

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
            clip_min=0.0,
        )

    def predict(self, horizon: int = SLOTS_PER_DAY) -> list[LoadForecastSlot]:
        """Forecast ``horizon`` half-hours ahead as probabilistic load slots."""
        times, preds = self.predict_quantiles(horizon)
        median, lower, upper = preds[0.5], preds[self.quantiles[0]], preds[self.quantiles[-1]]
        return [
            LoadForecastSlot(
                valid_from=times[h].to_pydatetime(),
                load_kwh=round(float(median[h]), 4),
                lower_kwh=round(float(min(lower[h], median[h])), 4),
                upper_kwh=round(float(max(upper[h], median[h])), 4),
            )
            for h in range(horizon)
        ]

    def predict_load_kwh(self, horizon: int = SLOTS_PER_DAY) -> list[float]:
        """Median forecast as a plain ``list[float]`` — drop-in for ``load_kwh``."""
        _, preds = self.predict_quantiles(horizon)
        return [round(float(v), 4) for v in preds[0.5]]


# --------------------------------------------------------------------------------------
# Backtest (rolling origin) + MLflow logging
# --------------------------------------------------------------------------------------
def _optional_mlflow() -> Any | None:
    try:
        import mlflow  # noqa: PLC0415
    except ImportError:
        return None
    return mlflow


def backtest_demand(
    series: pd.Series,
    horizon: int = SLOTS_PER_DAY,
    n_folds: int = 4,
    step: int | None = None,
    backend: str = "auto",
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    log_mlflow: bool = False,
    run_name: str | None = None,
) -> DemandBacktestReport:
    """Rolling-origin backtest: score the forecaster vs seasonal-naive (MAPE + pinball)."""
    series = ConformalForecaster._validate_series(series)
    values = series.to_numpy(dtype=float)
    step = step or horizon
    qs = sorted(float(q) for q in quantiles)

    origins = [len(values) - horizon - k * step for k in range(n_folds)][::-1]
    origins = [o for o in origins if o - 2 * SLOTS_PER_WEEK > 0]
    if not origins:
        raise ValueError("not enough history for the requested horizon/folds")

    y_true, model_median, naive_pred = [], [], []
    model_q: dict[float, list[np.ndarray]] = {q: [] for q in qs}
    covered = total = 0
    backend_used = "unfitted"

    for origin in origins:
        train = series.iloc[:origin]
        actual = values[origin : origin + horizon]
        fc = DemandForecaster(quantiles=qs, backend=backend).fit(train)
        backend_used = fc.backend
        _, preds = fc.predict_quantiles(horizon)
        naive = np.asarray(seasonal_naive_forecast(train.to_numpy().tolist(), horizon))

        y_true.append(actual)
        model_median.append(preds[0.5])
        naive_pred.append(naive)
        for q in qs:
            model_q[q].append(preds[q])
        lo, hi = preds[qs[0]], preds[qs[-1]]
        covered += int(np.sum((actual >= lo) & (actual <= hi)))
        total += len(actual)

    yt = np.concatenate(y_true)
    med = np.concatenate(model_median)
    naive = np.concatenate(naive_pred)
    model_pin = float(np.mean([pinball_loss(yt, np.concatenate(model_q[q]), q) for q in qs]))
    baseline_pin = float(np.mean([pinball_loss(yt, naive, q) for q in qs]))

    report = DemandBacktestReport(
        horizon=horizon,
        n_folds=len(origins),
        model_backend=backend_used,
        quantiles=qs,
        nominal_coverage=round(qs[-1] - qs[0], 4),
        model_mape=round(mape(yt, med), 4),
        baseline_mape=round(mape(yt, naive), 4),
        model_pinball=round(model_pin, 6),
        baseline_pinball=round(baseline_pin, 6),
        coverage=round(covered / total, 4) if total else 0.0,
    )
    logger.info(
        "Backtest: model MAPE %.2f%% vs naive %.2f%% (%.1f%% better); coverage %.2f @ %.0f%%",
        report.model_mape,
        report.baseline_mape,
        report.mape_improvement_pct,
        report.coverage,
        report.nominal_coverage * 100,
    )
    if log_mlflow:
        _log_backtest_to_mlflow(report, run_name)
    return report


def _log_backtest_to_mlflow(report: DemandBacktestReport, run_name: str | None) -> None:
    """Log backtest params + metrics to MLflow when available; otherwise no-op."""
    mlflow = _optional_mlflow()
    if mlflow is None:
        logger.info("MLflow not installed (pip install '.[ml]') — skipping run logging")
        return
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment("zephyrus-demand")
    with mlflow.start_run(run_name=run_name or "demand-backtest"):
        mlflow.log_params(
            {
                "horizon": report.horizon,
                "n_folds": report.n_folds,
                "model_backend": report.model_backend,
                "quantiles": report.quantiles,
                "nominal_coverage": report.nominal_coverage,
            }
        )
        mlflow.log_metrics(
            {
                "model_mape": report.model_mape,
                "baseline_mape": report.baseline_mape,
                "mape_improvement_pct": report.mape_improvement_pct,
                "model_pinball": report.model_pinball,
                "baseline_pinball": report.baseline_pinball,
                "coverage": report.coverage,
            }
        )
    logger.info("Logged backtest to MLflow at %s", get_settings().mlflow_tracking_uri)


# --------------------------------------------------------------------------------------
# Wiring into the optimiser
# --------------------------------------------------------------------------------------
def forecast_load_kwh(
    history_kwh: list[float],
    horizon: int,
    start: datetime | None = None,
    backend: str = "auto",
) -> list[float]:
    """Median demand forecast as ``list[float]`` — the optimiser's ``load_kwh`` input.

    Falls back to :func:`seasonal_naive_forecast` when history is too short to train.
    """
    if len(history_kwh) <= 2 * max(DEFAULT_LAGS):
        return seasonal_naive_forecast(history_kwh, horizon)
    series = _series_from_history(history_kwh, start)
    return DemandForecaster(backend=backend).fit(series).predict_load_kwh(horizon)


def demand_forecast_slots(
    n_slots: int,
    start: datetime | None = None,
    history_weeks: int = 6,
    backend: str = "auto",
) -> list[LoadForecastSlot]:
    """Forecast ``n_slots`` half-hours from ``start`` as probabilistic load slots.

    Convenience for the API/dashboard/CLI: builds a synthetic metered history (stand-in
    for real smart-meter data until M1's data layer lands) ending just before ``start``,
    fits :class:`DemandForecaster`, and returns its forecast.
    """
    start = ensure_utc(start) if start else _floor_now()
    history_start = start - history_weeks * 7 * timedelta(days=1)
    history = synthetic_metered_history(history_start, days=history_weeks * 7)
    return DemandForecaster(backend=backend).fit(history).predict(n_slots)


def _series_from_history(history_kwh: Sequence[float], start: datetime | None) -> pd.Series:
    start = ensure_utc(start) if start else _floor_now()
    index = pd.date_range(start=start, periods=len(history_kwh), freq=SLOT)
    return pd.Series(np.asarray(history_kwh, dtype=float), index=index, name="load_kwh")


def _floor_now() -> datetime:
    now = datetime.now(UTC)
    return now.replace(minute=0 if now.minute < 30 else 30, second=0, microsecond=0)
