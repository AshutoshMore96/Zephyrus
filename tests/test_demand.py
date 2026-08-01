"""Demand forecaster (M3) spec — runs offline on the numpy-ridge fallback (no ml extra).

These tests double as the executable contract for project #1: the forecaster must produce
non-crossing, non-negative probabilistic slots, beat the seasonal-naive baseline on MAPE
and pinball, deliver near-nominal conformal coverage, and drop straight into the
optimiser's ``load_kwh``.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime

import numpy as np
import pytest

from zephyrus.forecast.demand import (
    DemandForecaster,
    backtest_demand,
    demand_forecast_slots,
    forecast_load_kwh,
    mape,
    pinball_loss,
    seasonal_naive_forecast,
    synthetic_household_load,
    synthetic_metered_history,
)
from zephyrus.optimise.milp import optimise_schedule
from zephyrus.schemas import BatterySpec, DemandBacktestReport, LoadForecastSlot

MONDAY = datetime(2025, 1, 6, tzinfo=UTC)

_HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None
_HAS_MLFLOW = importlib.util.find_spec("mlflow") is not None
no_lightgbm = pytest.mark.skipif(
    _HAS_LIGHTGBM, reason="asserts the numpy fallback (ml extra absent)"
)
no_mlflow = pytest.mark.skipif(_HAS_MLFLOW, reason="asserts the no-op path (ml extra absent)")


@pytest.fixture
def history():
    """Six weeks of weekly-seasonal half-hourly load (naive-beatable structure)."""
    return synthetic_metered_history(MONDAY, days=42)


# -- baselines stay stable --------------------------------------------------------------
def test_seasonal_naive_repeats_previous_day():
    hist = list(range(48)) + [0.0] * 48
    out = seasonal_naive_forecast([float(x) for x in hist], horizon=48, period=48)
    assert out == [0.0] * 48  # last full day was all zeros


def test_seasonal_naive_short_history_uses_mean():
    assert seasonal_naive_forecast([2.0, 4.0], horizon=3) == [3.0, 3.0, 3.0]


def test_synthetic_household_load_shape_and_energy():
    load = synthetic_household_load(48, daily_kwh=9.0)
    assert len(load) == 48
    assert abs(sum(load) - 9.0) < 0.5
    assert min(load) >= 0.02


# -- forecaster contract ----------------------------------------------------------------
def test_predict_returns_monotone_nonnegative_slots(history):
    slots = DemandForecaster().fit(history).predict(48)
    assert len(slots) == 48
    assert all(isinstance(s, LoadForecastSlot) for s in slots)
    for s in slots:
        assert s.lower_kwh <= s.load_kwh <= s.upper_kwh
        assert s.lower_kwh >= 0.0


def test_predict_load_kwh_matches_slot_medians(history):
    forecaster = DemandForecaster().fit(history)
    medians = [s.load_kwh for s in forecaster.predict(24)]
    assert forecaster.predict_load_kwh(24) == medians


@no_lightgbm
def test_fallback_backend_is_ridge_without_ml_extra(history):
    # CI installs only [dev]; the numpy ridge fallback must be selected transparently.
    assert DemandForecaster().fit(history).backend == "ridge"


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        DemandForecaster().predict(10)


def test_rejects_series_without_datetime_index():
    import pandas as pd

    with pytest.raises(TypeError):
        DemandForecaster().fit(pd.Series([1.0, 2.0, 3.0]))


def test_quantiles_must_include_median():
    with pytest.raises(ValueError):
        DemandForecaster(quantiles=(0.1, 0.9))


# -- backtest: beats the baseline, near-nominal coverage --------------------------------
def test_backtest_beats_seasonal_naive(history):
    report = backtest_demand(history, horizon=48, n_folds=4)
    assert isinstance(report, DemandBacktestReport)
    assert report.model_backend == "ridge"
    # The weekly structure is invisible to "repeat yesterday" — the model must win.
    assert report.model_mape < report.baseline_mape
    assert report.model_pinball < report.baseline_pinball
    assert report.beats_baseline is True


def test_backtest_coverage_near_nominal(history):
    report = backtest_demand(history, horizon=48, n_folds=4)
    # 80% nominal interval — allow slack for the multi-step conformal approximation.
    assert report.coverage >= 0.6
    assert report.coverage <= 1.0


def test_backtest_does_not_log_mlflow_by_default(history):
    # Default must be side-effect free (no ./mlruns) so the offline suite stays clean.
    report = backtest_demand(history, horizon=48, n_folds=3, log_mlflow=False)
    assert report.n_folds == 3


@no_mlflow
def test_backtest_log_mlflow_noops_without_ml_extra(history):
    # log_mlflow=True must degrade gracefully (no crash, no ./mlruns) when MLflow is absent.
    report = backtest_demand(history, horizon=48, n_folds=3, log_mlflow=True)
    assert report.beats_baseline is True


@no_lightgbm
def test_lightgbm_backend_requires_ml_extra(history):
    # Explicitly requesting LightGBM without the extra installed is a clear ImportError.
    with pytest.raises(ImportError):
        DemandForecaster(backend="lightgbm").fit(history)


# -- metrics ----------------------------------------------------------------------------
def test_mape_zero_for_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    assert mape(y, y) == pytest.approx(0.0)


def test_pinball_loss_is_asymmetric():
    y = np.array([1.0, 1.0])
    over = pinball_loss(y, np.array([2.0, 2.0]), quantile=0.9)
    under = pinball_loss(y, np.array([0.0, 0.0]), quantile=0.9)
    # At q=0.9 under-prediction is penalised more heavily than over-prediction.
    assert under > over


# -- wiring into the optimiser ----------------------------------------------------------
def test_forecast_load_kwh_feeds_optimiser(history, price_carbon_series):
    prices, carbon = price_carbon_series()
    load = forecast_load_kwh(history.tolist(), horizon=len(prices), start=MONDAY)
    assert len(load) == len(prices)
    assert all(isinstance(x, float) and x >= 0.0 for x in load)

    result = optimise_schedule(prices, carbon, BatterySpec(capacity_kwh=5), load_kwh=load)
    assert result.cost_saving_gbp >= -1e-6  # optimiser never worse than the baseline


def test_forecast_load_kwh_short_history_falls_back_to_naive():
    short = synthetic_household_load(60)  # < two weeks -> cannot train
    assert forecast_load_kwh(short, horizon=10) == seasonal_naive_forecast(short, 10)


def test_demand_forecast_slots_convenience_feeds_api_and_dashboard():
    # Shared helper behind the API /forecast + /optimise and the dashboard: builds its
    # own synthetic history, so callers need only a slot count and a start time.
    slots = demand_forecast_slots(48, start=MONDAY, history_weeks=4)
    assert len(slots) == 48
    assert all(isinstance(s, LoadForecastSlot) for s in slots)
    assert all(s.lower_kwh <= s.load_kwh <= s.upper_kwh for s in slots)
