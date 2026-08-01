"""Shared forecasting engine: calendar/lag features + split-conformal quantiles.

Both the demand forecaster (project #1) and the price forecaster (project #4) are the
same recipe — a gradient-boosting (LightGBM) or numpy-ridge point model over lag +
calendar features, wrapped in split-conformal prediction intervals. That recipe lives
here once as :class:`ConformalForecaster`; the domain forecasters subclass it and only
map the raw quantile arrays onto their typed slot schema.

Heavy deps (LightGBM) are imported lazily so the package imports — and the numpy
fallback runs — without the ``ml`` extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Self

import numpy as np
import pandas as pd

from ..logging import get_logger

logger = get_logger(__name__)

SLOTS_PER_DAY = 48
SLOTS_PER_WEEK = SLOTS_PER_DAY * 7  # 336
SLOT = timedelta(minutes=30)

DEFAULT_LAGS: tuple[int, ...] = (1, 2, 3, SLOTS_PER_DAY, SLOTS_PER_DAY * 2, SLOTS_PER_WEEK)
DEFAULT_QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)


def optional_lightgbm() -> Any | None:
    try:
        import lightgbm  # noqa: PLC0415
    except ImportError:
        return None
    return lightgbm


# --- feature engineering ---------------------------------------------------------------
def calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Cyclic time-of-day + day-of-week features plus a weekend flag (fixed column order)."""
    hh = index.hour.to_numpy() * 2 + index.minute.to_numpy() // 30  # 0..47
    dow = index.dayofweek.to_numpy()  # 0=Mon .. 6=Sun
    return np.column_stack(
        [
            np.sin(2 * np.pi * hh / SLOTS_PER_DAY),
            np.cos(2 * np.pi * hh / SLOTS_PER_DAY),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            (dow >= 5).astype(float),
        ]
    )


def lag_matrix(values: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """Lagged copies of ``values``; rows before a lag is available hold NaN."""
    n = len(values)
    out = np.full((n, len(lags)), np.nan)
    for j, lag in enumerate(lags):
        if lag < n:
            out[lag:, j] = values[: n - lag]
    return out


class RidgeRegressor:
    """Closed-form L2-regularised linear regression (numpy only) — the offline fallback."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> RidgeRegressor:
        self._mean = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0.0] = 1.0
        self._std = std
        xs = (x - self._mean) / self._std
        self._intercept = float(y.mean())
        yc = y - self._intercept
        gram = xs.T @ xs + self.alpha * np.eye(xs.shape[1])
        self._coef = np.linalg.solve(gram, xs.T @ yc)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._coef is None or self._mean is None or self._std is None:
            raise RuntimeError("regressor is not fitted")
        xs = (x - self._mean) / self._std
        return xs @ self._coef + self._intercept


# --- metrics ---------------------------------------------------------------------------
def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean absolute percentage error (%). ``eps`` guards near-zero actuals."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Pinball (quantile) loss — the proper score for a quantile forecast."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


class ConformalForecaster:
    """Recursive point forecaster with split-conformal quantile intervals.

    Subclasses map :meth:`predict_quantiles` onto their typed slot schema. ``clip_min``
    bounds the recursive path (0 for loads; ``None`` for prices, which can go negative).
    """

    def __init__(
        self,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        lags: Sequence[int] = DEFAULT_LAGS,
        backend: str = "auto",
        calibration_frac: float = 0.2,
        random_state: int = 42,
        clip_min: float | None = 0.0,
    ) -> None:
        qs = sorted(float(q) for q in quantiles)
        if not qs or any(not 0.0 < q < 1.0 for q in qs):
            raise ValueError("quantiles must lie strictly in (0, 1)")
        if 0.5 not in qs:
            raise ValueError("quantiles must include the median (0.5)")
        if not 0.0 < calibration_frac < 1.0:
            raise ValueError("calibration_frac must lie strictly in (0, 1)")
        self.quantiles = qs
        self.lags = tuple(int(lag) for lag in lags)
        self.max_lag = max(self.lags)
        self.calibration_frac = calibration_frac
        self.random_state = random_state
        self.requested_backend = backend
        self.clip_min = clip_min

        self._model: Any | None = None
        self._offsets: dict[float, float] = {}
        self.backend: str = "unfitted"
        self._history: np.ndarray | None = None
        self._last_time: pd.Timestamp | None = None

    # -- fitting ------------------------------------------------------------------------
    def _make_model(self) -> Any:
        want = self.requested_backend
        lgb = optional_lightgbm() if want in {"auto", "lightgbm"} else None
        if want == "lightgbm" and lgb is None:
            raise ImportError("backend='lightgbm' requires the 'ml' extra (pip install '.[ml]')")
        if lgb is not None:
            self.backend = "lightgbm"
            return lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.9,
                random_state=self.random_state,
                verbosity=-1,
            )
        self.backend = "ridge"
        return RidgeRegressor(alpha=1.0)

    def fit(self, series: pd.Series) -> Self:
        """Fit the point model and calibrate conformal offsets on a held-out tail."""
        series = self._validate_series(series)
        values = series.to_numpy(dtype=float)
        index = series.index

        features = np.hstack([lag_matrix(values, self.lags), calendar_features(index)])
        valid = ~np.isnan(features).any(axis=1)
        x_all, y_all = features[valid], values[valid]
        if len(y_all) < 2 * self.max_lag:
            raise ValueError(
                f"need > {2 * self.max_lag} usable slots to fit (got {len(y_all)}); "
                "supply more history or smaller lags"
            )

        n_cal = max(SLOTS_PER_DAY, int(round(len(y_all) * self.calibration_frac)))
        n_cal = min(n_cal, len(y_all) // 2)
        x_train, y_train = x_all[:-n_cal], y_all[:-n_cal]
        x_cal, y_cal = x_all[-n_cal:], y_all[-n_cal:]

        self._model = self._make_model().fit(x_train, y_train)
        residuals = y_cal - self._model.predict(x_cal)
        self._offsets = {q: float(np.quantile(residuals, q)) for q in self.quantiles}

        self._history = values
        self._last_time = index[-1]
        logger.info(
            "%s fitted: backend=%s, train=%d, calib=%d",
            type(self).__name__,
            self.backend,
            len(y_train),
            len(y_cal),
        )
        return self

    # -- prediction ---------------------------------------------------------------------
    def _feature_row(self, buffer: list[float], ts: pd.Timestamp) -> np.ndarray:
        lags = np.array([buffer[-lag] for lag in self.lags], dtype=float)
        cal = calendar_features(pd.DatetimeIndex([ts]))[0]
        return np.hstack([lags, cal])

    def _forecast_point_path(self, horizon: int) -> tuple[list[pd.Timestamp], np.ndarray]:
        if self._model is None or self._history is None or self._last_time is None:
            raise RuntimeError("forecaster is not fitted; call .fit(series) first")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        buffer = self._history.tolist()
        ts = self._last_time
        times: list[pd.Timestamp] = []
        path = np.empty(horizon)
        for h in range(horizon):
            ts = ts + SLOT
            yhat = float(self._model.predict(self._feature_row(buffer, ts).reshape(1, -1))[0])
            if self.clip_min is not None:
                yhat = max(yhat, self.clip_min)
            path[h] = yhat
            times.append(ts)
            buffer.append(yhat)
        return times, path

    def predict_quantiles(self, horizon: int) -> tuple[list[pd.Timestamp], dict[float, np.ndarray]]:
        """Future timestamps and a ``{quantile: values}`` map (conformalised, monotone)."""
        times, point = self._forecast_point_path(horizon)
        preds = {}
        for q in self.quantiles:
            vals = point + self._offsets[q]
            if self.clip_min is not None:
                vals = np.clip(vals, self.clip_min, None)
            preds[q] = vals
        return times, preds

    @staticmethod
    def _validate_series(series: pd.Series) -> pd.Series:
        if not isinstance(series, pd.Series):
            raise TypeError("expected a pandas Series indexed by a half-hourly DatetimeIndex")
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError("series must be indexed by a DatetimeIndex")
        series = series.sort_index()
        if series.isna().any():
            raise ValueError("series contains NaNs; clean or interpolate before fitting")
        return series


def ensure_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
