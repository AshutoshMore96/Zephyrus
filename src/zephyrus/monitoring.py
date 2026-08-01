"""Drift & performance monitoring with alerting (project ops, M7).

Two guards run in production between retrains:

* **Feature drift** — Population Stability Index (PSI) between a reference window and live
  data. PSI > 0.2 is the usual "material shift, investigate/retrain" line.
* **Performance regression** — live error (e.g. MAPE) vs the backtest baseline; a breach
  means the deployed model has decayed.

Both return a typed :class:`~zephyrus.schemas.DriftReport`; :func:`raise_alert` logs a
warning (the hook a real alerter — PagerDuty/Slack — would replace).
"""

from __future__ import annotations

import numpy as np

from .logging import get_logger
from .schemas import DriftReport

logger = get_logger(__name__)

PSI_THRESHOLD = 0.2  # > 0.2 = significant population shift


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-6
) -> float:
    """PSI between two samples using quantile bins of the reference distribution."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        raise ValueError("reference and current must be non-empty")
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:  # reference is (near-)constant
        edges = np.array([reference.min() - eps, reference.max() + eps])
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_feature_drift(
    reference: np.ndarray, current: np.ndarray, threshold: float = PSI_THRESHOLD
) -> DriftReport:
    """Flag distribution drift between ``reference`` and ``current`` via PSI."""
    psi = round(population_stability_index(reference, current), 4)
    report = DriftReport(
        metric="psi",
        value=psi,
        threshold=threshold,
        detail=f"PSI={psi:.3f} vs threshold {threshold}",
    )
    raise_alert(report)
    return report


def detect_performance_regression(
    live_mape: float, baseline_mape: float, tolerance: float = 1.5
) -> DriftReport:
    """Flag model decay when live MAPE exceeds ``tolerance × baseline_mape``."""
    limit = baseline_mape * tolerance
    report = DriftReport(
        metric="mape_ratio",
        value=round(live_mape, 4),
        threshold=round(limit, 4),
        detail=f"live MAPE {live_mape:.2f}% vs limit {limit:.2f}% "
        f"({tolerance:g}× baseline {baseline_mape:.2f}%)",
    )
    raise_alert(report)
    return report


def raise_alert(report: DriftReport) -> None:
    """Log a warning when a report is in alert (the pluggable alerting hook)."""
    if report.alert:
        logger.warning("ALERT [%s]: %s", report.metric, report.detail)
    else:
        logger.info("OK [%s]: %s", report.metric, report.detail)
