"""Drift & performance monitoring + model registry (M7)."""

from __future__ import annotations

import numpy as np
import pytest

from zephyrus.monitoring import (
    detect_feature_drift,
    detect_performance_regression,
    population_stability_index,
)
from zephyrus.registry import ModelRegistry
from zephyrus.schemas import DriftReport, ModelVersion


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(0, 1, 5000)
    assert population_stability_index(ref, cur) < 0.1


def test_psi_large_for_shifted_distribution():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(3, 1, 5000)  # big mean shift
    assert population_stability_index(ref, cur) > 0.25


def test_detect_feature_drift_flags_shift():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 4000)
    drifted = detect_feature_drift(ref, rng.normal(2, 1, 4000))
    assert isinstance(drifted, DriftReport)
    assert drifted.alert is True
    steady = detect_feature_drift(ref, rng.normal(0, 1, 4000))
    assert steady.alert is False


def test_performance_regression_alert():
    assert detect_performance_regression(live_mape=6.0, baseline_mape=3.0).alert is True
    assert detect_performance_regression(live_mape=3.2, baseline_mape=3.0).alert is False


def test_psi_rejects_empty():
    with pytest.raises(ValueError):
        population_stability_index(np.array([]), np.array([1.0]))


# -- registry ---------------------------------------------------------------------------
def test_registry_versions_increment(tmp_path):
    reg = ModelRegistry(base_dir=tmp_path)
    v1 = reg.register("demand", {"mape": 9.0}, primary_metric="mape")
    v2 = reg.register("demand", {"mape": 7.5}, primary_metric="mape")
    assert (v1.version, v2.version) == (1, 2)
    assert isinstance(reg.latest("demand"), ModelVersion)
    assert reg.latest("demand").version == 2


def test_registry_best_by_metric(tmp_path):
    reg = ModelRegistry(base_dir=tmp_path)
    reg.register("demand", {"mape": 9.0}, primary_metric="mape")
    reg.register("demand", {"mape": 6.0}, primary_metric="mape")
    reg.register("demand", {"mape": 8.0}, primary_metric="mape")
    assert reg.best("demand", lower_is_better=True).metrics["mape"] == 6.0


def test_registry_rejects_missing_primary_metric(tmp_path):
    reg = ModelRegistry(base_dir=tmp_path)
    with pytest.raises(ValueError):
        reg.register("demand", {"mape": 9.0}, primary_metric="rmse")


def test_registry_best_raises_without_versions(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelRegistry(base_dir=tmp_path).best("missing")
