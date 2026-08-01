"""Scheduled retrain: backtest the demand forecaster, register it, guard against decay.

    python scripts/retrain.py

Runs the rolling-origin backtest (logging to MLflow when installed), registers the run in
the local model registry, and checks the new model against the previous best for
performance regression. Wired to a weekly GitHub Actions cron (see
``.github/workflows/retrain.yml``). Uses synthetic metered history as a stand-in until the
M1 data layer ingests real meter data.
"""

from __future__ import annotations

from datetime import UTC, datetime

import typer

from zephyrus.forecast.demand import backtest_demand, synthetic_metered_history
from zephyrus.monitoring import detect_performance_regression
from zephyrus.registry import ModelRegistry


def main(weeks: int = 8, horizon: int = 48, folds: int = 4, mlflow: bool = True) -> None:
    start = datetime(2025, 1, 6, tzinfo=UTC)  # a Monday, for stable calendar features
    history = synthetic_metered_history(start, days=weeks * 7)

    report = backtest_demand(history, horizon=horizon, n_folds=folds, log_mlflow=mlflow)
    typer.echo(
        f"Backtest: model MAPE {report.model_mape:.2f}% vs naive {report.baseline_mape:.2f}% "
        f"({report.mape_improvement_pct:+.1f}%), backend={report.model_backend}"
    )

    registry = ModelRegistry()
    previous_best = None
    try:
        previous_best = registry.best("demand", lower_is_better=True)
    except FileNotFoundError:
        typer.echo("No previous model registered yet — this is the first version.")

    entry = registry.register(
        "demand",
        metrics={
            "mape": report.model_mape,
            "pinball": report.model_pinball,
            "coverage": report.coverage,
        },
        primary_metric="mape",
        params={"backend": report.model_backend, "horizon": str(horizon), "folds": str(folds)},
    )
    typer.secho(f"Registered demand v{entry.version} (MAPE {report.model_mape:.2f}%).", fg="green")

    # Guard: a real deployment would gate promotion on this not alerting.
    baseline = previous_best.metrics["mape"] if previous_best else report.baseline_mape
    regression = detect_performance_regression(report.model_mape, baseline)
    if regression.alert:
        typer.secho(f"⚠️  Performance regression: {regression.detail}", fg="red")
        raise typer.Exit(code=1)
    typer.secho("Model within tolerance — safe to promote.", fg="green")


if __name__ == "__main__":
    typer.run(main)
