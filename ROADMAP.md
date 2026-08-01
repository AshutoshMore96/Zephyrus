# Roadmap

Milestoned backlog. Each item is sized to be a self-contained PR. Pick the next unchecked
box; follow `CLAUDE.md` for conventions and definition of done. `✅` = shipped in the
scaffold, `⬜` = to build.

## M0 — Scaffold ✅
- ✅ Monorepo, packaging (`pyproject`), tooling (ruff/black/mypy/pytest), CI, Docker
- ✅ Config, logging, typed schemas, shared HTTP client
- ✅ Live Octopus + Carbon Intensity clients
- ✅ Core MILP battery scheduler + naive baseline
- ✅ FastAPI, Streamlit, Typer CLI, demo scripts
- ✅ Offline test suite (respx)

## M1 — Data foundation ✅
- ✅ Persist snapshots to `data/` with a tiny local cache (`io/store.py`, parquet + TTL)
- ✅ Add Open-Meteo client for weather features (`io/weather.py`, `WeatherSlot`)
- ✅ Data-quality checks (gaps, DST folds, unit sanity) with clear errors (`quality.py`)
- ✅ Great-Expectations-style assertions on ingested frames (`expect_*` / `validate`);
  wired into the `zephyrus snapshot` command

## M2 — Optimiser hardening ✅
- ✅ On-site solar + export (Outgoing) tariff modelling end-to-end (`export_prices_gbp`,
  curtailment + export-capped-to-local-supply so grid arbitrage stays bounded)
- ✅ EV mode: departure-time SoC target + plug-in windows as constraints (`EVConstraints`)
- ✅ Degradation cost term (£/kWh throughput) in the objective (`degradation_gbp_per_kwh`)
- ✅ Sensitivity sweep (`optimise/sensitivity.py::sweep_savings`, savings vs size/power)

## M3 — Demand forecasting (project #1) ✅
- ✅ `DemandForecaster` (LightGBM lags/calendar features; numpy-ridge fallback keeps the
  offline suite green without the `ml` extra — weather features slot in with M1's Open-Meteo)
- ✅ Split-conformal prediction intervals; `backtest_demand(..., log_mlflow=True)` logs
  params + metrics to MLflow
- ✅ Backtest: MAPE + pinball loss vs the seasonal-naive baseline (`zephyrus backtest`)
- ✅ Wire forecast output into the optimiser's `load_kwh` (`forecast_load_kwh`,
  `zephyrus forecast`)

## M4 — Price forecasting & arbitrage (project #4) ✅
- ✅ Implement `ElexonClient` (system/imbalance prices in p/kWh, FUELHH generation mix, NIV)
- ✅ `PriceForecaster` with quantile outputs (spike-aware; shares the conformal core)
- ✅ Battery-arbitrage backtest: P&L, Sharpe, hit-rate vs persistence + oracle capture
  (`backtest_arbitrage`, `zephyrus arbitrage`)
- ✅ Feed forecast prices into the optimiser (`predict_prices` -> `HalfHourlyPrice`)

## M5 — Virtual power plant (aggregation) ✅
- ✅ Optimise N heterogeneous assets in one MILP (`optimise.vpp.optimise_portfolio`) +
  `aggregate` KPI roll-up
- ✅ Coincident-peak reduction + network-headroom constraint (shared per-slot import cap)
- ✅ Balancing-market bid + settlement (`peak_bid_gbp_per_kw`, `bid_revenue_gbp`);
  exposed via `POST /vpp` and `zephyrus vpp`

## M6 — Serving & UX ✅
- ✅ Persist schedules (`io/store.py::save_result`/`load_result`, `/optimise?persist_as`);
  added `/forecast` and `/vpp` API endpoints
- ✅ Dashboard: VPP portfolio view + scenario-compare (Plotly savings-vs-size)
- ✅ Auth (optional `X-API-Key`) + in-process rate limiting on the API (`api/security.py`)

## M7 — MLOps & deploy ✅
- ✅ Model registry (`registry.py`) + scheduled retrain (`scripts/retrain.py`,
  `.github/workflows/retrain.yml` weekly cron)
- ✅ Drift (PSI) & performance-regression monitoring with alerting (`monitoring.py`)
- ✅ One-click deploy: `fly.toml` (API, scale-to-zero) + Streamlit/HF for the dashboard
- ✅ Coverage badge + `make coverage`; MLOps docs (`docs/mlops.md`)
