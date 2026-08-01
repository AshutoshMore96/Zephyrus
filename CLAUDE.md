# CLAUDE.md — working agreement for Claude Code

This file orients an AI coding agent (and any new contributor) working in this repo.
Read it before writing code. Keep it up to date when structure or conventions change.

## What this project is

**Zephyrus** turns home batteries / EVs into a virtual power plant. It is one repo that
integrates three portfolio projects:

| Component | Package | Role in the system |
|-----------|---------|--------------------|
| #1 Demand forecasting | `zephyrus.forecast.demand` | Predicts household load -> feeds `load_kwh` to the optimiser |
| #4 Price forecasting | `zephyrus.forecast.price` | Predicts wholesale/imbalance prices -> feeds prices to the optimiser for backtests |
| #2 Flexibility optimiser (**core**) | `zephyrus.optimise` | MILP schedules the battery against prices + carbon; aggregates assets into a VPP |

The design principle is **forecast-driven optimisation**: forecasts are upstream signals
that flow into the optimiser. Everything is wired end-to-end today using transparent
baselines; the ROADMAP replaces baselines with real models without changing interfaces.

## Repo map

```
src/zephyrus/
  config.py         # Pydantic settings (env-driven). Import via get_settings().
  logging.py        # get_logger(__name__)
  schemas.py        # SINGLE source of truth for data contracts. Change here first.
  utils.py          # align_price_carbon() and other small helpers
  io/               # external data clients (all free APIs)
    base.py         # BaseHTTPClient: retry/timeout/logging — all HTTP goes through here
    octopus.py      # ✅ live Octopus Agile prices
    carbon.py       # ✅ live National Grid carbon intensity
    elexon.py       # ⛔ STUB — implement in M4
  forecast/
    demand.py       # ✅ baseline (seasonal-naive + synthetic load); model TODO M3
    price.py        # ✅ baseline (persistence); model TODO M4
  optimise/
    battery.py      # ✅ battery helpers + naive baseline controller
    milp.py         # ✅ CORE MILP scheduler (PuLP/CBC)
    vpp.py          # ⛔ STUB — portfolio aggregation, M5
  api/main.py       # FastAPI service
  dashboard/app.py  # Streamlit demo (recruiter-facing)
  cli.py            # Typer CLI (`zephyrus ...`)
scripts/            # fetch_sample_data.py, run_demo.py
tests/              # pytest; APIs mocked with respx (no network in CI)
configs/default.yaml, .env.example  # configuration
docs/               # architecture.md + ADRs
ROADMAP.md          # the backlog — pick the next unchecked item here
```

## How to run

```bash
make install     # editable install with all extras
make test        # pytest (offline; respx mocks the APIs)
make lint        # ruff + mypy
make demo        # end-to-end optimisation on LIVE data
make api         # uvicorn FastAPI on :8000
make dashboard   # Streamlit demo
make data        # snapshot live data into data/raw/
```

## Conventions (please follow)

- **Python 3.11+, fully type-hinted.** `from __future__ import annotations` at the top.
- **Units are fixed and must not drift:** money in **£** (schemas expose a `_gbp` helper
  from pence), energy in **kWh**, power in **kW**, carbon in **gCO2/kWh** (convert to kg
  only at the point of reporting). If you need a new quantity, add it to `schemas.py`.
- **Data contracts live in `schemas.py`.** Prefer passing Pydantic models over loose
  dicts. Change the schema first, then the producers/consumers.
- **All outbound HTTP goes through `io/base.py`** so retries/timeouts stay uniform and
  tests can mock one place.
- **Config via `get_settings()`** — never read `os.environ` directly.
- **Logging via `get_logger(__name__)`** — no bare `print` in library code (scripts may
  print for CLI UX).
- **Keep public signatures stable.** When you replace a baseline with a real model, match
  the existing function/'`predict`' signature so downstream code and the optimiser are
  untouched. Look for `TODO(Mx)` markers that name the milestone.
- **Style:** ruff + black, line length 100. Run `make fmt` before committing.

## Testing strategy

- Unit tests must run **offline**. Mock Octopus/Carbon with `respx` (see
  `tests/test_octopus.py`, `tests/test_carbon.py` for the pattern).
- The optimiser tests (`tests/test_milp.py`) are pure and deterministic — use them as the
  spec: savings ≥ 0 vs baseline, SoC stays within bounds, never charge+discharge together.
- Add a test with every behaviour change. CI runs lint + mypy + pytest.

## Definition of done for a task

1. Code + type hints + docstring; `make lint` and `make test` are green.
2. A test that would fail without your change.
3. If you touched a data contract, `schemas.py` and all call sites updated together.
4. If you finished a ROADMAP item, tick its box in `ROADMAP.md`.
5. No secrets committed; no network calls in unit tests.

## Good first tasks

See `ROADMAP.md`. The highest-value next steps are **M3** (real demand forecaster with
MLflow + backtest vs the seasonal-naive baseline) and **M4** (Elexon price model + battery
arbitrage backtest). Both slot into existing interfaces.
