# ADR 0001 — Monorepo with forecast-driven optimisation

- **Status:** Accepted
- **Date:** 2026-07-26

## Context

Three related projects — demand forecasting (#1), price forecasting (#4), and a battery
flexibility optimiser (#2) — could ship as three separate repos or one. They share data
contracts (prices, carbon, battery state), the same ingestion clients, and the same
serving/MLOps surface. The optimiser is only compelling when it consumes real forecasts.

## Decision

Build a **single monorepo** organised around **forecast-driven optimisation**: the
forecasters are upstream signal producers whose outputs feed the optimiser through stable,
typed interfaces (`load_kwh` from demand; forecast prices for arbitrage backtests). The
system runs end-to-end from day one using transparent baselines (seasonal-naive demand,
persistence prices), which are swapped for real models later without changing call sites.

## Consequences

- **Positive:** one coherent story for reviewers; shared schemas/clients/tests; the
  optimiser demonstrates value immediately; forecasting work plugs in incrementally
  (ROADMAP M3/M4) behind fixed interfaces.
- **Positive:** mirrors how commercial flexibility platforms are structured (forecast →
  optimise → aggregate → settle).
- **Negative / trade-off:** heavier single repo; ML dependencies are isolated behind the
  `ml` optional-extra so the core stays lightweight.
- **Guardrail:** data contracts change in `schemas.py` first; baseline and model share a
  signature so downstream code is untouched.
