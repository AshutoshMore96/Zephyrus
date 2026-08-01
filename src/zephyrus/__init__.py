"""Zephyrus — forecast-driven flexibility optimisation for distributed energy assets.

Zephyrus turns home batteries and EVs into a virtual power plant by combining three
integrated components:

  * demand forecasting          (``zephyrus.forecast.demand``)  — project #1
  * imbalance-price forecasting (``zephyrus.forecast.price``)    — project #4
  * a battery/EV MILP optimiser (``zephyrus.optimise``)          — project #2 (core)

The forecasts are *upstream signals*: they feed the optimiser, which schedules a
battery against live half-hourly prices and carbon intensity, then aggregates many
assets into a virtual power plant.

See ``README.md`` for the architecture diagram and ``ROADMAP.md`` for the delivery plan.
"""

from __future__ import annotations

__version__ = "0.1.0"
