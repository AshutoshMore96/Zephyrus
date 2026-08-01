"""Fetch a live snapshot of Agile prices + carbon forecast into ``data/raw/``.

Handy for offline development and notebook exploration::

    python scripts/fetch_sample_data.py --hours 48 --region C
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import typer

from zephyrus.config import get_settings
from zephyrus.io.carbon import CarbonIntensityClient
from zephyrus.io.octopus import OctopusClient


def main(hours: int = 48, region: str = "C") -> None:
    out = get_settings().data_dir / "raw"
    out.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    prices = OctopusClient().get_agile_prices(
        start, start + timedelta(hours=hours), gsp_group=region
    )
    carbon = CarbonIntensityClient().get_forecast_48h(start)

    (out / "agile_prices.json").write_text(
        json.dumps([p.model_dump(mode="json") for p in prices], indent=2)
    )
    (out / "carbon_forecast.json").write_text(
        json.dumps([c.model_dump(mode="json") for c in carbon], indent=2)
    )
    typer.echo(f"Saved {len(prices)} price slots and {len(carbon)} carbon slots to {out}")


if __name__ == "__main__":
    typer.run(main)
