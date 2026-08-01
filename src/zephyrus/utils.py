"""Small shared helpers (kept dependency-light)."""

from __future__ import annotations

from .schemas import CarbonSlot, HalfHourlyPrice


def align_price_carbon(
    prices: list[HalfHourlyPrice], carbon: list[CarbonSlot]
) -> tuple[list[HalfHourlyPrice], list[CarbonSlot]]:
    """Restrict two half-hourly series to their matching timestamps.

    Octopus and the Carbon Intensity API both publish on :00/:30 boundaries but may
    start/end at different points; the optimiser needs them index-aligned. Matching is
    on ``valid_from`` (both are tz-aware UTC instants).
    """
    carbon_by_start = {c.valid_from: c for c in carbon}
    aligned_p: list[HalfHourlyPrice] = []
    aligned_c: list[CarbonSlot] = []
    for p in prices:
        match = carbon_by_start.get(p.valid_from)
        if match is not None:
            aligned_p.append(p)
            aligned_c.append(match)
    return aligned_p, aligned_c
