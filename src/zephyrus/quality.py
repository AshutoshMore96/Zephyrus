"""Data-quality checks for half-hourly frames (project #1, M1).

A tiny Great-Expectations-style layer: each ``expect_*`` returns an :class:`Expectation`
(pass/fail + human detail); :func:`validate` runs a batch and, by default, raises a
:class:`DataQualityError` listing every failure. The specific checks cover the failure
modes that actually bite half-hourly energy data: missing/duplicated slots (gaps),
daylight-saving folds, non-monotonic time, nulls and out-of-range units.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class DataQualityError(ValueError):
    """Raised when one or more data-quality expectations fail."""


@dataclass(frozen=True)
class Expectation:
    """The outcome of a single check."""

    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}: {self.detail}" if self.detail else f"[{mark}] {self.name}"


def find_gaps(index: pd.DatetimeIndex, freq: str = "30min") -> list[pd.Timestamp]:
    """Return timestamps expected on a regular ``freq`` grid but missing from ``index``."""
    if len(index) < 2:
        return []
    expected = pd.date_range(index.min(), index.max(), freq=freq, tz=index.tz)
    return list(expected.difference(index))


def expect_regular_intervals(index: pd.DatetimeIndex, freq: str = "30min") -> Expectation:
    """No missing slots on the regular grid, and no duplicated timestamps."""
    dupes = int(index.duplicated().sum())
    gaps = find_gaps(index, freq)
    ok = not gaps and dupes == 0
    detail = ""
    if not ok:
        detail = f"{len(gaps)} gap(s), {dupes} duplicate(s)"
        if gaps:
            detail += f"; first gap at {gaps[0]}"
    return Expectation("regular_intervals", ok, detail)


def expect_monotonic_increasing(index: pd.DatetimeIndex) -> Expectation:
    """Timestamps strictly increase — a DST fold or unsorted merge breaks this."""
    ok = bool(index.is_monotonic_increasing) and not bool(index.duplicated().any())
    return Expectation("monotonic_increasing", ok, "" if ok else "index not strictly increasing")


def expect_tz_aware_utc(index: pd.DatetimeIndex) -> Expectation:
    """Index must be tz-aware UTC so it aligns with prices/carbon and dodges DST folds."""
    tz = getattr(index, "tz", None)
    ok = tz is not None and str(tz) in {"UTC", "utc"}
    return Expectation("tz_aware_utc", ok, "" if ok else f"tz is {tz!r}, expected UTC")


def expect_no_nulls(df: pd.DataFrame, columns: list[str]) -> Expectation:
    """Named columns contain no nulls."""
    present = [c for c in columns if c in df.columns]
    bad = {c: int(df[c].isna().sum()) for c in present if df[c].isna().any()}
    return Expectation("no_nulls", not bad, "" if not bad else f"nulls: {bad}")


def expect_in_range(df: pd.DataFrame, column: str, low: float, high: float) -> Expectation:
    """Values in ``column`` lie within ``[low, high]`` (unit sanity)."""
    if column not in df.columns:
        return Expectation(f"in_range[{column}]", False, "column missing")
    series = df[column]
    n_bad = int(((series < low) | (series > high)).sum())
    detail = "" if n_bad == 0 else f"{n_bad} value(s) outside [{low}, {high}]"
    return Expectation(f"in_range[{column}]", n_bad == 0, detail)


def validate(expectations: list[Expectation], raise_on_fail: bool = True) -> list[Expectation]:
    """Return the expectations; if any failed and ``raise_on_fail``, raise with details."""
    failures = [e for e in expectations if not e.passed]
    if failures and raise_on_fail:
        raise DataQualityError("; ".join(str(f) for f in failures))
    return expectations


# Standard sanity ranges for the quantities Zephyrus handles (units per schemas.py).
PRICE_RANGE_P = (-50.0, 200.0)  # pence/kWh (Agile can go negative and spike)
CARBON_RANGE = (0.0, 1000.0)  # gCO2/kWh
LOAD_RANGE = (0.0, 100.0)  # kWh per half-hour (generous upper bound)


def check_halfhourly_frame(
    df: pd.DataFrame,
    value_columns: list[str] | None = None,
    ranges: dict[str, tuple[float, float]] | None = None,
    raise_on_fail: bool = True,
) -> list[Expectation]:
    """Run the standard structural + unit checks on a half-hourly, time-indexed frame.

    ``df`` must have a tz-aware UTC :class:`~pandas.DatetimeIndex`. Structural checks
    (regular grid, monotonicity, tz, nulls) always run; range checks run for any column
    named in ``ranges``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataQualityError("frame must be indexed by a DatetimeIndex")
    value_columns = value_columns or list(df.columns)
    checks = [
        expect_tz_aware_utc(df.index),
        expect_monotonic_increasing(df.index),
        expect_regular_intervals(df.index),
        expect_no_nulls(df, value_columns),
    ]
    for column, (low, high) in (ranges or {}).items():
        checks.append(expect_in_range(df, column, low, high))
    return validate(checks, raise_on_fail=raise_on_fail)
