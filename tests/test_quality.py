"""Data-quality expectations (M1)."""

from __future__ import annotations

import pandas as pd
import pytest

from zephyrus.quality import (
    DataQualityError,
    check_halfhourly_frame,
    expect_in_range,
    expect_regular_intervals,
    find_gaps,
)


def _hh_index(n: int = 6):
    return pd.date_range("2026-01-01", periods=n, freq="30min", tz="UTC")


def _good_frame(n: int = 6) -> pd.DataFrame:
    idx = _hh_index(n)
    return pd.DataFrame({"price_p_per_kwh": [10.0] * n, "load_kwh": [0.3] * n}, index=idx)


def test_find_gaps_detects_missing_slot():
    idx = _hh_index(6).delete(3)  # drop one slot -> a gap
    gaps = find_gaps(idx)
    assert len(gaps) == 1


def test_regular_intervals_passes_on_clean_grid():
    assert expect_regular_intervals(_hh_index()).passed


def test_regular_intervals_flags_duplicates():
    idx = _hh_index(4).append(_hh_index(1))  # duplicate first slot
    result = expect_regular_intervals(idx)
    assert not result.passed
    assert "duplicate" in result.detail


def test_in_range_flags_out_of_bounds():
    frame = _good_frame()
    frame.loc[frame.index[0], "price_p_per_kwh"] = 9999.0
    assert not expect_in_range(frame, "price_p_per_kwh", -50, 200).passed


def test_check_halfhourly_frame_raises_with_clear_message():
    frame = _good_frame(6)
    frame = frame.drop(frame.index[2])  # introduce a gap
    with pytest.raises(DataQualityError) as exc:
        check_halfhourly_frame(frame, ranges={"price_p_per_kwh": (-50, 200)})
    assert "regular_intervals" in str(exc.value)


def test_check_halfhourly_frame_passes_clean_frame():
    frame = _good_frame()
    results = check_halfhourly_frame(
        frame, ranges={"price_p_per_kwh": (-50, 200), "load_kwh": (0, 100)}
    )
    assert all(r.passed for r in results)


def test_check_rejects_non_datetime_index():
    with pytest.raises(DataQualityError):
        check_halfhourly_frame(pd.DataFrame({"x": [1, 2, 3]}))
