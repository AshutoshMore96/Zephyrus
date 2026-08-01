"""API auth + rate-limiting helpers (M6). Skipped without the ``api`` extra."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi", reason="requires the 'api' extra")

from fastapi import HTTPException  # noqa: E402

from zephyrus.api.security import (  # noqa: E402
    rate_limited,
    require_api_key,
    reset_rate_limits,
)
from zephyrus.config import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_settings_and_limits(monkeypatch):
    reset_rate_limits()
    get_settings.cache_clear()
    yield
    monkeypatch.undo()
    get_settings.cache_clear()
    reset_rate_limits()


def _request(host: str = "1.2.3.4"):
    return SimpleNamespace(client=SimpleNamespace(host=host))


def test_api_key_not_required_when_unset():
    # No key configured -> open (keeps local demos and CI frictionless).
    assert require_api_key(None) is None


def test_api_key_enforced_when_set(monkeypatch):
    monkeypatch.setenv("ZEPHYRUS_API_KEY", "s3cret")
    get_settings.cache_clear()
    assert require_api_key("s3cret") is None
    with pytest.raises(HTTPException) as exc:
        require_api_key("wrong")
    assert exc.value.status_code == 401
    with pytest.raises(HTTPException):
        require_api_key(None)


def test_rate_limit_trips_after_configured_calls(monkeypatch):
    monkeypatch.setenv("ZEPHYRUS_RATE_LIMIT_PER_MIN", "3")
    get_settings.cache_clear()
    req = _request()
    for _ in range(3):
        rate_limited(req)  # first 3 allowed
    with pytest.raises(HTTPException) as exc:
        rate_limited(req)  # 4th over the limit
    assert exc.value.status_code == 429


def test_rate_limit_is_per_client(monkeypatch):
    monkeypatch.setenv("ZEPHYRUS_RATE_LIMIT_PER_MIN", "2")
    get_settings.cache_clear()
    rate_limited(_request("a"))
    rate_limited(_request("a"))
    # A different client has its own budget.
    rate_limited(_request("b"))


def test_rate_limit_disabled_when_zero(monkeypatch):
    monkeypatch.setenv("ZEPHYRUS_RATE_LIMIT_PER_MIN", "0")
    get_settings.cache_clear()
    req = _request()
    for _ in range(50):
        rate_limited(req)  # no cap
