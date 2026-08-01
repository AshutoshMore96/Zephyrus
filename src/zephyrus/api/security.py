"""API-key auth + a simple in-process rate limiter (project serving, M6).

Both are opt-in and safe by default so local demos and the offline test-suite stay
frictionless: auth is enforced only when ``ZEPHYRUS_API_KEY`` is set, and the rate limit
disables at ``rate_limit_per_min <= 0``. For real multi-instance deployment swap the
in-memory limiter for Redis; the dependency signature stays the same.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

from ..config import get_settings
from ..logging import get_logger

logger = get_logger(__name__)

_WINDOW_S = 60.0
_hits: dict[str, deque[float]] = defaultdict(deque)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Enforce ``X-API-Key`` only when an API key is configured; otherwise allow."""
    expected = get_settings().api_key
    if expected is None:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def rate_limited(request: Request) -> None:
    """Fixed-window per-client rate limit; no-op when ``rate_limit_per_min <= 0``."""
    limit = get_settings().rate_limit_per_min
    if limit <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    bucket = _hits[client]
    while bucket and now - bucket[0] > _WINDOW_S:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded; retry shortly")
    bucket.append(now)


def reset_rate_limits() -> None:
    """Clear all rate-limit counters (used by tests)."""
    _hits.clear()
