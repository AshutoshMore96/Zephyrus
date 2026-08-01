"""Shared HTTP client: consistent timeouts, retry/back-off and logging.

Every external call routes through here so retry policy is uniform and testable
(the unit tests mock this layer with ``respx`` — no network needed in CI).
"""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import get_settings
from ..logging import get_logger

logger = get_logger(__name__)


class BaseHTTPClient:
    """Thin wrapper around :class:`httpx.Client` with retries on transient errors."""

    def __init__(self, base_url: str, timeout: float | None = None) -> None:
        settings = get_settings()
        self.base_url = base_url.rstrip("/")
        self._retries = settings.http_retries
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout or settings.http_timeout_s,
            headers={"User-Agent": "zephyrus/0.1"},
        )

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET ``path`` and return parsed JSON, retrying transient failures."""

        @retry(
            stop=stop_after_attempt(self._retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
            retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
            reraise=True,
        )
        def _do() -> dict[str, Any]:
            logger.debug("GET %s params=%s", path, params)
            resp = self._client.get(path, params=params)
            resp.raise_for_status()
            return resp.json()

        return _do()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> BaseHTTPClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
