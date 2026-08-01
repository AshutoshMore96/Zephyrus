"""A tiny file-based model registry (project ops, M7).

Records each trained model version's metrics + params as JSON under ``data/registry`` and
lets serving pick the best version by a metric. Deliberately dependency-free (works with
or without MLflow); MLflow's own registry can drop in later behind the same interface.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .config import get_settings
from .logging import get_logger
from .schemas import ModelVersion

logger = get_logger(__name__)


class ModelRegistry:
    """Append-only, versioned store of model metrics keyed by model name."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else get_settings().data_dir / "registry"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _dir(self, name: str) -> Path:
        path = self.base_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def versions(self, name: str) -> list[ModelVersion]:
        """All registered versions for ``name``, oldest first."""
        files = sorted(self._dir(name).glob("v*.json"))
        return [ModelVersion.model_validate_json(f.read_text()) for f in files]

    def register(
        self,
        name: str,
        metrics: dict[str, float],
        primary_metric: str,
        params: dict[str, str] | None = None,
    ) -> ModelVersion:
        """Record a new version (auto-incremented) and return it."""
        if primary_metric not in metrics:
            raise ValueError(f"primary_metric {primary_metric!r} not in metrics")
        version = len(self.versions(name)) + 1
        entry = ModelVersion(
            name=name,
            version=version,
            primary_metric=primary_metric,
            metrics=metrics,
            params=params or {},
            created_at=datetime.now(UTC),
        )
        (self._dir(name) / f"v{version:04d}.json").write_text(
            json.dumps(entry.model_dump(mode="json"), indent=2)
        )
        logger.info(
            "Registered %s v%d (%s=%.4f)", name, version, primary_metric, metrics[primary_metric]
        )
        return entry

    def best(self, name: str, lower_is_better: bool = True) -> ModelVersion:
        """Return the version with the best ``primary_metric``."""
        versions = self.versions(name)
        if not versions:
            raise FileNotFoundError(f"No registered versions for {name!r}")
        key = lambda v: v.metrics[v.primary_metric]  # noqa: E731
        return min(versions, key=key) if lower_is_better else max(versions, key=key)

    def latest(self, name: str) -> ModelVersion:
        """Return the most recently registered version."""
        versions = self.versions(name)
        if not versions:
            raise FileNotFoundError(f"No registered versions for {name!r}")
        return versions[-1]
