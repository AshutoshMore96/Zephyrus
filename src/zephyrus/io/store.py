"""Local parquet snapshot store with a tiny TTL cache (project #1, M1).

Persists half-hourly frames to ``data/`` as parquet so development and backtests can run
offline, and so repeated calls within a TTL reuse the snapshot instead of hitting the
free APIs. Pydantic model lists convert to/from tidy, time-indexed frames via the helpers
here, keeping the ``schemas.py`` contracts as the single source of truth.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel

from ..config import get_settings
from ..logging import get_logger
from ..schemas import OptimisationResult

logger = get_logger(__name__)

M = TypeVar("M", bound=BaseModel)


def _schedules_dir() -> Path:
    path = get_settings().data_dir / "schedules"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_result(name: str, result: OptimisationResult, base_dir: Path | None = None) -> Path:
    """Persist an optimisation result (schedule + savings) as JSON (M6)."""
    directory = Path(base_dir) if base_dir else _schedules_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(result.model_dump(mode="json"), indent=2))
    logger.info("Persisted schedule %s (%d slots) -> %s", name, len(result.slots), path)
    return path


def load_result(name: str, base_dir: Path | None = None) -> OptimisationResult:
    """Load a previously persisted optimisation result by name."""
    directory = Path(base_dir) if base_dir else _schedules_dir()
    path = directory / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No schedule named {name!r} at {path}")
    return OptimisationResult.model_validate_json(path.read_text())


def models_to_frame(models: Sequence[BaseModel], index: str = "valid_from") -> pd.DataFrame:
    """Tidy time-indexed frame from a list of pydantic models (computed fields included)."""
    rows = [m.model_dump() for m in models]
    frame = pd.DataFrame(rows)
    if index in frame.columns:
        frame[index] = pd.to_datetime(frame[index], utc=True)
        frame = frame.set_index(index).sort_index()
    return frame


def frame_to_models(frame: pd.DataFrame, model: type[M], index: str = "valid_from") -> list[M]:
    """Reconstruct a list of pydantic models from a time-indexed frame."""
    reset = frame.reset_index()
    if index not in reset.columns and frame.index.name == index:
        reset = reset.rename(columns={"index": index})
    fields = set(model.model_fields)
    records = reset.to_dict(orient="records")
    return [model(**{k: v for k, v in rec.items() if k in fields}) for rec in records]


class SnapshotStore:
    """Read/write named parquet snapshots under a base directory, with a TTL cache."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else get_settings().data_dir / "snapshots"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        return self.base_dir / f"{name}.parquet"

    def exists(self, name: str) -> bool:
        return self.path_for(name).exists()

    def age_seconds(self, name: str) -> float:
        """Seconds since the snapshot was written; ``inf`` if it does not exist."""
        path = self.path_for(name)
        return time.time() - path.stat().st_mtime if path.exists() else float("inf")

    def save(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.path_for(name)
        frame.to_parquet(path)
        logger.info("Saved snapshot %s (%d rows) -> %s", name, len(frame), path)
        return path

    def load(self, name: str) -> pd.DataFrame:
        path = self.path_for(name)
        if not path.exists():
            raise FileNotFoundError(f"No snapshot named {name!r} at {path}")
        return pd.read_parquet(path)

    def cached_or_build(
        self, name: str, builder: Callable[[], pd.DataFrame], ttl_seconds: float = 3600.0
    ) -> pd.DataFrame:
        """Return a fresh-enough cached snapshot, else rebuild via ``builder`` and save it."""
        if self.exists(name) and self.age_seconds(name) <= ttl_seconds:
            logger.info(
                "Snapshot %s is fresh (%.0fs old) — using cache", name, self.age_seconds(name)
            )
            return self.load(name)
        frame = builder()
        self.save(name, frame)
        return frame
