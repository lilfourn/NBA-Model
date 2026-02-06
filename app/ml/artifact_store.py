"""DB-backed model artifact storage.

Saves and loads serialized model artifacts (joblib, pt, json) to/from the
``model_artifacts`` Postgres table.  A local file cache avoids hitting the DB
on every scoring request.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

_CACHE_TTL_SECONDS = float(os.getenv("MODEL_CACHE_TTL_SECONDS", "600"))  # 10 min
_cache_dir: Path | None = None
_cache_lock = threading.Lock()
# In-memory index: model_name -> (written_ts, local_path)
_cache_index: dict[str, tuple[float, Path]] = {}


def _get_cache_dir() -> Path:
    global _cache_dir
    if _cache_dir is None:
        _cache_dir = Path(tempfile.gettempdir()) / "nba_model_cache"
        _cache_dir.mkdir(parents=True, exist_ok=True)
    return _cache_dir


def save_artifact(
    engine: Engine,
    *,
    model_name: str,
    data: bytes,
    artifact_format: str,
) -> str:
    """Upload a model artifact to the DB.  Returns the new row's UUID."""
    row_id = str(uuid4())
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO model_artifacts (id, model_name, created_at, artifact_data, artifact_format, size_bytes) "
                "VALUES (:id, :model_name, :created_at, :artifact_data, :artifact_format, :size_bytes)"
            ),
            {
                "id": row_id,
                "model_name": model_name,
                "created_at": now,
                "artifact_data": data,
                "artifact_format": artifact_format,
                "size_bytes": len(data),
            },
        )
    # Prune old artifacts (keep last 5 per model_name)
    _prune_old_artifacts(engine, model_name, keep=5)
    return row_id


def _prune_old_artifacts(engine: Engine, model_name: str, *, keep: int = 5) -> None:
    """Delete all but the most recent *keep* artifacts for a given model_name."""
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "DELETE FROM model_artifacts WHERE id IN ("
                    "  SELECT id FROM model_artifacts"
                    "  WHERE model_name = :model_name"
                    "  ORDER BY created_at DESC"
                    "  OFFSET :keep"
                    ")"
                ),
                {"model_name": model_name, "keep": keep},
            )
    except Exception:  # noqa: BLE001
        pass


def load_latest_artifact(engine: Engine, model_name: str) -> bytes | None:
    """Return raw bytes of the most recent artifact for *model_name*, or None."""
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT artifact_data FROM model_artifacts "
                "WHERE model_name = :model_name "
                "ORDER BY created_at DESC LIMIT 1"
            ),
            {"model_name": model_name},
        ).fetchone()
    if row is None:
        return None
    return bytes(row[0])


def load_latest_artifact_as_file(
    engine: Engine,
    model_name: str,
    *,
    suffix: str = "",
) -> Path | None:
    """Load the latest artifact and write it to a local cached file.

    Returns the local ``Path`` or ``None`` if no artifact exists.
    Uses a TTL-based cache so repeated calls within the window skip the DB.
    """
    now = time.monotonic()
    with _cache_lock:
        entry = _cache_index.get(model_name)
        if entry is not None:
            ts, cached_path = entry
            if (now - ts) < _CACHE_TTL_SECONDS and cached_path.exists():
                return cached_path

    data = load_latest_artifact(engine, model_name)
    if data is None:
        return None

    cache_dir = _get_cache_dir()
    # Deterministic filename so we overwrite stale versions
    name_hash = hashlib.md5(model_name.encode()).hexdigest()[:8]
    filename = f"{model_name}_{name_hash}{suffix}"
    local_path = cache_dir / filename
    local_path.write_bytes(data)

    with _cache_lock:
        _cache_index[model_name] = (now, local_path)

    return local_path


def upload_file(engine: Engine, *, model_name: str, file_path: str | Path) -> str:
    """Convenience: read a local file and upload it as an artifact."""
    p = Path(file_path)
    data = p.read_bytes()
    suffix = p.suffix.lstrip(".")
    fmt_map = {"joblib": "joblib", "pt": "pt", "json": "json"}
    artifact_format = fmt_map.get(suffix, suffix or "bin")
    return save_artifact(engine, model_name=model_name, data=data, artifact_format=artifact_format)
