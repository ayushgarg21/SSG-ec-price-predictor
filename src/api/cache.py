"""Simple in-memory TTL cache for prediction results."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any


class PredictionCache:
    """LRU-like cache with TTL eviction for prediction responses.

    Keys are deterministic hashes of the input features,
    so identical requests within the TTL window return cached results.
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 10_000) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._store: dict[str, tuple[float, Any]] = {}

    @staticmethod
    def _make_key(features: dict[str, Any]) -> str:
        raw = json.dumps(features, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, features: dict[str, Any]) -> Any | None:
        key = self._make_key(features)
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.monotonic() - ts > self.ttl_seconds:
            del self._store[key]
            return None
        return value

    def set(self, features: dict[str, Any], value: Any) -> None:
        if len(self._store) >= self.max_size:
            # Evict oldest entry
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest_key]
        key = self._make_key(features)
        self._store[key] = (time.monotonic(), value)

    @property
    def size(self) -> int:
        return len(self._store)
