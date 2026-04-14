"""Tests for prediction cache."""

from __future__ import annotations

import time

from src.api.cache import PredictionCache


class TestPredictionCache:
    def test_set_and_get(self) -> None:
        cache = PredictionCache(ttl_seconds=60)
        features = {"district": 19, "area_sqm": 95.0}
        cache.set(features, 850000.0)
        assert cache.get(features) == 850000.0

    def test_cache_miss(self) -> None:
        cache = PredictionCache(ttl_seconds=60)
        assert cache.get({"district": 99}) is None

    def test_ttl_expiry(self) -> None:
        cache = PredictionCache(ttl_seconds=0)  # Immediate expiry
        features = {"district": 19}
        cache.set(features, 850000.0)
        time.sleep(0.01)
        assert cache.get(features) is None

    def test_max_size_eviction(self) -> None:
        cache = PredictionCache(ttl_seconds=60, max_size=2)
        cache.set({"a": 1}, 100)
        cache.set({"b": 2}, 200)
        cache.set({"c": 3}, 300)  # Should evict oldest
        assert cache.size == 2

    def test_deterministic_keys(self) -> None:
        cache = PredictionCache(ttl_seconds=60)
        f1 = {"district": 19, "area": 95.0}
        f2 = {"area": 95.0, "district": 19}  # Same keys, different order
        cache.set(f1, 850000.0)
        assert cache.get(f2) == 850000.0  # Should hit due to sorted keys
