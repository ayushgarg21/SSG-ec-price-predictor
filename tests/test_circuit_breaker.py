"""Tests for URA client circuit breaker and retry logic."""

from __future__ import annotations

import time

from src.ingestion.ura_client import CircuitBreaker


class TestCircuitBreaker:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker(threshold=3)
        assert cb.allow_request() is True

    def test_opens_after_threshold(self) -> None:
        cb = CircuitBreaker(threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.allow_request() is False

    def test_resets_on_success(self) -> None:
        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        cb.record_failure()
        assert cb.allow_request() is True  # Only 2 consecutive failures

    def test_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(threshold=2, reset_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        assert cb.allow_request() is False
        time.sleep(0.02)
        assert cb.allow_request() is True  # Reset after timeout
