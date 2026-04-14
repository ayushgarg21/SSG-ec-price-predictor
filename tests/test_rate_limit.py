"""Tests for rate limiter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from src.api.rate_limit import SlidingWindowRateLimiter


def _make_request(ip: str = "127.0.0.1") -> MagicMock:
    req = MagicMock()
    req.client.host = ip
    return req


class TestSlidingWindowRateLimiter:
    def test_allows_within_limit(self) -> None:
        limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
        req = _make_request()
        for _ in range(5):
            limiter.check(req)  # Should not raise

    def test_rejects_over_limit(self) -> None:
        limiter = SlidingWindowRateLimiter(max_requests=3, window_seconds=60)
        req = _make_request()
        for _ in range(3):
            limiter.check(req)
        with pytest.raises(HTTPException) as exc_info:
            limiter.check(req)
        assert exc_info.value.status_code == 429

    def test_separate_ips(self) -> None:
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
        r1 = _make_request("10.0.0.1")
        r2 = _make_request("10.0.0.2")
        for _ in range(2):
            limiter.check(r1)
            limiter.check(r2)
        # r1 is at limit, r2 is at limit
        with pytest.raises(HTTPException):
            limiter.check(r1)
        with pytest.raises(HTTPException):
            limiter.check(r2)
