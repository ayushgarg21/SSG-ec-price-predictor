"""In-memory sliding window rate limiter."""

from __future__ import annotations

import time
from collections import deque

from fastapi import HTTPException, Request


class SlidingWindowRateLimiter:
    """Per-IP sliding window rate limiter.

    Tracks request timestamps per client IP and rejects requests
    that exceed `max_requests` within the `window_seconds` window.
    """

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = {}

    def _cleanup(self, ip: str, now: float) -> None:
        """Remove timestamps older than the window."""
        q = self._requests.get(ip)
        if q:
            cutoff = now - self.window_seconds
            while q and q[0] < cutoff:
                q.popleft()

    def check(self, request: Request) -> None:
        """Check if the request is within rate limits. Raises 429 if not."""
        ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        self._cleanup(ip, now)

        q = self._requests.setdefault(ip, deque())
        if len(q) >= self.max_requests:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds}s.",
            )
        q.append(now)
