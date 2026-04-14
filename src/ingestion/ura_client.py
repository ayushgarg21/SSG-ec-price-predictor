"""Client for URA Private Residential Property Transactions API.

Uses curl subprocess because URA's bot protection blocks Python HTTP clients.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any

import structlog

logger = structlog.get_logger()

URA_TOKEN_URL = "https://eservice.ura.gov.sg/uraDataService/insertNewToken/v1"
URA_DATA_URL = "https://eservice.ura.gov.sg/uraDataService/invokeUraDS/v1"


class URAClientError(Exception):
    """Raised when the URA API returns an unrecoverable error."""


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open (too many consecutive failures)."""


class CircuitBreaker:
    """Simple circuit breaker: opens after `threshold` consecutive failures,
    resets after `reset_timeout` seconds."""

    def __init__(self, threshold: int = 5, reset_timeout: float = 60.0) -> None:
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._is_open = False

    def record_success(self) -> None:
        self._failure_count = 0
        self._is_open = False

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.threshold:
            self._is_open = True
            logger.error("circuit_breaker_opened", failures=self._failure_count)

    def allow_request(self) -> bool:
        if not self._is_open:
            return True
        if time.monotonic() - self._last_failure_time > self.reset_timeout:
            logger.info("circuit_breaker_half_open")
            self._is_open = False
            self._failure_count = 0
            return True
        return False


def _curl_get(url: str, headers: dict[str, str], timeout: int = 60) -> dict[str, Any]:
    """Execute a GET request via curl and return parsed JSON."""
    cmd = ["curl", "-s", "-f", "--max-time", str(timeout)]
    for key, val in headers.items():
        cmd.extend(["-H", f"{key}: {val}"])
    cmd.append(url)

    result = subprocess.run(cmd, capture_output=True, timeout=timeout + 10)
    if result.returncode != 0:
        raise URAClientError(f"curl failed (exit {result.returncode}): {result.stderr.decode(errors='replace').strip()}")
    if not result.stdout.strip():
        raise URAClientError("Empty response from URA API")
    # URA responses sometimes contain non-UTF-8 chars in project/street names
    text = result.stdout.decode("utf-8", errors="replace")
    return json.loads(text)


class URAClient:
    """Resilient URA API client using curl with retry and circuit breaker."""

    def __init__(
        self,
        access_key: str,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
    ) -> None:
        self.access_key = access_key
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._token: str | None = None
        self._breaker = CircuitBreaker()

    def _get_token(self) -> str:
        """Fetch a fresh daily token from URA."""
        data = _curl_get(URA_TOKEN_URL, {"AccessKey": self.access_key}, timeout=30)
        token = data.get("Result")
        if not token:
            raise URAClientError(f"Failed to obtain URA token: {data}")
        logger.info("ura_token_obtained")
        self._token = token
        return token

    @property
    def token(self) -> str:
        if self._token is None:
            self._get_token()
        return self._token  # type: ignore[return-value]

    def _headers(self) -> dict[str, str]:
        return {"AccessKey": self.access_key, "Token": self.token}

    def _request_with_retry(self, batch: int) -> list[dict[str, Any]]:
        """Execute a single batch request with exponential backoff retry."""
        if not self._breaker.allow_request():
            raise CircuitOpenError("Circuit breaker is open — URA API unavailable")

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                url = f"{URA_DATA_URL}?service=PMI_Resi_Transaction&batch={batch}"
                data = _curl_get(url, self._headers(), timeout=120)
                self._breaker.record_success()
                return data.get("Result", [])
            except (URAClientError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                last_exc = exc
                self._breaker.record_failure()
                if attempt < self.max_retries:
                    wait = self.retry_backoff * (2 ** (attempt - 1))
                    logger.warning(
                        "ura_request_retry",
                        batch=batch,
                        attempt=attempt,
                        wait_seconds=wait,
                        error=str(exc),
                    )
                    time.sleep(wait)

        raise URAClientError(
            f"Failed to fetch batch {batch} after {self.max_retries} retries"
        ) from last_exc

    def fetch_transactions(self, batch: int = 1) -> list[dict[str, Any]]:
        """Fetch a single batch of private residential transactions."""
        results = self._request_with_retry(batch)
        logger.info("ura_batch_fetched", batch=batch, projects=len(results))
        return results

    def fetch_all_transactions(self) -> list[dict[str, Any]]:
        """Fetch all batches (1-4) and merge results."""
        all_results: list[dict[str, Any]] = []
        for batch in range(1, 5):
            try:
                results = self.fetch_transactions(batch)
                if not results:
                    break
                all_results.extend(results)
            except (URAClientError, CircuitOpenError) as exc:
                logger.warning("ura_batch_skipped", batch=batch, reason=str(exc))
                break
        logger.info("ura_total_fetched", total_projects=len(all_results))
        return all_results
