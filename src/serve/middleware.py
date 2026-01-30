"""Security middleware for the FastAPI application.

This module provides:
- Rate limiting with token bucket algorithm
- API key authentication
- Request logging and metrics
"""

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

from fastapi import Request, status
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


# =============================================================================
# Rate Limiting
# =============================================================================


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(init=False)
    last_update: float = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)
        self.last_update = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            True if tokens were consumed, False if rate limited.
        """
        now = time.time()
        elapsed = now - self.last_update

        # Refill tokens
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate time until tokens are available.

        Args:
            tokens: Number of tokens needed.

        Returns:
            Seconds until tokens are available.
        """
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.refill_rate


class RateLimiter:
    """Rate limiter using token bucket algorithm per client."""

    def __init__(
        self,
        requests_per_minute: int = 100,
        burst: int = 20,
        enabled: bool = True,
        max_buckets: int = 10000,
    ) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_minute: Allowed requests per minute.
            burst: Maximum burst size.
            enabled: Whether rate limiting is enabled.
            max_buckets: Maximum number of tracked clients to prevent memory exhaustion.
        """
        self.enabled = enabled
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.max_buckets = max_buckets
        self.refill_rate = requests_per_minute / 60.0  # tokens per second
        self.buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()

    def _get_client_key(self, request: Request) -> str:
        """Get unique key for the client.

        Uses X-Forwarded-For if behind proxy, otherwise client host.
        """
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take first IP in chain
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup_old_buckets(self) -> None:
        """Remove stale buckets to prevent memory leaks."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        # Remove buckets that haven't been used in 10 minutes
        stale_threshold = now - 600
        stale_keys = [
            key for key, bucket in self.buckets.items() if bucket.last_update < stale_threshold
        ]
        for key in stale_keys:
            del self.buckets[key]

        self._last_cleanup = now
        if stale_keys:
            logger.debug(f"Cleaned up {len(stale_keys)} stale rate limit buckets")

    def check(self, request: Request) -> tuple[bool, Optional[float]]:
        """Check if request should be allowed.

        Thread-safe via internal lock on bucket access.

        Args:
            request: FastAPI request object.

        Returns:
            Tuple of (allowed, retry_after_seconds).
        """
        if not self.enabled:
            return True, None

        client_key = self._get_client_key(request)

        with self._lock:
            self._cleanup_old_buckets()

            if client_key not in self.buckets:
                # Evict oldest bucket if at capacity
                if len(self.buckets) >= self.max_buckets:
                    oldest_key = min(self.buckets, key=lambda k: self.buckets[k].last_update)
                    del self.buckets[oldest_key]
                self.buckets[client_key] = TokenBucket(
                    capacity=self.burst,
                    refill_rate=self.refill_rate,
                )

            bucket = self.buckets[client_key]

            if bucket.consume():
                return True, None

            retry_after = bucket.time_until_available()
            return False, retry_after


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""

    def __init__(
        self,
        app: Callable,
        requests_per_minute: int = 100,
        burst: int = 20,
        enabled: bool = True,
        exclude_paths: Optional[Set[str]] = None,
    ) -> None:
        super().__init__(app)
        self.limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst=burst,
            enabled=enabled,
        )
        self.exclude_paths = exclude_paths or {"/health", "/metrics", "/"}

    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        """Process request with rate limiting."""
        # Skip rate limiting for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        allowed, retry_after = self.limiter.check(request)

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {self.limiter._get_client_key(request)}"
            )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit exceeded. Try again in {retry_after:.1f} seconds.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        return await call_next(request)


# =============================================================================
# API Key Authentication
# =============================================================================


class APIKeyAuth:
    """API key authentication handler."""

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        api_key_hashes: Optional[List[str]] = None,
        header_name: str = "X-API-Key",
        enabled: bool = False,
    ) -> None:
        """Initialize API key authenticator.

        Args:
            api_keys: List of valid API keys (will be hashed).
            api_key_hashes: List of pre-hashed API keys (for production).
            header_name: Header to check for API key.
            enabled: Whether authentication is enabled.
        """
        import json
        import os

        self.enabled = enabled
        self.header_name = header_name
        # Store hashed keys for security
        self._valid_keys: Set[str] = set()

        # Load from provided keys
        if api_keys:
            for key in api_keys:
                self._valid_keys.add(self._hash_key(key))

        # Load from pre-hashed keys (production use)
        if api_key_hashes:
            for key_hash in api_key_hashes:
                self._valid_keys.add(key_hash)

        # Load from environment variable (JSON array of hashes)
        env_hashes = os.environ.get("SEMANTIC_KD_API_KEY_HASHES")
        if env_hashes:
            try:
                hashes = json.loads(env_hashes)
                for h in hashes:
                    self._valid_keys.add(h)
                logger.info(f"Loaded {len(hashes)} API key hashes from environment")
            except json.JSONDecodeError:
                logger.warning("Failed to parse SEMANTIC_KD_API_KEY_HASHES env var")

    def _hash_key(self, key: str, salt: Optional[str] = None) -> str:
        """Hash an API key with PBKDF2-HMAC-SHA256 for secure storage.

        Args:
            key: The API key to hash.
            salt: Optional salt. If None, uses a fixed derivation for
                  verification against stored unsalted hashes (backward compat).
        """
        if salt:
            return hashlib.pbkdf2_hmac(
                "sha256", key.encode(), salt.encode(), 100_000
            ).hex()
        # Backward-compatible: plain SHA256 for existing keys
        return hashlib.sha256(key.encode()).hexdigest()

    def add_key(self, key: str) -> None:
        """Add a valid API key."""
        self._valid_keys.add(self._hash_key(key))

    def remove_key(self, key: str) -> None:
        """Remove an API key."""
        self._valid_keys.discard(self._hash_key(key))

    def verify(self, request: Request) -> bool:
        """Verify API key from request.

        Args:
            request: FastAPI request object.

        Returns:
            True if authenticated, False otherwise.
        """
        if not self.enabled:
            return True

        api_key = request.headers.get(self.header_name)
        if not api_key:
            return False

        return self._hash_key(api_key) in self._valid_keys


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""

    def __init__(
        self,
        app: Callable,
        api_keys: Optional[List[str]] = None,
        header_name: str = "X-API-Key",
        enabled: bool = False,
        exclude_paths: Optional[Set[str]] = None,
    ) -> None:
        super().__init__(app)
        self.auth = APIKeyAuth(
            api_keys=api_keys,
            header_name=header_name,
            enabled=enabled,
        )
        self.exclude_paths = exclude_paths or {"/health", "/", "/docs", "/openapi.json"}

    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        """Process request with authentication."""
        # Skip auth for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        if not self.auth.verify(request):
            logger.warning(f"Invalid API key from {request.client.host if request.client else 'unknown'}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "INVALID_API_KEY",
                    "message": "Invalid or missing API key",
                },
                headers={"WWW-Authenticate": f'ApiKey realm="API"'},
            )

        return await call_next(request)


# =============================================================================
# Request Logging
# =============================================================================


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging with privacy controls."""

    def __init__(
        self,
        app: Callable,
        log_queries: bool = False,
        log_latencies: bool = True,
        log_headers: bool = False,
    ) -> None:
        super().__init__(app)
        self.log_queries = log_queries
        self.log_latencies = log_latencies
        self.log_headers = log_headers

    def _hash_query(self, query: str) -> str:
        """Hash query for privacy-preserving logging."""
        return hashlib.sha256(query.encode()).hexdigest()[:12]

    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        """Log request details."""
        start_time = time.time()

        # Build log context
        log_context = {
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown",
        }

        response = await call_next(request)

        # Add response info
        latency_ms = (time.time() - start_time) * 1000
        log_context["status_code"] = response.status_code

        if self.log_latencies:
            log_context["latency_ms"] = f"{latency_ms:.2f}"

        # Log based on status code
        if response.status_code >= 500:
            logger.error("Request failed", **log_context)
        elif response.status_code >= 400:
            logger.warning("Request error", **log_context)
        else:
            logger.info("Request completed", **log_context)

        return response


# =============================================================================
# Security Headers
# =============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to responses."""

    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
    }

    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        """Add security headers to response."""
        response = await call_next(request)

        for header, value in self.SECURITY_HEADERS.items():
            response.headers[header] = value

        return response
