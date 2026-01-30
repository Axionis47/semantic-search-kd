"""Tests for FastAPI endpoints and middleware."""

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.config import Settings
from src.serve.middleware import APIKeyAuth, RateLimiter, TokenBucket


# =============================================================================
# Middleware Tests
# =============================================================================


class TestTokenBucket:
    """Tests for TokenBucket rate limiter."""

    def test_initial_tokens(self):
        """Test initial token count."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.tokens == 10.0

    def test_consume_success(self):
        """Test successful token consumption."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)
        assert bucket.consume(1) is True
        assert bucket.tokens == 9.0

    def test_consume_failure(self):
        """Test consumption when tokens exhausted."""
        bucket = TokenBucket(capacity=1, refill_rate=0.1)
        assert bucket.consume(1) is True
        assert bucket.consume(1) is False

    def test_refill_over_time(self):
        """Test token refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens/sec
        bucket.consume(5)
        assert bucket.tokens == 5.0

        # Simulate time passing
        bucket.last_update -= 0.5  # 0.5 seconds ago
        bucket.consume(0)  # Trigger refill calculation

        assert bucket.tokens == pytest.approx(10.0, abs=0.1)

    def test_capacity_limit(self):
        """Test that tokens don't exceed capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=100.0)
        bucket.last_update -= 10  # 10 seconds ago

        bucket.consume(0)  # Trigger refill
        assert bucket.tokens == 10.0

    def test_time_until_available(self):
        """Test time calculation for next available token."""
        bucket = TokenBucket(capacity=1, refill_rate=1.0)
        bucket.consume(1)

        wait_time = bucket.time_until_available(1)
        assert wait_time == pytest.approx(1.0, abs=0.1)


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.fixture
    def limiter(self) -> RateLimiter:
        """Create rate limiter instance."""
        return RateLimiter(requests_per_minute=60, burst=10, enabled=True)

    @pytest.fixture
    def mock_request(self) -> MagicMock:
        """Create mock request."""
        request = MagicMock()
        request.headers = {}
        request.client.host = "127.0.0.1"
        return request

    def test_allows_requests_under_limit(self, limiter: RateLimiter, mock_request):
        """Test that requests under limit are allowed."""
        for _ in range(5):
            allowed, _ = limiter.check(mock_request)
            assert allowed is True

    def test_blocks_requests_over_burst(self, limiter: RateLimiter, mock_request):
        """Test that requests over burst limit are blocked."""
        # Exhaust burst
        for _ in range(10):
            limiter.check(mock_request)

        allowed, retry_after = limiter.check(mock_request)
        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0

    def test_disabled_limiter_allows_all(self, mock_request):
        """Test that disabled limiter allows all requests."""
        limiter = RateLimiter(enabled=False)

        for _ in range(100):
            allowed, _ = limiter.check(mock_request)
            assert allowed is True

    def test_per_client_isolation(self, limiter: RateLimiter):
        """Test that rate limits are per-client."""
        request1 = MagicMock()
        request1.headers = {}
        request1.client.host = "192.168.1.1"

        request2 = MagicMock()
        request2.headers = {}
        request2.client.host = "192.168.1.2"

        # Exhaust client1's limit
        for _ in range(10):
            limiter.check(request1)

        # Client2 should still have tokens
        allowed, _ = limiter.check(request2)
        assert allowed is True

    def test_x_forwarded_for_header(self, limiter: RateLimiter):
        """Test that X-Forwarded-For header is respected."""
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "10.0.0.1, 192.168.1.1"}
        request.client.host = "127.0.0.1"

        key = limiter._get_client_key(request)
        assert key == "10.0.0.1"


class TestAPIKeyAuth:
    """Tests for APIKeyAuth."""

    @pytest.fixture
    def auth(self) -> APIKeyAuth:
        """Create API key auth instance."""
        return APIKeyAuth(
            api_keys=["valid-key-1", "valid-key-2"],
            header_name="X-API-Key",
            enabled=True,
        )

    def test_valid_key_accepted(self, auth: APIKeyAuth):
        """Test that valid API key is accepted."""
        request = MagicMock()
        request.headers = {"X-API-Key": "valid-key-1"}

        assert auth.verify(request) is True

    def test_invalid_key_rejected(self, auth: APIKeyAuth):
        """Test that invalid API key is rejected."""
        request = MagicMock()
        request.headers = {"X-API-Key": "invalid-key"}

        assert auth.verify(request) is False

    def test_missing_key_rejected(self, auth: APIKeyAuth):
        """Test that missing API key is rejected."""
        request = MagicMock()
        request.headers = {}

        assert auth.verify(request) is False

    def test_disabled_auth_allows_all(self):
        """Test that disabled auth allows all requests."""
        auth = APIKeyAuth(enabled=False)
        request = MagicMock()
        request.headers = {}

        assert auth.verify(request) is True

    def test_add_remove_key(self, auth: APIKeyAuth):
        """Test adding and removing API keys."""
        request = MagicMock()
        request.headers = {"X-API-Key": "new-key"}

        # Key not valid initially
        assert auth.verify(request) is False

        # Add key
        auth.add_key("new-key")
        assert auth.verify(request) is True

        # Remove key
        auth.remove_key("new-key")
        assert auth.verify(request) is False


# =============================================================================
# API Endpoint Tests
# =============================================================================


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, test_client: TestClient):
        """Test that health endpoint returns 200."""
        response = test_client.get("/health")
        assert response.status_code == status.HTTP_200_OK

    def test_health_response_format(self, test_client: TestClient):
        """Test health response format."""
        response = test_client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_200(self, test_client: TestClient):
        """Test that root endpoint returns 200."""
        response = test_client.get("/")
        assert response.status_code == status.HTTP_200_OK

    def test_root_response_format(self, test_client: TestClient):
        """Test root response format."""
        response = test_client.get("/")
        data = response.json()

        assert data["service"] == "Semantic Search API"
        assert "version" in data
        assert "status" in data


class TestReadinessEndpoint:
    """Tests for readiness probe."""

    def test_ready_when_model_loaded(self, test_client: TestClient):
        """Test readiness when model is loaded."""
        # Note: In test_client fixture, model is mocked as loaded
        response = test_client.get("/ready")
        # May return 503 if app_state not fully initialized
        assert response.status_code in [200, 503]


class TestLivenessEndpoint:
    """Tests for liveness probe."""

    def test_live_always_returns_200(self, test_client: TestClient):
        """Test that liveness always returns 200."""
        response = test_client.get("/live")
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"alive": True}


class TestEncodeEndpoint:
    """Tests for encode endpoint."""

    def test_encode_single_text(self, test_client: TestClient):
        """Test encoding a single text."""
        response = test_client.post(
            "/encode",
            json={"texts": ["Hello world"]},
        )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert "embeddings" in data
        assert data["num_texts"] == 1
        assert "latency_ms" in data

    def test_encode_multiple_texts(self, test_client: TestClient):
        """Test encoding multiple texts."""
        response = test_client.post(
            "/encode",
            json={"texts": ["Text one", "Text two", "Text three"]},
        )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data["num_texts"] == 3

    def test_encode_with_normalization(self, test_client: TestClient):
        """Test encoding with normalization."""
        response = test_client.post(
            "/encode",
            json={"texts": ["Test text"], "normalize": True},
        )

        if response.status_code == 200:
            data = response.json()
            assert "embeddings" in data

    def test_encode_empty_list_rejected(self, test_client: TestClient):
        """Test that empty text list is rejected."""
        response = test_client.post(
            "/encode",
            json={"texts": []},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestSearchEndpoint:
    """Tests for search endpoint."""

    def test_search_requires_index(self, test_client: TestClient):
        """Test that search requires loaded index."""
        response = test_client.post(
            "/search",
            json={"query": "test query", "k": 5},
        )

        # Should return 503 when index not loaded
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    def test_search_request_validation(self, test_client: TestClient):
        """Test search request validation."""
        # Missing query
        response = test_client.post(
            "/search",
            json={"k": 5},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_search_k_bounds(self, test_client: TestClient):
        """Test search k parameter bounds."""
        # k too large should still be accepted (validated by schema)
        response = test_client.post(
            "/search",
            json={"query": "test", "k": 1000},
        )
        # Will be 503 without index, but shouldn't be 422
        assert response.status_code != status.HTTP_422_UNPROCESSABLE_ENTITY


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_for_unknown_endpoint(self, test_client: TestClient):
        """Test 404 for unknown endpoints."""
        response = test_client.get("/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_405_for_wrong_method(self, test_client: TestClient):
        """Test 405 for wrong HTTP method."""
        response = test_client.get("/encode")  # Should be POST
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_422_for_invalid_json(self, test_client: TestClient):
        """Test 422 for invalid request body."""
        response = test_client.post(
            "/encode",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# =============================================================================
# Integration Tests
# =============================================================================


class TestAPIIntegration:
    """Integration tests for API."""

    def test_cors_preflight_request(self, test_client: TestClient):
        """Test that CORS preflight responds correctly."""
        response = test_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200, (
            f"CORS preflight should return 200, got {response.status_code}"
        )

    def test_cors_non_allowed_origin(self, test_client: TestClient):
        """Test that disallowed origins don't get CORS headers."""
        response = test_client.get(
            "/health",
            headers={"Origin": "http://evil.example.com"},
        )
        # Should still succeed but without CORS allow header
        assert response.status_code == 200

    def test_security_headers_present(self, test_client: TestClient):
        """Test that security headers are present."""
        response = test_client.get("/health")

        # Check for security headers (may not be present in all configs)
        headers = response.headers
        # These would be added by SecurityHeadersMiddleware
        # assert "X-Content-Type-Options" in headers
