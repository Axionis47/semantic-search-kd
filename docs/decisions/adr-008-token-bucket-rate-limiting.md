# ADR-008: In-Process Token Bucket Rate Limiting per Client IP

**Status:** Accepted  
**Date:** 2024-12-01  
**Deciders:** Project team

## Context

The search API must protect against abuse and overload. Without rate limiting, a single client (malicious or misconfigured) can consume all available compute, starving legitimate users. The rate limiting mechanism must balance protection with simplicity, given the current single-instance deployment model.

Deployment context:
- The API runs on a single Cloud Run instance (auto-scaling is disabled during initial deployment for cost control).
- Each search request triggers query encoding (1ms) and FAISS search (10ms), consuming CPU for approximately 12ms total.
- At full utilization, a single instance handles approximately 80 queries per second.
- The expected legitimate traffic pattern is bursty: a client may send 10-15 rapid queries during active search, then pause.

The rate limiting solution must accommodate legitimate bursts while capping sustained throughput per client.

## Decision

Implement **in-process token bucket rate limiting** keyed by client IP address, with the following parameters:

- **Rate:** 10 tokens per second (sustained throughput limit)
- **Burst capacity:** 20 tokens (maximum burst size)
- **Stale bucket cleanup:** Buckets for inactive IPs are evicted after 5 minutes to prevent memory exhaustion

The implementation lives in the application code (a middleware or decorator) with no external dependencies. Each incoming request consumes one token. If the bucket is empty, the request receives a 429 Too Many Requests response with a Retry-After header.

## Alternatives Considered

### Alternative 1: Redis-based distributed rate limiter
- **Pros:** Works across multiple instances. Shared state ensures global rate limits are enforced regardless of which instance handles the request. Industry standard for distributed rate limiting. Supports more sophisticated algorithms (sliding window, leaky bucket with persistence).
- **Cons:** Introduces an external dependency (Redis instance). Adds network latency to every request (1-5ms round trip to Redis). Requires Redis provisioning, monitoring, and maintenance. Connection failure handling adds complexity: should requests pass or fail when Redis is unreachable? Cost of running a Redis instance for a single-instance API.
- **Why rejected:** The current deployment is a single Cloud Run instance. All requests are handled by one process, so in-process state is sufficient for accurate rate limiting. Adding Redis for a single instance introduces operational complexity and cost with no functional benefit. This decision should be revisited when the deployment scales to multiple instances, as in-process rate limiting cannot coordinate across instances.

### Alternative 2: Cloud Run concurrency limits
- **Pros:** Zero application code. Configured entirely through Cloud Run settings. Cloud Run automatically queues or rejects requests when the concurrency limit is exceeded. Managed by the platform.
- **Cons:** Concurrency limits are global, not per-client. A limit of 80 concurrent requests does not prevent a single client from consuming all 80 slots. No ability to differentiate between clients or apply different limits to different usage tiers. The limit controls concurrent in-flight requests, not request rate: a client sending short requests (12ms each) can achieve very high throughput within a moderate concurrency limit.
- **Why rejected:** Cloud Run concurrency limits protect the instance from overload but do not implement per-client fairness. A single client sending 80 concurrent requests would consume the full capacity while all other clients are rejected. The rate limiter must be per-client to be useful.

### Alternative 3: API gateway (Cloud Endpoints / API Gateway)
- **Pros:** Fully managed rate limiting with per-API-key quotas. No application code changes. Supports sophisticated policies (daily quotas, per-endpoint limits, tiered plans). Built-in authentication and API key management.
- **Cons:** Adds an additional network hop (gateway sits in front of the Cloud Run service). Configuration is through YAML/OpenAPI specs, separate from the application code. More complex deployment pipeline. Requires API key distribution and management for each client. Overkill for the current scale and usage pattern.
- **Why rejected:** The overhead of setting up and managing an API gateway is not justified for a single-instance deployment with a small number of clients. The application-level rate limiter is simpler to implement, test, and modify. If the service grows to require API key management, tiered pricing, or multiple endpoints with different limits, an API gateway would be reconsidered.

## Consequences

### Positive
- Zero external dependencies. The rate limiter is pure application code, running in the same process as the request handler. No network calls, no connection management, no failure modes from external services.
- Token bucket algorithm naturally handles bursty traffic. A client can send 20 rapid requests (exhausting the burst capacity) and then continue at 10 requests/second. This matches the expected usage pattern of interactive search sessions.
- Per-IP keying provides client-level fairness without requiring authentication. Each client gets its own bucket regardless of how many total clients are active.
- Stale bucket cleanup (eviction after 5 minutes of inactivity) bounds memory usage. Even under a distributed denial-of-service scenario with many unique IPs, the memory consumed by buckets is proportional to the number of IPs active within the last 5 minutes, not all-time unique IPs.
- The 429 response with Retry-After header follows HTTP standards, enabling well-behaved clients to automatically back off and retry.

### Negative
- In-process state is lost on instance restart. After a restart, all clients start with full buckets. A client that was rate-limited before the restart gets a fresh allowance. This is acceptable for the current deployment but means rate limiting is not persistent.
- Per-IP limiting can be circumvented by clients using multiple IP addresses (proxies, VPNs, rotating IPs). This is a known limitation of IP-based rate limiting that API key-based approaches would address.
- Single-instance assumption means this solution must be replaced or augmented when scaling to multiple Cloud Run instances. With multiple instances, each instance maintains its own buckets, and a client's effective rate limit becomes N times the configured limit (where N is the instance count).
- The cleanup interval (5 minutes) is a heuristic. Too short and legitimate clients returning after a brief pause lose their bucket history. Too long and memory usage grows during traffic spikes with many unique IPs.

### Trade-offs
- Chose simplicity over distributed correctness. The in-process approach works perfectly for one instance and fails gracefully (over-permissive, not over-restrictive) with multiple instances. This failure mode is acceptable: the worst case is that clients get higher effective limits, not that legitimate clients are incorrectly blocked.
- Chose IP-based identification over API keys. This avoids the overhead of key management but sacrifices the ability to apply per-client policies. Suitable for the current stage where the API does not require authentication.
- The burst capacity of 20 tokens (2x the per-second rate) is calibrated for interactive search. Batch or programmatic clients with different access patterns would benefit from separate rate limit tiers, which the current implementation does not support.
