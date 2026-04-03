# API Reference

Complete reference for the Semantic Search API (v1.1.0).

## Base URL and Versioning

The API is served at the root path with no version prefix. The current version is `1.1.0`, reported in the root endpoint and health check responses.

- **Local development:** `http://localhost:8080`
- **Production (Cloud Run):** your deployed Cloud Run URL

## Authentication

Authentication is controlled by the `auth.enabled` configuration flag. When enabled, every request must include a valid API key in the `X-API-Key` header.

**Generating keys:**

```bash
python scripts/manage_api_keys.py generate --name "client-app-1"
```

**Using keys:**

```bash
curl -H "X-API-Key: sk_live_abc123..." http://localhost:8080/search ...
```

Keys are stored in `./artifacts/api_keys/keys.json` locally and in GCP Secret Manager for production deployments. Use `manage_api_keys.py list` to view active keys and `manage_api_keys.py revoke --key-id <id>` to revoke them.

When auth is enabled and a request is missing or has an invalid key, the API returns:

```json
{
  "error": "Invalid or missing API key",
  "detail": null
}
```

with HTTP status `401`.

## Rate Limiting

Rate limiting is enabled by default.

| Parameter | Default |
|---|---|
| Requests per minute | 100 |
| Burst allowance | 20 |

When the limit is exceeded, the API returns HTTP `429`:

```json
{
  "error": "Rate limit exceeded: 100 requests per 60 seconds",
  "detail": null
}
```

---

## Endpoints

### GET /

Service information and status.

**Response body:**

| Field | Type | Description |
|---|---|---|
| `service` | string | Service name (`"Semantic Search API"`) |
| `version` | string | API version (e.g. `"1.1.0"`) |
| `status` | string | `"running"` or `"starting"` |
| `environment` | string | `"development"`, `"staging"`, or `"production"` |

**Example request:**

```bash
curl http://localhost:8080/
```

**Example response:**

```json
{
  "service": "Semantic Search API",
  "version": "1.1.0",
  "status": "running",
  "environment": "development"
}
```

---

### GET /health

Health check endpoint for load balancers and orchestrators.

**Response body (`HealthResponse`):**

| Field | Type | Description |
|---|---|---|
| `status` | string | `"healthy"` or `"unhealthy"` |
| `model_loaded` | boolean | Whether the student model is loaded |
| `index_loaded` | boolean | Whether the FAISS index is loaded |
| `index_size` | integer | Number of documents in the index |
| `version` | string | Service version |

**Example request:**

```bash
curl http://localhost:8080/health
```

**Example response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "index_loaded": true,
  "index_size": 10000,
  "version": "1.1.0"
}
```

---

### GET /ready

Kubernetes readiness probe. Returns `200` when the service is ready to accept traffic.

**Response body (when ready):**

```json
{ "ready": true }
```

**Error response (when not ready):**

Returns HTTP `503`:

```json
{
  "error": "Service not ready",
  "detail": null
}
```

**Example request:**

```bash
curl http://localhost:8080/ready
```

---

### GET /live

Kubernetes liveness probe. Always returns `200` if the process is alive.

**Response body:**

```json
{ "alive": true }
```

**Example request:**

```bash
curl http://localhost:8080/live
```

---

### POST /search

Search for semantically similar documents. Encodes the query with the student bi-encoder, retrieves candidates from the FAISS index, and optionally reranks with the teacher cross-encoder.

**Request body (`SearchRequest`):**

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | (required) | Search query text. Min 1, max 1000 characters. |
| `k` | integer | `10` | Number of results to return. Range: 1-100. |
| `rerank` | boolean | `false` | Whether to rerank results with the teacher model. |
| `rerank_top_k` | integer | `50` | Number of candidates to retrieve before reranking. Range: 1-200. |

**Response body (`SearchResponse`):**

| Field | Type | Description |
|---|---|---|
| `query` | string | The original query text |
| `results` | array of `SearchResult` | Ranked list of results |
| `total_results` | integer | Number of results returned |
| `reranked` | boolean | Whether results were reranked by the teacher |
| `latency_ms` | float | End-to-end latency in milliseconds |

**`SearchResult` object:**

| Field | Type | Description |
|---|---|---|
| `doc_id` | string | Document identifier |
| `text` | string | Document text content |
| `score` | float | Relevance score (higher is better) |
| `rank` | integer | Rank position, 1-indexed |

**Example request:**

```bash
curl -X POST http://localhost:8080/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "k": 5,
    "rerank": true,
    "rerank_top_k": 50
  }'
```

**Example response:**

```json
{
  "query": "What is machine learning?",
  "results": [
    {
      "doc_id": "doc_123",
      "text": "Machine learning is a subset of AI...",
      "score": 0.95,
      "rank": 1
    },
    {
      "doc_id": "doc_456",
      "text": "ML algorithms learn patterns from data...",
      "score": 0.87,
      "rank": 2
    }
  ],
  "total_results": 5,
  "reranked": true,
  "latency_ms": 42.3
}
```

**Error responses:**

| Status | Condition |
|---|---|
| `422` | Validation error (query too long, k out of range) |
| `503` | Student model or index not loaded |
| `500` | Internal search failure |

---

### POST /encode

Encode texts into dense vector embeddings using the student bi-encoder.

**Request body (`EncodeRequest`):**

| Field | Type | Default | Description |
|---|---|---|---|
| `texts` | array of string | (required) | List of texts to encode. Min 1, max 100 items. |
| `normalize` | boolean | `true` | Whether to L2-normalize the embeddings. |

**Response body (`EncodeResponse`):**

| Field | Type | Description |
|---|---|---|
| `embeddings` | array of array of float | List of embedding vectors |
| `dimension` | integer | Embedding dimensionality (e.g. 384) |
| `num_texts` | integer | Number of texts encoded |
| `latency_ms` | float | Encoding latency in milliseconds |

**Example request:**

```bash
curl -X POST http://localhost:8080/encode \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["What is machine learning?", "How does AI work?"],
    "normalize": true
  }'
```

**Example response:**

```json
{
  "embeddings": [[0.1, 0.2, 0.3, "..."], [0.4, 0.5, 0.6, "..."]],
  "dimension": 384,
  "num_texts": 2,
  "latency_ms": 5.2
}
```

**Error responses:**

| Status | Condition |
|---|---|
| `422` | Validation error (empty texts list, exceeds 100 items) |
| `503` | Model not loaded |
| `500` | Encoding failure |

---

### POST /index/load

Load a FAISS index from disk at runtime. If the index directory contains a `texts.json` file, document texts are loaded alongside the index for use in search responses.

**Request body:**

| Parameter | Type | Description |
|---|---|---|
| `index_path` | string (query param) | Path to the directory containing the FAISS index files. |

**Response body:**

| Field | Type | Description |
|---|---|---|
| `status` | string | `"loaded"` |
| `index_path` | string | Resolved path to the loaded index |
| `num_documents` | integer | Number of documents in the index |

**Example request:**

```bash
curl -X POST "http://localhost:8080/index/load?index_path=./artifacts/indexes/20251020-1430_a3f2b1c"
```

**Example response:**

```json
{
  "status": "loaded",
  "index_path": "artifacts/indexes/20251020-1430_a3f2b1c",
  "num_documents": 8841823
}
```

**Error responses:**

| Status | Condition |
|---|---|
| `404` | Index directory not found at the specified path |
| `500` | Failed to load index (corrupt files, dimension mismatch) |

---

## Error Response Format

All errors follow a consistent JSON structure:

```json
{
  "error": "Human-readable error message",
  "detail": "Additional context (null in production)"
}
```

The `detail` field is populated in development and staging environments but set to `null` in production to avoid leaking internal information.

## Status Codes Reference

| Code | Meaning | When returned |
|---|---|---|
| `200` | Success | All successful requests |
| `401` | Unauthorized | Missing or invalid API key (when auth is enabled) |
| `404` | Not Found | Index path does not exist (`/index/load`) |
| `422` | Validation Error | Request body fails Pydantic validation |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Unhandled exception during processing |
| `503` | Service Unavailable | Model or index not loaded, service not ready |
