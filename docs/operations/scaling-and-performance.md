# Scaling and Performance

This document covers scaling strategies, FAISS tuning guidance, caching, batch optimization, and memory budgeting for the semantic-kd service.

---

## Cloud Run Scaling Configuration

The service is deployed on Google Cloud Run. Scaling parameters are configured in `service.yaml` under the `gcp` section.

### Current Configuration

```yaml
gcp:
  memory: "32Gi"
  cpu: "8"
  max_instances: 100
  min_instances: 0
  concurrency: 80
  timeout: 300  # seconds
```

### Parameter Guide

| Parameter | Current Value | Description | Tuning Notes |
|-----------|--------------|-------------|--------------|
| `memory` | 32Gi | RAM per instance | Must fit model + index + overhead |
| `cpu` | 8 | vCPUs per instance | ONNX Runtime benefits from multiple cores |
| `max_instances` | 100 | Upper autoscale bound | Set based on budget and traffic ceiling |
| `min_instances` | 0 | Lower autoscale bound (cold start risk) | Set to 1+ for production to avoid cold starts |
| `concurrency` | 80 | Max concurrent requests per instance | Lower if latency degrades under load |
| `timeout` | 300s | Max request duration | Must exceed worst-case reranking time |

### Scaling Recommendations by Traffic Profile

**Low traffic (< 10 QPS):**
- `min_instances: 1` (avoids cold starts)
- `max_instances: 5`
- `concurrency: 80`

**Medium traffic (10-100 QPS):**
- `min_instances: 2`
- `max_instances: 20`
- `concurrency: 80`

**High traffic (100+ QPS):**
- `min_instances: 5`
- `max_instances: 100`
- `concurrency: 40` (lower concurrency per instance to maintain latency SLAs)
- Consider disabling on-demand reranking or using async reranking

### Cold Start Mitigation

Cold starts on Cloud Run include model and index loading, which can take 10-30 seconds depending on artifact sizes. Strategies to reduce impact:

1. Set `min_instances: 1` or higher to keep warm instances available
2. Use CPU-always-allocated mode (not just during request processing)
3. Optimize model loading by using memory-mapped FAISS indexes
4. Pre-warm with a startup health check that loads all artifacts

---

## FAISS Tuning by Corpus Size

The service uses FAISS for approximate nearest-neighbor search. The optimal index type and parameters depend on corpus size.

### Under 100K Vectors: HNSW (Current Default)

The current configuration uses HNSW (Hierarchical Navigable Small World), which provides excellent recall with low latency for smaller corpora.

**Current parameters (from `service.yaml`):**

```yaml
search:
  ef_search: 64
```

**Recommended HNSW parameters:**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `M` | 32 | Number of connections per node. Higher = better recall, more memory |
| `ef_construction` | 200 | Build-time search depth. Higher = better graph quality, slower build |
| `ef_search` | 64 | Query-time search depth. Higher = better recall, slower search |

**Expected performance:**
- Index build time: < 5 minutes
- Memory: ~4 bytes/dim/vector + graph overhead (~1.5x)
- Query latency: < 5ms for top-10
- Recall@10: > 0.95

### 100K to 1M Vectors: HNSW with Adjusted Parameters

HNSW remains viable but requires tuning to balance memory and performance.

**Recommended adjustments:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `M` | 48 | More connections improve recall at scale |
| `ef_construction` | 400 | Better graph quality for larger corpus |
| `ef_search` | 128 | Maintain recall as the graph grows |

**Expected performance:**
- Index build time: 10-60 minutes
- Memory: 2-8 GB (depending on embedding dimension)
- Query latency: 5-20ms for top-10
- Recall@10: > 0.93

**Memory estimate for 1M vectors at 384 dimensions:**
- Raw vectors: 1M x 384 x 4 bytes = ~1.5 GB
- HNSW graph overhead (M=48): ~2.5 GB
- Total: ~4 GB

### 1M to 50M Vectors: IVF-PQ Migration Path

At this scale, HNSW memory usage becomes prohibitive. Migrate to IVF (Inverted File Index) with Product Quantization (PQ) for compression.

**Recommended configuration:**

```python
import faiss

dim = 384
nlist = 4096          # Number of Voronoi cells
m_pq = 48            # Number of PQ sub-quantizers
nbits = 8            # Bits per sub-quantizer

quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, nbits)
```

**Tuning parameters:**

| Parameter | Range | Effect |
|-----------|-------|--------|
| `nlist` | sqrt(N) to 4*sqrt(N) | More cells = faster search, slower build |
| `nprobe` | 16-128 | Cells to visit at query time. Higher = better recall |
| `m_pq` | 32-64 | Sub-quantizers. Higher = better accuracy, more memory |

**Expected performance (10M vectors):**
- Index build time: 1-4 hours
- Memory: ~2-5 GB (significant compression vs HNSW)
- Query latency: 10-50ms for top-10 (nprobe=32)
- Recall@10: > 0.85 (tune nprobe for recall/speed tradeoff)

**Migration steps:**
1. Generate embeddings as usual with `make embed`
2. Update `configs/index.yaml` to specify IVF-PQ index type
3. Train the quantizer on a representative sample (10-50K vectors)
4. Build the index with `make index-build`
5. Validate recall against a held-out query set with `make eval-offline`

### 50M+ Vectors: Sharding Strategies

Beyond 50M vectors, a single FAISS index on one machine becomes impractical. Consider these approaches:

**Option A: Index sharding across multiple Cloud Run instances**
- Split the corpus into N shards, each with its own IVF-PQ index
- Route queries to all shards in parallel, merge results
- Each shard fits within the 32Gi memory limit
- Requires a thin routing layer or load balancer with fan-out

**Option B: Managed vector database**
- Migrate to a managed service (Vertex AI Vector Search, Pinecone, Weaviate)
- Offloads index management, scaling, and replication
- The semantic-kd service becomes a query orchestrator

**Option C: Hierarchical indexing**
- First-stage coarse retrieval with a heavily quantized index (OPQ + IVF-PQ)
- Second-stage re-scoring with full-precision vectors for the top candidates
- Fits the existing reranking pipeline architecture

---

## Caching Strategies

The service supports optional caching, configured in `service.yaml` under `cache`.

### Query Result Cache

Cache full search results keyed by the normalized query string and search parameters.

```yaml
cache:
  enabled: true
  backend: "redis"
  redis_url: "redis://localhost:6379"
  ttl_seconds: 3600
  max_size: 10000
```

**When to enable:** High query repetition rate (e.g., autocomplete, popular searches). Monitor cache hit rate to validate effectiveness.

**Cache key format:** `SHA256(normalize(query) + top_k + ef_search)`

**Invalidation:** TTL-based (default 3600s). Flush the cache after index rebuilds or model updates:
```bash
redis-cli FLUSHDB
```

### Embedding Cache

Cache query embeddings to skip ONNX inference for repeated queries. This is especially useful when the same query is searched with different parameters (different top_k, with/without reranking).

**Implementation notes:**
- Key: `SHA256(normalize(query))`
- Value: numpy array (384 floats = 1.5 KB per entry)
- 10,000 cached embeddings = ~15 MB memory
- TTL should match the model version lifecycle

### In-Memory Cache (Development/Small Scale)

For development or low-traffic deployments, use the in-memory cache to avoid a Redis dependency:

```yaml
cache:
  enabled: true
  backend: "in-memory"
  max_size: 10000
```

Note: in-memory cache is per-instance and not shared across Cloud Run instances.

---

## Batch Encoding Optimization

When building the FAISS index, the service encodes the entire corpus. Batch size tuning has a significant impact on throughput.

### Encoding Throughput by Batch Size

| Batch Size | Throughput (docs/sec, CPU) | Throughput (docs/sec, GPU) | Peak Memory |
|------------|---------------------------|---------------------------|-------------|
| 32         | ~200                      | ~2,000                    | Low         |
| 64         | ~350                      | ~4,000                    | Medium      |
| 128        | ~500                      | ~7,000                    | High        |
| 256        | ~550                      | ~10,000                   | Very High   |

**Recommendations:**
- CPU encoding (ONNX INT8): batch size 64-128
- GPU encoding (PyTorch): batch size 128-256
- Monitor memory during encoding. Reduce batch size if OOM occurs.

### Parallel Encoding

For large corpora, use multiple workers:

```bash
# Encode with 4 parallel workers
poetry run python -m src.cli.main index embed --workers 4 --batch-size 128
```

Each worker loads its own copy of the ONNX model, so memory scales linearly with worker count.

---

## Memory Budgeting

Understanding memory consumption is critical for setting the correct Cloud Run memory limit and concurrency.

### Component Memory Breakdown

| Component | Memory | Notes |
|-----------|--------|-------|
| ONNX INT8 student model | 100-200 MB | Depends on model architecture |
| FAISS HNSW index (100K vectors, 384d) | ~300 MB | Scales with corpus size |
| FAISS HNSW index (1M vectors, 384d) | ~4 GB | Consider IVF-PQ at this scale |
| Teacher model (bge-reranker-large) | ~1.3 GB | Loaded on-demand for reranking |
| Python runtime + libraries | 300-500 MB | FastAPI, torch, transformers, etc. |
| Per-request overhead | ~5-20 MB | Query embedding + result buffers |
| Rate limiter buckets | ~1 MB per 10K clients | Capped at max_buckets (10000) |

### Memory Formula

```
Total = ONNX_model + FAISS_index + teacher_model (if reranking) + runtime + (concurrency * per_request)
```

**Example for 100K corpus with reranking:**
```
200 MB + 300 MB + 1300 MB + 400 MB + (80 * 10 MB) = 3.0 GB
```

**Example for 1M corpus with reranking:**
```
200 MB + 4000 MB + 1300 MB + 400 MB + (80 * 10 MB) = 6.7 GB
```

### Memory Limit Recommendations

| Corpus Size | Reranking | Recommended Memory | Recommended Concurrency |
|-------------|-----------|-------------------|------------------------|
| < 100K      | Off       | 2Gi               | 80                     |
| < 100K      | On        | 4Gi               | 80                     |
| 100K-1M     | Off       | 8Gi               | 60                     |
| 100K-1M     | On        | 16Gi              | 40                     |
| 1M-10M      | Off       | 16Gi              | 40                     |
| 1M-10M      | On        | 32Gi              | 20                     |

---

## Performance Benchmarks

These benchmarks were measured on a Cloud Run instance with 8 vCPUs and 32Gi memory, using the ONNX INT8 student model with a 100K-vector HNSW index.

### Search Latency (Without Reranking)

| Percentile | Latency |
|------------|---------|
| p50        | 12ms    |
| p95        | 35ms    |
| p99        | 78ms    |

### Search Latency (With Reranking, batch_size=10)

| Percentile | Latency |
|------------|---------|
| p50        | 180ms   |
| p95        | 450ms   |
| p99        | 920ms   |

### Throughput (Without Reranking)

| Concurrency | Requests/sec | p95 Latency |
|-------------|-------------|-------------|
| 1           | 80          | 15ms        |
| 10          | 450         | 35ms        |
| 40          | 1,200       | 55ms        |
| 80          | 1,800       | 120ms       |

### Index Build Time

| Corpus Size | Index Type | Build Time | Index Size on Disk |
|-------------|-----------|------------|-------------------|
| 10K         | HNSW      | 15s        | 30 MB             |
| 100K        | HNSW      | 3 min      | 300 MB            |
| 1M          | HNSW      | 45 min     | 4 GB              |
| 1M          | IVF-PQ    | 20 min     | 800 MB            |
| 10M         | IVF-PQ    | 3 hours    | 5 GB              |

---

## Tuning Checklist

Use this checklist when optimizing for a new deployment or traffic pattern:

- [ ] Set `min_instances` to avoid cold starts in production
- [ ] Right-size memory based on the memory budgeting formula above
- [ ] Set concurrency based on corpus size and reranking configuration
- [ ] Choose the correct FAISS index type for your corpus size
- [ ] Tune `ef_search` (HNSW) or `nprobe` (IVF) for your recall/latency target
- [ ] Enable caching if query repetition is high
- [ ] Set `rerank.confidence_threshold` to control reranking frequency
- [ ] Monitor p95 latency and memory usage after deployment
- [ ] Run `make eval-offline` to verify recall has not degraded after index changes
