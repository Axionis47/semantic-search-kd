# Configuration Reference

Complete reference for every configuration knob in the semantic-search-kd system.

## Configuration Hierarchy

Settings are loaded in the following order of precedence (highest first):

1. **Environment variables** (prefixed with `SEMANTIC_KD_`)
2. **YAML configuration file** (specified via `SEMANTIC_KD_CONFIG_PATH`)
3. **Default values** (defined in `src/config.py`)

A value set by an environment variable always overrides the same value from YAML, which in turn overrides the built-in default.

## How to Set Configuration

### YAML file

Point to your YAML config by setting the `SEMANTIC_KD_CONFIG_PATH` environment variable:

```bash
export SEMANTIC_KD_CONFIG_PATH=./configs/service.yaml
```

Or load programmatically:

```python
from src.config import Settings
settings = Settings.from_yaml(Path("./configs/service.yaml"))
```

### Environment variables

All environment variables use the prefix `SEMANTIC_KD_` and double underscores (`__`) as delimiters for nested keys. The prefix is case-insensitive.

```bash
# Top-level key
export SEMANTIC_KD_ENVIRONMENT=production

# Nested key: service.port
export SEMANTIC_KD_SERVICE__PORT=9090

# Deeply nested: service.cors.allow_origins
export SEMANTIC_KD_SERVICE__CORS__ALLOW_ORIGINS='["https://example.com"]'
```

---

## Top-Level Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `environment` | string | `"development"` | Runtime environment. One of: `development`, `staging`, `production`. |
| `debug` | boolean | `false` | Enable debug mode. Must be `false` in production. |

---

## Student Model Config (`student`)

Configuration for the KD-trained bi-encoder model used for query and document encoding.

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `model_name` | string | `"./artifacts/models/kd_student_production"` | - | HuggingFace model name or local path |
| `max_length` | integer | `512` | 32-8192 | Maximum input token length |
| `embedding_dim` | integer | `384` | 64-4096 | Output embedding dimensionality |
| `normalize_embeddings` | boolean | `true` | - | Apply L2 normalization to embeddings |
| `device` | string | `"cpu"` | `cpu` or `cuda:N` | Compute device |

---

## Teacher Model Config (`teacher`)

Configuration for the cross-encoder model used for reranking and knowledge distillation scoring.

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `model_name` | string | `"BAAI/bge-reranker-large"` | - | HuggingFace model name or local path |
| `max_length` | integer | `512` | 32-8192 | Maximum input token length |
| `device` | string | `"cpu"` | `cpu` or `cuda:N` | Compute device |
| `batch_size` | integer | `32` | 1-512 | Batch size for scoring |

---

## Training Config (`training`)

Hyperparameters for the knowledge distillation training pipeline.

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `epochs` | integer | `3` | 1-100 | Number of training epochs |
| `batch_size` | integer | `32` | 1-512 | Training batch size |
| `gradient_accumulation_steps` | integer | `2` | >= 1 | Steps to accumulate gradients before update |
| `learning_rate` | float | `2e-5` | > 0.0, <= 1.0 | Optimizer learning rate |
| `warmup_steps` | integer | `1000` | >= 0 | Linear warmup steps |
| `weight_decay` | float | `0.01` | 0.0-1.0 | Weight decay for regularization |
| `max_grad_norm` | float | `1.0` | > 0.0 | Maximum gradient norm for clipping |
| `fp16` | boolean | `true` | - | Enable mixed-precision (FP16) training |
| `early_stopping_patience` | integer | `2` | >= 1 | Epochs without improvement before stopping |
| `early_stopping_metric` | string | `"ndcg@10"` | - | Metric to monitor for early stopping |
| `save_steps` | integer | `500` | >= 1 | Save checkpoint every N steps |
| `eval_steps` | integer | `500` | >= 1 | Run evaluation every N steps |
| `logging_steps` | integer | `100` | >= 1 | Log metrics every N steps |

---

## Loss Config (`training.loss`)

Weights and temperature for the combined knowledge distillation loss function. Loss weights must sum to approximately 1.0 (within 0.01 tolerance).

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `contrastive_weight` | float | `0.2` | 0.0-1.0 | Weight for contrastive loss |
| `margin_mse_weight` | float | `0.6` | 0.0-1.0 | Weight for margin MSE loss |
| `listwise_kd_weight` | float | `0.2` | 0.0-1.0 | Weight for listwise KD loss |
| `contrastive_temperature` | float | `0.05` | > 0.0, <= 1.0 | Temperature for contrastive loss |
| `temperature_start` | float | `4.0` | > 0.0 | KD temperature at start of training |
| `temperature_end` | float | `2.0` | > 0.0 | KD temperature at end of training (annealed) |

---

## Mining Config (`mining`)

Configuration for hard negative mining across the 3-stage curriculum.

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `bm25_top_k` | integer | `100` | 10-1000 | Number of BM25 candidates to retrieve |
| `teacher_top_k` | integer | `50` | 5-500 | Number of teacher-scored candidates |
| `ance_enabled` | boolean | `true` | - | Enable ANCE (iterative student mining) in stage 3 |
| `ance_warmup_steps` | integer | `1000` | >= 0 | Steps before ANCE mining begins |
| `negatives_per_query` | integer | `7` | 1-50 | Number of hard negatives per query |

---

## FAISS Index Config (`faiss`)

Configuration for the FAISS vector index.

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `index_type` | string | `"HNSW"` | `Flat`, `IVF`, `HNSW`, `PQ` | Index algorithm |
| `metric` | string | `"inner_product"` | `l2`, `inner_product` | Distance metric |
| `hnsw_m` | integer | `32` | 8-128 | HNSW: bi-directional links per node |
| `hnsw_ef_construction` | integer | `200` | 50-500 | HNSW: search depth during index build |
| `hnsw_ef_search` | integer | `64` | 16-256 | HNSW: search depth during query |
| `ivf_nlist` | integer | `100` | 1-10000 | IVF: number of clusters |
| `ivf_nprobe` | integer | `10` | 1-1000 | IVF: number of clusters to search at query time |

---

## Service Config (`service`)

Configuration for the FastAPI HTTP server.

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `host` | string | `"0.0.0.0"` | - | Bind address |
| `port` | integer | `8080` | 1024-65535 | Listen port |
| `workers` | integer | `1` | 1-32 | Number of Uvicorn workers |
| `reload` | boolean | `false` | - | Enable auto-reload (development only) |
| `log_level` | string | `"info"` | `debug`, `info`, `warning`, `error`, `critical` | Log verbosity |

---

## CORS Config (`service.cors`)

Cross-Origin Resource Sharing settings.

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `enabled` | boolean | `true` | - | Enable CORS middleware |
| `allow_origins` | list of string | `["http://localhost:3000"]` | - | Allowed origins. Warns if `*` is used. |
| `allow_methods` | list of string | `["GET", "POST"]` | - | Allowed HTTP methods |
| `allow_headers` | list of string | `["*"]` | - | Allowed request headers |
| `allow_credentials` | boolean | `false` | - | Allow credentials (cookies, auth headers) |

---

## Rate Limit Config (`service.rate_limit`)

Token bucket rate limiter applied per client.

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `enabled` | boolean | `true` | - | Enable rate limiting |
| `requests_per_minute` | integer | `100` | 1-10000 | Maximum sustained request rate |
| `burst` | integer | `20` | 1-100 | Maximum burst allowance above the sustained rate |

---

## Auth Config (`service.auth`)

API key authentication settings.

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `enabled` | boolean | `false` | - | Enable API key authentication. Should be `true` in production. |
| `api_key_header` | string | `"X-API-Key"` | - | HTTP header name for the API key |
| `api_keys` | list of string | `[]` | - | List of valid API keys |

---

## Monitoring Config (`service.monitoring`)

Observability and telemetry settings.

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `prometheus_enabled` | boolean | `true` | - | Enable Prometheus metrics endpoint |
| `prometheus_port` | integer | `9090` | 1024-65535 | Port for Prometheus metrics |
| `prometheus_path` | string | `"/metrics"` | - | HTTP path for metrics scraping |
| `opentelemetry_enabled` | boolean | `false` | - | Enable OpenTelemetry tracing |
| `opentelemetry_endpoint` | string | `"http://localhost:4317"` | - | OTLP exporter endpoint |
| `service_name` | string | `"semantic-kd"` | - | Service name in traces and metrics |
| `log_queries` | boolean | `false` | - | Log raw query text (disable for privacy) |
| `log_latencies` | boolean | `true` | - | Log request latency metrics |

---

## Search Config (`search`)

Runtime search behavior.

| Key | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `default_top_k` | integer | `10` | 1-1000 | Default number of results when `k` is not specified |
| `max_top_k` | integer | `100` | 1-10000 | Maximum allowed value for `k` |
| `rerank_enabled` | boolean | `true` | - | Load the teacher model for reranking at startup |
| `rerank_top_k` | integer | `50` | 1-500 | Number of candidates to pass to the reranker |
| `rerank_confidence_threshold` | float | `0.6` | 0.0-1.0 | Minimum teacher confidence to include a result |
| `rerank_timeout_ms` | integer | `5000` | 100-30000 | Reranking timeout (circuit breaker) |

---

## Data Config (`data`)

Data paths and chunking parameters.

| Key | Type | Default | Description |
|---|---|---|---|
| `raw_data_dir` | path | `./data/raw` | Directory for raw downloaded data |
| `chunks_dir` | path | `./data/chunks` | Directory for processed chunks |
| `artifacts_dir` | path | `./artifacts` | Directory for models, indexes, and exports |
| `max_chunk_tokens` | integer | `512` | Maximum tokens per chunk (range: 64-8192) |
| `chunk_stride` | integer | `80` | Overlap stride between chunks (range: 0-256) |
| `dataset_name` | string | `"ms_marco"` | Dataset identifier |
| `dataset_version` | string | `"v2.1"` | Dataset version |

---

## Production Validation

When `environment` is set to `production`, the system checks for common misconfigurations and emits warnings for:

- `service.auth.enabled` is `false`
- `service.cors.allow_origins` contains `*`
- `service.rate_limit.enabled` is `false`
- `debug` is `true`
- `service.reload` is `true`

You can run validation programmatically:

```python
from src.config import get_settings
issues = get_settings().validate_for_production()
for issue in issues:
    print(f"WARNING: {issue}")
```
