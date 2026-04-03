# Environment Variables Reference

Complete reference for all environment variables used by the semantic-search-kd system.

## How Environment Variables Work

The application uses Pydantic Settings with the prefix `SEMANTIC_KD_`. Nested configuration keys are separated by double underscores (`__`). The prefix matching is case-insensitive.

**Pattern:**

```
SEMANTIC_KD_<SECTION>__<KEY>
SEMANTIC_KD_<SECTION>__<SUBSECTION>__<KEY>
```

**Examples:**

```bash
SEMANTIC_KD_ENVIRONMENT=production          # maps to: environment
SEMANTIC_KD_SERVICE__PORT=9090              # maps to: service.port
SEMANTIC_KD_SERVICE__AUTH__ENABLED=true     # maps to: service.auth.enabled
SEMANTIC_KD_TRAINING__LOSS__MARGIN_MSE_WEIGHT=0.7  # maps to: training.loss.margin_mse_weight
```

For list values, pass a JSON array string:

```bash
SEMANTIC_KD_SERVICE__CORS__ALLOW_ORIGINS='["https://app.example.com","https://admin.example.com"]'
SEMANTIC_KD_SERVICE__AUTH__API_KEYS='["sk_live_abc123","sk_live_def456"]'
```

## Config Path Variable

| Variable | Type | Default | Description |
|---|---|---|---|
| `SEMANTIC_KD_CONFIG_PATH` | string | not set | Path to YAML config file. When set, the YAML file is loaded first and then environment variables override its values. |

---

## SEMANTIC_KD_ Prefixed Variables

### Top-Level

| Variable | Type | Default | Description |
|---|---|---|---|
| `SEMANTIC_KD_ENVIRONMENT` | string | `development` | Runtime environment: `development`, `staging`, or `production` |
| `SEMANTIC_KD_DEBUG` | boolean | `false` | Enable debug mode |

### Student Model (`SEMANTIC_KD_STUDENT__`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `SEMANTIC_KD_STUDENT__MODEL_NAME` | string | `./artifacts/models/kd_student_production` | Model name or path |
| `SEMANTIC_KD_STUDENT__MAX_LENGTH` | integer | `512` | Max input token length |
| `SEMANTIC_KD_STUDENT__EMBEDDING_DIM` | integer | `384` | Embedding dimensionality |
| `SEMANTIC_KD_STUDENT__NORMALIZE_EMBEDDINGS` | boolean | `true` | L2-normalize embeddings |
| `SEMANTIC_KD_STUDENT__DEVICE` | string | `cpu` | Compute device |

### Teacher Model (`SEMANTIC_KD_TEACHER__`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `SEMANTIC_KD_TEACHER__MODEL_NAME` | string | `BAAI/bge-reranker-large` | Model name or path |
| `SEMANTIC_KD_TEACHER__MAX_LENGTH` | integer | `512` | Max input token length |
| `SEMANTIC_KD_TEACHER__DEVICE` | string | `cpu` | Compute device |
| `SEMANTIC_KD_TEACHER__BATCH_SIZE` | integer | `32` | Scoring batch size |

### Training (`SEMANTIC_KD_TRAINING__`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `SEMANTIC_KD_TRAINING__EPOCHS` | integer | `3` | Training epochs |
| `SEMANTIC_KD_TRAINING__BATCH_SIZE` | integer | `32` | Training batch size |
| `SEMANTIC_KD_TRAINING__GRADIENT_ACCUMULATION_STEPS` | integer | `2` | Gradient accumulation steps |
| `SEMANTIC_KD_TRAINING__LEARNING_RATE` | float | `2e-5` | Learning rate |
| `SEMANTIC_KD_TRAINING__WARMUP_STEPS` | integer | `1000` | Warmup steps |
| `SEMANTIC_KD_TRAINING__WEIGHT_DECAY` | float | `0.01` | Weight decay |
| `SEMANTIC_KD_TRAINING__MAX_GRAD_NORM` | float | `1.0` | Gradient clipping norm |
| `SEMANTIC_KD_TRAINING__FP16` | boolean | `true` | Mixed-precision training |
| `SEMANTIC_KD_TRAINING__EARLY_STOPPING_PATIENCE` | integer | `2` | Early stopping patience |
| `SEMANTIC_KD_TRAINING__EARLY_STOPPING_METRIC` | string | `ndcg@10` | Metric for early stopping |
| `SEMANTIC_KD_TRAINING__SAVE_STEPS` | integer | `500` | Checkpoint interval |
| `SEMANTIC_KD_TRAINING__EVAL_STEPS` | integer | `500` | Evaluation interval |
| `SEMANTIC_KD_TRAINING__LOGGING_STEPS` | integer | `100` | Logging interval |

### Loss (`SEMANTIC_KD_TRAINING__LOSS__`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `SEMANTIC_KD_TRAINING__LOSS__CONTRASTIVE_WEIGHT` | float | `0.2` | Contrastive loss weight |
| `SEMANTIC_KD_TRAINING__LOSS__MARGIN_MSE_WEIGHT` | float | `0.6` | Margin MSE loss weight |
| `SEMANTIC_KD_TRAINING__LOSS__LISTWISE_KD_WEIGHT` | float | `0.2` | Listwise KD loss weight |
| `SEMANTIC_KD_TRAINING__LOSS__CONTRASTIVE_TEMPERATURE` | float | `0.05` | Contrastive loss temperature |
| `SEMANTIC_KD_TRAINING__LOSS__TEMPERATURE_START` | float | `4.0` | KD temperature at training start |
| `SEMANTIC_KD_TRAINING__LOSS__TEMPERATURE_END` | float | `2.0` | KD temperature at training end |

### Mining (`SEMANTIC_KD_MINING__`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `SEMANTIC_KD_MINING__BM25_TOP_K` | integer | `100` | BM25 retrieval depth |
| `SEMANTIC_KD_MINING__TEACHER_TOP_K` | integer | `50` | Teacher scoring depth |
| `SEMANTIC_KD_MINING__ANCE_ENABLED` | boolean | `true` | Enable ANCE mining |
| `SEMANTIC_KD_MINING__ANCE_WARMUP_STEPS` | integer | `1000` | ANCE warmup steps |
| `SEMANTIC_KD_MINING__NEGATIVES_PER_QUERY` | integer | `7` | Hard negatives per query |

### FAISS Index (`SEMANTIC_KD_FAISS__`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `SEMANTIC_KD_FAISS__INDEX_TYPE` | string | `HNSW` | Index algorithm: `Flat`, `IVF`, `HNSW`, `PQ` |
| `SEMANTIC_KD_FAISS__METRIC` | string | `inner_product` | Distance metric: `l2`, `inner_product` |
| `SEMANTIC_KD_FAISS__HNSW_M` | integer | `32` | HNSW links per node |
| `SEMANTIC_KD_FAISS__HNSW_EF_CONSTRUCTION` | integer | `200` | HNSW build search depth |
| `SEMANTIC_KD_FAISS__HNSW_EF_SEARCH` | integer | `64` | HNSW query search depth |
| `SEMANTIC_KD_FAISS__IVF_NLIST` | integer | `100` | IVF cluster count |
| `SEMANTIC_KD_FAISS__IVF_NPROBE` | integer | `10` | IVF clusters searched |

### Service (`SEMANTIC_KD_SERVICE__`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `SEMANTIC_KD_SERVICE__HOST` | string | `0.0.0.0` | Bind address |
| `SEMANTIC_KD_SERVICE__PORT` | integer | `8080` | Listen port |
| `SEMANTIC_KD_SERVICE__WORKERS` | integer | `1` | Uvicorn worker count |
| `SEMANTIC_KD_SERVICE__RELOAD` | boolean | `false` | Auto-reload (dev only) |
| `SEMANTIC_KD_SERVICE__LOG_LEVEL` | string | `info` | Log verbosity |

### CORS (`SEMANTIC_KD_SERVICE__CORS__`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `SEMANTIC_KD_SERVICE__CORS__ENABLED` | boolean | `true` | Enable CORS |
| `SEMANTIC_KD_SERVICE__CORS__ALLOW_ORIGINS` | JSON list | `["http://localhost:3000"]` | Allowed origins |
| `SEMANTIC_KD_SERVICE__CORS__ALLOW_METHODS` | JSON list | `["GET", "POST"]` | Allowed HTTP methods |
| `SEMANTIC_KD_SERVICE__CORS__ALLOW_HEADERS` | JSON list | `["*"]` | Allowed headers |
| `SEMANTIC_KD_SERVICE__CORS__ALLOW_CREDENTIALS` | boolean | `false` | Allow credentials |

### Rate Limiting (`SEMANTIC_KD_SERVICE__RATE_LIMIT__`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `SEMANTIC_KD_SERVICE__RATE_LIMIT__ENABLED` | boolean | `true` | Enable rate limiting |
| `SEMANTIC_KD_SERVICE__RATE_LIMIT__REQUESTS_PER_MINUTE` | integer | `100` | Max requests per minute |
| `SEMANTIC_KD_SERVICE__RATE_LIMIT__BURST` | integer | `20` | Burst allowance |

### Authentication (`SEMANTIC_KD_SERVICE__AUTH__`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `SEMANTIC_KD_SERVICE__AUTH__ENABLED` | boolean | `false` | Enable API key auth |
| `SEMANTIC_KD_SERVICE__AUTH__API_KEY_HEADER` | string | `X-API-Key` | Header name |
| `SEMANTIC_KD_SERVICE__AUTH__API_KEYS` | JSON list | `[]` | Valid API keys |

### Monitoring (`SEMANTIC_KD_SERVICE__MONITORING__`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `SEMANTIC_KD_SERVICE__MONITORING__PROMETHEUS_ENABLED` | boolean | `true` | Enable Prometheus |
| `SEMANTIC_KD_SERVICE__MONITORING__PROMETHEUS_PORT` | integer | `9090` | Prometheus port |
| `SEMANTIC_KD_SERVICE__MONITORING__PROMETHEUS_PATH` | string | `/metrics` | Metrics path |
| `SEMANTIC_KD_SERVICE__MONITORING__OPENTELEMETRY_ENABLED` | boolean | `false` | Enable OTLP tracing |
| `SEMANTIC_KD_SERVICE__MONITORING__OPENTELEMETRY_ENDPOINT` | string | `http://localhost:4317` | OTLP endpoint |
| `SEMANTIC_KD_SERVICE__MONITORING__SERVICE_NAME` | string | `semantic-kd` | Service name in telemetry |
| `SEMANTIC_KD_SERVICE__MONITORING__LOG_QUERIES` | boolean | `false` | Log raw query text |
| `SEMANTIC_KD_SERVICE__MONITORING__LOG_LATENCIES` | boolean | `true` | Log request latencies |

### Search (`SEMANTIC_KD_SEARCH__`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `SEMANTIC_KD_SEARCH__DEFAULT_TOP_K` | integer | `10` | Default result count |
| `SEMANTIC_KD_SEARCH__MAX_TOP_K` | integer | `100` | Maximum allowed result count |
| `SEMANTIC_KD_SEARCH__RERANK_ENABLED` | boolean | `true` | Load teacher for reranking |
| `SEMANTIC_KD_SEARCH__RERANK_TOP_K` | integer | `50` | Candidates to rerank |
| `SEMANTIC_KD_SEARCH__RERANK_CONFIDENCE_THRESHOLD` | float | `0.6` | Minimum rerank confidence |
| `SEMANTIC_KD_SEARCH__RERANK_TIMEOUT_MS` | integer | `5000` | Rerank timeout in ms |

### Data (`SEMANTIC_KD_DATA__`)

| Variable | Type | Default | Description |
|---|---|---|---|
| `SEMANTIC_KD_DATA__RAW_DATA_DIR` | string | `./data/raw` | Raw data directory |
| `SEMANTIC_KD_DATA__CHUNKS_DIR` | string | `./data/chunks` | Chunks directory |
| `SEMANTIC_KD_DATA__ARTIFACTS_DIR` | string | `./artifacts` | Artifacts directory |
| `SEMANTIC_KD_DATA__MAX_CHUNK_TOKENS` | integer | `512` | Max tokens per chunk |
| `SEMANTIC_KD_DATA__CHUNK_STRIDE` | integer | `80` | Chunk overlap stride |
| `SEMANTIC_KD_DATA__DATASET_NAME` | string | `ms_marco` | Dataset name |
| `SEMANTIC_KD_DATA__DATASET_VERSION` | string | `v2.1` | Dataset version |

---

## Non-Prefixed Environment Variables

These variables are used by shell scripts and Docker but are not part of the Pydantic Settings system.

| Variable | Default | Used by | Description |
|---|---|---|---|
| `GCP_PROJECT_ID` | `plotpointe` | `deploy.sh`, `setup_gcp.sh` | GCP project ID |
| `GCP_REGION` | `us-central1` | `deploy.sh`, `setup_gcp.sh` | GCP region |
| `GCS_BUCKET_MODELS` | `gs://plotpointe-semantic-kd-models` | deployment scripts | GCS bucket for model artifacts |
| `GCS_BUCKET_INDEXES` | `gs://plotpointe-semantic-kd-indexes` | deployment scripts | GCS bucket for index files |
| `GCS_BUCKET_DATA` | `gs://plotpointe-semantic-kd-data` | deployment scripts | GCS bucket for training data |
| `GRAFANA_ADMIN_USER` | `admin` | docker-compose | Grafana admin username |
| `GRAFANA_ADMIN_PASSWORD` | - | docker-compose | Grafana admin password |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | OpenTelemetry SDK | OTLP collector endpoint |

---

## Common Override Examples

### Development

```bash
export SEMANTIC_KD_ENVIRONMENT=development
export SEMANTIC_KD_DEBUG=true
export SEMANTIC_KD_SERVICE__RELOAD=true
export SEMANTIC_KD_SERVICE__LOG_LEVEL=debug
export SEMANTIC_KD_SERVICE__AUTH__ENABLED=false
export SEMANTIC_KD_SERVICE__MONITORING__LOG_QUERIES=true
```

### Production

```bash
export SEMANTIC_KD_ENVIRONMENT=production
export SEMANTIC_KD_DEBUG=false
export SEMANTIC_KD_SERVICE__PORT=8080
export SEMANTIC_KD_SERVICE__WORKERS=1
export SEMANTIC_KD_SERVICE__RELOAD=false
export SEMANTIC_KD_SERVICE__LOG_LEVEL=info
export SEMANTIC_KD_SERVICE__AUTH__ENABLED=true
export SEMANTIC_KD_SERVICE__AUTH__API_KEYS='["sk_live_your_key_here"]'
export SEMANTIC_KD_SERVICE__CORS__ALLOW_ORIGINS='["https://your-app.example.com"]'
export SEMANTIC_KD_SERVICE__RATE_LIMIT__REQUESTS_PER_MINUTE=100
export SEMANTIC_KD_SERVICE__MONITORING__PROMETHEUS_ENABLED=true
export SEMANTIC_KD_SERVICE__MONITORING__OPENTELEMETRY_ENABLED=true
export SEMANTIC_KD_SERVICE__MONITORING__OPENTELEMETRY_ENDPOINT=http://otel-collector:4317
export SEMANTIC_KD_SERVICE__MONITORING__LOG_QUERIES=false
export SEMANTIC_KD_STUDENT__DEVICE=cpu
export SEMANTIC_KD_SEARCH__RERANK_ENABLED=true
```

### GPU Training

```bash
export SEMANTIC_KD_STUDENT__DEVICE=cuda:0
export SEMANTIC_KD_TEACHER__DEVICE=cuda:0
export SEMANTIC_KD_TRAINING__FP16=true
export SEMANTIC_KD_TRAINING__BATCH_SIZE=64
```
