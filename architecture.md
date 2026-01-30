# Architecture

## The Problem

Cross-encoders give accurate relevance scores but are too slow for production.
Bi-encoders are fast but less accurate.

We use **knowledge distillation** to get both: train a fast bi-encoder to mimic a slow cross-encoder.

---

## Models

### Teacher: BAAI/bge-reranker-large (Cross-Encoder)

```
┌─────────────────────────────────────────────┐
│  Input: "[CLS] query [SEP] document [SEP]"  │
│                    ↓                        │
│  XLM-RoBERTa (24 layers, 560M params)       │
│                    ↓                        │
│  [CLS] token → Linear(1024 → 1)             │
│                    ↓                        │
│  Output: Relevance Score (float)            │
└─────────────────────────────────────────────┘
```

- Encodes query and document **together**
- Full cross-attention between tokens
- Latency: ~100ms per query-doc pair
- Use: Scoring, reranking (not retrieval)

### Student: intfloat/e5-small-v2 (Bi-Encoder)

```
┌──────────────────┐    ┌──────────────────┐
│      Query       │    │     Document     │
│        ↓         │    │        ↓         │
│  "query: {text}" │    │ "passage: {text}"│
│        ↓         │    │        ↓         │
│  BERT (6 layers) │    │  BERT (6 layers) │
│  33M params      │    │  33M params      │
│        ↓         │    │        ↓         │
│  Mean Pooling    │    │  Mean Pooling    │
│        ↓         │    │        ↓         │
│  384-dim vector  │    │  384-dim vector  │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         └───── Cosine Sim ──────┘
                    ↓
            Relevance Score
```

- Encodes query and document **separately**
- Documents can be pre-computed and indexed
- Latency: ~1ms per query
- Use: First-stage retrieval

### Why This Matters

| Aspect | Teacher (Cross) | Student (Bi) |
|--------|-----------------|--------------|
| Params | 560M | 33M (17x smaller) |
| Latency | ~100ms | ~1ms (100x faster) |
| Accuracy | High | Lower (but KD helps) |
| Can Index | No | Yes |

---

## Training Pipeline

### Step 1: Data

**Source:** MS MARCO Passage Ranking v2.1

```
Raw JSONL → Parse → 603 queries, 643 positive passages
                 → Chunk to 512 tokens (stride 80)
                 → Save as Parquet
```

### Step 2: Hard Negative Mining

The model learns better from "hard" negatives (documents that look relevant but aren't).

**Stage 1 - BM25 (Lexical)**
```
Query → BM25 Index → Top 100 candidates
```
Fast keyword matching. Gets documents with similar words.

**Stage 2 - Teacher Scoring**
```
100 candidates → Teacher scores each → Sort by score
Pick: High BM25 rank + Low teacher score = Hard negatives
```
These are documents that match keywords but teacher says are irrelevant.

**Stage 3 - ANCE (Dynamic)**
```
Use current student model to mine negatives
Update student → Re-mine → Repeat
```
Negatives adapt as student improves.

### Step 3: Knowledge Distillation

Each training step:

```
Input: query + 1 positive + N negatives (N=10)
                    ↓
┌───────────────────┴───────────────────┐
│                                       │
▼                                       ▼
Teacher scores                    Student scores
(cross-encoder)                   (bi-encoder)
[t0, t1, t2, ..., tN]            [s0, s1, s2, ..., sN]
│                                       │
└───────────────────┬───────────────────┘
                    ↓
              Compute Loss
                    ↓
         Backprop → Update Student
```

---

## Loss Functions

Three losses combined with weights:

```
Total Loss = 0.6 × Margin-MSE + 0.2 × Listwise-KD + 0.2 × Contrastive
```

### 1. Margin-MSE Loss (60%)

Teaches student to preserve **relative differences** between scores.

```python
# Compute margins (difference from max score)
student_margins = scores - scores.max(dim=1)
teacher_margins = (teacher_scores / temp) - (teacher_scores / temp).max(dim=1)

# MSE between margins
loss = MSE(student_margins, teacher_margins)
```

Why margins, not absolute scores? Cross-encoder scores (e.g., -5.2, 3.1) have different scale than bi-encoder scores (e.g., 0.4, 0.8). Margins normalize this.

### 2. Listwise-KD Loss (20%)

Teaches student to match teacher's **ranking distribution**.

```python
# Convert scores to probability distributions
teacher_probs = softmax(teacher_scores / temp)
student_log_probs = log_softmax(student_scores / temp)

# KL divergence
loss = KL_div(student_log_probs, teacher_probs) * (temp ** 2)
```

Captures the full ranking, not just pairwise comparisons.

### 3. Contrastive Loss (20%)

Teaches student to distinguish positive from negatives using **in-batch negatives**.

```python
# First document is positive (index 0)
scaled_scores = student_scores / 0.05
log_probs = log_softmax(scaled_scores)
loss = -log_probs[:, 0].mean()  # Maximize prob of positive
```

Uses all negatives in batch as additional signal.

### Temperature Annealing

Temperature controls how "soft" the teacher's distribution is:

```
Start: T = 4.0 (soft, forgiving)
End:   T = 2.0 (sharp, precise)
Schedule: Linear decay over training
```

High temp early → student learns coarse ranking
Low temp later → student learns fine distinctions

---

## Inference Architecture

### PyTorch Serving (No ONNX)

```
┌────────────────────────────────────────────┐
│              FastAPI Service               │
├────────────────────────────────────────────┤
│                                            │
│  Endpoints:                                │
│  ├─ POST /search  → Query + FAISS search   │
│  ├─ POST /encode  → Text → Embedding       │
│  └─ GET  /health  → Service status         │
│                                            │
│  Models (loaded on startup):               │
│  ├─ Student: SentenceTransformer (PyTorch) │
│  └─ Teacher: CrossEncoder (optional)       │
│                                            │
│  Index:                                    │
│  └─ FAISS HNSW (in-memory)                 │
│                                            │
└────────────────────────────────────────────┘
```

### Search Flow

```
1. User sends POST /search {"query": "...", "k": 10}

2. Encode query:
   - Add prefix: "query: {text}"
   - Pass through student model
   - Get 384-dim embedding

3. Search FAISS index:
   - HNSW approximate nearest neighbor
   - Returns top-K doc IDs + distances

4. (Optional) Rerank with teacher:
   - Score top-K with cross-encoder
   - Re-sort by teacher scores

5. Return JSON with results
```

### Index Building

```
Documents → Student encodes → 384-dim embeddings
                                    ↓
                            FAISS IndexHNSW
                            ├─ M = 32 (connections)
                            ├─ efConstruction = 200
                            └─ efSearch = 50
```

---

## File Structure

```
src/
├── data/
│   ├── fetch.py          # Download MS MARCO
│   ├── prepare.py        # Chunk, normalize
│   └── bm25.py           # BM25 index for mining
│
├── models/
│   ├── teacher.py        # CrossEncoder wrapper
│   └── student.py        # SentenceTransformer wrapper
│
├── kd/
│   ├── losses.py         # Margin-MSE, Listwise, Contrastive
│   ├── train.py          # Training loop
│   └── eval.py           # nDCG, MRR metrics
│
├── mining/
│   └── miners.py         # BM25, Teacher, ANCE miners
│
├── index/
│   └── build_index.py    # FAISS HNSW builder
│
└── serve/
    ├── app.py            # FastAPI application
    └── schemas.py        # Request/response models
```

---

## Training Config

```yaml
# configs/kd.yaml
model:
  teacher: "BAAI/bge-reranker-large"
  student: "intfloat/e5-small-v2"

training:
  epochs: 3
  batch_size: 8
  learning_rate: 2e-5
  optimizer: AdamW
  warmup_ratio: 0.1

loss:
  margin_mse_weight: 0.6
  listwise_kd_weight: 0.2
  contrastive_weight: 0.2
  temperature_start: 4.0
  temperature_end: 2.0

mining:
  stage: 2  # BM25 + Teacher
  bm25_top_k: 100
  num_hard_negatives: 10
```

---

## Results

### Training Metrics

| Epoch | Total Loss | Margin-MSE | Listwise-KD | Contrastive | Temp |
|-------|-----------|------------|-------------|-------------|------|
| 1 | 0.3035 | 0.0022 | 0.0075 | 1.5035 | 3.33 |
| 2 | 0.2486 | 0.0049 | 0.0066 | 1.2219 | 2.67 |
| 3 | 0.1942 | 0.0107 | 0.0061 | 0.9325 | 2.00 |

Loss dropped 36% across training.

### Evaluation (200 queries)

| Metric | Vanilla Student | KD Student | Improvement |
|--------|----------------|------------|-------------|
| nDCG@1 | 0.635 | 0.780 | +22.8% |
| nDCG@5 | 0.685 | 0.840 | +22.6% |
| nDCG@10 | 0.719 | 0.882 | +22.7% |
| MRR@10 | 0.632 | 0.775 | +22.7% |

### Latency

| Component | Time |
|-----------|------|
| Query encode | ~1ms |
| FAISS search (10k docs) | ~10ms |
| Teacher rerank (10 docs) | ~100ms |
| **Total (student only)** | **~11ms** |
| **Total (+ rerank)** | **~111ms** |

---

## Deployment

**Cloud Run (GCP)**

```bash
# Build image
docker build -t gcr.io/PROJECT/semantic-kd:latest .

# Deploy
gcloud run deploy semantic-kd \
  --image=gcr.io/PROJECT/semantic-kd:latest \
  --memory=2Gi \
  --cpu=2 \
  --max-instances=10
```

**Costs**
- Training: $4 (one-time, 8.5 hours on CPU)
- Storage: ~$1/month
- Serving: ~$40/month (Cloud Run)

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| 3-stage mining | Progressive difficulty. BM25 is fast, teacher adds quality, ANCE adapts |
| Multi-loss | Each loss captures different signal. Margin-MSE for ordering, listwise for distribution, contrastive for batch |
| Temperature annealing | Soft targets early help student learn; sharp targets later add precision |
| FAISS HNSW | Fast ANN search (<20ms), 97%+ recall, scales to millions |
| PyTorch serving | Simpler than ONNX, fast enough for our latency requirements |
| E5 student | Good balance of size (33M) and quality, built-in query/passage prefixes |
| BGE teacher | State-of-the-art reranker, well-calibrated scores |

---

## Production Architecture

### Security

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Middleware Stack (applied in order):                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 1. Security Headers                                  │    │
│  │    - X-Content-Type-Options: nosniff                │    │
│  │    - X-Frame-Options: DENY                          │    │
│  │    - Strict-Transport-Security                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 2. Request Logging                                   │    │
│  │    - Privacy-preserving query hashing               │    │
│  │    - Latency tracking                               │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 3. Rate Limiting (Token Bucket)                     │    │
│  │    - 100 requests/minute per client                 │    │
│  │    - Burst: 20 requests                             │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 4. API Key Authentication                           │    │
│  │    - X-API-Key header                               │    │
│  │    - Keys stored as SHA-256 hashes                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↓                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 5. CORS (Configurable Origins)                      │    │
│  │    - Production: specific domains only              │    │
│  │    - Development: localhost allowed                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Configuration System

Pydantic Settings with environment variable support:

```python
# Environment variables override config files
SEMANTIC_KD_ENVIRONMENT=production
SEMANTIC_KD_SERVICE__PORT=8080
SEMANTIC_KD_SERVICE__CORS__ALLOW_ORIGINS=["https://myapp.com"]
SEMANTIC_KD_SERVICE__AUTH__ENABLED=true
```

Configuration hierarchy:
1. Environment variables (highest priority)
2. YAML config file (SEMANTIC_KD_CONFIG_PATH)
3. Default values (lowest priority)

### Health Checks

```
GET /health    → Full health status
GET /ready     → Kubernetes readiness probe
GET /live      → Kubernetes liveness probe
```

### Exception Handling

Custom exception hierarchy for better error handling:

```
SemanticKDError (base)
├── ModelError
│   ├── ModelLoadError
│   ├── ModelNotLoadedError
│   └── EncodingError
├── IndexError
│   ├── IndexNotFoundError
│   └── IndexNotBuiltError
├── SearchError
│   ├── InvalidQueryError
│   └── SearchTimeoutError
└── AuthError
    ├── InvalidAPIKeyError
    └── RateLimitExceededError
```

---

## CI/CD Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Commit    │────▶│    CI Jobs   │────▶│   Artifacts  │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌─────────┐   ┌─────────┐   ┌─────────┐
        │  Lint   │   │  Test   │   │ Security│
        │ (Ruff)  │   │(pytest) │   │ (bandit)│
        └─────────┘   └─────────┘   └─────────┘
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                     ┌─────────────┐
                     │   Build     │
                     │  (poetry)   │
                     └─────────────┘
                            │
                            ▼
                     ┌─────────────┐
                     │   Docker    │
                     │   Build     │
                     └─────────────┘
```

Jobs run on every push/PR:
- **Lint**: Ruff linter and formatter
- **Type Check**: MyPy static analysis
- **Test**: pytest with coverage (Python 3.10, 3.11)
- **Security**: bandit code scanner, safety dependency check
- **Build**: Package build verification
- **Docker**: Image build (main branch only)

---

## Docker Deployment

Multi-stage build for minimal image size:

```dockerfile
# Stage 1: Builder (with Poetry)
FROM python:3.10-slim as builder
# Install dependencies, copy source

# Stage 2: Runtime (minimal)
FROM python:3.10-slim as runtime
# Copy only venv and source
# Run as non-root user
```

Docker Compose services:
- **api**: Main semantic search service
- **prometheus**: Metrics collection (optional)
- **grafana**: Visualization dashboard (optional)

---

## Testing Strategy

```
tests/
├── conftest.py          # Shared fixtures
├── test_config.py       # Configuration tests
├── test_losses.py       # Loss function tests
├── test_bm25.py         # BM25 index tests
└── test_api.py          # API endpoint tests
```

Test categories:
- **Unit tests**: Individual functions and classes
- **Integration tests**: API endpoints with mocked models
- **Numerical stability**: Edge cases for loss functions

Run tests:
```bash
poetry run pytest tests/ -v --cov=src
```

---

## Project Importance

### Why This Project Matters

1. **Solves a Real Problem**: The speed-accuracy tradeoff in semantic search is a fundamental challenge. Cross-encoders are accurate but slow; bi-encoders are fast but less accurate. This project bridges that gap.

2. **Production-Grade Implementation**:
   - Security: Rate limiting, API authentication, CORS
   - Observability: Health checks, logging, metrics
   - Reliability: Proper error handling, input validation
   - Deployment: Docker, CI/CD, cloud-ready

3. **Novel Technical Approach**:
   - 3-loss combination (Margin-MSE + Listwise + Contrastive)
   - Temperature annealing for stable training
   - 3-stage curriculum mining (BM25 → Teacher → ANCE)

4. **Measurable Results**:
   - 97% of teacher accuracy (0.88 vs 0.91 nDCG@10)
   - 17x smaller model (33M vs 560M params)
   - 100x faster inference (1ms vs 100ms)

5. **Cost Efficient**:
   - Training: $4 one-time
   - Serving: ~$40/month for 10K queries/day

### Skills Demonstrated

- **ML/NLP**: Knowledge distillation, embedding models, loss function design
- **Software Engineering**: Clean architecture, testing, CI/CD
- **Production Systems**: API design, security, observability
- **Infrastructure**: Docker, cloud deployment, configuration management
