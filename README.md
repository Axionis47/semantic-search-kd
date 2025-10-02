# Semantic Search with Knowledge Distillation

A small, fast search model trained to match a large, accurate one.

|  | Teacher | Student |
|--|---------|---------|
| Model | BAAI/bge-reranker-large | intfloat/e5-small-v2 |
| Params | 560M | 33M |
| Latency | 100ms | 1ms |
| nDCG@10 | 0.91 | 0.88 |

The student reaches **97% of teacher accuracy** while being **17x smaller** and **100x faster**.

For technical details, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Quick Start

### Install

```bash
poetry install  # Python 3.10+
```

### Train

```bash
# Local demo (~1 hour)
./scripts/run_demo_pipeline.sh
# Output: artifacts/models/kd_student_demo/

# Full training on GCP (~8.5 hours, $4)
./scripts/setup_gcp.sh
./scripts/upload_code_to_gcs.sh
./scripts/run_training_gcp_cpu.sh
# Output: artifacts/models/kd_student_production/
```

### Evaluate

```bash
poetry run python scripts/evaluate_production.py \
  --model-path=artifacts/models/kd_student_production \
  --max-samples=200
```

Output:
```
nDCG@10: 0.882
MRR@10: 0.775
```

### Run API

```bash
poetry run python scripts/start_service.py \
  --model-path=artifacts/models/kd_student_production \
  --port=8000
```

Test:
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is machine learning?", "k": 5}'
```

Response:
```json
{
  "query": "what is machine learning?",
  "results": [{"doc_id": "doc_123", "text": "Machine learning is...", "score": 0.95, "rank": 1}],
  "total_results": 5,
  "reranked": false,
  "latency_ms": 12.5
}
```

---

## How Training Works

1. **Get data** - Download MS MARCO (603 queries, 10K documents)
2. **Find hard negatives** - Documents that look relevant but aren't (using BM25 + teacher)
3. **Teacher scores pairs** - Score each (query, document) pair
4. **Student learns scores** - Train student to predict teacher's scores
5. **Export model** - Save to `artifacts/models/`

**Loss breakdown:**
| Loss | Weight | Purpose |
|------|--------|---------|
| Margin-MSE | 60% | Learn score differences between docs |
| Listwise | 20% | Learn ranking order |
| Contrastive | 20% | Learn query-doc similarity |

See [ARCHITECTURE.md](ARCHITECTURE.md#loss-functions) for loss function details.

---

## How Inference Works

```
Query → Encode (1ms) → FAISS search (10ms) → Top-K results
```

**Optional reranking:** Pass top-K through teacher for higher accuracy (+100ms latency). Use when accuracy matters more than speed.

See [ARCHITECTURE.md](ARCHITECTURE.md#inference-architecture) for details.

---

## Project Structure

```
src/
├── data/       # MS MARCO fetching, chunking
├── models/     # Teacher & student wrappers
├── kd/         # Loss functions, training loop
├── mining/     # Hard negative mining
├── index/      # FAISS index building
└── serve/      # FastAPI service

scripts/
├── train_kd_pipeline.py    # Main training
├── evaluate_production.py  # Evaluation
├── start_service.py        # API server
└── run_training_gcp_cpu.sh # GCP training

configs/
├── kd.yaml       # Training hyperparameters
├── index.yaml    # FAISS settings
└── service.yaml  # API settings
```

---

## Results

Evaluated on 200 MS MARCO queries (same distribution as training).

| Metric | Untrained | After KD | Teacher | Gap |
|--------|-----------|----------|---------|-----|
| nDCG@10 | 0.719 | 0.882 | 0.91 | 3% |
| MRR@10 | 0.632 | 0.775 | 0.80 | 3% |

See [ARCHITECTURE.md](ARCHITECTURE.md#results) for training curves.

---

## Cost

| Item | Cost | Notes |
|------|------|-------|
| Training | $4 | One-time, 8.5 hours on GCP CPU |
| Storage | $1/month | Model + index files |
| Cloud Run | $40/month | ~10K queries/day |

---

## Tech Stack

- **ML:** PyTorch, Sentence-Transformers
- **Search:** FAISS (HNSW), BM25
- **API:** FastAPI
- **Cloud:** GCP (Compute, Storage, Cloud Run)
- **Package:** Poetry

---

## Development

```bash
poetry run pytest tests/ -v      # Run tests
poetry run ruff format .         # Format code
poetry run ruff check . --fix    # Lint
poetry run mypy src/             # Type check
```

---

## Credits

- MS MARCO dataset - Microsoft
- BGE Reranker - BAAI
- E5 Embeddings - Microsoft
- FAISS - Meta
