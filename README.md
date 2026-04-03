# Semantic Search with Knowledge Distillation

A production semantic search system that uses knowledge distillation to compress a large, accurate cross-encoder (BAAI/bge-reranker-large, 560M params) into a fast bi-encoder student (intfloat/e5-small-v2, 33M params). The student reaches 97% of teacher accuracy while running 100x faster, making real-time neural search practical without sacrificing relevance.

## Teacher vs Student

|  | Teacher | Student |
|--|---------|---------|
| Model | BAAI/bge-reranker-large | intfloat/e5-small-v2 |
| Parameters | 560M | 33M |
| Latency | ~100ms per pair | ~1ms per query |
| nDCG@10 | 0.91 | 0.88 |
| Architecture | Cross-encoder (joint) | Bi-encoder (independent) |
| Use case | Offline scoring, reranking | Real-time search |

## Quickstart

```bash
# 1. Install
poetry install  # Python 3.10+

# 2. Train (local demo, ~1 hour)
./scripts/run_demo_pipeline.sh

# 3. Run API
poetry run python scripts/start_service.py \
  --model-path=artifacts/models/kd_student_demo \
  --port=8000
```

Test it:
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is machine learning?", "k": 5}'
```

## Documentation

### Overview

| Document | Description |
|----------|-------------|
| [Problem and Approach](docs/overview/problem-and-approach.md) | Why knowledge distillation, why these models, why this design |
| [Results and Benchmarks](docs/overview/results-and-benchmarks.md) | Training metrics, evaluation results, latency, costs |

### Architecture (C4 Diagrams)

| Document | Description |
|----------|-------------|
| [C4 Context](docs/architecture/c4-context.md) | Level 1: System context with actors and external systems |
| [C4 Containers](docs/architecture/c4-container.md) | Level 2: Training pipeline, index builder, API service |
| [C4 Training Components](docs/architecture/c4-component-training.md) | Level 3: Data, mining, KD trainer internals |
| [C4 Serving Components](docs/architecture/c4-component-serving.md) | Level 3: Middleware stack, search flow, endpoints |
| [C4 Loss Functions](docs/architecture/c4-code-losses.md) | Level 4: Loss math, code, temperature annealing |

### Decisions (ADRs)

| Document | Description |
|----------|-------------|
| [ADR-001: Student Model](docs/decisions/adr-001-student-model-choice.md) | Why intfloat/e5-small-v2 |
| [ADR-002: Teacher Model](docs/decisions/adr-002-teacher-model-choice.md) | Why BAAI/bge-reranker-large |
| [ADR-003: Mining Curriculum](docs/decisions/adr-003-three-stage-mining.md) | Why 3-stage BM25, teacher, ANCE |
| [ADR-004: Multi-Loss](docs/decisions/adr-004-multi-loss-combination.md) | Why 60/20/20 Margin-MSE, Listwise, Contrastive |
| [ADR-005: Temperature](docs/decisions/adr-005-temperature-annealing.md) | Why linear annealing from 4.0 to 2.0 |
| [ADR-006: FAISS HNSW](docs/decisions/adr-006-faiss-hnsw-over-ivfpq.md) | Why HNSW over IVF-PQ |
| [ADR-007: PyTorch Serving](docs/decisions/adr-007-pytorch-over-onnx-serving.md) | Why native PyTorch over ONNX |
| [ADR-008: Rate Limiting](docs/decisions/adr-008-token-bucket-rate-limiting.md) | Why in-process token bucket |

### Guides

| Document | Description |
|----------|-------------|
| [Quickstart](docs/guides/quickstart.md) | 5-minute local demo |
| [Training Guide](docs/guides/training-guide.md) | Full training walkthrough (local + GCP) |
| [Deployment Guide](docs/guides/deployment-guide.md) | Docker through Cloud Run |
| [Hyperparameter Tuning](docs/guides/hyperparameter-tuning.md) | What to tune, in what order |
| [Custom Datasets](docs/guides/custom-dataset-guide.md) | Bring your own data |

### Reference

| Document | Description |
|----------|-------------|
| [API Reference](docs/reference/api-reference.md) | Endpoints, schemas, error codes |
| [Configuration Reference](docs/reference/configuration-reference.md) | Every config knob |
| [CLI and Makefile](docs/reference/cli-and-makefile.md) | Make targets and scripts |
| [Environment Variables](docs/reference/environment-variables.md) | All SEMANTIC_KD_ variables |

### Operations

| Document | Description |
|----------|-------------|
| [Runbook](docs/operations/runbook.md) | Symptom-based troubleshooting |
| [Monitoring and Alerting](docs/operations/monitoring-and-alerting.md) | Prometheus, Grafana, alert thresholds |
| [Scaling and Performance](docs/operations/scaling-and-performance.md) | FAISS tuning, Cloud Run scaling |

## Tech Stack

PyTorch, Sentence-Transformers, FAISS (HNSW), FastAPI, GCP (Compute Engine, Cloud Storage, Cloud Run), Poetry, Docker

## Dev Commands

```bash
poetry run pytest tests/ -v          # tests
poetry run ruff format .             # format
poetry run ruff check . --fix        # lint
poetry run mypy src/                 # type check
```
