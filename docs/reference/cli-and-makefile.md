# CLI and Makefile Reference

All Make targets and scripts available in the semantic-search-kd project.

## Makefile Targets

Run any target with `make <target>`. Run `make help` to see the full list in your terminal.

### Setup and Development

| Target | Description | Dependencies |
|---|---|---|
| `help` | Show all available targets with descriptions | - |
| `install` | Install dependencies with Poetry and set up pre-commit hooks | Poetry |
| `fmt` | Format code with Black and Ruff (auto-fix) | `install` |
| `lint` | Run linters: Ruff for style, mypy for type checking | `install` |
| `test` | Run test suite with pytest | `install` |
| `test-cov` | Run tests with coverage report (HTML + terminal) | `install` |
| `clean` | Remove `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `htmlcov/` | - |

### Data Pipeline

| Target | Description | Dependencies |
|---|---|---|
| `data-fetch` | Fetch MS MARCO and BEIR datasets | `install` |
| `data-check` | Run data integrity checks and validation tests | `data-fetch` |
| `data-prepare` | Convert raw JSONL to Parquet chunks | `data-fetch` |
| `bm25-build` | Build BM25 index with Pyserini | `data-prepare` |

### Baselines and EDA

| Target | Description | Dependencies |
|---|---|---|
| `baselines` | Run baseline evaluations (BM25, vanilla bi-encoder) | `data-prepare`, `bm25-build` |
| `eda` | Execute EDA Jupyter notebooks in place | `data-fetch` |

### Hard Negative Mining

| Target | Description | Dependencies |
|---|---|---|
| `mine-stage1` | Mine negatives with BM25 | `bm25-build` |
| `mine-stage2` | Mine negatives with the teacher cross-encoder | `mine-stage1` |
| `mine-stage3` | Mine negatives with ANCE (iterative student mining) | `mine-stage2` |

### Training

| Target | Description | Dependencies |
|---|---|---|
| `train-kd` | Train the student model with knowledge distillation. Uses `configs/kd.yaml`. | Mining stages |
| `eval-offline` | Evaluate trained models offline against baselines | `train-kd` |

### Export and Index

| Target | Description | Dependencies |
|---|---|---|
| `export-onnx` | Export the student model to ONNX format | `train-kd` |
| `quantize` | Quantize the ONNX model to INT8 | `export-onnx` |
| `embed` | Generate embeddings for the full corpus | `export-onnx` or `train-kd` |
| `index-build` | Build the FAISS index. Uses `configs/index.yaml`. | `embed` |

### Service

| Target | Description | Dependencies |
|---|---|---|
| `serve-local` | Start the FastAPI service locally with auto-reload on port 8080 | Model + index |
| `smoke` | Run smoke tests against a running service | Service running |

### Infrastructure and Deployment

| Target | Description | Dependencies |
|---|---|---|
| `tf-init` | Initialize Terraform in `infra/terraform/` | Terraform installed |
| `tf-plan` | Preview Terraform infrastructure changes | `tf-init` |
| `tf-apply` | Apply Terraform infrastructure changes | `tf-plan` |
| `deploy-gcp` | Deploy to GCP Cloud Run via Cloud Build | GCP auth, Docker |

### Documentation

| Target | Description | Dependencies |
|---|---|---|
| `docs` | Build the MkDocs documentation site to `site/` | `install` |
| `docs-serve` | Serve documentation locally with live reload | `install` |

### Composite Workflows

| Target | Runs | Description |
|---|---|---|
| `pipeline-data` | `data-fetch` -> `data-check` -> `data-prepare` -> `bm25-build` | Complete data pipeline |
| `pipeline-train` | `mine-stage1` -> `mine-stage2` -> `train-kd` -> `eval-offline` | Complete training pipeline |
| `pipeline-deploy` | `export-onnx` -> `quantize` -> `embed` -> `index-build` | Complete deployment artifact pipeline |
| `all` | `pipeline-data` -> `baselines` -> `pipeline-train` -> `pipeline-deploy` | Full end-to-end pipeline |

---

## Scripts Reference

All scripts are in the `scripts/` directory. Run Python scripts with `poetry run python scripts/<name>.py` and shell scripts with `bash scripts/<name>.sh`.

### Training Scripts

#### `train_kd_pipeline.py`

End-to-end knowledge distillation training pipeline. Fetches data (if needed), prepares chunks, builds BM25 index, loads models, runs 3-stage curriculum training, and evaluates.

```bash
poetry run python scripts/train_kd_pipeline.py \
  --data-dir ./data \
  --max-samples 10000 \
  --teacher-model BAAI/bge-reranker-large \
  --student-model intfloat/e5-small-v2 \
  --epochs 3 \
  --batch-size 8 \
  --lr 2e-5 \
  --patience 2 \
  --stage 2 \
  --output-dir ./artifacts/models/kd_student \
  --device cpu \
  --seed 42 \
  --log-level INFO
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `./data` | Root data directory |
| `--max-samples` | `10000` | Maximum training samples (useful for quick tests) |
| `--teacher-model` | `BAAI/bge-reranker-large` | Teacher model name or path |
| `--student-model` | `intfloat/e5-small-v2` | Student model name or path |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `8` | Training batch size |
| `--lr` | `2e-5` | Learning rate |
| `--patience` | `2` | Early stopping patience |
| `--stage` | `1` | Mining stage: 1 (BM25), 2 (teacher), or 3 (ANCE) |
| `--output-dir` | `./artifacts/models/kd_student` | Directory for saved model |
| `--device` | `cpu` | Compute device (`cpu` or `cuda`) |
| `--seed` | `42` | Random seed for reproducibility |

#### `run_demo_pipeline.sh`

Quick demo pipeline for local testing. Uses a small data subset (200 samples, 2 epochs, batch size 4, BM25-only mining) to verify the pipeline works end to end.

```bash
bash scripts/run_demo_pipeline.sh
```

No arguments. Configuration is hardcoded: 200 samples, 2 epochs, batch size 4, CPU only.

#### `run_training_gcp_cpu.sh`

Launch a full training run on a GCP Compute Engine VM (n1-highmem-8). Creates the VM, SSHs in, runs training, uploads results to GCS, and tears down the VM.

```bash
bash scripts/run_training_gcp_cpu.sh
```

No arguments. Configuration is hardcoded inside the script: 1000 samples, 3 epochs, batch size 8, stage 2 mining. Estimated runtime is approximately 2.5 hours.

---

### Evaluation Scripts

#### `evaluate_production.py`

Evaluate a trained KD model against baselines on a test set. Computes NDCG@k and MRR@k for both the student bi-encoder and teacher cross-encoder.

```bash
poetry run python scripts/evaluate_production.py \
  --model-path ./artifacts/models/kd_student_production \
  --test-data ./data/chunks/msmarco/test.parquet \
  --max-samples 1000 \
  --device cpu
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--model-path` | (required) | Path to the trained student model |
| `--test-data` | (required) | Path to test data parquet file |
| `--max-samples` | `1000` | Maximum number of test queries |
| `--device` | `cpu` | Compute device |

---

### Service Scripts

#### `start_service.py`

Start the FastAPI semantic search service with a specified model.

```bash
poetry run python scripts/start_service.py \
  --model-path ./artifacts/models/kd_student_production \
  --host 0.0.0.0 \
  --port 8000 \
  --device cpu \
  --reload
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--model-path` | (required) | Path to the trained student model |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Listen port |
| `--device` | `cpu` | Compute device |
| `--reload` | disabled | Enable auto-reload for development |

---

### Deployment Scripts

#### `deploy.sh`

Deploy the service to different environments. Builds a Docker image, tags it with the current git SHA, and pushes to the target environment.

```bash
bash scripts/deploy.sh local       # Run locally with uvicorn
bash scripts/deploy.sh docker      # Run locally with Docker
bash scripts/deploy.sh staging     # Deploy to GCP Cloud Run (staging)
bash scripts/deploy.sh production  # Deploy to GCP Cloud Run (production)
```

Requires `GCP_PROJECT_ID` and `GCP_REGION` environment variables (defaults: `plotpointe`, `us-central1`).

#### `setup_gcp.sh`

Initialize GCP infrastructure: create GCS buckets for data, models, and indexes, and configure the gcloud project.

```bash
bash scripts/setup_gcp.sh
```

No arguments. Reads `GCP_PROJECT_ID` and `GCP_REGION` from `.env` or environment variables.

---

### Utility Scripts

#### `manage_api_keys.py`

Generate, list, revoke, and rotate API keys. Keys are hashed with PBKDF2-HMAC-SHA256 and can optionally be stored in GCP Secret Manager.

```bash
# Generate a new key
python scripts/manage_api_keys.py generate --name "client-app-1"

# List all active keys
python scripts/manage_api_keys.py list

# Revoke a key
python scripts/manage_api_keys.py revoke --key-id "abc123"

# Rotate a key (revoke old, generate new)
python scripts/manage_api_keys.py rotate --key-id "abc123"
```

#### `export_to_onnx.py`

Export a trained student model to ONNX format with optional INT8 quantization and validation.

```bash
poetry run python scripts/export_to_onnx.py \
  --model-path ./artifacts/models/kd_student_production \
  --output-dir ./artifacts/models/onnx \
  --device cpu \
  --skip-quantize \
  --skip-validate
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--model-path` | (required) | Path to the trained PyTorch model |
| `--output-dir` | (required) | Directory for ONNX output files |
| `--device` | `cpu` | Compute device |
| `--skip-quantize` | disabled | Skip INT8 quantization step |
| `--skip-validate` | disabled | Skip output validation against PyTorch |
