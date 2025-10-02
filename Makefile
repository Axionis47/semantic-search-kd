.PHONY: help install fmt lint test clean
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)semantic-kd Makefile targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# ============================================================================
# Setup & Development
# ============================================================================

install: ## Install dependencies with Poetry
	@echo "$(BLUE)Installing dependencies...$(NC)"
	poetry install
	poetry run pre-commit install
	@echo "$(GREEN)✓ Installation complete$(NC)" | tee -a RUNS.md

fmt: ## Format code with black and ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	poetry run black src tests
	poetry run ruff check --fix src tests
	@echo "$(GREEN)✓ Formatting complete$(NC)"

lint: ## Run linters (ruff, mypy)
	@echo "$(BLUE)Running linters...$(NC)"
	poetry run ruff check src tests
	poetry run mypy src
	@echo "$(GREEN)✓ Linting complete$(NC)"

test: ## Run tests with pytest
	@echo "$(BLUE)Running tests...$(NC)"
	poetry run pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-cov: ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	poetry run pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

clean: ## Clean temporary files and caches
	@echo "$(BLUE)Cleaning...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# ============================================================================
# Data Pipeline
# ============================================================================

data-fetch: ## Fetch MS MARCO and BEIR datasets
	@echo "$(BLUE)Fetching datasets...$(NC)"
	poetry run python -m src.cli.main data fetch
	@echo "$(GREEN)✓ Data fetch complete - see data/manifests/$(NC)" | tee -a RUNS.md

data-check: ## Run data integrity checks
	@echo "$(BLUE)Running data integrity checks...$(NC)"
	poetry run python -m src.cli.main data check
	poetry run pytest tests/test_data_integrity.py -v
	@echo "$(GREEN)✓ Data integrity verified$(NC)" | tee -a RUNS.md

data-prepare: ## Prepare data (JSONL → Parquet chunks)
	@echo "$(BLUE)Preparing data...$(NC)"
	poetry run python -m src.cli.main data prepare
	@echo "$(GREEN)✓ Data preparation complete - see data/chunks/$(NC)" | tee -a RUNS.md

bm25-build: ## Build BM25 index with Pyserini
	@echo "$(BLUE)Building BM25 index...$(NC)"
	poetry run python -m src.cli.main bm25 build
	@echo "$(GREEN)✓ BM25 index built$(NC)" | tee -a RUNS.md

# ============================================================================
# Baselines & EDA
# ============================================================================

baselines: ## Run baseline evaluations (BM25, vanilla bi-encoder)
	@echo "$(BLUE)Running baseline evaluations...$(NC)"
	poetry run python -m src.cli.main eval baselines
	@echo "$(GREEN)✓ Baselines complete - see docs/_artifacts/baselines.json$(NC)" | tee -a RUNS.md

eda: ## Run EDA notebooks
	@echo "$(BLUE)Running EDA notebooks...$(NC)"
	poetry run jupyter nbconvert --execute --to notebook --inplace notebooks/00_eda_msmarco.ipynb
	poetry run jupyter nbconvert --execute --to notebook --inplace notebooks/01_baselines_eval.ipynb
	@echo "$(GREEN)✓ EDA complete$(NC)"

# ============================================================================
# Mining
# ============================================================================

mine-stage1: ## Mine negatives with BM25
	@echo "$(BLUE)Mining stage 1 (BM25)...$(NC)"
	poetry run python -m src.cli.main mine stage1
	@echo "$(GREEN)✓ Stage 1 mining complete$(NC)" | tee -a RUNS.md

mine-stage2: ## Mine negatives with teacher
	@echo "$(BLUE)Mining stage 2 (Teacher)...$(NC)"
	poetry run python -m src.cli.main mine stage2
	@echo "$(GREEN)✓ Stage 2 mining complete$(NC)" | tee -a RUNS.md

mine-stage3: ## Mine negatives with ANCE
	@echo "$(BLUE)Mining stage 3 (ANCE)...$(NC)"
	poetry run python -m src.cli.main mine stage3
	@echo "$(GREEN)✓ Stage 3 mining complete$(NC)" | tee -a RUNS.md

# ============================================================================
# Training
# ============================================================================

train-kd: ## Train student with knowledge distillation
	@echo "$(BLUE)Training student with KD...$(NC)"
	poetry run python -m src.cli.main train kd --config configs/kd.yaml
	@echo "$(GREEN)✓ KD training complete - see artifacts/models/$(NC)" | tee -a RUNS.md

eval-offline: ## Evaluate trained models offline
	@echo "$(BLUE)Running offline evaluation...$(NC)"
	poetry run python -m src.cli.main eval offline
	@echo "$(GREEN)✓ Evaluation complete - see docs/KD_REPORT.md$(NC)" | tee -a RUNS.md

# ============================================================================
# Export & Index
# ============================================================================

export-onnx: ## Export student model to ONNX
	@echo "$(BLUE)Exporting to ONNX...$(NC)"
	poetry run python -m src.cli.main export onnx
	@echo "$(GREEN)✓ ONNX export complete$(NC)" | tee -a RUNS.md

quantize: ## Quantize ONNX model to INT8
	@echo "$(BLUE)Quantizing model...$(NC)"
	poetry run python -m src.cli.main export quantize
	@echo "$(GREEN)✓ Quantization complete$(NC)" | tee -a RUNS.md

embed: ## Generate embeddings for corpus
	@echo "$(BLUE)Generating embeddings...$(NC)"
	poetry run python -m src.cli.main index embed
	@echo "$(GREEN)✓ Embeddings generated$(NC)" | tee -a RUNS.md

index-build: ## Build FAISS index
	@echo "$(BLUE)Building FAISS index...$(NC)"
	poetry run python -m src.cli.main index build --config configs/index.yaml
	@echo "$(GREEN)✓ Index built - see artifacts/indexes/$(NC)" | tee -a RUNS.md

# ============================================================================
# Service
# ============================================================================

serve-local: ## Run FastAPI service locally
	@echo "$(BLUE)Starting local service...$(NC)"
	poetry run uvicorn src.serve.app:app --host 0.0.0.0 --port 8080 --reload

smoke: ## Run smoke tests against service
	@echo "$(BLUE)Running smoke tests...$(NC)"
	poetry run pytest tests/test_api_smoke.py -v
	@echo "$(GREEN)✓ Smoke tests passed$(NC)" | tee -a RUNS.md

# ============================================================================
# Infrastructure & Deployment
# ============================================================================

tf-init: ## Initialize Terraform
	@echo "$(BLUE)Initializing Terraform...$(NC)"
	cd infra/terraform && terraform init

tf-plan: ## Plan Terraform changes
	@echo "$(BLUE)Planning Terraform changes...$(NC)"
	cd infra/terraform && terraform plan

tf-apply: ## Apply Terraform changes
	@echo "$(BLUE)Applying Terraform changes...$(NC)"
	cd infra/terraform && terraform apply
	@echo "$(GREEN)✓ Infrastructure deployed$(NC)" | tee -a RUNS.md

deploy-gcp: ## Deploy to GCP Cloud Run
	@echo "$(BLUE)Deploying to GCP...$(NC)"
	gcloud builds submit --config=infra/cloudbuild.yaml
	@echo "$(GREEN)✓ Deployment complete$(NC)" | tee -a RUNS.md

# ============================================================================
# Documentation
# ============================================================================

docs: ## Build MkDocs documentation site
	@echo "$(BLUE)Building documentation...$(NC)"
	poetry run mkdocs build
	@echo "$(GREEN)✓ Docs built in site/$(NC)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	poetry run mkdocs serve

# ============================================================================
# Complete Workflows
# ============================================================================

pipeline-data: data-fetch data-check data-prepare bm25-build ## Complete data pipeline
	@echo "$(GREEN)✓ Data pipeline complete$(NC)"

pipeline-train: mine-stage1 mine-stage2 train-kd eval-offline ## Complete training pipeline
	@echo "$(GREEN)✓ Training pipeline complete$(NC)"

pipeline-deploy: export-onnx quantize embed index-build ## Complete deployment pipeline
	@echo "$(GREEN)✓ Deployment pipeline complete$(NC)"

all: pipeline-data baselines pipeline-train pipeline-deploy ## Run complete end-to-end pipeline
	@echo "$(GREEN)✓✓✓ All pipelines complete ✓✓✓$(NC)"

