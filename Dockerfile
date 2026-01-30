# =============================================================================
# Multi-stage Dockerfile for Semantic Search with Knowledge Distillation
# =============================================================================
#
# Build stages:
# 1. builder - Install dependencies with Poetry
# 2. runtime - Minimal production image
#
# Usage:
#   docker build -t semantic-kd:latest .
#   docker run -p 8080:8080 semantic-kd:latest
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=1.7.0 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

# Add Poetry to PATH
ENV PATH="$POETRY_HOME/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies (production only)
RUN poetry install --no-root --only main

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Install project
RUN poetry install --only main

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM python:3.10-slim as runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    # App configuration
    SEMANTIC_KD_ENVIRONMENT=production \
    SEMANTIC_KD_SERVICE__HOST=0.0.0.0 \
    SEMANTIC_KD_SERVICE__PORT=8080 \
    # GCS model path (must be set at deploy time via --set-env-vars or docker run -e)
    GCS_MODEL_PATH=""

# Install system dependencies including gsutil for model download
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs

# Copy entrypoint script
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create directories for artifacts
RUN mkdir -p /app/artifacts/models/kd_student_production /app/artifacts/indexes /app/data && \
    chown -R appuser:appgroup /app

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check (extended start-period for model download)
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Use entrypoint to download model from GCS before starting
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8080"]

# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
LABEL org.opencontainers.image.title="Semantic Search with Knowledge Distillation" \
      org.opencontainers.image.description="Production-grade semantic search API" \
      org.opencontainers.image.version="1.1.0" \
      org.opencontainers.image.source="https://github.com/example/semantic-kd"
