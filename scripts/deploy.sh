#!/bin/bash
# =============================================================================
# Deployment Script for Semantic Search with Knowledge Distillation
# =============================================================================
#
# This script handles deployment to different environments:
# - local: Run locally with uvicorn
# - docker: Run locally with Docker
# - staging: Deploy to GCP Cloud Run (staging)
# - production: Deploy to GCP Cloud Run (production)
#
# Usage:
#   ./scripts/deploy.sh local
#   ./scripts/deploy.sh docker
#   ./scripts/deploy.sh staging
#   ./scripts/deploy.sh production
#
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-plotpointe}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="semantic-kd"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
MODEL_PATH="./artifacts/models/kd_student_production"

# Get version from git
VERSION=$(git rev-parse --short HEAD 2>/dev/null || echo "local")
IMAGE_TAG="${IMAGE_NAME}:${VERSION}"

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_model_exists() {
    if [ ! -d "${MODEL_PATH}" ]; then
        echo_error "Model not found at ${MODEL_PATH}"
        echo_info "Please ensure the trained model is available."
        echo_info "You can download it from GCS:"
        echo_info "  gsutil -m cp -r gs://plotpointe-semantic-kd-models/kd_student_production_cpu_20251020_012956/best_model/* ${MODEL_PATH}/"
        exit 1
    fi
    echo_success "Model found at ${MODEL_PATH}"
}

run_model_validation() {
    echo_info "Running model validation tests..."
    poetry run pytest tests/test_model_validation.py -v --tb=short
    if [ $? -eq 0 ]; then
        echo_success "Model validation passed!"
    else
        echo_error "Model validation failed!"
        exit 1
    fi
}

deploy_local() {
    echo_info "Starting local deployment..."
    check_model_exists

    # Optionally run validation
    if [ "${SKIP_VALIDATION:-false}" != "true" ]; then
        run_model_validation
    fi

    echo_info "Starting uvicorn server..."
    echo_info "API will be available at http://localhost:8080"
    echo_info "Press Ctrl+C to stop"

    poetry run uvicorn src.serve.app:app \
        --host 0.0.0.0 \
        --port 8080 \
        --reload
}

deploy_docker() {
    echo_info "Starting Docker deployment..."
    check_model_exists

    # Build image
    echo_info "Building Docker image..."
    docker build -t ${SERVICE_NAME}:local .

    # Run container
    echo_info "Starting container..."
    docker run -d \
        --name ${SERVICE_NAME}-local \
        -p 8080:8080 \
        -v "$(pwd)/${MODEL_PATH}:/app/artifacts/models/kd_student_production:ro" \
        -e SEMANTIC_KD_ENVIRONMENT=development \
        ${SERVICE_NAME}:local

    echo_success "Container started!"
    echo_info "API available at http://localhost:8080"
    echo_info "View logs: docker logs -f ${SERVICE_NAME}-local"
    echo_info "Stop: docker stop ${SERVICE_NAME}-local && docker rm ${SERVICE_NAME}-local"
}

deploy_staging() {
    echo_info "Starting staging deployment..."
    check_model_exists

    # Run validation
    run_model_validation

    # Build and push image
    echo_info "Building and pushing Docker image..."
    docker build -t ${IMAGE_TAG} .
    docker push ${IMAGE_TAG}

    # Deploy to Cloud Run
    echo_info "Deploying to Cloud Run (staging)..."
    gcloud run deploy ${SERVICE_NAME}-staging \
        --image ${IMAGE_TAG} \
        --region ${REGION} \
        --platform managed \
        --memory 2Gi \
        --cpu 2 \
        --timeout 60 \
        --concurrency 100 \
        --min-instances 0 \
        --max-instances 5 \
        --set-env-vars "SEMANTIC_KD_ENVIRONMENT=staging" \
        --allow-unauthenticated

    # Get service URL
    SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME}-staging --region ${REGION} --format 'value(status.url)')

    echo_success "Staging deployment complete!"
    echo_info "Service URL: ${SERVICE_URL}"

    # Run smoke tests
    echo_info "Running smoke tests..."
    curl -f "${SERVICE_URL}/health" && echo ""
    echo_success "Smoke tests passed!"
}

deploy_production() {
    echo_info "Starting production deployment..."
    check_model_exists

    # Confirm production deployment
    echo_warning "You are about to deploy to PRODUCTION!"
    read -p "Are you sure? (yes/no): " confirm
    if [ "${confirm}" != "yes" ]; then
        echo_info "Deployment cancelled."
        exit 0
    fi

    # Run validation
    run_model_validation

    # Check staging is healthy
    STAGING_URL=$(gcloud run services describe ${SERVICE_NAME}-staging --region ${REGION} --format 'value(status.url)' 2>/dev/null || echo "")
    if [ -z "${STAGING_URL}" ]; then
        echo_warning "Staging service not found. Consider deploying to staging first."
        read -p "Continue anyway? (yes/no): " confirm
        if [ "${confirm}" != "yes" ]; then
            echo_info "Deployment cancelled."
            exit 0
        fi
    else
        echo_info "Verifying staging is healthy..."
        if ! curl -sf "${STAGING_URL}/health" > /dev/null; then
            echo_error "Staging health check failed!"
            exit 1
        fi
        echo_success "Staging is healthy"
    fi

    # Build and push image
    echo_info "Building and pushing Docker image..."
    docker build -t ${IMAGE_TAG} -t ${IMAGE_NAME}:latest .
    docker push ${IMAGE_TAG}
    docker push ${IMAGE_NAME}:latest

    # Deploy to Cloud Run
    echo_info "Deploying to Cloud Run (production)..."
    gcloud run deploy ${SERVICE_NAME} \
        --image ${IMAGE_TAG} \
        --region ${REGION} \
        --platform managed \
        --memory 4Gi \
        --cpu 4 \
        --timeout 60 \
        --concurrency 200 \
        --min-instances 1 \
        --max-instances 20 \
        --set-env-vars "SEMANTIC_KD_ENVIRONMENT=production,SEMANTIC_KD_SERVICE__AUTH__ENABLED=true,SEMANTIC_KD_SERVICE__RATE_LIMIT__ENABLED=true" \
        --no-allow-unauthenticated

    # Get service URL
    SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --format 'value(status.url)')

    echo_success "Production deployment complete!"
    echo_info "Service URL: ${SERVICE_URL}"
    echo_info "Image: ${IMAGE_TAG}"

    # Update model registry
    echo_info "Updating model registry..."
    poetry run python scripts/model_registry.py promote \
        --name kd-student \
        --version "v${VERSION}" \
        --to-stage production || true

    echo_success "Deployment complete!"
}

rollback() {
    ENV="${1:-production}"
    echo_info "Rolling back ${ENV} deployment..."

    # Get previous revision
    REVISIONS=$(gcloud run revisions list \
        --service ${SERVICE_NAME}$([ "$ENV" == "staging" ] && echo "-staging") \
        --region ${REGION} \
        --format 'value(metadata.name)' \
        --sort-by '~metadata.creationTimestamp' \
        --limit 2)

    PREVIOUS_REVISION=$(echo "${REVISIONS}" | tail -n 1)

    if [ -z "${PREVIOUS_REVISION}" ]; then
        echo_error "No previous revision found!"
        exit 1
    fi

    echo_info "Rolling back to revision: ${PREVIOUS_REVISION}"

    gcloud run services update-traffic \
        ${SERVICE_NAME}$([ "$ENV" == "staging" ] && echo "-staging") \
        --region ${REGION} \
        --to-revisions ${PREVIOUS_REVISION}=100

    echo_success "Rollback complete!"
}

print_usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  local       Run locally with uvicorn (hot reload)"
    echo "  docker      Run locally with Docker"
    echo "  staging     Deploy to GCP Cloud Run (staging)"
    echo "  production  Deploy to GCP Cloud Run (production)"
    echo "  rollback    Rollback to previous revision"
    echo ""
    echo "Options:"
    echo "  SKIP_VALIDATION=true  Skip model validation tests"
    echo ""
    echo "Examples:"
    echo "  $0 local"
    echo "  $0 staging"
    echo "  SKIP_VALIDATION=true $0 local"
}

# Main
case "${1}" in
    local)
        deploy_local
        ;;
    docker)
        deploy_docker
        ;;
    staging)
        deploy_staging
        ;;
    production)
        deploy_production
        ;;
    rollback)
        rollback "${2}"
        ;;
    *)
        print_usage
        exit 1
        ;;
esac
