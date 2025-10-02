#!/bin/bash
set -e

echo "=========================================="
echo "Semantic-KD Full Training Pipeline"
echo "=========================================="
echo ""

# Configuration
PROJECT_ID="plotpointe"
REGION="us-central1"
REPO_NAME="semantic-kd"
IMAGE_NAME="train"
IMAGE_TAG="latest"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# Training parameters
MAX_SAMPLES=${MAX_SAMPLES:-50000}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-32}
STAGE=${STAGE:-3}
DEVICE=${DEVICE:-cuda}

echo "Configuration:"
echo "  Project: ${PROJECT_ID}"
echo "  Region: ${REGION}"
echo "  Image: ${IMAGE_URI}"
echo "  Max Samples: ${MAX_SAMPLES}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Mining Stage: ${STAGE}"
echo "  Device: ${DEVICE}"
echo ""

# Step 1: Build Docker image
echo "Step 1: Building Docker image..."
cd "$(dirname "$0")/.."
docker build -f infra/Dockerfile.train -t ${IMAGE_URI} .
echo "✓ Docker image built"
echo ""

# Step 2: Push to Artifact Registry
echo "Step 2: Pushing to Artifact Registry..."
docker push ${IMAGE_URI}
echo "✓ Image pushed to ${IMAGE_URI}"
echo ""

# Step 3: Submit Vertex AI training job
echo "Step 3: Submitting Vertex AI training job..."

JOB_NAME="semantic-kd-training-$(date +%Y%m%d-%H%M%S)"

gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${JOB_NAME} \
  --worker-pool-spec=machine-type=g2-standard-4,replica-count=1,accelerator-type=NVIDIA_L4,accelerator-count=1,container-image-uri=${IMAGE_URI} \
  --args="--max-samples=${MAX_SAMPLES},--epochs=${EPOCHS},--batch-size=${BATCH_SIZE},--stage=${STAGE},--device=${DEVICE},--log-level=INFO"

echo "✓ Training job submitted: ${JOB_NAME}"
echo ""
echo "Monitor job at:"
echo "https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo ""
echo "To stream logs:"
echo "gcloud ai custom-jobs stream-logs ${JOB_NAME} --region=${REGION}"

