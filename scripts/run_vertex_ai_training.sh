#!/bin/bash
set -e

echo "=========================================="
echo "Vertex AI Custom Training Job"
echo "=========================================="
echo ""

# Configuration
PROJECT_ID="plotpointe"
REGION="us-central1"
DISPLAY_NAME="kd-training-production-$(date +%s)"

# Training configuration
MAX_SAMPLES=1000
EPOCHS=3
BATCH_SIZE=16
STAGE=2
DEVICE="cuda"

# GCS paths
GCS_DATA_BUCKET="gs://plotpointe-semantic-kd-data"
GCS_MODEL_BUCKET="gs://plotpointe-semantic-kd-models"
GCS_OUTPUT_DIR="${GCS_MODEL_BUCKET}/kd_student_production_$(date +%Y%m%d_%H%M%S)"

echo "Configuration:"
echo "  Project: ${PROJECT_ID}"
echo "  Region: ${REGION}"
echo "  Display Name: ${DISPLAY_NAME}"
echo "  Max Samples: ${MAX_SAMPLES}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Mining Stage: ${STAGE}"
echo "  Output: ${GCS_OUTPUT_DIR}"
echo ""

# Step 1: Upload data to GCS
echo "=========================================="
echo "Step 1: Verify Data in GCS"
echo "=========================================="
echo ""

echo "Checking GCS data..."
if gsutil ls ${GCS_DATA_BUCKET}/raw/msmarco/ &>/dev/null; then
    echo "✓ MS MARCO data found in GCS"
else
    echo "Uploading MS MARCO data..."
    gsutil -m cp -r data/raw/msmarco ${GCS_DATA_BUCKET}/raw/
fi

if gsutil ls ${GCS_DATA_BUCKET}/chunks/msmarco/ &>/dev/null; then
    echo "✓ Chunks found in GCS"
else
    echo "Uploading chunks..."
    gsutil -m cp -r data/chunks/msmarco ${GCS_DATA_BUCKET}/chunks/
fi

if gsutil ls ${GCS_DATA_BUCKET}/indexes/bm25_msmarco/ &>/dev/null; then
    echo "✓ BM25 index found in GCS"
else
    echo "Uploading BM25 index..."
    gsutil -m cp -r artifacts/indexes/bm25_msmarco ${GCS_DATA_BUCKET}/indexes/
fi

echo ""
echo "✓ All data available in GCS"
echo ""

# Step 2: Submit using gcloud (simpler than Docker)
echo "=========================================="
echo "Step 2: Submit Vertex AI Training Job"
echo "=========================================="
echo ""

# Try with Python script approach (no Docker needed)
echo "Submitting training job using Python script..."

gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name="${DISPLAY_NAME}" \
  --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,python-package-uris=gs://plotpointe-semantic-kd-data/package/semantic_kd-0.1.0.tar.gz,python-module=scripts.train_kd_pipeline,args="--max-samples=${MAX_SAMPLES},--epochs=${EPOCHS},--batch-size=${BATCH_SIZE},--stage=${STAGE},--device=${DEVICE},--gcs-output-dir=${GCS_OUTPUT_DIR}" \
  2>&1 | tee /tmp/vertex_submit.log

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Training job submitted!"
    echo ""

    # Get job ID from output
    JOB_ID=$(grep -oP 'projects/\d+/locations/[^/]+/customJobs/\d+' /tmp/vertex_submit.log | tail -1)

    echo "Job ID: ${JOB_ID}"
    echo "Output: ${GCS_OUTPUT_DIR}"
    echo ""
    echo "To monitor:"
    echo "  gcloud ai custom-jobs list --region=${REGION}"
    echo "  gcloud ai custom-jobs stream-logs ${JOB_ID}"
    echo ""
else
    echo ""
    echo "❌ Failed to submit job"
    echo ""
    echo "This might be due to:"
    echo "1. GPU quota limits"
    echo "2. Missing Python package"
    echo "3. API not enabled"
    echo ""
    echo "Trying alternative approach with pre-built container..."
    echo ""

    # Try with pre-built PyTorch container
    gcloud ai custom-jobs create \
      --region=${REGION} \
      --display-name="${DISPLAY_NAME}" \
      --worker-pool-spec=machine-type=n1-standard-4,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri=gcr.io/deeplearning-platform-release/pytorch-gpu.2-0:latest \
      2>&1

    if [ $? -eq 0 ]; then
        echo "✓ Job submitted with pre-built container"
    else
        echo "❌ All submission methods failed"
        echo ""
        echo "Please check:"
        echo "1. GPU quota: https://console.cloud.google.com/iam-admin/quotas?project=${PROJECT_ID}"
        echo "2. Vertex AI API: https://console.cloud.google.com/apis/library/aiplatform.googleapis.com?project=${PROJECT_ID}"
        echo ""
        exit 1
    fi
fi

