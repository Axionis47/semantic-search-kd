#!/bin/bash
set -e

echo "=========================================="
echo "GCP CPU Training (Cloud Compute)"
echo "=========================================="
echo ""

# Configuration
PROJECT_ID="plotpointe"
REGION="us-central1"
ZONE="us-central1-a"
VM_NAME="semantic-kd-cpu-training-$(date +%s)"
MACHINE_TYPE="n1-highmem-8"  # 8 vCPUs, 52GB RAM (good for CPU training)
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
BOOT_DISK_SIZE="100GB"

# Training configuration
MAX_SAMPLES=1000
EPOCHS=3
BATCH_SIZE=8
STAGE=2
GCS_OUTPUT="gs://plotpointe-semantic-kd-models/kd_student_production_cpu_$(date +%Y%m%d_%H%M%S)"

echo "Configuration:"
echo "  Project: ${PROJECT_ID}"
echo "  Region: ${REGION}"
echo "  Zone: ${ZONE}"
echo "  VM Name: ${VM_NAME}"
echo "  Machine Type: ${MACHINE_TYPE} (8 vCPUs, 52GB RAM)"
echo "  Device: CPU"
echo "  Max Samples: ${MAX_SAMPLES}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Mining Stage: ${STAGE}"
echo "  Output: ${GCS_OUTPUT}"
echo "  Estimated Time: ~2.5 hours"
echo ""

# Step 1: Create VM with startup script
echo "=========================================="
echo "Creating GCP VM with CPU"
echo "=========================================="
echo ""

gcloud compute instances create $VM_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-ssd \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata=startup-script='#!/bin/bash
set -e

echo "=========================================="
echo "VM Startup - Installing Dependencies"
echo "=========================================="

# Update system
apt-get update
apt-get install -y git python3-pip curl

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"

# Create working directory
mkdir -p /workspace
cd /workspace

echo "Downloading code from GCS..."
gsutil -m cp -r gs://plotpointe-semantic-kd-data/package/src .
gsutil -m cp -r gs://plotpointe-semantic-kd-data/package/scripts .
gsutil -m cp -r gs://plotpointe-semantic-kd-data/package/configs .
gsutil cp gs://plotpointe-semantic-kd-data/package/pyproject.toml .
gsutil cp gs://plotpointe-semantic-kd-data/package/poetry.lock .

echo "Downloading data from GCS..."
mkdir -p data/raw data/chunks artifacts/indexes

gsutil -m cp -r gs://plotpointe-semantic-kd-data/raw/msmarco data/raw/
gsutil -m cp -r gs://plotpointe-semantic-kd-data/chunks/msmarco data/chunks/
gsutil -m cp -r gs://plotpointe-semantic-kd-data/indexes/bm25_msmarco artifacts/indexes/

echo "Installing dependencies with Poetry..."
poetry install --only main --no-root

echo "Starting training..."
export PYTHONPATH=/workspace:\$PYTHONPATH
poetry run python scripts/train_kd_pipeline.py \
    --max-samples='$MAX_SAMPLES' \
    --epochs='$EPOCHS' \
    --batch-size='$BATCH_SIZE' \
    --stage='$STAGE' \
    --device=cpu \
    --output-dir=/workspace/kd_student \
    --gcs-output-dir='$GCS_OUTPUT' \
    --log-level=INFO 2>&1 | tee /workspace/training.log

# Upload logs
gsutil cp /workspace/training.log '$GCS_OUTPUT'/training.log

echo "Training complete! Shutting down..."
sudo shutdown -h now
'

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ VM Created Successfully!"
    echo "=========================================="
    echo ""
    echo "VM Name: ${VM_NAME}"
    echo "Zone: ${ZONE}"
    echo "Machine: ${MACHINE_TYPE} (8 vCPUs, 52GB RAM)"
    echo "Device: CPU"
    echo "Output: ${GCS_OUTPUT}"
    echo ""
    echo "Expected completion: ~2.5 hours"
    echo ""
    echo "To monitor progress:"
    echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command='tail -f /workspace/training.log'"
    echo ""
    echo "To view VM logs:"
    echo "  gcloud compute instances get-serial-port-output ${VM_NAME} --zone=${ZONE}"
    echo ""
    echo "To check status:"
    echo "  gcloud compute instances describe ${VM_NAME} --zone=${ZONE} --format='get(status)'"
    echo ""
    echo "To stop VM manually:"
    echo "  gcloud compute instances delete ${VM_NAME} --zone=${ZONE}"
    echo ""
    echo "The VM will auto-shutdown after training completes."
    echo ""
else
    echo ""
    echo "❌ Failed to create VM"
    echo ""
    exit 1
fi

