#!/bin/bash
set -e

echo "=========================================="
echo "GCP Training with L4 GPU (US East)"
echo "=========================================="
echo ""

# Configuration
PROJECT_ID="plotpointe"
REGION="us-east1"
ZONE="us-east1-b"  # L4 available here
VM_NAME="semantic-kd-training-$(date +%s)"
MACHINE_TYPE="g2-standard-4"  # L4 GPU machine
ACCELERATOR="type=nvidia-l4,count=1"
IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_SIZE="100GB"

# Training configuration
MAX_SAMPLES=1000
EPOCHS=3
BATCH_SIZE=32  # Can use larger batch with A100
STAGE=2
GCS_OUTPUT="gs://plotpointe-semantic-kd-models/kd_student_production_$(date +%Y%m%d_%H%M%S)"

echo "Configuration:"
echo "  Project: ${PROJECT_ID}"
echo "  Region: ${REGION}"
echo "  Zone: ${ZONE}"
echo "  VM Name: ${VM_NAME}"
echo "  Machine Type: ${MACHINE_TYPE}"
echo "  GPU: NVIDIA L4 (24GB)"
echo "  Max Samples: ${MAX_SAMPLES}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Mining Stage: ${STAGE}"
echo "  Output: ${GCS_OUTPUT}"
echo ""

# Step 1: Verify data in GCS
echo "=========================================="
echo "Step 1: Verify Data in GCS"
echo "=========================================="
echo ""

if gsutil ls gs://plotpointe-semantic-kd-data/raw/msmarco/ &>/dev/null; then
    echo "✓ MS MARCO data found in GCS"
else
    echo "Uploading MS MARCO data..."
    gsutil -m cp -r data/raw/msmarco gs://plotpointe-semantic-kd-data/raw/
fi

if gsutil ls gs://plotpointe-semantic-kd-data/chunks/msmarco/ &>/dev/null; then
    echo "✓ Chunks found in GCS"
else
    echo "Uploading chunks..."
    gsutil -m cp -r data/chunks/msmarco gs://plotpointe-semantic-kd-data/chunks/
fi

if gsutil ls gs://plotpointe-semantic-kd-data/indexes/bm25_msmarco/ &>/dev/null; then
    echo "✓ BM25 index found in GCS"
else
    echo "Uploading BM25 index..."
    gsutil -m cp -r artifacts/indexes/bm25_msmarco gs://plotpointe-semantic-kd-data/indexes/
fi

echo ""
echo "✓ All data available in GCS"
echo ""

# Step 2: Create VM with startup script
echo "=========================================="
echo "Step 2: Creating GCP VM with A100 GPU"
echo "=========================================="
echo ""

gcloud compute instances create $VM_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --accelerator=$ACCELERATOR \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata=startup-script='#!/bin/bash
set -e

echo "=========================================="
echo "VM Startup - Installing Dependencies"
echo "=========================================="

# Wait for GPU drivers
sleep 30
nvidia-smi

# Update system
apt-get update
apt-get install -y git python3-pip curl

# Create working directory
mkdir -p /workspace
cd /workspace

# Download all code from GCS
echo "Downloading code and data from GCS..."
mkdir -p data/raw data/chunks artifacts/indexes src scripts configs

gsutil -m cp -r gs://plotpointe-semantic-kd-data/package/src . || echo "Downloading src failed"
gsutil -m cp -r gs://plotpointe-semantic-kd-data/package/scripts . || echo "Downloading scripts failed"
gsutil -m cp -r gs://plotpointe-semantic-kd-data/package/configs . || echo "Downloading configs failed"

gsutil -m cp -r gs://plotpointe-semantic-kd-data/raw/msmarco data/raw/
gsutil -m cp -r gs://plotpointe-semantic-kd-data/chunks/msmarco data/chunks/
gsutil -m cp -r gs://plotpointe-semantic-kd-data/indexes/bm25_msmarco artifacts/indexes/

# Install required packages
echo "Installing Python packages..."
pip3 install torch transformers sentence-transformers pandas loguru rank-bm25 tqdm pyyaml

# Set Python path
export PYTHONPATH=/workspace:$PYTHONPATH

echo "Starting training..."
python3 scripts/train_kd_pipeline.py \
    --max-samples='$MAX_SAMPLES' \
    --epochs='$EPOCHS' \
    --batch-size='$BATCH_SIZE' \
    --stage='$STAGE' \
    --device=cuda \
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
    echo "GPU: A100 80GB"
    echo "Output: ${GCS_OUTPUT}"
    echo ""
    echo "To monitor progress:"
    echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command='tail -f /workspace/training.log'"
    echo ""
    echo "To check GPU usage:"
    echo "  gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command='nvidia-smi'"
    echo ""
    echo "To view VM logs:"
    echo "  gcloud compute instances get-serial-port-output ${VM_NAME} --zone=${ZONE}"
    echo ""
    echo "To stop VM manually:"
    echo "  gcloud compute instances delete ${VM_NAME} --zone=${ZONE}"
    echo ""
    echo "Expected completion: ~15-30 minutes (A100 is very fast!)"
    echo ""
else
    echo ""
    echo "❌ Failed to create VM"
    echo ""
    exit 1
fi

