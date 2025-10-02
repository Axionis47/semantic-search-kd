#!/bin/bash
set -e

echo "=========================================="
echo "Trying Multiple Zones for GPU Training"
echo "=========================================="
echo ""

# Configuration
PROJECT_ID="plotpointe"
REGION="us-central1"
VM_NAME="semantic-kd-training-$(date +%s)"
MACHINE_TYPE="n1-standard-4"
ACCELERATOR="type=nvidia-tesla-t4,count=1"
IMAGE_FAMILY="pytorch-2-7-cu128-ubuntu-2204-nvidia-570"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_SIZE="100GB"

# Training configuration
MAX_SAMPLES=1000
EPOCHS=3
BATCH_SIZE=16
STAGE=2
GCS_OUTPUT="gs://plotpointe-semantic-kd-models/kd_student_production_$(date +%Y%m%d_%H%M%S)"

# List of zones to try (in order of preference)
ZONES=(
    "us-central1-c"
    "us-central1-f"
    "us-central1-b"
    "us-west1-c"
    "us-west1-a"
    "us-east1-c"
    "us-east1-d"
    "us-east1-b"
)

echo "Will try zones in order:"
for zone in "${ZONES[@]}"; do
    echo "  - $zone"
done
echo ""

# Try each zone
for ZONE in "${ZONES[@]}"; do
    echo "=========================================="
    echo "Trying zone: $ZONE"
    echo "=========================================="
    echo ""
    
    # Try to create VM
    if gcloud compute instances create $VM_NAME \
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

# Install dependencies
apt-get update
apt-get install -y git python3-pip

# Clone repo (replace with your repo)
cd /home
git clone https://github.com/YOUR_USERNAME/semantic-kd.git || echo "Using existing code"
cd semantic-kd

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
/root/.local/bin/poetry install --no-dev

# Download data from GCS
gsutil -m cp -r gs://plotpointe-semantic-kd-data/raw/msmarco data/raw/
gsutil -m cp -r gs://plotpointe-semantic-kd-data/chunks/msmarco data/chunks/
gsutil -m cp -r gs://plotpointe-semantic-kd-data/indexes/bm25_msmarco artifacts/indexes/

# Run training
/root/.local/bin/poetry run python scripts/train_kd_pipeline.py \
    --max-samples='$MAX_SAMPLES' \
    --epochs='$EPOCHS' \
    --batch-size='$BATCH_SIZE' \
    --stage='$STAGE' \
    --device=cuda \
    --output-dir=/tmp/kd_student \
    --gcs-output-dir='$GCS_OUTPUT' \
    --log-level=INFO

# Shutdown VM after completion
sudo shutdown -h now
' 2>&1; then
        
        echo ""
        echo "✓ VM created successfully in zone: $ZONE"
        echo ""
        echo "VM Name: $VM_NAME"
        echo "Zone: $ZONE"
        echo "Output: $GCS_OUTPUT"
        echo ""
        echo "To monitor:"
        echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --command='tail -f /var/log/syslog'"
        echo ""
        echo "To check status:"
        echo "  gcloud compute instances describe $VM_NAME --zone=$ZONE"
        echo ""
        exit 0
    else
        echo ""
        echo "✗ Failed to create VM in zone: $ZONE"
        echo "Trying next zone..."
        echo ""
        sleep 2
    fi
done

echo ""
echo "=========================================="
echo "❌ Failed to create VM in any zone"
echo "=========================================="
echo ""
echo "This is likely due to GPU quota limits."
echo ""
echo "Options:"
echo "1. Request GPU quota increase: https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT_ID"
echo "2. Use Google Colab (free T4 GPU): notebooks/train_on_colab.ipynb"
echo "3. Use Vertex AI Training (may have separate quotas)"
echo ""
exit 1

