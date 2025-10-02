#!/bin/bash
set -e

echo "=========================================="
echo "GCP VM Training Setup"
echo "=========================================="
echo ""

# Configuration
PROJECT_ID="plotpointe"
REGION="us-central1"
ZONE="us-west1-b"
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
GCS_OUTPUT="gs://plotpointe-semantic-kd-models/kd_student_production"

echo "Configuration:"
echo "  Project: ${PROJECT_ID}"
echo "  Zone: ${ZONE}"
echo "  VM Name: ${VM_NAME}"
echo "  Machine Type: ${MACHINE_TYPE}"
echo "  Accelerator: ${ACCELERATOR}"
echo "  Max Samples: ${MAX_SAMPLES}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Mining Stage: ${STAGE}"
echo ""

# Create startup script
cat > /tmp/startup-script.sh << 'EOF'
#!/bin/bash
set -e

echo "=========================================="
echo "VM Startup - Installing Dependencies"
echo "=========================================="

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/root/.local/bin:$PATH"

# Clone or copy code (will be uploaded separately)
cd /root

# Install dependencies
poetry install --no-dev

echo "✓ Dependencies installed"
echo ""
echo "Ready for training!"
EOF

echo "Creating GCP VM with L4 GPU..."
echo ""

gcloud compute instances create ${VM_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --machine-type=${MACHINE_TYPE} \
  --accelerator=${ACCELERATOR} \
  --image-family=${IMAGE_FAMILY} \
  --image-project=${IMAGE_PROJECT} \
  --boot-disk-size=${BOOT_DISK_SIZE} \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --metadata-from-file=startup-script=/tmp/startup-script.sh \
  --scopes=https://www.googleapis.com/auth/cloud-platform

echo ""
echo "✓ VM created: ${VM_NAME}"
echo ""

# Wait for VM to be ready
echo "Waiting for VM to be ready..."
sleep 30

# Copy code to VM
echo "Copying code to VM..."
gcloud compute scp --recurse \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  . ${VM_NAME}:/root/semantic-kd/ \
  --exclude=".git" \
  --exclude="data/raw" \
  --exclude="artifacts" \
  --exclude="*.log" \
  --exclude="__pycache__" \
  --exclude=".venv"

echo "✓ Code copied"
echo ""

# Copy data to VM
echo "Copying MS MARCO data to VM..."
gcloud compute scp --recurse \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  data/raw/msmarco ${VM_NAME}:/root/semantic-kd/data/raw/

echo "✓ Data copied"
echo ""

# Run training
echo "=========================================="
echo "Starting Training on VM"
echo "=========================================="
echo ""

gcloud compute ssh ${VM_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --command="
    set -e
    cd /root/semantic-kd
    export PATH=\"/root/.local/bin:\$PATH\"
    
    echo 'Installing dependencies...'
    poetry install --no-dev
    
    echo ''
    echo 'Starting training...'
    poetry run python scripts/train_kd_pipeline.py \
      --max-samples ${MAX_SAMPLES} \
      --epochs ${EPOCHS} \
      --batch-size ${BATCH_SIZE} \
      --stage ${STAGE} \
      --device cuda \
      --output-dir artifacts/models/kd_student_production \
      --gcs-output-dir ${GCS_OUTPUT} \
      --log-level INFO
  "

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""

# Copy results back
echo "Copying results from VM..."
gcloud compute scp --recurse \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  ${VM_NAME}:/root/semantic-kd/artifacts/models/kd_student_production \
  ./artifacts/models/

echo "✓ Results copied"
echo ""

# Cleanup
echo "Cleaning up VM..."
gcloud compute instances delete ${VM_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --quiet

echo "✓ VM deleted"
echo ""

echo "=========================================="
echo "Complete!"
echo "=========================================="
echo ""
echo "📁 Artifacts:"
echo "  - Local: ./artifacts/models/kd_student_production"
echo "  - GCS: ${GCS_OUTPUT}"
echo ""
echo "🎯 Next: Run evaluation"
echo "  poetry run python scripts/evaluate_and_compare.py \\"
echo "    --vanilla-model intfloat/e5-small-v2 \\"
echo "    --kd-model ./artifacts/models/kd_student_production/best_model \\"
echo "    --teacher-model BAAI/bge-reranker-large \\"
echo "    --data-path data/raw/msmarco/test.jsonl \\"
echo "    --output-dir artifacts/evaluation_production"
echo ""

