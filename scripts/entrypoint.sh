#!/bin/bash
# =============================================================================
# Container Entrypoint Script
# =============================================================================
# Downloads the model from GCS if not present, then starts the application.
# =============================================================================

set -e

MODEL_DIR="/app/artifacts/models/kd_student_production"
GCS_MODEL_PATH="${GCS_MODEL_PATH:-gs://plotpointe-semantic-kd-models/kd_student_production_cpu_20251020_012956/best_model}"

echo "=========================================="
echo "Semantic KD Service Startup"
echo "=========================================="

# Check if model exists locally
if [ -f "${MODEL_DIR}/model.safetensors" ]; then
    echo "Model already exists at ${MODEL_DIR}"
else
    echo "Downloading model from GCS..."
    echo "Source: ${GCS_MODEL_PATH}"
    echo "Destination: ${MODEL_DIR}"

    mkdir -p "${MODEL_DIR}"

    # Download model from GCS
    if command -v gsutil &> /dev/null; then
        gsutil -m cp -r "${GCS_MODEL_PATH}/*" "${MODEL_DIR}/"
        echo "Model downloaded successfully!"
    else
        echo "ERROR: gsutil not available. Please mount the model directory or install gsutil."
        exit 1
    fi
fi

# Verify model files
if [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
    echo "ERROR: Model file not found at ${MODEL_DIR}/model.safetensors"
    exit 1
fi

echo "Model verified. Size: $(du -sh ${MODEL_DIR} | cut -f1)"
echo "=========================================="
echo "Starting application..."
echo "=========================================="

# Execute the main command
exec "$@"
