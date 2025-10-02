#!/bin/bash
set -e

echo "=========================================="
echo "Upload Training Code to GCS"
echo "=========================================="
echo ""

GCS_BUCKET="gs://plotpointe-semantic-kd-data"

# Upload all source code
echo "Uploading source code..."
gsutil -m cp -r src ${GCS_BUCKET}/package/
gsutil -m cp -r scripts ${GCS_BUCKET}/package/
gsutil -m cp -r configs ${GCS_BUCKET}/package/
gsutil cp pyproject.toml ${GCS_BUCKET}/package/
gsutil cp poetry.lock ${GCS_BUCKET}/package/
gsutil cp README.md ${GCS_BUCKET}/package/

echo ""
echo "✓ Code uploaded to ${GCS_BUCKET}/package/"
echo ""

