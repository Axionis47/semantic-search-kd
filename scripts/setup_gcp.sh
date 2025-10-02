#!/bin/bash
# Setup GCP infrastructure for semantic-kd

set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

PROJECT_ID=${GCP_PROJECT_ID:-plotpointe}
REGION=${GCP_REGION:-us-central1}

echo "=== Setting up GCP infrastructure ==="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Set project
gcloud config set project $PROJECT_ID

# Create GCS buckets
echo ""
echo "Creating GCS buckets..."

BUCKETS=(
    "plotpointe-semantic-kd-data"
    "plotpointe-semantic-kd-models"
    "plotpointe-semantic-kd-indexes"
)

for BUCKET in "${BUCKETS[@]}"; do
    if gsutil ls -b gs://$BUCKET 2>/dev/null; then
        echo "✓ Bucket gs://$BUCKET already exists"
    else
        echo "Creating bucket gs://$BUCKET..."
        gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET
        echo "✓ Created gs://$BUCKET"
    fi
done

# Enable required APIs (if not already enabled)
echo ""
echo "Ensuring required APIs are enabled..."

APIS=(
    "aiplatform.googleapis.com"
    "storage.googleapis.com"
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "artifactregistry.googleapis.com"
)

for API in "${APIS[@]}"; do
    if gcloud services list --enabled --filter="name:$API" --format="value(name)" | grep -q "$API"; then
        echo "✓ $API already enabled"
    else
        echo "Enabling $API..."
        gcloud services enable $API
    fi
done

# Create Artifact Registry repository for Docker images
echo ""
echo "Setting up Artifact Registry..."

REPO_NAME="semantic-kd"
REPO_LOCATION=$REGION

if gcloud artifacts repositories describe $REPO_NAME --location=$REPO_LOCATION 2>/dev/null; then
    echo "✓ Artifact Registry repository '$REPO_NAME' already exists"
else
    echo "Creating Artifact Registry repository..."
    gcloud artifacts repositories create $REPO_NAME \
        --repository-format=docker \
        --location=$REPO_LOCATION \
        --description="Docker images for semantic-kd"
    echo "✓ Created repository '$REPO_NAME'"
fi

# Create service account for Vertex AI training
echo ""
echo "Setting up service account..."

SA_NAME="semantic-kd-training"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if gcloud iam service-accounts describe $SA_EMAIL 2>/dev/null; then
    echo "✓ Service account $SA_EMAIL already exists"
else
    echo "Creating service account..."
    gcloud iam service-accounts create $SA_NAME \
        --display-name="Semantic KD Training Service Account"
    echo "✓ Created service account $SA_EMAIL"
fi

# Grant necessary permissions
echo "Granting permissions to service account..."

ROLES=(
    "roles/storage.objectAdmin"
    "roles/aiplatform.user"
    "roles/logging.logWriter"
)

for ROLE in "${ROLES[@]}"; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SA_EMAIL" \
        --role="$ROLE" \
        --condition=None \
        --quiet
done

echo "✓ Permissions granted"

echo ""
echo "=== GCP Setup Complete ==="
echo ""
echo "Buckets created:"
for BUCKET in "${BUCKETS[@]}"; do
    echo "  - gs://$BUCKET"
done
echo ""
echo "Service account: $SA_EMAIL"
echo "Artifact Registry: $REPO_LOCATION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME"
echo ""
echo "Next steps:"
echo "  1. Run 'make data-fetch' to download datasets"
echo "  2. Run 'make pipeline-data' to prepare data and upload to GCS"
echo "  3. Run 'make train-vertex' to start training on Vertex AI"

