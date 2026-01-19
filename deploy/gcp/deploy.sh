#!/bin/bash
# OpenSight GCP Cloud Run Deployment Script

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="opensight"
IMAGE_NAME="gcr.io/$PROJECT_ID/opensight"

echo "OpenSight GCP Cloud Run Deployment"
echo "==================================="

# Check for required tools
command -v gcloud >/dev/null 2>&1 || { echo "gcloud CLI is required but not installed."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed."; exit 1; }

# Check for project ID
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        echo "Error: GCP_PROJECT_ID not set and no default project configured."
        echo "Run: gcloud config set project YOUR_PROJECT_ID"
        exit 1
    fi
fi

echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo ""

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID
gcloud services enable run.googleapis.com --project=$PROJECT_ID
gcloud services enable containerregistry.googleapis.com --project=$PROJECT_ID

# Configure Docker for GCR
echo "Configuring Docker for GCR..."
gcloud auth configure-docker --quiet

# Build Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME:latest ../../

# Push to Container Registry
echo "Pushing image to GCR..."
docker push $IMAGE_NAME:latest

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME:latest \
    --platform managed \
    --region $REGION \
    --port 7860 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --max-instances 10 \
    --allow-unauthenticated \
    --set-env-vars "PYTHONPATH=src,LOG_LEVEL=INFO" \
    --project $PROJECT_ID

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --platform managed --region $REGION --project $PROJECT_ID --format 'value(status.url)')

echo ""
echo "Deployment complete!"
echo "===================="
echo "Service URL: $SERVICE_URL"
echo ""
echo "Health check: curl $SERVICE_URL/health"
