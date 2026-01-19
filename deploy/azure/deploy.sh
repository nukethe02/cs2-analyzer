#!/bin/bash
# OpenSight Azure Container Apps Deployment Script

set -e

# Configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-opensight-rg}"
LOCATION="${AZURE_LOCATION:-eastus}"
ACR_NAME="${AZURE_ACR_NAME:-opensightacr}"
ENVIRONMENT_NAME="${AZURE_ENV_NAME:-opensight-env}"
APP_NAME="opensight"

echo "OpenSight Azure Container Apps Deployment"
echo "=========================================="

# Check for required tools
command -v az >/dev/null 2>&1 || { echo "Azure CLI is required but not installed."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed."; exit 1; }

# Login check
az account show >/dev/null 2>&1 || { echo "Please login with: az login"; exit 1; }

SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo "Subscription: $SUBSCRIPTION_ID"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo ""

# Create resource group if it doesn't exist
echo "Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION --output none 2>/dev/null || true

# Create Azure Container Registry if it doesn't exist
echo "Creating Azure Container Registry..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true \
    --output none 2>/dev/null || true

# Get ACR credentials
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query 'passwords[0].value' -o tsv)

# Login to ACR
echo "Logging into ACR..."
docker login $ACR_LOGIN_SERVER -u $ACR_USERNAME -p $ACR_PASSWORD

# Build and push Docker image
IMAGE_NAME="$ACR_LOGIN_SERVER/$APP_NAME:latest"
echo "Building Docker image..."
docker build -t $IMAGE_NAME ../../

echo "Pushing image to ACR..."
docker push $IMAGE_NAME

# Create Container Apps environment if it doesn't exist
echo "Creating Container Apps environment..."
az containerapp env create \
    --name $ENVIRONMENT_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --output none 2>/dev/null || true

# Deploy Container App
echo "Deploying Container App..."
az containerapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $ENVIRONMENT_NAME \
    --image $IMAGE_NAME \
    --target-port 7860 \
    --ingress external \
    --registry-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --cpu 1.0 \
    --memory 2Gi \
    --min-replicas 0 \
    --max-replicas 10 \
    --env-vars "PYTHONPATH=src" "LOG_LEVEL=INFO"

# Get the app URL
APP_URL=$(az containerapp show --name $APP_NAME --resource-group $RESOURCE_GROUP --query 'properties.configuration.ingress.fqdn' -o tsv)

echo ""
echo "Deployment complete!"
echo "===================="
echo "App URL: https://$APP_URL"
echo ""
echo "Health check: curl https://$APP_URL/health"
