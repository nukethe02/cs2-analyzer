# OpenSight Cloud Deployment

This directory contains deployment configurations for AWS, GCP, and Azure.

## Prerequisites

- Docker installed and running
- Cloud provider CLI installed and authenticated

## AWS (ECS Fargate)

### Quick Deploy

```bash
cd aws
export AWS_REGION=us-east-1
export VPC_ID=vpc-xxx
export SUBNET_IDS=subnet-xxx,subnet-yyy
chmod +x deploy.sh
./deploy.sh
```

### Files

- `task-definition.json` - ECS task definition for Fargate
- `cloudformation.yaml` - Full CloudFormation stack
- `deploy.sh` - Automated deployment script

### Manual Deployment

1. Create ECR repository:
   ```bash
   aws ecr create-repository --repository-name opensight
   ```

2. Build and push image:
   ```bash
   docker build -t opensight .
   docker tag opensight:latest ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/opensight:latest
   docker push ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/opensight:latest
   ```

3. Deploy CloudFormation:
   ```bash
   aws cloudformation deploy --template-file cloudformation.yaml --stack-name opensight
   ```

## GCP (Cloud Run)

### Quick Deploy

```bash
cd gcp
export GCP_PROJECT_ID=your-project-id
export GCP_REGION=us-central1
chmod +x deploy.sh
./deploy.sh
```

### Files

- `cloudbuild.yaml` - Cloud Build configuration
- `cloudrun-service.yaml` - Knative service definition
- `deploy.sh` - Automated deployment script

### Using Cloud Build

```bash
gcloud builds submit --config cloudbuild.yaml ../..
```

### Manual Deployment

1. Build and push image:
   ```bash
   docker build -t gcr.io/PROJECT_ID/opensight .
   docker push gcr.io/PROJECT_ID/opensight
   ```

2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy opensight \
       --image gcr.io/PROJECT_ID/opensight \
       --platform managed \
       --region us-central1 \
       --port 7860 \
       --allow-unauthenticated
   ```

## Azure (Container Apps)

### Quick Deploy

```bash
cd azure
export AZURE_RESOURCE_GROUP=opensight-rg
export AZURE_LOCATION=eastus
export AZURE_ACR_NAME=opensightacr
chmod +x deploy.sh
./deploy.sh
```

### Files

- `containerapp.yaml` - Container Apps YAML definition
- `deploy.sh` - Automated deployment script

### Manual Deployment

1. Create resource group and ACR:
   ```bash
   az group create --name opensight-rg --location eastus
   az acr create --name opensightacr --resource-group opensight-rg --sku Basic
   ```

2. Build and push image:
   ```bash
   az acr build --registry opensightacr --image opensight:latest .
   ```

3. Deploy Container App:
   ```bash
   az containerapp create \
       --name opensight \
       --resource-group opensight-rg \
       --environment opensight-env \
       --image opensightacr.azurecr.io/opensight:latest \
       --target-port 7860 \
       --ingress external
   ```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHONPATH` | Python path | `src` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Resources

| Provider | CPU | Memory | Max Instances |
|----------|-----|--------|---------------|
| AWS Fargate | 1 vCPU | 2 GB | 10 |
| GCP Cloud Run | 2 vCPU | 2 GB | 10 |
| Azure Container Apps | 1 vCPU | 2 GB | 10 |

### Health Check

All deployments include a health check at `/health` endpoint.

## HuggingFace Spaces (Free)

For the simplest deployment, push to a HuggingFace Space:

```bash
# Clone the space
git clone https://huggingface.co/spaces/YOUR_USERNAME/opensight

# Copy files
cp -r * ../opensight/

# Push
cd ../opensight
git add .
git commit -m "Deploy OpenSight"
git push
```

The README.md has the required YAML frontmatter for HuggingFace Spaces.

## Monitoring

### AWS
- CloudWatch Logs: `/ecs/opensight`
- CloudWatch Container Insights

### GCP
- Cloud Logging
- Cloud Monitoring

### Azure
- Log Analytics
- Application Insights

## Cost Optimization

### Free Tier Usage

- **AWS**: Fargate doesn't have free tier, but use FARGATE_SPOT for 70% savings
- **GCP**: Cloud Run has generous free tier (2 million requests/month)
- **Azure**: Container Apps has free tier with limits
- **HuggingFace**: Free tier with CPU-only

### Scaling to Zero

All configurations support scaling to zero when not in use.
