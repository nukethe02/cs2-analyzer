#!/bin/bash
# OpenSight AWS Deployment Script
# Deploys to AWS ECS Fargate

set -e

# Configuration
REGION="${AWS_REGION:-us-east-1}"
ECR_REPO="${ECR_REPO:-opensight}"
STACK_NAME="${STACK_NAME:-opensight-stack}"
CLUSTER_NAME="${CLUSTER_NAME:-opensight-cluster}"
SERVICE_NAME="${SERVICE_NAME:-opensight-service}"

echo "OpenSight AWS Deployment"
echo "========================"
echo "Region: $REGION"
echo "ECR Repository: $ECR_REPO"
echo ""

# Check for required tools
command -v aws >/dev/null 2>&1 || { echo "AWS CLI is required but not installed."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed."; exit 1; }

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account: $ACCOUNT_ID"

# Create ECR repository if it doesn't exist
echo "Creating ECR repository..."
aws ecr describe-repositories --repository-names $ECR_REPO --region $REGION 2>/dev/null || \
    aws ecr create-repository --repository-name $ECR_REPO --region $REGION

# Get ECR login
echo "Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Build and push Docker image
IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest"
echo "Building Docker image..."
docker build -t $ECR_REPO:latest ../../

echo "Tagging image..."
docker tag $ECR_REPO:latest $IMAGE_URI

echo "Pushing image to ECR..."
docker push $IMAGE_URI

# Deploy CloudFormation stack (if VPC parameters provided)
if [ -n "$VPC_ID" ] && [ -n "$SUBNET_IDS" ]; then
    echo "Deploying CloudFormation stack..."
    aws cloudformation deploy \
        --template-file cloudformation.yaml \
        --stack-name $STACK_NAME \
        --parameter-overrides \
            ImageUri=$IMAGE_URI \
            VpcId=$VPC_ID \
            SubnetIds=$SUBNET_IDS \
        --capabilities CAPABILITY_NAMED_IAM \
        --region $REGION

    echo "Stack deployed successfully!"
    echo ""
    echo "To get the service URL, run:"
    echo "  aws ecs describe-tasks --cluster $CLUSTER_NAME --tasks \$(aws ecs list-tasks --cluster $CLUSTER_NAME --query 'taskArns[0]' --output text) --query 'tasks[0].attachments[0].details[?name==\`networkInterfaceId\`].value' --output text | xargs aws ec2 describe-network-interfaces --network-interface-ids --query 'NetworkInterfaces[0].Association.PublicIp' --output text"
else
    echo ""
    echo "Image pushed to ECR: $IMAGE_URI"
    echo ""
    echo "To deploy with CloudFormation, set VPC_ID and SUBNET_IDS:"
    echo "  export VPC_ID=vpc-xxx"
    echo "  export SUBNET_IDS=subnet-xxx,subnet-yyy"
    echo "  ./deploy.sh"
fi

echo ""
echo "Deployment complete!"
