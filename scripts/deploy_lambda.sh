#!/bin/bash

# Deploy OCT MLOps Lambda Functions
# This script packages and deploys the inference Lambda function to AWS

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "OCT MLOps - Lambda Deployment"
echo "================================================"

# Load environment variables
if [ -f "aws_config/.env.aws" ]; then
    export $(cat aws_config/.env.aws | grep -v '^#' | xargs)
    echo "✓ Loaded AWS configuration"
else
    echo -e "${RED}✗ AWS configuration not found!${NC}"
    echo "Please create aws_config/.env.aws from the template"
    exit 1
fi

# Configuration
FUNCTION_NAME="${PROJECT_NAME}-inference"
LAMBDA_ROLE_NAME="${PROJECT_NAME}-lambda-execution-role"
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
MEMORY_SIZE="${LAMBDA_MEMORY_SIZE:-1024}"
TIMEOUT="${LAMBDA_TIMEOUT:-300}"

echo ""
echo "Configuration:"
echo "  Function Name: $FUNCTION_NAME"
echo "  Region: $REGION"
echo "  Memory: ${MEMORY_SIZE}MB"
echo "  Timeout: ${TIMEOUT}s"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}✗ AWS CLI not installed!${NC}"
    echo "Install from: https://aws.amazon.com/cli/"
    exit 1
fi

# Check AWS credentials
echo "Checking AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}✗ AWS credentials not configured!${NC}"
    echo "Run: aws configure"
    exit 1
fi
echo -e "${GREEN}✓ AWS credentials valid${NC}"

# Get IAM role ARN
echo ""
echo "Getting IAM role ARN..."
ROLE_ARN=$(aws iam get-role --role-name $LAMBDA_ROLE_NAME --query 'Role.Arn' --output text 2>/dev/null || echo "")

if [ -z "$ROLE_ARN" ]; then
    echo -e "${YELLOW}⚠ Lambda execution role not found. Creating...${NC}"
    
    # Create trust policy
    cat > /tmp/lambda-trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
    
    # Create role
    ROLE_ARN=$(aws iam create-role \
        --role-name $LAMBDA_ROLE_NAME \
        --assume-role-policy-document file:///tmp/lambda-trust-policy.json \
        --query 'Role.Arn' \
        --output text)
    
    # Attach policies
    aws iam attach-role-policy \
        --role-name $LAMBDA_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
    
    # Attach custom policy
    if [ -f "aws_config/iam_policies/lambda_execution_role_policy.json" ]; then
        aws iam put-role-policy \
            --role-name $LAMBDA_ROLE_NAME \
            --policy-name ${PROJECT_NAME}-lambda-policy \
            --policy-document file://aws_config/iam_policies/lambda_execution_role_policy.json
    fi
    
    echo -e "${GREEN}✓ Created IAM role${NC}"
    echo "  Waiting 10s for role to propagate..."
    sleep 10
fi

echo -e "${GREEN}✓ IAM Role ARN: $ROLE_ARN${NC}"

# Create deployment package
echo ""
echo "Creating deployment package..."
DEPLOY_DIR="lambda_deploy"
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

# Copy Lambda handler
cp lambda/inference_handler.py $DEPLOY_DIR/

# Copy MLOps modules (minimal)
mkdir -p $DEPLOY_DIR/mlops
cp mlops/__init__.py $DEPLOY_DIR/mlops/
cp mlops/config.py $DEPLOY_DIR/mlops/
cp mlops/model_registry.py $DEPLOY_DIR/mlops/

# Copy environment config (template only, no secrets)
mkdir -p $DEPLOY_DIR/aws_config
cat > $DEPLOY_DIR/aws_config/.env.aws <<EOF
AWS_DEFAULT_REGION=$REGION
PROJECT_NAME=$PROJECT_NAME
S3_MODELS_BUCKET=$S3_MODELS_BUCKET
DYNAMODB_MODELS_TABLE=$DYNAMODB_MODELS_TABLE
EOF

# Install dependencies
echo "Installing Python dependencies..."
pip install --target $DEPLOY_DIR -r docker/requirements.lambda.txt --quiet

# Create ZIP package
echo "Creating ZIP archive..."
cd $DEPLOY_DIR
zip -r9 ../lambda_function.zip . -q
cd ..
echo -e "${GREEN}✓ Deployment package created: lambda_function.zip${NC}"

# Get package size
PACKAGE_SIZE=$(du -h lambda_function.zip | cut -f1)
echo "  Package size: $PACKAGE_SIZE"

# Check if function exists
FUNCTION_EXISTS=$(aws lambda get-function --function-name $FUNCTION_NAME --region $REGION 2>/dev/null || echo "")

if [ -z "$FUNCTION_EXISTS" ]; then
    # Create new function
    echo ""
    echo "Creating Lambda function..."
    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime python3.9 \
        --role $ROLE_ARN \
        --handler inference_handler.lambda_handler \
        --zip-file fileb://lambda_function.zip \
        --timeout $TIMEOUT \
        --memory-size $MEMORY_SIZE \
        --region $REGION \
        --environment "Variables={S3_MODELS_BUCKET=$S3_MODELS_BUCKET,PROJECT_NAME=$PROJECT_NAME}" \
        --description "OCT image inference (classification and segmentation)" \
        > /dev/null
    
    echo -e "${GREEN}✓ Lambda function created!${NC}"
else
    # Update existing function
    echo ""
    echo "Updating Lambda function code..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb://lambda_function.zip \
        --region $REGION \
        > /dev/null
    
    echo "Updating Lambda function configuration..."
    aws lambda update-function-configuration \
        --function-name $FUNCTION_NAME \
        --timeout $TIMEOUT \
        --memory-size $MEMORY_SIZE \
        --environment "Variables={S3_MODELS_BUCKET=$S3_MODELS_BUCKET,PROJECT_NAME=$PROJECT_NAME}" \
        --region $REGION \
        > /dev/null
    
    echo -e "${GREEN}✓ Lambda function updated!${NC}"
fi

# Get function ARN
FUNCTION_ARN=$(aws lambda get-function --function-name $FUNCTION_NAME --region $REGION --query 'Configuration.FunctionArn' --output text)
echo ""
echo "Function ARN: $FUNCTION_ARN"

# Create/Update API Gateway (optional)
echo ""
read -p "Do you want to create an API Gateway endpoint? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating API Gateway..."
    
    API_NAME="${PROJECT_NAME}-api"
    
    # Check if API exists
    API_ID=$(aws apigatewayv2 get-apis --region $REGION --query "Items[?Name=='$API_NAME'].ApiId" --output text 2>/dev/null || echo "")
    
    if [ -z "$API_ID" ]; then
        # Create HTTP API
        API_ID=$(aws apigatewayv2 create-api \
            --name $API_NAME \
            --protocol-type HTTP \
            --target $FUNCTION_ARN \
            --region $REGION \
            --query 'ApiId' \
            --output text)
        
        echo -e "${GREEN}✓ Created API Gateway${NC}"
    else
        echo -e "${YELLOW}⚠ API Gateway already exists${NC}"
    fi
    
    # Add Lambda permission for API Gateway
    aws lambda add-permission \
        --function-name $FUNCTION_NAME \
        --statement-id apigateway-invoke \
        --action lambda:InvokeFunction \
        --principal apigateway.amazonaws.com \
        --region $REGION \
        2>/dev/null || echo "Permission already exists"
    
    # Get API endpoint
    API_ENDPOINT=$(aws apigatewayv2 get-apis --region $REGION --query "Items[?Name=='$API_NAME'].ApiEndpoint" --output text)
    
    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}✅ Deployment Complete!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo "API Endpoint: $API_ENDPOINT"
    echo ""
    echo "Test with:"
    echo "  curl -X POST $API_ENDPOINT \\"
    echo "    -H 'Content-Type: application/json' \\"
    echo "    -d '{\"image\": \"<base64_image>\", \"model_type\": \"classification\"}'"
else
    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}✅ Deployment Complete!${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo "To invoke the function directly:"
    echo "  aws lambda invoke --function-name $FUNCTION_NAME \\"
    echo "    --payload '{\"body\": \"{\\\"model_type\\\": \\\"classification\\\"}\"}' \\"
    echo "    response.json"
fi

# Cleanup
echo ""
echo "Cleaning up temporary files..."
rm -rf $DEPLOY_DIR lambda_function.zip /tmp/lambda-trust-policy.json
echo -e "${GREEN}✓ Cleanup complete${NC}"

echo ""
echo "================================================"
echo "Deployment finished successfully!"
echo "================================================"

