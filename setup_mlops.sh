#!/bin/bash

# OCT MLOps Quick Setup Script
# This script guides you through the complete setup process

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "======================================================================="
echo "    OCT MLOps Pipeline - Interactive Setup"
echo "======================================================================="
echo -e "${NC}"
echo ""
echo "This script will guide you through setting up your MLOps pipeline."
echo "You can run this setup without actually starting any AWS services."
echo ""

# Check Python
echo -e "${YELLOW}Step 1: Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found!${NC}"
    echo "Please install Python 3.9 or higher from python.org"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
echo ""

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}✗ pip not found!${NC}"
    echo "Please install pip"
    exit 1
fi
echo -e "${GREEN}✓ pip found${NC}"
echo ""

# Check if virtual environment exists
echo -e "${YELLOW}Step 2: Setting up Python virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || {
    echo -e "${YELLOW}⚠ Please activate manually:${NC}"
    echo "  Linux/Mac: source venv/bin/activate"
    echo "  Windows: venv\\Scripts\\activate"
    exit 1
}
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Install dependencies
echo -e "${YELLOW}Step 3: Installing dependencies...${NC}"
echo "This may take a few minutes..."
pip install -q --upgrade pip
pip install -q -r requirements_mlops.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# AWS Configuration
echo -e "${YELLOW}Step 4: AWS Configuration${NC}"
if [ ! -f "aws_config/.env.aws" ]; then
    echo "Creating AWS configuration file..."
    cp aws_config/.env.aws.template aws_config/.env.aws
    echo -e "${GREEN}✓ Configuration template created${NC}"
    echo ""
    echo -e "${YELLOW}⚠ IMPORTANT: You need to edit aws_config/.env.aws with your AWS credentials${NC}"
    echo ""
    read -p "Do you want to configure AWS credentials now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Please enter your AWS credentials:"
        read -p "AWS Access Key ID: " aws_key_id
        read -p "AWS Secret Access Key: " aws_secret_key
        read -p "AWS Region (default: us-east-1): " aws_region
        aws_region=${aws_region:-us-east-1}
        
        # Generate unique bucket suffix
        UNIQUE_SUFFIX=$(date +%s | tail -c 5)
        
        # Update .env.aws file
        sed -i.bak "s/AWS_ACCESS_KEY_ID=.*/AWS_ACCESS_KEY_ID=$aws_key_id/" aws_config/.env.aws
        sed -i.bak "s/AWS_SECRET_ACCESS_KEY=.*/AWS_SECRET_ACCESS_KEY=$aws_secret_key/" aws_config/.env.aws
        sed -i.bak "s/AWS_DEFAULT_REGION=.*/AWS_DEFAULT_REGION=$aws_region/" aws_config/.env.aws
        sed -i.bak "s/S3_DATA_BUCKET=.*/S3_DATA_BUCKET=oct-mlops-data-dev-$UNIQUE_SUFFIX/" aws_config/.env.aws
        sed -i.bak "s/S3_MODELS_BUCKET=.*/S3_MODELS_BUCKET=oct-mlops-models-dev-$UNIQUE_SUFFIX/" aws_config/.env.aws
        sed -i.bak "s/S3_ARTIFACTS_BUCKET=.*/S3_ARTIFACTS_BUCKET=oct-mlops-artifacts-dev-$UNIQUE_SUFFIX/" aws_config/.env.aws
        rm aws_config/.env.aws.bak 2>/dev/null || true
        
        echo -e "${GREEN}✓ AWS credentials configured${NC}"
    else
        echo ""
        echo -e "${YELLOW}Please edit aws_config/.env.aws manually before proceeding${NC}"
        echo "Template location: aws_config/.env.aws"
    fi
else
    echo -e "${GREEN}✓ AWS configuration already exists${NC}"
fi
echo ""

# Check AWS CLI
echo -e "${YELLOW}Step 5: Checking AWS CLI...${NC}"
if command -v aws &> /dev/null; then
    echo -e "${GREEN}✓ AWS CLI installed${NC}"
    AWS_IDENTITY=$(aws sts get-caller-identity 2>/dev/null || echo "")
    if [ -n "$AWS_IDENTITY" ]; then
        echo -e "${GREEN}✓ AWS credentials valid${NC}"
    else
        echo -e "${YELLOW}⚠ AWS credentials not configured in AWS CLI${NC}"
        echo "You can configure them later with: aws configure"
    fi
else
    echo -e "${YELLOW}⚠ AWS CLI not installed${NC}"
    echo "Download from: https://aws.amazon.com/cli/"
    echo "This is optional - you can proceed without it"
fi
echo ""

# Test local setup
echo -e "${YELLOW}Step 6: Testing local setup...${NC}"
python -c "from mlops import ExperimentTracker, ModelRegistry; print('MLOps modules imported successfully')" 2>/dev/null && {
    echo -e "${GREEN}✓ MLOps modules working${NC}"
} || {
    echo -e "${RED}✗ MLOps modules import failed${NC}"
    exit 1
}
echo ""

# Summary
echo -e "${BLUE}"
echo "======================================================================="
echo "    Setup Complete!"
echo "======================================================================="
echo -e "${NC}"
echo ""
echo -e "${GREEN}✓ Python environment ready${NC}"
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo -e "${GREEN}✓ Configuration files created${NC}"
echo -e "${GREEN}✓ MLOps modules working${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo "1. Test locally (no AWS needed):"
echo "   ${BLUE}python -m mlops.training_pipeline --experiment-name test --epochs 5 --local${NC}"
echo ""
echo "2. Setup AWS resources (preview first):"
echo "   ${BLUE}python scripts/setup_aws.py --dry-run${NC}"
echo ""
echo "3. Actually create AWS resources:"
echo "   ${BLUE}python scripts/setup_aws.py${NC}"
echo ""
echo "4. Deploy Lambda function:"
echo "   ${BLUE}bash scripts/deploy_lambda.sh${NC}"
echo ""
echo "5. Check costs:"
echo "   ${BLUE}python scripts/cost_monitor.py${NC}"
echo ""
echo "6. Read the documentation:"
echo "   - Quick Start: ${BLUE}QUICK_START.md${NC}"
echo "   - Full Guide: ${BLUE}README_MLOPS.md${NC}"
echo "   - Summary: ${BLUE}MLOPS_SUMMARY.md${NC}"
echo ""
echo -e "${GREEN}Happy MLOps! 🚀${NC}"
echo ""

