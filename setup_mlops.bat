@echo off
REM OCT MLOps Quick Setup Script for Windows
REM This script guides you through the complete setup process

echo ======================================================================
echo     OCT MLOps Pipeline - Interactive Setup (Windows)
echo ======================================================================
echo.
echo This script will guide you through setting up your MLOps pipeline.
echo You can run this setup without actually starting any AWS services.
echo.

REM Check Python
echo Step 1: Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo Please install Python 3.9 or higher from python.org
    pause
    exit /b 1
)
echo [OK] Python found
echo.

REM Check pip
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip not found!
    echo Please install pip
    pause
    exit /b 1
)
echo [OK] pip found
echo.

REM Create virtual environment
echo Step 2: Setting up Python virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Install dependencies
echo Step 3: Installing dependencies...
echo This may take a few minutes...
pip install -q --upgrade pip
pip install -q -r requirements_mlops.txt
echo [OK] Dependencies installed
echo.

REM AWS Configuration
echo Step 4: AWS Configuration
if not exist "aws_config\.env.aws" (
    echo Creating AWS configuration file...
    copy aws_config\.env.aws.template aws_config\.env.aws >nul
    echo [OK] Configuration template created
    echo.
    echo [WARNING] You need to edit aws_config\.env.aws with your AWS credentials
    echo.
    echo Please edit aws_config\.env.aws manually with your:
    echo   - AWS Access Key ID
    echo   - AWS Secret Access Key
    echo   - AWS Region
    echo   - Unique bucket names
) else (
    echo [OK] AWS configuration already exists
)
echo.

REM Check AWS CLI
echo Step 5: Checking AWS CLI...
aws --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] AWS CLI installed
) else (
    echo [WARNING] AWS CLI not installed
    echo Download from: https://aws.amazon.com/cli/
    echo This is optional - you can proceed without it
)
echo.

REM Test local setup
echo Step 6: Testing local setup...
python -c "from mlops import ExperimentTracker, ModelRegistry; print('MLOps modules imported successfully')" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] MLOps modules working
) else (
    echo [ERROR] MLOps modules import failed
    pause
    exit /b 1
)
echo.

REM Summary
echo ======================================================================
echo     Setup Complete!
echo ======================================================================
echo.
echo [OK] Python environment ready
echo [OK] Dependencies installed
echo [OK] Configuration files created
echo [OK] MLOps modules working
echo.
echo Next Steps:
echo.
echo 1. Test locally (no AWS needed):
echo    python -m mlops.training_pipeline --experiment-name test --epochs 5 --local
echo.
echo 2. Setup AWS resources (preview first):
echo    python scripts\setup_aws.py --dry-run
echo.
echo 3. Actually create AWS resources:
echo    python scripts\setup_aws.py
echo.
echo 4. Deploy Lambda function:
echo    bash scripts\deploy_lambda.sh
echo    (Requires Git Bash or WSL)
echo.
echo 5. Check costs:
echo    python scripts\cost_monitor.py
echo.
echo 6. Read the documentation:
echo    - Quick Start: QUICK_START.md
echo    - Full Guide: README_MLOPS.md
echo    - Summary: MLOPS_SUMMARY.md
echo.
echo Happy MLOps!
echo.
pause

