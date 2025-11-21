@echo off
echo ================================================
echo OCT Retina Segmentation Web Application
echo ================================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Python is not installed or not in PATH!
    pause
    exit /b 1
)
echo.

echo Checking model file...
if not exist "unet_retina_segmentation_last_fold.pth" (
    echo WARNING: Model file not found!
    echo Please ensure unet_retina_segmentation_last_fold.pth is in this directory.
    pause
    exit /b 1
)
echo Model file found!
echo.

echo Installing dependencies...
pip install -r requirements_webapp.txt
if errorlevel 1 (
    echo Failed to install dependencies!
    pause
    exit /b 1
)
echo.

echo Starting web application...
echo Open your browser and go to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python app.py

pause

