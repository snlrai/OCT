#!/bin/bash

echo "================================================"
echo "OCT Retina Segmentation Web Application"
echo "================================================"
echo ""

echo "Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Python3 is not installed!"
    exit 1
fi
echo ""

echo "Checking model file..."
if [ ! -f "unet_retina_segmentation_last_fold.pth" ]; then
    echo "WARNING: Model file not found!"
    echo "Please ensure unet_retina_segmentation_last_fold.pth is in this directory."
    exit 1
fi
echo "Model file found!"
echo ""

echo "Installing dependencies..."
pip3 install -r requirements_webapp.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies!"
    exit 1
fi
echo ""

echo "Starting web application..."
echo "Open your browser and go to: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""
python3 app.py

