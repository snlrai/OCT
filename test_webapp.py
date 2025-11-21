"""
Quick test script to verify the webapp setup
This doesn't actually run the webapp, just checks if everything is in place
"""

import os
import sys

def check_file(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False

def check_directory(dirpath, description):
    """Check if a directory exists"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✓ {description}: {dirpath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {dirpath}")
        return False

def main():
    print("="*60)
    print("OCT ANALYSIS WEB APPLICATION - SETUP VERIFICATION")
    print("="*60)
    print()
    
    all_good = True
    
    # Check main application files
    print("1. Checking Main Application Files:")
    print("-" * 60)
    all_good &= check_file("app.py", "Main Flask application")
    all_good &= check_file("templates/index.html", "HTML template")
    all_good &= check_file("static/script.js", "JavaScript file")
    all_good &= check_file("static/style.css", "CSS stylesheet")
    print()
    
    # Check model files
    print("2. Checking Model Files:")
    print("-" * 60)
    all_good &= check_file("unet_combined_best.pth", "Segmentation model")
    all_good &= check_file("classification_models/best_oct_classifier.pth", "Classification model")
    all_good &= check_file("classification_models/model_summary.json", "Model summary JSON")
    print()
    
    # Check visualization files
    print("3. Checking Visualization Files:")
    print("-" * 60)
    all_good &= check_file("training_metrics.png", "Segmentation training metrics")
    all_good &= check_file("classification_models/confusion_matrix.png", "Confusion matrix")
    all_good &= check_file("classification_models/training_history.png", "Training history")
    print()
    
    # Check directories
    print("4. Checking Required Directories:")
    print("-" * 60)
    all_good &= check_directory("uploads", "Upload directory")
    all_good &= check_directory("results", "Results directory")
    all_good &= check_directory("predictions", "Predictions directory")
    all_good &= check_directory("static", "Static files directory")
    all_good &= check_directory("templates", "Templates directory")
    print()
    
    # Check Python imports
    print("5. Checking Python Dependencies:")
    print("-" * 60)
    
    try:
        import flask
        print(f"✓ Flask installed (version {flask.__version__})")
    except ImportError:
        print("✗ Flask NOT installed")
        all_good = False
    
    try:
        import torch
        print(f"✓ PyTorch installed (version {torch.__version__})")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⚠ CUDA not available (will use CPU)")
    except ImportError:
        print("✗ PyTorch NOT installed")
        all_good = False
    
    try:
        import torchvision
        print(f"✓ Torchvision installed (version {torchvision.__version__})")
    except ImportError:
        print("✗ Torchvision NOT installed")
        all_good = False
    
    try:
        import cv2
        print(f"✓ OpenCV installed (version {cv2.__version__})")
    except ImportError:
        print("✗ OpenCV NOT installed")
        all_good = False
    
    try:
        import numpy
        print(f"✓ NumPy installed (version {numpy.__version__})")
    except ImportError:
        print("✗ NumPy NOT installed")
        all_good = False
    
    try:
        from PIL import Image
        import PIL
        print(f"✓ Pillow installed (version {PIL.__version__})")
    except ImportError:
        print("✗ Pillow NOT installed")
        all_good = False
    
    print()
    
    # Summary
    print("="*60)
    if all_good:
        print("✓ ALL CHECKS PASSED!")
        print()
        print("You can now start the web application:")
        print("  python app.py")
        print()
        print("Then open your browser to: http://localhost:5000")
    else:
        print("✗ SOME CHECKS FAILED!")
        print()
        print("Please fix the issues above before running the webapp.")
        print()
        print("To install missing Python packages, run:")
        print("  pip install flask torch torchvision opencv-python numpy pillow")
        print()
        print("Make sure all model files are in the correct locations.")
    print("="*60)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())

