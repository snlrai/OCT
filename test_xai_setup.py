"""
Test script to verify XAI module setup
Run this before starting the Flask app to check for issues
"""

import sys
import os

print("="*70)
print("XAI Module Setup Verification")
print("="*70)

# Test 1: Check Python version
print("\n[1/8] Checking Python version...")
print(f"  Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
if sys.version_info >= (3, 7):
    print("  ✓ Python version OK")
else:
    print("  ✗ Python 3.7+ required")
    sys.exit(1)

# Test 2: Check required files exist
print("\n[2/8] Checking required files...")
required_files = [
    'xai_utils.py',
    'xai_routes.py',
    'requirements_xai.txt',
    'OCT_XAI.ipynb',
    'classification_models/best_oct_classifier.pth'
]

all_files_exist = True
for file in required_files:
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"  {status} {file}")
    if not exists:
        all_files_exist = False

if not all_files_exist:
    print("\n  ⚠️  Some files are missing!")
    if not os.path.exists('classification_models/best_oct_classifier.pth'):
        print("  → Train the classification model first using OCT_Classification.ipynb")
    sys.exit(1)

# Test 3: Check XAI dependencies
print("\n[3/8] Checking XAI dependencies...")
missing_deps = []

try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
except ImportError:
    print("  ✗ PyTorch not installed")
    missing_deps.append('torch')

try:
    import captum
    print(f"  ✓ Captum installed")
except ImportError:
    print("  ✗ Captum not installed")
    missing_deps.append('captum')

try:
    import lime
    print(f"  ✓ LIME installed")
except ImportError:
    print("  ✗ LIME not installed")
    missing_deps.append('lime')

try:
    import cv2
    print(f"  ✓ OpenCV installed")
except ImportError:
    print("  ✗ OpenCV not installed")
    missing_deps.append('opencv-python')

try:
    import matplotlib
    print(f"  ✓ Matplotlib installed")
except ImportError:
    print("  ✗ Matplotlib not installed")
    missing_deps.append('matplotlib')

if missing_deps:
    print(f"\n  ⚠️  Missing dependencies: {', '.join(missing_deps)}")
    print("  → Run: pip install -r requirements_xai.txt")
    sys.exit(1)

# Test 4: Import xai_utils
print("\n[4/8] Testing xai_utils.py import...")
try:
    from xai_utils import (
        preprocess_image_for_xai,
        generate_gradcam,
        generate_integrated_gradients,
        generate_lime_explanation,
        CLASS_NAMES
    )
    print(f"  ✓ xai_utils.py imported successfully")
    print(f"  ✓ CLASS_NAMES: {CLASS_NAMES}")
except Exception as e:
    print(f"  ✗ Failed to import xai_utils.py: {e}")
    sys.exit(1)

# Test 5: Import xai_routes
print("\n[5/8] Testing xai_routes.py import...")
try:
    from xai_routes import register_xai_routes, xai_bp
    print(f"  ✓ xai_routes.py imported successfully")
except Exception as e:
    print(f"  ✗ Failed to import xai_routes.py: {e}")
    sys.exit(1)

# Test 6: Check Flask
print("\n[6/8] Checking Flask...")
try:
    from flask import Flask
    print(f"  ✓ Flask installed")
except ImportError:
    print(f"  ✗ Flask not installed")
    print("  → Run: pip install flask")
    sys.exit(1)

# Test 7: Test model loading
print("\n[7/8] Testing model loading...")
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    
    def create_classification_model(num_classes=4):
        model = models.resnet50(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        return model
    
    model = create_classification_model(num_classes=4)
    checkpoint = torch.load('classification_models/best_oct_classifier.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  ✓ Model loaded successfully")
    print(f"  ✓ Validation accuracy: {checkpoint['val_acc']:.2f}%")
except Exception as e:
    print(f"  ✗ Failed to load model: {e}")
    print("  → Make sure you've trained the classification model first")
    sys.exit(1)

# Test 8: Test XAI function
print("\n[8/8] Testing XAI function with dummy data...")
try:
    import torch
    import numpy as np
    
    # Create dummy input
    dummy_tensor = torch.randn(1, 3, 224, 224)
    
    # Test Grad-CAM
    from xai_utils import generate_gradcam
    heatmap, pred_class, confidence, probs = generate_gradcam(model, dummy_tensor)
    
    print(f"  ✓ Grad-CAM works!")
    print(f"  ✓ Output shape: {heatmap.shape}")
    print(f"  ✓ Predicted class: {pred_class}")
    print(f"  ✓ Confidence: {confidence*100:.1f}%")
except Exception as e:
    print(f"  ✗ XAI function test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed!
print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nYour XAI module is properly set up!")
print("\nNext steps:")
print("  1. Start the Flask app: python app.py")
print("  2. Navigate to http://localhost:5000")
print("  3. Upload an image and classify it")
print("  4. Click 'Explain This Prediction'")
print("\nOr use the Jupyter notebook:")
print("  jupyter notebook OCT_XAI.ipynb")
print("\n" + "="*70)

