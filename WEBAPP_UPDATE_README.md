# OCT Analysis Web Application - Update Summary

## Overview
The web application has been successfully updated to provide both **Segmentation** and **Classification** functionality for OCT (Optical Coherence Tomography) retinal images.

## New Features

### 1. Dual Analysis Modes
Users can now choose between two analysis types after uploading an image:

#### **Segmentation Mode**
- Uses the U-Net model (`unet_combined_best.pth`)
- Identifies and segments 13 different retinal layers and pathologies:
  - Background
  - GCL (Ganglion Cell Layer)
  - INL (Inner Nuclear Layer)
  - IPL (Inner Plexiform Layer)
  - ONL (Outer Nuclear Layer)
  - OPL (Outer Plexiform Layer)
  - RNFL (Retinal Nerve Fiber Layer)
  - RPE (Retinal Pigment Epithelium)
  - CHOROID
  - INTRA-RETINAL-FLUID
  - SUB-RETINAL-FLUID
  - PED (Pigment Epithelial Detachment)
  - DRUSENOID-PED

- **Outputs**:
  - Original image
  - Colored segmentation mask
  - Overlay visualization
  - Layer distribution chart
  - Training metrics visualization

#### **Classification Mode**
- Uses the ResNet50 model (`classification_models/best_oct_classifier.pth`)
- Classifies images into 4 disease categories:
  - **CNV** (Choroidal Neovascularization)
  - **DME** (Diabetic Macular Edema)
  - **DRUSEN**
  - **NORMAL** (Healthy retina)

- **Outputs**:
  - Original image
  - Predicted diagnosis with confidence score
  - Class probabilities for all categories
  - Confusion matrix
  - Training history plot
  - Model summary (architecture details, accuracy metrics)

## Updated Files

### 1. `app.py` - Flask Backend
**Changes**:
- Added support for both segmentation and classification models
- Created two new endpoints:
  - `/segment` - for segmentation analysis
  - `/classify` - for classification analysis
- Updated preprocessing functions:
  - `preprocess_for_segmentation()` - prepares images for U-Net (512×512, RGB, normalized)
  - `preprocess_for_classification()` - prepares images for ResNet50 (224×224, ImageNet normalization)
- Enhanced postprocessing for 13-class segmentation
- Loads both models on startup with status reporting

### 2. `templates/index.html` - Frontend Structure
**Changes**:
- Updated header title to "OCT Retina Analysis Platform"
- Added analysis options selector with two large buttons (Segment / Classify)
- Created two separate result sections:
  - **Segmentation Results**: Shows original, mask, overlay, metrics, and layer distribution
  - **Classification Results**: Shows original image, prediction badge, probabilities, confusion matrix, training history, and model summary
- Improved visual hierarchy and user flow

### 3. `static/script.js` - Frontend Logic
**Changes**:
- Implemented dual-mode analysis system
- Added separate result display functions:
  - `displaySegmentationResults()` - handles segmentation visualization
  - `displayClassificationResults()` - handles classification visualization
  - `displayClassProbabilities()` - creates probability bars with predicted class highlighting
  - `displayModelSummary()` - formats and displays model metadata
- Enhanced error handling with mode-specific messages
- Updated health check to verify both models

### 4. `static/style.css` - Styling
**Changes**:
- Added styles for analysis options grid
- Created prediction badge styles with color coding:
  - CNV: Red gradient
  - DME: Orange gradient
  - Drusen: Purple gradient
  - Normal: Green gradient
- Added probability bar styles with highlighting for predicted class
- Created model summary grid styles
- Added metrics grid for side-by-side comparison views
- Improved responsive design for mobile devices

## How to Use

### 1. Start the Application
```bash
python app.py
```

The application will:
- Load both models (segmentation and classification)
- Report loading status for each model
- Start the Flask server on `http://localhost:5000`

### 2. Upload an Image
- Drag and drop an OCT image or click "Choose File"
- Supported formats: PNG, JPG, JPEG (max 16MB)

### 3. Choose Analysis Type
Two options will appear:
- **Segment Image**: Click to perform layer segmentation
- **Classify Disease**: Click to diagnose disease category

### 4. View Results
- Results are displayed in a comprehensive dashboard
- Download individual result images
- View training metrics and model performance
- Analyze class/layer distributions

### 5. Analyze Another Image
- Click "Analyze Another Image" to start over

## Model Files Required

Ensure these model files are in place:

### Segmentation Model
- **Location**: `unet_combined_best.pth` (root directory)
- **Architecture**: U-Net with 3 input channels, 13 output classes
- **Input Size**: 512×512 RGB

### Classification Model
- **Location**: `classification_models/best_oct_classifier.pth`
- **Architecture**: ResNet50 with custom classifier
- **Input Size**: 224×224 RGB
- **Associated Files**:
  - `classification_models/confusion_matrix.png`
  - `classification_models/training_history.png`
  - `classification_models/model_summary.json`

### Segmentation Metrics
- **Location**: `training_metrics.png` (root directory)

## Technical Details

### Preprocessing Differences

#### Segmentation
```python
- Denoise with fastNlMeans
- Apply unsharp masking
- CLAHE enhancement
- Resize to 512×512
- Normalize: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
```

#### Classification
```python
- Resize to 224×224
- Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  (ImageNet normalization)
```

### API Endpoints

#### GET `/`
Returns the main HTML page

#### POST `/segment`
- **Input**: Image file (multipart/form-data)
- **Output**: JSON with segmentation results
- **Response Fields**:
  - `success`: Boolean
  - `original_image`: Base64-encoded original image
  - `segmented_mask`: Base64-encoded colored mask
  - `overlay_image`: Base64-encoded overlay
  - `class_distribution`: Object with layer percentages
  - `training_metrics`: Base64-encoded training plot
  - `message`: Status message

#### POST `/classify`
- **Input**: Image file (multipart/form-data)
- **Output**: JSON with classification results
- **Response Fields**:
  - `success`: Boolean
  - `original_image`: Base64-encoded original image
  - `predicted_class`: String (CNV/DME/DRUSEN/NORMAL)
  - `class_probabilities`: Object with probabilities for each class
  - `confusion_matrix`: Base64-encoded confusion matrix plot
  - `training_history`: Base64-encoded training history plot
  - `model_summary`: Object with model metadata
  - `message`: Status message

#### GET `/health`
- **Output**: Server and model status
- **Response Fields**:
  - `status`: "healthy"
  - `segmentation_model_loaded`: Boolean
  - `classification_model_loaded`: Boolean
  - `device`: "cuda" or "cpu"

## Browser Compatibility

Tested and works on:
- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Dependencies

Python packages required:
```
flask
torch
torchvision
opencv-python
numpy
pillow
werkzeug
```

## Troubleshooting

### Model Not Loading
- Check that model files exist at specified paths
- Verify model architecture matches saved checkpoint
- Check console output for detailed error messages

### Memory Issues
- Reduce batch size if processing multiple images
- Use CPU mode if GPU memory is insufficient
- Close other applications to free up RAM

### Slow Inference
- Segmentation takes ~2-5 seconds on GPU, ~10-30 seconds on CPU
- Classification takes ~0.5-2 seconds on GPU, ~3-10 seconds on CPU
- Consider using GPU for faster processing

## Future Enhancements

Potential improvements:
1. Batch processing for multiple images
2. Export results as PDF report
3. Side-by-side comparison tool
4. Image enhancement tools
5. User authentication and result history
6. REST API documentation
7. Docker containerization

## Credits

- **Segmentation Model**: U-Net trained on combined OCT dataset (CNV, DME, Drusen, Normal)
- **Classification Model**: ResNet50 trained on OCT disease dataset
- **Frontend**: Custom HTML/CSS/JS with modern UI design
- **Backend**: Flask framework with PyTorch inference

---

**Project**: OCT Major Project 2025  
**Updated**: November 2025  
**Version**: 2.0

