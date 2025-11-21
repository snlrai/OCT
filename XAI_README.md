# Explainable AI (XAI) Module for OCT Classification

## Overview

This XAI module adds explainability to the OCT retinal image classification model, making AI predictions interpretable and trustworthy for clinical use. It implements three state-of-the-art explanation techniques:

- **Grad-CAM** (Gradient-weighted Class Activation Mapping)
- **LIME** (Local Interpretable Model-agnostic Explanations)
- **Integrated Gradients** (Attribution-based explanations)

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Jupyter Notebook Usage](#jupyter-notebook-usage)
4. [Web Application Usage](#web-application-usage)
5. [Understanding the Explanations](#understanding-the-explanations)
6. [API Reference](#api-reference)
7. [Clinical Interpretation Guide](#clinical-interpretation-guide)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: Install XAI Dependencies

```bash
pip install -r requirements_xai.txt
```

This will install:
- `captum` - For Grad-CAM and Integrated Gradients
- `lime` - For LIME explanations
- Supporting libraries for visualization

### Step 2: Verify Installation

```python
import captum
import lime
print("XAI libraries installed successfully!")
```

### Requirements

- Python 3.7+
- PyTorch 1.9+
- Trained OCT classification model (`classification_models/best_oct_classifier.pth`)
- At least 4GB RAM (8GB recommended for faster processing)
- GPU optional but recommended for faster explanations

---

## Quick Start

### Option 1: Jupyter Notebook

```python
# Open OCT_XAI.ipynb
# Set your image path and run:

image_path = 'path/to/your/oct_image.jpg'
results = explain_prediction(image_path, save_results=True)
```

### Option 2: Web Application

```bash
# Start the Flask server
python app.py

# Navigate to http://localhost:5000
# Upload an image → Classify → Click "Explain This Prediction"
```

### Option 3: Command Line (using Python)

```python
from xai_utils import *
import torch
from pathlib import Path

# Load model
model = create_classification_model(num_classes=4)
checkpoint = torch.load('classification_models/best_oct_classifier.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate explanations
image_tensor, original = preprocess_image_for_xai('image.jpg')
heatmap, pred_class, confidence, probs = generate_gradcam(model, image_tensor)

# Visualize
overlay = apply_colormap_on_image(original, heatmap)
cv2.imwrite('explanation.png', overlay)
```

---

## Jupyter Notebook Usage

### OCT_XAI.ipynb Features

The notebook provides a comprehensive interface for generating and analyzing explanations:

#### 1. Single Image Explanation

```python
# Configure output directory
CONFIG = {
    'model_path': 'classification_models/best_oct_classifier.pth',
    'output_dir': 'xai_explanations'
}

# Explain prediction
results = explain_prediction('path/to/image.jpg', save_results=True)
```

**Outputs:**
- Comparison dashboard (all 3 methods side-by-side)
- Individual explanation images (Grad-CAM, LIME, Integrated Gradients)
- Quantitative metrics (JSON file)
- Bounding boxes highlighting key regions

#### 2. Batch Processing

```python
# Process multiple images
batch_results = batch_explain('uploads/', max_images=10, save_results=True)
```

#### 3. Class-Specific Analysis

```python
# See what model looks at for each possible diagnosis
explain_all_classes('path/to/image.jpg')
```

This generates a 2x2 grid showing Grad-CAM heatmaps for all 4 classes (CNV, DME, DRUSEN, NORMAL), helping you understand how the model discriminates between conditions.

#### 4. Clinical Report Generation

```python
# Create comprehensive report with all visualizations
report_path = create_clinical_report('path/to/image.jpg')
```

### Notebook Sections

1. **Setup and Installation** - Import libraries and check dependencies
2. **Configuration** - Set paths and parameters
3. **Load Model** - Load pre-trained classification model
4. **Single Image Explanation** - Main explanation function
5. **Example Usage** - Quick start examples
6. **Batch Processing** - Process multiple images
7. **Interpretation Guide** - How to read the explanations
8. **Interactive Widgets** - Upload and explain (if supported)

---

## Web Application Usage

### Starting the Server

```bash
python app.py
```

The server will automatically load the XAI module if the classification model is available.

### Using the Web Interface

1. **Upload Image**
   - Navigate to http://localhost:5000
   - Upload an OCT retinal image

2. **Classify**
   - Click "Classify Disease"
   - Wait for prediction results

3. **Explain**
   - Click "Explain This Prediction" button
   - Wait 30-60 seconds for explanations to generate
   - View all XAI visualizations

### Web Features

- **Prediction Summary**: Predicted class, confidence, key metrics
- **Visual Explanations**: Grad-CAM, LIME, Integrated Gradients, Comparison dashboard
- **Quantitative Metrics**: Attribution coverage, peak locations, attribution statistics
- **Interpretation Guide**: Built-in clinical validation checklist
- **Download**: All visualizations can be downloaded individually

### API Endpoints

#### `/explain` - Generate XAI Explanations

**Request:**
```bash
curl -X POST http://localhost:5000/explain \
  -F "file=@path/to/image.jpg"
```

**Response:**
```json
{
  "success": true,
  "predicted_class": "CNV",
  "confidence": 0.89,
  "class_probabilities": {
    "CNV": 0.89,
    "DME": 0.06,
    "DRUSEN": 0.03,
    "NORMAL": 0.02
  },
  "images": {
    "gradcam": "data:image/png;base64,...",
    "lime": "data:image/png;base64,...",
    "integrated_gradients": "data:image/png;base64,...",
    "gradcam_with_boxes": "data:image/png;base64,...",
    "comparison": "data:image/png;base64,..."
  },
  "metrics": {
    "gradcam": {
      "attribution_coverage_percent": 45.2,
      "peak_activation_location": {"x": 112, "y": 98},
      "mean_attribution": 0.234,
      "max_attribution": 0.987
    },
    "integrated_gradients": {
      "attribution_coverage_percent": 38.7,
      "peak_activation_location": {"x": 115, "y": 102},
      "mean_attribution": 0.189,
      "max_attribution": 0.943
    }
  },
  "interpretation": {
    "gradcam_description": "Regions highlighted in red...",
    "clinical_notes": "The model predicted CNV with 89.0% confidence..."
  }
}
```

#### `/explain/batch` - Batch Explanations

Upload multiple files for batch processing (returns Grad-CAM only for speed).

#### `/explain/health` - Health Check

Check if XAI module is available.

---

## Understanding the Explanations

### Grad-CAM (Gradient-weighted Class Activation Mapping)

**What it shows:** Regions of the image that most strongly influence the model's prediction.

**Colors:**
- 🔴 Red = High importance
- 🟡 Yellow = Medium importance
- 🔵 Blue = Low importance

**How to interpret:**
- Red areas are where the model "looks" to make its decision
- Multiple red regions indicate distributed attention
- Concentrated red area suggests focused decision-making

**Clinical use:**
- Verify the model focuses on pathological features (fluid accumulation, deposits, structural changes)
- Ensure the model ignores artifacts (shadows, edges, imaging noise)
- Compare with clinical findings

### LIME (Local Interpretable Model-agnostic Explanations)

**What it shows:** Superpixels (image regions) that contribute to the classification.

**Colors:**
- Highlighted boundaries show important superpixels
- Typically shows top 5 most important regions

**How to interpret:**
- The model's decision is based on these specific image patches
- Larger highlighted areas indicate more distributed reasoning
- Smaller regions suggest focused feature detection

**Clinical use:**
- Understand which anatomical structures drive the diagnosis
- Verify consistency with known disease patterns
- Identify if model relies on spurious correlations

### Integrated Gradients

**What it shows:** Pixel-level attribution showing each pixel's contribution.

**Colors:**
- 🔴 Red/bright areas = High positive contribution
- 🔵 Blue/dark areas = Low/negative contribution

**How to interpret:**
- Quantifies exactly how much each pixel influences the prediction
- More granular than Grad-CAM
- Baseline comparison shows change from black image

**Clinical use:**
- Fine-grained analysis of important features
- Useful for identifying subtle pathological changes
- Helps understand boundary effects

---

## API Reference

### Core Functions (xai_utils.py)

#### `generate_gradcam(model, image_tensor, target_class=None)`
Generates Grad-CAM heatmap.

**Parameters:**
- `model`: PyTorch model
- `image_tensor`: Input tensor (1, C, H, W)
- `target_class`: Target class index (None = predicted class)

**Returns:**
- `heatmap`: Numpy array (H, W) in [0, 1]
- `predicted_class`: Int
- `confidence`: Float
- `all_probs`: Numpy array of probabilities

#### `generate_lime_explanation(model, image_tensor, original_image, num_samples=1000, num_features=5)`
Generates LIME explanation.

**Parameters:**
- `model`: PyTorch model
- `image_tensor`: Input tensor
- `original_image`: Original RGB image
- `num_samples`: Number of samples for LIME
- `num_features`: Number of superpixels to highlight

**Returns:**
- `explanation_image`: Visualized explanation
- `mask`: Binary mask of important regions
- `predicted_class`: Int
- `confidence`: Float
- `all_probs`: Numpy array

#### `generate_integrated_gradients(model, image_tensor, target_class=None, n_steps=50)`
Generates Integrated Gradients attribution.

**Parameters:**
- `model`: PyTorch model
- `image_tensor`: Input tensor
- `target_class`: Target class (None = predicted)
- `n_steps`: Integration steps

**Returns:**
- `attribution_map`: Attribution heatmap
- `predicted_class`: Int
- `confidence`: Float
- `all_probs`: Numpy array

#### `create_comparison_plot(original, gradcam, lime, ig, pred_class, confidence, probs)`
Creates side-by-side comparison visualization.

#### `calculate_explanation_metrics(heatmap, threshold=0.5)`
Calculates quantitative metrics for explanations.

**Returns dictionary with:**
- `attribution_coverage_percent`: Coverage percentage
- `peak_activation_location`: (x, y) coordinates
- `mean_attribution`: Average attribution
- `max_attribution`: Maximum attribution
- `std_attribution`: Standard deviation

---

## Clinical Interpretation Guide

### Validation Checklist

Before trusting model predictions, verify:

✅ **Does the model focus on relevant anatomical structures?**
- Retinal layers (GCL, INL, ONL, RPE, etc.)
- Pathological features (fluid, deposits, blood vessels)

✅ **Are pathological features highlighted?**
- **CNV**: Subretinal fluid, choroidal neovascularization
- **DME**: Intraretinal fluid, cystoid spaces
- **DRUSEN**: Deposits between RPE and Bruch's membrane
- **NORMAL**: No highlighted pathology

✅ **Is the model ignoring artifacts?**
- Image borders/edges
- Shadows or reflections
- Imaging noise
- Text overlays

✅ **Do all three XAI methods agree?**
- Grad-CAM, LIME, and Integrated Gradients should highlight similar regions
- Disagreement may indicate uncertainty or complex reasoning

✅ **Does explanation align with clinical knowledge?**
- Compare with standard clinical diagnostic criteria
- Verify against domain expert interpretation
- Check consistency across similar cases

### Understanding Metrics

#### Attribution Coverage
- **High (>50%)**: Model considers large portion of image
- **Medium (20-50%)**: Focused attention on specific regions
- **Low (<20%)**: Very concentrated decision-making

**Clinical implication:**
- High coverage: Diffuse pathology or considering multiple factors
- Low coverage: Localized pathology or single dominant feature

#### Peak Activation Location
Coordinates of the most important region.

**Use:** Verify peak corresponds to expected anatomical location for the diagnosis.

#### Mean vs Max Attribution
- **High mean, high max**: Strong, distributed signal
- **Low mean, high max**: Sparse but strong focal features
- **High mean, low max**: Diffuse, weak signals

### Red Flags 🚩

Be cautious if:
- Model focuses on image borders or artifacts
- Explanations are inconsistent across methods
- Peak activation in unexpected anatomical region
- Very low attribution coverage (<10%)
- High confidence with weak explanations

### Best Practices

1. **Always review explanations** before accepting predictions
2. **Compare with previous cases** of the same condition
3. **Document** any unexpected explanation patterns
4. **Validate** against ground truth when available
5. **Update understanding** as you see more cases

---

## Troubleshooting

### Common Issues

#### 1. "XAI module not available"

**Cause:** XAI dependencies not installed or import error

**Solution:**
```bash
pip install -r requirements_xai.txt
```

#### 2. "Model not loaded"

**Cause:** Classification model not found

**Solution:**
- Train the classification model first using `OCT_Classification.ipynb`
- Verify model path: `classification_models/best_oct_classifier.pth`

#### 3. "CUDA out of memory"

**Cause:** GPU memory insufficient

**Solution:**
- Use CPU instead: Set `DEVICE = 'cpu'` in config
- Reduce image size
- Process one image at a time

#### 4. LIME is very slow

**Cause:** High `num_samples` parameter

**Solution:**
- Reduce `num_samples` to 500 or 300
- LIME is inherently slow (30-60 seconds per image)
- For faster results, use only Grad-CAM

#### 5. Explanations look strange/random

**Possible causes:**
- Model not properly trained
- Image preprocessing mismatch
- Wrong model architecture

**Solution:**
- Verify model accuracy on test set
- Check image preprocessing matches training
- Ensure model architecture matches checkpoint

#### 6. Web interface not showing XAI button

**Cause:** XAI routes not registered

**Solution:**
- Check console for "XAI module loaded successfully"
- Verify `xai_routes.py` and `xai_utils.py` are in project root
- Restart Flask server

### Performance Optimization

**For faster explanations:**

1. **Use GPU** if available
   ```python
   device = torch.device('cuda')
   ```

2. **Reduce LIME samples**
   ```python
   lime_image = generate_lime_explanation(..., num_samples=300)
   ```

3. **Reduce Integrated Gradients steps**
   ```python
   ig_map = generate_integrated_gradients(..., n_steps=25)
   ```

4. **Process in batches**
   - Use `batch_explain()` for multiple images
   - Generates only Grad-CAM for speed

**For better quality:**

1. **Increase LIME samples**
   ```python
   lime_image = generate_lime_explanation(..., num_samples=2000)
   ```

2. **Increase IG steps**
   ```python
   ig_map = generate_integrated_gradients(..., n_steps=100)
   ```

3. **Use full resolution**
   - Process at higher resolution if GPU memory allows

---

## File Structure

```
oct_major_project/
├── OCT_XAI.ipynb                 # Main XAI notebook
├── xai_utils.py                   # Core XAI utility functions
├── xai_routes.py                  # Flask routes for web app
├── requirements_xai.txt           # XAI dependencies
├── XAI_README.md                  # This file
├── xai_explanations/              # Output directory
│   ├── image_CNV_comparison.png
│   ├── image_CNV_gradcam.png
│   ├── image_CNV_lime.png
│   ├── image_CNV_ig.png
│   ├── image_CNV_gradcam_bbox.png
│   └── image_CNV_metrics.json
├── app.py                         # Flask app (with XAI integration)
├── templates/
│   └── index.html                 # Web UI (with XAI section)
└── static/
    ├── style.css                  # Styles (with XAI styles)
    └── script.js                  # JavaScript (with XAI handlers)
```

---

## Examples

### Example 1: Quick Explanation

```python
from xai_utils import *

# Load and explain
image_path = 'test_images/cnv_example.jpg'
tensor, original = preprocess_image_for_xai(image_path)
heatmap, pred, conf, probs = generate_gradcam(model, tensor)

# Visualize
overlay = apply_colormap_on_image(original, heatmap)
cv2.imwrite('explanation.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print(f"Predicted: {CLASS_NAMES[pred]} ({conf*100:.1f}%)")
```

### Example 2: Compare All Methods

```python
# Generate all explanations
gradcam_map, pred, conf, probs = generate_gradcam(model, tensor)
ig_map, _, _, _ = generate_integrated_gradients(model, tensor)
lime_img, _, _, _, _ = generate_lime_explanation(model, tensor, original)

# Create comparison
fig = create_comparison_plot(original, gradcam_overlay, lime_img, ig_overlay,
                             pred, conf, probs)
fig.savefig('comparison.png', dpi=300)
```

### Example 3: Batch Processing with Progress

```python
from tqdm import tqdm
from pathlib import Path

image_dir = Path('test_images/')
results = []

for img_path in tqdm(list(image_dir.glob('*.jpg'))):
    try:
        tensor, original = preprocess_image_for_xai(str(img_path))
        heatmap, pred, conf, _ = generate_gradcam(model, tensor)
        
        results.append({
            'filename': img_path.name,
            'predicted_class': CLASS_NAMES[pred],
            'confidence': float(conf)
        })
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# Save results
import json
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## Citation

If you use this XAI module in your research or clinical work, please cite:

### XAI Methods

**Grad-CAM:**
```
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017).
Grad-cam: Visual explanations from deep networks via gradient-based localization.
In Proceedings of the IEEE international conference on computer vision (pp. 618-626).
```

**LIME:**
```
Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).
" Why should i trust you?" Explaining the predictions of any classifier.
In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).
```

**Integrated Gradients:**
```
Sundararajan, M., Taly, A., & Yan, Q. (2017).
Axiomatic attribution for deep networks.
In International conference on machine learning (pp. 3319-3328). PMLR.
```

---

## Support

For issues, questions, or contributions:

1. Check this README and troubleshooting section
2. Review the code comments in `xai_utils.py`
3. Test with the provided examples
4. Check the Jupyter notebook for detailed usage

---

## License

This XAI module is provided as part of the OCT Major Project 2025.

**Important:** This tool is for research and educational purposes. Always consult with qualified medical professionals for clinical diagnosis and treatment decisions.

---

**Last Updated:** November 2025  
**Version:** 1.0.0  
**Author:** OCT Major Project Team

