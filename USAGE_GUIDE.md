# OCT Analysis Web Application - Usage Guide

## Quick Start

### Step 1: Launch the Application

Open your terminal/command prompt and navigate to the project directory:

```bash
cd C:\Users\sneha\OneDrive\Desktop\oct_major_project
python app.py
```

You should see:
```
✓ Segmentation model loaded from unet_combined_best.pth
✓ Classification model loaded from classification_models/best_oct_classifier.pth

============================================================
OCT Retina Analysis Platform
============================================================
Device: cuda  # or 'cpu' if no GPU

Models Status:
  Segmentation Model: ✓ Loaded
  Classification Model: ✓ Loaded

Folders:
  Upload folder: uploads
  Result folder: results
  Predictions folder: predictions

Starting server...
Open your browser and navigate to: http://localhost:5000
============================================================
```

### Step 2: Open Your Browser

Navigate to: **http://localhost:5000**

---

## User Interface Walkthrough

### Main Page Layout

```
┌─────────────────────────────────────────────────────────┐
│          🔬 OCT Retina Analysis Platform               │
│   AI-Powered Retinal Disease Detection, Segmentation   │
│               & Classification                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                                                         │
│            📁 Upload OCT Image                         │
│                                                         │
│   Drag and drop your image here or click to browse    │
│         Supports: PNG, JPG, JPEG (Max 16MB)          │
│                                                         │
│              [Choose File Button]                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### After Uploading an Image

```
┌─────────────────────────────────────────────────────────┐
│                   Selected Image                        │
│            [Preview of your uploaded image]            │
│                                                         │
│              Choose Analysis Type                       │
│                                                         │
│   ┌──────────────────┐    ┌──────────────────┐       │
│   │   🔲 Segment    │    │   📊 Classify    │       │
│   │   Image         │    │   Disease         │       │
│   │                 │    │                   │       │
│   │ Identify and    │    │ Classify into     │       │
│   │ segment retinal │    │ CNV, DME, Drusen, │       │
│   │ layers          │    │ or Normal         │       │
│   └──────────────────┘    └──────────────────┘       │
│                                                         │
│        [Upload Different Image Button]                 │
└─────────────────────────────────────────────────────────┘
```

---

## Option 1: Segmentation

### When to Use
- Want to identify specific retinal layers
- Need to visualize fluid accumulation
- Analyzing retinal structure changes
- Research or educational purposes

### What You'll Get

#### 1. **Three Images Side-by-Side**
```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Original   │  │ Segmentation│  │   Overlay   │
│   Image     │  │    Mask     │  │    View     │
└─────────────┘  └─────────────┘  └─────────────┘
```

#### 2. **Training Metrics**
Shows how the segmentation model was trained:
- Training loss over epochs
- Validation accuracy progression
- IoU (Intersection over Union) scores

#### 3. **Layer Distribution Chart**
Horizontal bar chart showing percentage of each layer:
```
Background         ████████████████ 45.23%
GCL               ████████ 12.34%
INL               ███████ 10.11%
INTRA-RETINAL-FLUID ████ 5.67%
...
```

#### 4. **Color Legend**
Visual guide to the colors used in the segmentation:
- 🔴 Red = GCL
- 🟢 Green = INL
- 🔵 Blue = IPL
- 🟡 Yellow = ONL
- 🟣 Purple = CHOROID
- 🌸 Pink = INTRA-RETINAL-FLUID
- And more...

### Download Options
Each image has a download button to save:
- `original.png`
- `segmentation_mask.png`
- `overlay.png`

---

## Option 2: Classification

### When to Use
- Need a disease diagnosis
- Want to categorize the OCT scan
- Quick screening for abnormalities
- Patient report generation

### What You'll Get

#### 1. **Diagnosis Result**
```
┌─────────────────────────────────────────────┐
│         Predicted Diagnosis                  │
│                                              │
│              ┌──────────┐                    │
│              │   DME    │  <- Color coded   │
│              └──────────┘                    │
│           Confidence: 94.3%                  │
└─────────────────────────────────────────────┘
```

**Color Codes:**
- 🔴 **CNV**: Red badge (Choroidal Neovascularization)
- 🟠 **DME**: Orange badge (Diabetic Macular Edema)
- 🟣 **DRUSEN**: Purple badge
- 🟢 **NORMAL**: Green badge (Healthy)

#### 2. **Class Probabilities**
Bar chart showing confidence for each category:
```
DME     ██████████████████████ 94.30%  ✓ Predicted
DRUSEN  ████ 4.50%
CNV     █ 1.10%
NORMAL  █ 0.10%
```

#### 3. **Model Performance Metrics**

##### Confusion Matrix
Shows how the model performs across all disease categories on the test set:
```
              Predicted
           CNV  DME  DRUSEN  NORMAL
Actual CNV  [  10    0      0      0  ]
       DME  [   0   12      0      0  ]
    DRUSEN  [   0    1     11      0  ]
    NORMAL  [   0    0      0     10  ]
```

##### Training History
Two plots showing:
- **Loss**: How the model's error decreased during training
- **Accuracy**: How the model's accuracy improved over epochs

#### 4. **Model Summary**
Technical details about the classification model:
```
┌─────────────────────────────────────────────┐
│ Model Architecture:    ResNet50             │
│ Number of Classes:     4                    │
│ Image Size:            224×224              │
│ Training Samples:      95                   │
│ Validation Accuracy:   100.00%              │
│ Test Accuracy:         91.67%               │
│ Total Parameters:      24,559,172           │
└─────────────────────────────────────────────┘
```

---

## Interpretation Guide

### Segmentation Results Interpretation

#### Normal Retina
- Clear, well-defined layers
- Minimal or no fluid regions
- Smooth layer boundaries
- All major layers visible

#### DME (Diabetic Macular Edema)
- Presence of **INTRA-RETINAL-FLUID** (pink regions)
- Disrupted layer structure
- Thickened retina
- Fluid accumulation between layers

#### CNV (Choroidal Neovascularization)
- **SUB-RETINAL-FLUID** (teal regions) under the retina
- Possible **PED** (olive regions)
- Irregular RPE layer
- Fluid below retinal layers

#### DRUSEN
- **DRUSENOID-PED** (silver regions)
- Bumps or elevations in RPE
- May have associated fluid
- Deposits under the retina

### Classification Results Interpretation

#### Confidence Levels
- **>90%**: Very high confidence - strong prediction
- **70-90%**: High confidence - reliable prediction
- **50-70%**: Moderate confidence - consider manual review
- **<50%**: Low confidence - requires expert verification

#### When to Seek Expert Review
- Confidence below 70%
- Multiple classes with similar probabilities
- Unusual or ambiguous scan appearance
- Clinical symptoms don't match prediction

---

## Common Use Cases

### Research Workflow
1. Upload batch of OCT images
2. Use **Segment** for layer analysis
3. Export segmentation masks for quantitative analysis
4. Measure layer thickness, fluid volume, etc.

### Clinical Screening Workflow
1. Upload patient's OCT scan
2. Use **Classify** for quick diagnosis
3. Review confidence scores
4. Check training metrics to understand model reliability
5. Generate report with results

### Educational Workflow
1. Upload example images
2. Compare **Segmentation** and **Classification** results
3. Study layer distributions
4. Learn disease characteristics
5. Understand model performance metrics

---

## Tips for Best Results

### Image Quality
✅ **Good:**
- High resolution OCT scans
- Clear layer boundaries
- Minimal noise
- Centered on macula

❌ **Avoid:**
- Very low resolution images
- Heavily compressed JPEGs
- Motion artifacts
- Off-center or partial scans

### File Formats
- **Best**: PNG (lossless compression)
- **Good**: High-quality JPEG (90%+ quality)
- **Acceptable**: Standard JPEG

### Analysis Choice

**Choose Segmentation when:**
- You need detailed layer information
- Measuring fluid volumes
- Analyzing structural changes
- Creating quantitative reports

**Choose Classification when:**
- You need a quick diagnosis
- Screening multiple patients
- Categorizing large datasets
- Creating clinical reports

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + O` or `Cmd + O` | Open file dialog |
| `Esc` | Close previews/reset |
| `Ctrl + R` or `Cmd + R` | Refresh page |

---

## Troubleshooting

### Issue: "Model not loaded" error
**Solution**: 
- Check console output for specific model file missing
- Verify model files exist:
  - `unet_combined_best.pth`
  - `classification_models/best_oct_classifier.pth`

### Issue: Slow processing
**Possible Causes**:
- Running on CPU instead of GPU
- Large image file size
- System resource constraints

**Solutions**:
- Install CUDA-enabled PyTorch for GPU acceleration
- Resize images to recommended sizes before upload
- Close other applications

### Issue: Unexpected results
**Check**:
- Is the image actually an OCT scan?
- Is the image orientation correct?
- Is the image quality sufficient?
- Try the other analysis mode

### Issue: Can't upload image
**Check**:
- File size under 16MB
- File format is PNG, JPG, or JPEG
- File is not corrupted
- Browser has necessary permissions

---

## Support & Feedback

For issues or questions:
1. Check the console output for error messages
2. Review this guide
3. Check the main README documentation
4. Verify all model files are present

---

## Example Workflow

### Complete Analysis Example

1. **Start**: Launch `app.py`
2. **Upload**: Drag and drop `patient_oct_scan.jpg`
3. **Classify First**: Click "Classify Disease"
   - Result: **DME** with 96.5% confidence
   - Note: High confidence in DME diagnosis
4. **Go Back**: Click "Upload Different Image" (or re-upload same)
5. **Segment**: Click "Segment Image"
   - Result: Shows INTRA-RETINAL-FLUID at 12.3%
   - Confirms DME diagnosis with visible fluid
6. **Download**: Save all result images
7. **Report**: Use results for clinical documentation

---

**Happy Analyzing! 🔬👁️**

