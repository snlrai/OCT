# XAI Module - Quick Start Guide

## ⚡ 5-Minute Quick Start

### Step 1: Install Dependencies (1 minute)

```bash
pip install -r requirements_xai.txt
```

### Step 2: Choose Your Method

#### Method A: Web Interface (Easiest)

```bash
python app.py
```

Then:
1. Open http://localhost:5000 in your browser
2. Upload an OCT image
3. Click "Classify Disease"
4. Click "Explain This Prediction"
5. Wait ~1 minute
6. View all explanations!

#### Method B: Jupyter Notebook (Most Features)

```bash
jupyter notebook OCT_XAI.ipynb
```

Then:
1. Run cells 1-8 to set up
2. Update `image_path` in cell 12
3. Run cell 12 to generate explanations
4. View results!

#### Method C: Python Script (Most Flexible)

```python
from xai_utils import *
import torch

# Load model
model = torch.load('classification_models/best_oct_classifier.pth')

# Explain
image_tensor, original = preprocess_image_for_xai('your_image.jpg')
heatmap, pred, conf, probs = generate_gradcam(model, image_tensor)

# Visualize
overlay = apply_colormap_on_image(original, heatmap)
import cv2
cv2.imwrite('explanation.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
```

---

## 📋 What You'll Get

For each explained image:

✅ **4 explanation visualizations:**
- Grad-CAM heatmap
- LIME superpixels
- Integrated Gradients
- Comparison dashboard

✅ **Quantitative metrics:**
- Attribution coverage
- Peak focus locations
- Confidence scores

✅ **Interpretation guide:**
- What the model sees
- Clinical validation tips
- Trust indicators

---

## 🎯 Example Output

```
xai_explanations/
├── myimage_CNV_comparison.png       ← All methods side-by-side
├── myimage_CNV_gradcam.png          ← Heatmap overlay
├── myimage_CNV_lime.png             ← Superpixel explanation
├── myimage_CNV_ig.png               ← Attribution map
├── myimage_CNV_gradcam_bbox.png     ← Key regions marked
└── myimage_CNV_metrics.json         ← Quantitative data
```

---

## 🔍 How to Read the Explanations

### Grad-CAM
- **Red areas** = Model focuses here
- **Blue areas** = Model ignores this
- Check if red areas match expected pathology

### LIME
- **Highlighted regions** = Important for decision
- Check if regions are anatomically relevant

### Integrated Gradients
- **Bright pixels** = High importance
- Check for consistency with other methods

---

## ✅ Validation Checklist

Before trusting the prediction, verify:

- [ ] Model focuses on pathological features (not edges/artifacts)
- [ ] All 3 methods agree on important regions
- [ ] Peak focus is in expected anatomical location
- [ ] Confidence matches explanation strength
- [ ] No red flags (see below)

### 🚩 Red Flags

Be cautious if:
- Focus on image borders
- Very low attribution (<10%)
- Methods disagree completely
- High confidence but weak explanation

---

## 💡 Pro Tips

1. **For faster results:** Use Grad-CAM only
   ```python
   heatmap, _, _, _ = generate_gradcam(model, image_tensor)
   ```

2. **For batch processing:** Use the batch function
   ```python
   batch_results = batch_explain('folder/', max_images=10)
   ```

3. **For better LIME:** Increase samples (but slower)
   ```python
   lime_img = generate_lime_explanation(..., num_samples=2000)
   ```

4. **For reports:** Use the clinical report function
   ```python
   report_path = create_clinical_report('image.jpg')
   ```

---

## 🐛 Quick Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| Import error | `pip install -r requirements_xai.txt` |
| Model not found | Train classification model first |
| Out of memory | Use CPU: `device='cpu'` |
| LIME too slow | Reduce samples to 300 |
| Web button missing | Restart `python app.py` |

---

## 📚 More Info

- **Full documentation:** See `XAI_README.md`
- **Implementation details:** See `XAI_IMPLEMENTATION_SUMMARY.md`
- **Code examples:** See `OCT_XAI.ipynb`

---

## 🎉 That's It!

You're ready to generate explainable AI visualizations for your OCT classifications.

**Start with:** Web interface or Jupyter notebook  
**Time:** ~1-2 minutes per image  
**Output:** 5 images + metrics JSON

Happy explaining! 🚀

