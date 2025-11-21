# OCT Image Classification Model

## Overview
This Jupyter notebook implements a deep learning classification model to classify OCT (Optical Coherence Tomography) retinal images into four categories:
- **Normal**: Healthy retina
- **DME**: Diabetic Macular Edema
- **Drusen**: Deposits under the retina
- **CNV**: Choroidal Neovascularization

## Features

### ✅ Complete Training Pipeline
- Data loading from multiple sources
- Train/validation/test split (80%/10%/10%)
- Data augmentation for better generalization
- Transfer learning with pre-trained models

### ✅ Model Options
- **ResNet50** (default) - Best balance of accuracy and speed
- **ResNet18** - Faster, lighter model
- **VGG16** - Classic architecture
- **EfficientNet-B0** - State-of-the-art efficiency

### ✅ Training Features
- Automatic model checkpointing (saves best model)
- Learning rate scheduling
- Progress bars with real-time metrics
- Training history visualization

### ✅ Evaluation & Visualization
- Classification report with precision, recall, F1-score
- Confusion matrix (raw and normalized)
- Test set predictions with visual comparison
- Single image prediction with probability display

### ✅ Model Saving
- Best model checkpoint with full configuration
- Lightweight deployment model
- Model summary (JSON format)

## Quick Start Guide

### 1. Setup
```python
# Update these paths in Cell 7 (Configuration section):
BASE_PATH = '/content/drive/MyDrive/oct_major_project/'

DATA_PATHS = {
    'NORMAL': os.path.join(BASE_PATH, 'NORMAL 2.v1i.coco-segmentation/train'),
    'DME': os.path.join(BASE_PATH, 'DME 2.v1i.coco-segmentation/train'),
    'DRUSEN': os.path.join(BASE_PATH, 'drusen 3.v1i.coco-segmentation/train'),
    'CNV': os.path.join(BASE_PATH, 'CNV 2.v1i.coco-segmentation/train')
}
```

### 2. Configuration Options
```python
CONFIG = {
    'img_size': 224,          # Image size (224x224)
    'batch_size': 32,         # Batch size (adjust based on GPU memory)
    'num_epochs': 50,         # Number of training epochs
    'learning_rate': 0.001,   # Initial learning rate
    'model_name': 'resnet50', # Model architecture
}
```

### 3. Run the Notebook
1. Upload to Google Colab
2. Mount Google Drive (Cell 2)
3. Update paths in Configuration (Cell 7)
4. Run all cells sequentially

### 4. Training
The training loop will:
- Train for specified epochs
- Save best model automatically
- Create checkpoints every 10 epochs
- Display progress with metrics

### 5. Prediction on Single Image
```python
# Option 1: Upload image
from google.colab import files
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Option 2: Use existing image
image_path = '/content/drive/MyDrive/oct_major_project/test_image.jpg'

# Predict
predicted_class, probabilities, image = predict_single_image(
    model, image_path, val_test_transform, CONFIG['device'], CLASS_NAMES
)

# Visualize
visualize_prediction(image, predicted_class, probabilities)
```

## Output Files

After training, the following files will be saved in `classification_models/` folder:

1. **best_oct_classifier.pth**
   - Best model checkpoint with full training state
   - Includes optimizer state, config, and class mappings
   - Use this to resume training or for evaluation

2. **oct_classifier_deployment.pth**
   - Lightweight model for deployment
   - Contains only model weights and essential config
   - Smaller file size for production use

3. **model_summary.json**
   - Model configuration and performance metrics
   - Training statistics
   - Class mappings

4. **training_history.png**
   - Training and validation loss curves
   - Training and validation accuracy curves

5. **confusion_matrix.png**
   - Confusion matrix visualization (raw counts)
   
6. **confusion_matrix_normalized.png**
   - Normalized confusion matrix (percentages)

7. **test_predictions_sample.png**
   - Sample predictions on test images
   - Green = correct, Red = incorrect

## Performance Tips

### For Better Accuracy:
- Increase `num_epochs` (e.g., 75-100)
- Use data augmentation (already included)
- Try different model architectures
- Adjust learning rate

### For Faster Training:
- Reduce `batch_size` if running out of memory
- Use `resnet18` instead of `resnet50`
- Reduce `img_size` to 128 or 192

### For Google Colab:
- Use GPU runtime (Runtime → Change runtime type → GPU)
- Recommended: T4 GPU or better
- Training time: ~30-60 minutes for 50 epochs

## Loading Saved Model

To use the trained model in a new session:

```python
import torch
import torchvision.models as models

# Load checkpoint
checkpoint = torch.load('path/to/best_oct_classifier.pth')

# Recreate model
model = create_model(
    model_name=checkpoint['config']['model_name'],
    num_classes=checkpoint['config']['num_classes'],
    pretrained=False
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Now you can use predict_single_image() function
```

## Troubleshooting

### Issue: "Folder not found"
- Check that `BASE_PATH` points to correct folder
- Verify folder names match exactly (case-sensitive)
- Ensure data is uploaded to Google Drive

### Issue: Out of memory
- Reduce `batch_size` (try 16 or 8)
- Reduce `img_size` (try 192 or 128)
- Use smaller model (resnet18)

### Issue: Low accuracy
- Increase `num_epochs`
- Check data quality and balance
- Try different `learning_rate`
- Ensure correct data paths

### Issue: Training too slow
- Verify GPU is enabled in Colab
- Reduce `num_workers` to 0
- Use smaller model architecture

## Model Architectures Comparison

| Model | Parameters | Speed | Accuracy | Recommended Use |
|-------|-----------|-------|----------|-----------------|
| ResNet50 | ~25M | Medium | High | **Best overall choice** |
| ResNet18 | ~11M | Fast | Good | Quick experiments |
| VGG16 | ~138M | Slow | High | Maximum accuracy |
| EfficientNet-B0 | ~5M | Fast | High | Resource-constrained |

## Citation

If you use this code for your project, please acknowledge the use of:
- PyTorch for deep learning framework
- torchvision for pre-trained models
- Your OCT dataset source

## Next Steps

1. ✅ Train the model using this notebook
2. ✅ Evaluate performance on test set
3. ✅ Use for predictions on new images
4. 🔄 Integrate into web application
5. 🔄 Deploy for clinical use

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the code comments in the notebook
3. Verify all paths and configurations

---

**Good luck with your OCT classification project! 🚀**

*Last updated: November 2025*

