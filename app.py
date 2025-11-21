"""
OCT Retina Segmentation & Classification Web Application
Flask-based web interface for uploading, segmenting, and classifying OCT retinal images
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import io
import base64
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['PREDICTIONS_FOLDER'] = 'predictions'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICTIONS_FOLDER'], exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Unified category mapping for segmentation
UNIFIED_CATEGORIES = {
    0: 'background',
    1: 'GCL',
    2: 'INL',
    3: 'IPL',
    4: 'ONL',
    5: 'OPL',
    6: 'RNFL',
    7: 'RPE',
    8: 'CHOROID',
    9: 'INTRA-RETINAL-FLUID',
    10: 'SUB-RETINAL-FLUID',
    11: 'PED',
    12: 'DRUSENOID-PED'
}


# U-Net Model Definition (same as training)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=13, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (downsampling)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder (upsampling)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)


# ResNet50 Classification Model
def create_classification_model(num_classes=4, pretrained=False):
    """Create ResNet50 classifier for OCT image classification"""
    model = models.resnet50(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model


# Load models
SEGMENTATION_MODEL_PATH = 'unet_combined_best.pth'
CLASSIFICATION_MODEL_PATH = 'classification_models/best_oct_classifier.pth'
segmentation_model = None
classification_model = None

def load_models():
    """Load both segmentation and classification models"""
    global segmentation_model, classification_model
    
    # Load Segmentation Model
    try:
        segmentation_model = UNet(in_channels=3, out_channels=13)
        if os.path.exists(SEGMENTATION_MODEL_PATH):
            segmentation_model.load_state_dict(torch.load(SEGMENTATION_MODEL_PATH, map_location=DEVICE))
            segmentation_model.to(DEVICE)
            segmentation_model.eval()
            print(f"✓ Segmentation model loaded from {SEGMENTATION_MODEL_PATH}")
        else:
            print(f"⚠️  Segmentation model not found: {SEGMENTATION_MODEL_PATH}")
            segmentation_model = None
    except Exception as e:
        print(f"❌ Error loading segmentation model: {e}")
        segmentation_model = None
    
    # Load Classification Model
    try:
        classification_model = create_classification_model(num_classes=4)
        if os.path.exists(CLASSIFICATION_MODEL_PATH):
            checkpoint = torch.load(CLASSIFICATION_MODEL_PATH, map_location=DEVICE)
            classification_model.load_state_dict(checkpoint['model_state_dict'])
            classification_model.to(DEVICE)
            classification_model.eval()
            print(f"✓ Classification model loaded from {CLASSIFICATION_MODEL_PATH}")
        else:
            print(f"⚠️  Classification model not found: {CLASSIFICATION_MODEL_PATH}")
            classification_model = None
    except Exception as e:
        print(f"❌ Error loading classification model: {e}")
        classification_model = None
    
    return (segmentation_model is not None, classification_model is not None)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_for_segmentation(image_path):
    """
    Preprocess image for segmentation model (U-Net)
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Tuple of (preprocessed_tensor, original_image_array)
    """
    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Store original for display
    original = image.copy()
    
    # Preprocess (same as training)
    image = cv2.fastNlMeansDenoising(image, None, h=10, searchWindowSize=21)
    gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(image, 2.0, gaussian_3, -1.0, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(unsharp_image)
    
    # Convert to RGB for the model
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize to 512x512
    image_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    image_normalized = (image_normalized - mean) / std
    
    # Convert to tensor (C, H, W)
    image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    
    return image_tensor.to(DEVICE), original


def preprocess_for_classification(image_path):
    """
    Preprocess image for classification model (ResNet50)
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Tuple of (preprocessed_tensor, original_image_array)
    """
    # Read image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image_rgb.copy()
    
    # Define transform (same as training)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transform
    image_tensor = transform(image_rgb).unsqueeze(0)
    
    return image_tensor.to(DEVICE), original


def postprocess_mask(mask_tensor):
    """
    Convert model output to visualization-ready mask
    
    Args:
        mask_tensor: Model output tensor
        
    Returns:
        Colored segmentation mask as numpy array
    """
    # Get class predictions
    mask = torch.argmax(mask_tensor, dim=1).squeeze().cpu().numpy()
    
    # Create colored mask with distinct colors for 13 classes
    colors = [
        [0, 0, 0],          # 0: Background (black)
        [255, 0, 0],        # 1: GCL (red)
        [0, 255, 0],        # 2: INL (green)
        [0, 0, 255],        # 3: IPL (blue)
        [255, 255, 0],      # 4: ONL (yellow)
        [255, 0, 255],      # 5: OPL (magenta)
        [0, 255, 255],      # 6: RNFL (cyan)
        [255, 128, 0],      # 7: RPE (orange)
        [128, 0, 255],      # 8: CHOROID (purple)
        [255, 192, 203],    # 9: INTRA-RETINAL-FLUID (pink)
        [0, 128, 128],      # 10: SUB-RETINAL-FLUID (teal)
        [128, 128, 0],      # 11: PED (olive)
        [192, 192, 192],    # 12: DRUSENOID-PED (silver)
    ]
    
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for class_id, color in enumerate(colors):
        if class_id < len(colors):
            colored_mask[mask == class_id] = color
    
    return colored_mask


def create_overlay(original, mask, alpha=0.5):
    """
    Create an overlay of the mask on the original image
    
    Args:
        original: Original grayscale image
        mask: Colored segmentation mask
        alpha: Transparency factor
        
    Returns:
        Overlay image
    """
    # Resize mask to original size if needed
    if original.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
    
    # Convert grayscale to RGB
    if len(original.shape) == 2:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original
    
    # Create overlay
    overlay = cv2.addWeighted(original_rgb, 1-alpha, mask, alpha, 0)
    
    return overlay


def numpy_to_base64(image_array):
    """Convert numpy array to base64 encoded string for HTML display"""
    # Convert to PIL Image
    if len(image_array.shape) == 2:
        # Grayscale
        image_pil = Image.fromarray(image_array)
    else:
        # RGB
        image_pil = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    image_pil.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return f"data:image/png;base64,{image_base64}"


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/segment', methods=['POST'])
def segment():
    """Handle image segmentation"""
    if segmentation_model is None:
        return jsonify({'error': 'Segmentation model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        image_tensor, original_image = preprocess_for_segmentation(filepath)
        
        # Run inference
        with torch.no_grad():
            output = segmentation_model(image_tensor)
        
        # Postprocess mask
        colored_mask = postprocess_mask(output)
        
        # Create overlay
        overlay = create_overlay(original_image, colored_mask, alpha=0.4)
        
        # Convert images to base64 for display
        original_base64 = numpy_to_base64(original_image)
        mask_base64 = numpy_to_base64(colored_mask)
        overlay_base64 = numpy_to_base64(overlay)
        
        # Save results
        result_filename_base = filename.rsplit('.', 1)[0]
        mask_path = os.path.join(app.config['RESULT_FOLDER'], f"{result_filename_base}_mask.png")
        overlay_path = os.path.join(app.config['RESULT_FOLDER'], f"{result_filename_base}_overlay.png")
        
        cv2.imwrite(mask_path, colored_mask)
        cv2.imwrite(overlay_path, overlay)
        
        # Get unique classes in the prediction
        mask_classes = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        unique_classes = np.unique(mask_classes)
        class_distribution = {}
        
        for class_id in unique_classes:
            percentage = (mask_classes == class_id).sum() / mask_classes.size * 100
            class_name = UNIFIED_CATEGORIES.get(int(class_id), f'Class {class_id}')
            class_distribution[class_name] = float(percentage)
        
        # Read training metrics plot
        training_metrics_path = 'training_metrics.png'
        training_metrics_base64 = None
        if os.path.exists(training_metrics_path):
            with open(training_metrics_path, 'rb') as f:
                training_metrics_base64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
        
        return jsonify({
            'success': True,
            'original_image': original_base64,
            'segmented_mask': mask_base64,
            'overlay_image': overlay_base64,
            'class_distribution': class_distribution,
            'training_metrics': training_metrics_base64,
            'message': 'Segmentation completed successfully!'
        })
    
    except Exception as e:
        print(f"Error during segmentation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Segmentation failed: {str(e)}'}), 500


@app.route('/classify', methods=['POST'])
def classify():
    """Handle image classification"""
    if classification_model is None:
        return jsonify({'error': 'Classification model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        image_tensor, original_image = preprocess_for_classification(filepath)
        
        # Run inference
        with torch.no_grad():
            output = classification_model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
        
        # Class names
        CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        predicted_class = CLASS_NAMES[predicted_idx]
        
        # Get probabilities for each class
        class_probabilities = {}
        for i, class_name in enumerate(CLASS_NAMES):
            class_probabilities[class_name] = float(probabilities[0][i].item())
        
        # Convert original image to base64
        original_base64 = numpy_to_base64(original_image)
        
        # Load confusion matrix
        confusion_matrix_path = 'classification_models/confusion_matrix.png'
        confusion_matrix_base64 = None
        if os.path.exists(confusion_matrix_path):
            with open(confusion_matrix_path, 'rb') as f:
                confusion_matrix_base64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
        
        # Load training history
        training_history_path = 'classification_models/training_history.png'
        training_history_base64 = None
        if os.path.exists(training_history_path):
            with open(training_history_path, 'rb') as f:
                training_history_base64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
        
        # Load model summary
        model_summary_path = 'classification_models/model_summary.json'
        model_summary = None
        if os.path.exists(model_summary_path):
            with open(model_summary_path, 'r') as f:
                model_summary = json.load(f)
        
        return jsonify({
            'success': True,
            'original_image': original_base64,
            'predicted_class': predicted_class,
            'class_probabilities': class_probabilities,
            'confusion_matrix': confusion_matrix_base64,
            'training_history': training_history_base64,
            'model_summary': model_summary,
            'message': f'Classification completed! Predicted: {predicted_class}'
        })
    
    except Exception as e:
        print(f"Error during classification: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'segmentation_model_loaded': segmentation_model is not None,
        'classification_model_loaded': classification_model is not None,
        'device': str(DEVICE)
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files"""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


# Register XAI routes (must be done after model loading)
def register_xai():
    """Register XAI routes if classification model is available"""
    try:
        if classification_model is not None:
            from xai_routes import register_xai_routes
            register_xai_routes(app, classification_model)
            return True
    except Exception as e:
        print(f"⚠️  Could not load XAI module: {e}")
    return False


if __name__ == '__main__':
    # Load models on startup
    seg_loaded, cls_loaded = load_models()
    
    # Register XAI routes if classification model is loaded
    xai_loaded = False
    if cls_loaded and classification_model is not None:
        xai_loaded = register_xai()
        if xai_loaded:
            print("✓ XAI module loaded successfully")
    
    print("\n" + "="*60)
    print("OCT Retina Segmentation & Classification Web App")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"\nModels Status:")
    print(f"  Segmentation Model: {'✓ Loaded' if seg_loaded else '✗ Not Loaded'}")
    print(f"  Classification Model: {'✓ Loaded' if cls_loaded else '✗ Not Loaded'}")
    print(f"  XAI Module: {'✓ Loaded' if xai_loaded else '✗ Not Loaded'}")
    print(f"\nFolders:")
    print(f"  Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"  Result folder: {app.config['RESULT_FOLDER']}")
    print(f"  Predictions folder: {app.config['PREDICTIONS_FOLDER']}")
    print("\nStarting server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("="*60 + "\n")
    
    if not seg_loaded and not cls_loaded:
        print("⚠️  WARNING: No models loaded! The app may not work properly.")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

