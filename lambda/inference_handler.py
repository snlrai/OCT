"""
AWS Lambda Handler for OCT Image Inference
Supports both classification and segmentation models
"""

import json
import base64
import io
import os
import boto3
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from typing import Dict, Any, Tuple
import time

# Model caching (Lambda /tmp directory)
MODEL_CACHE_DIR = '/tmp/models'
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# S3 client
s3_client = boto3.client('s3')

# Device configuration
DEVICE = torch.device('cpu')  # Lambda uses CPU


class UNet(nn.Module):
    """U-Net architecture for segmentation"""
    def __init__(self, in_channels=1, out_channels=13, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.downs.append(self._double_conv(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(self._double_conv(feature*2, feature))

        self.bottleneck = self._double_conv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = transforms.functional.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


def load_model_from_s3(model_type: str, model_version: str = 'latest') -> Tuple[nn.Module, str]:
    """
    Load model from S3 with caching
    
    Args:
        model_type: 'classification' or 'segmentation'
        model_version: Model version or 'latest'
    
    Returns:
        Loaded model and model path
    """
    # Check cache first
    cache_key = f"{model_type}_{model_version}.pth"
    cache_path = os.path.join(MODEL_CACHE_DIR, cache_key)
    
    if os.path.exists(cache_path):
        print(f"Loading model from cache: {cache_path}")
        return _load_model_weights(model_type, cache_path), cache_path
    
    # Download from S3
    bucket = os.getenv('S3_MODELS_BUCKET', 'oct-mlops-models-dev')
    
    if model_type == 'classification':
        s3_key = 'models/oct-classifier/latest/best_oct_classifier.pth'
    else:
        s3_key = 'models/oct-segmentation/latest/unet_combined_best.pth'
    
    print(f"Downloading model from S3: s3://{bucket}/{s3_key}")
    
    try:
        s3_client.download_file(bucket, s3_key, cache_path)
        return _load_model_weights(model_type, cache_path), cache_path
    except Exception as e:
        print(f"Error downloading model from S3: {e}")
        # Fallback to default model location
        return _load_model_weights(model_type, None), "default"


def _load_model_weights(model_type: str, model_path: str = None) -> nn.Module:
    """Load model architecture and weights"""
    
    if model_type == 'classification':
        # Load ResNet50 classifier
        model = models.resnet50(pretrained=False)
        num_classes = 4
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
    
    elif model_type == 'segmentation':
        # Load U-Net segmentation model
        model = UNet(in_channels=1, out_channels=13)
        
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(state_dict)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(DEVICE)
    model.eval()
    return model


# Global model cache
MODELS_CACHE = {}


def get_model(model_type: str) -> nn.Module:
    """Get model from cache or load it"""
    if model_type not in MODELS_CACHE:
        print(f"Loading model: {model_type}")
        model, _ = load_model_from_s3(model_type)
        MODELS_CACHE[model_type] = model
    return MODELS_CACHE[model_type]


def preprocess_image_classification(image_bytes: bytes) -> torch.Tensor:
    """Preprocess image for classification"""
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0).to(DEVICE)


def preprocess_image_segmentation(image_bytes: bytes) -> Tuple[torch.Tensor, tuple]:
    """Preprocess image for segmentation"""
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    original_size = image.size
    
    image = image.resize((256, 256))
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    return image_tensor, original_size


def classify_image(model: nn.Module, image_tensor: torch.Tensor) -> Dict[str, Any]:
    """Run classification inference"""
    class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        # Get all class probabilities
        all_probs = {
            class_name: float(probabilities[0][i])
            for i, class_name in enumerate(class_names)
        }
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence_score,
        'probabilities': all_probs
    }


def segment_image(model: nn.Module, image_tensor: torch.Tensor, original_size: tuple) -> Dict[str, Any]:
    """Run segmentation inference"""
    class_names = [
        'background', 'GCL', 'INL', 'IPL', 'ONL', 'OPL', 'RNFL', 'RPE',
        'CHOROID', 'INTRA-RETINAL-FLUID', 'SUB-RETINAL-FLUID', 'PED', 'DRUSENOID-PED'
    ]
    
    with torch.no_grad():
        outputs = model(image_tensor)
        predictions = torch.argmax(outputs, dim=1)
        predictions_np = predictions.squeeze(0).cpu().numpy().astype(np.uint8)
        
        # Resize to original size
        predictions_resized = cv2.resize(
            predictions_np,
            original_size,
            interpolation=cv2.INTER_NEAREST
        )
        
        # Count pixels per class
        unique, counts = np.unique(predictions_resized, return_counts=True)
        class_distribution = {
            class_names[cls_id]: int(count)
            for cls_id, count in zip(unique, counts)
        }
        
        # Encode mask as base64
        _, buffer = cv2.imencode('.png', predictions_resized)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'mask_base64': mask_base64,
        'class_distribution': class_distribution,
        'detected_classes': [class_names[cls_id] for cls_id in unique.tolist()]
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler
    
    Expected event structure:
    {
        "body": {
            "image": "base64_encoded_image",
            "model_type": "classification" or "segmentation",
            "model_version": "latest" (optional)
        }
    }
    """
    start_time = time.time()
    
    try:
        # Parse input
        if 'body' in event:
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
        else:
            body = event
        
        # Extract parameters
        image_base64 = body.get('image')
        model_type = body.get('model_type', 'classification')
        model_version = body.get('model_version', 'latest')
        
        if not image_base64:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No image provided'})
            }
        
        # Decode image
        image_bytes = base64.b64decode(image_base64)
        
        # Load model
        model = get_model(model_type)
        
        # Run inference
        if model_type == 'classification':
            image_tensor = preprocess_image_classification(image_bytes)
            result = classify_image(model, image_tensor)
        
        elif model_type == 'segmentation':
            image_tensor, original_size = preprocess_image_segmentation(image_bytes)
            result = segment_image(model, image_tensor, original_size)
        
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': f'Unknown model type: {model_type}'})
            }
        
        # Add metadata
        inference_time = time.time() - start_time
        result['model_type'] = model_type
        result['model_version'] = model_version
        result['inference_time_ms'] = round(inference_time * 1000, 2)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }
    
    except Exception as e:
        print(f"Error in lambda_handler: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'error_type': type(e).__name__
            })
        }


# For local testing
if __name__ == '__main__':
    # Test with a sample image
    test_event = {
        'body': json.dumps({
            'model_type': 'classification',
            'image': ''  # Add base64 encoded image here for testing
        })
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))

