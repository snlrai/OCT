"""
XAI Utilities for OCT Image Classification
Provides Grad-CAM, LIME, and Integrated Gradients explanations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import io
import base64

# XAI Libraries
from captum.attr import IntegratedGradients, LayerGradCam
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Class names for OCT classification
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']


def preprocess_image_for_xai(image_path, img_size=224):
    """
    Preprocess image for XAI explanations
    
    Args:
        image_path: Path to image file
        img_size: Target image size (default 224 for ResNet)
        
    Returns:
        Tuple of (tensor, original_rgb_image)
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image_rgb.copy()
    
    # Resize
    image_resized = cv2.resize(image_rgb, (img_size, img_size))
    
    # Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (image_resized / 255.0 - mean) / std
    
    # Convert to tensor (C, H, W)
    image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1)).float()
    
    return image_tensor, original


def denormalize_image(tensor):
    """
    Denormalize image tensor back to [0, 1] range
    
    Args:
        tensor: Normalized image tensor (C, H, W)
        
    Returns:
        Denormalized numpy array (H, W, C)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Convert to numpy and transpose
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    
    # Denormalize
    img = img * std + mean
    img = np.clip(img, 0, 1)
    
    return img


def generate_gradcam(model, image_tensor, target_class=None, target_layer=None):
    """
    Generate Grad-CAM heatmap
    
    Args:
        model: PyTorch model
        image_tensor: Input image tensor (1, C, H, W) or (C, H, W)
        target_class: Target class index (if None, uses predicted class)
        target_layer: Target layer for Grad-CAM (if None, uses model.layer4)
        
    Returns:
        Tuple of (heatmap, predicted_class, confidence, all_probs)
    """
    # Ensure batch dimension
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Set model to eval mode
    model.eval()
    
    # Get target layer (default: last conv layer in ResNet)
    if target_layer is None:
        if hasattr(model, 'layer4'):
            target_layer = model.layer4
        else:
            raise ValueError("Could not find suitable target layer. Please specify target_layer.")
    
    # Create LayerGradCam
    layer_gc = LayerGradCam(model, target_layer)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()
    
    # Use predicted class if target not specified
    if target_class is None:
        target_class = predicted_class
    
    # Generate attribution
    attributions = layer_gc.attribute(image_tensor, target=target_class)
    
    # Upsample to input size
    attributions = LayerGradCam.interpolate(attributions, image_tensor.shape[2:])
    
    # Convert to numpy and normalize
    heatmap = attributions.squeeze().cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap, predicted_class, confidence, probs[0].cpu().numpy()


def generate_integrated_gradients(model, image_tensor, target_class=None, n_steps=50):
    """
    Generate Integrated Gradients attribution
    
    Args:
        model: PyTorch model
        image_tensor: Input image tensor (1, C, H, W) or (C, H, W)
        target_class: Target class index (if None, uses predicted class)
        n_steps: Number of steps for path integration
        
    Returns:
        Tuple of (attribution_map, predicted_class, confidence, all_probs)
    """
    # Ensure batch dimension
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Set model to eval mode
    model.eval()
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()
    
    # Use predicted class if target not specified
    if target_class is None:
        target_class = predicted_class
    
    # Create IntegratedGradients
    ig = IntegratedGradients(model)
    
    # Create baseline (black image)
    baseline = torch.zeros_like(image_tensor)
    
    # Generate attributions
    attributions = ig.attribute(image_tensor, baseline, target=target_class, n_steps=n_steps)
    
    # Convert to numpy
    attributions_np = attributions.squeeze().cpu().detach().numpy()
    
    # Sum across color channels
    attribution_map = np.sum(np.abs(attributions_np), axis=0)
    
    # Normalize
    if attribution_map.max() > 0:
        attribution_map = attribution_map / attribution_map.max()
    
    return attribution_map, predicted_class, confidence, probs[0].cpu().numpy()


def generate_lime_explanation(model, image_tensor, original_image, target_class=None, num_samples=1000, num_features=5):
    """
    Generate LIME explanation
    
    Args:
        model: PyTorch model
        image_tensor: Input image tensor (for getting prediction)
        original_image: Original RGB image (H, W, C) in [0, 255] range
        target_class: Target class index (if None, uses predicted class)
        num_samples: Number of samples for LIME
        num_features: Number of superpixels to highlight
        
    Returns:
        Tuple of (explanation_image, mask, predicted_class, confidence, all_probs)
    """
    # Ensure batch dimension
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    # Set model to eval mode
    model.eval()
    device = next(model.parameters()).device
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()
    
    # Use predicted class if target not specified
    if target_class is None:
        target_class = predicted_class
    
    # Prepare image for LIME (resize to model input size)
    lime_image_input = cv2.resize(original_image, (224, 224))
    
    # Define prediction function for LIME
    def predict_fn(images):
        """Prediction function for LIME"""
        batch_tensors = []
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        for img in images:
            # Normalize
            img_normalized = (img / 255.0 - mean) / std
            # Convert to tensor
            tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()
            batch_tensors.append(tensor)
        
        # Stack into batch
        batch = torch.stack(batch_tensors).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(batch)
            probs = F.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    # Create LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Generate explanation
    explanation = explainer.explain_instance(
        lime_image_input,
        predict_fn,
        top_labels=4,
        hide_color=0,
        num_samples=num_samples
    )
    
    # Get mask for target class
    temp, mask = explanation.get_image_and_mask(
        target_class,
        positive_only=True,
        num_features=num_features,
        hide_rest=False
    )
    
    # Create visualization
    explanation_image = mark_boundaries(lime_image_input / 255.0, mask)
    explanation_image = (explanation_image * 255).astype(np.uint8)
    
    return explanation_image, mask, predicted_class, confidence, probs[0].cpu().numpy()


def apply_colormap_on_image(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Apply heatmap overlay on image
    
    Args:
        image: Original image (H, W, C) in [0, 255] or [0, 1] range
        heatmap: Heatmap (H, W) in [0, 1] range
        alpha: Overlay transparency (0-1)
        colormap: OpenCV colormap
        
    Returns:
        Overlay image (H, W, C) in [0, 255] range
    """
    # Ensure image is in [0, 255] range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Resize heatmap to match image size if needed
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert heatmap to uint8
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def create_comparison_plot(original_image, gradcam_overlay, lime_overlay, ig_overlay, 
                          predicted_class, confidence, class_probs, save_path=None):
    """
    Create side-by-side comparison of all XAI methods
    
    Args:
        original_image: Original RGB image
        gradcam_overlay: Grad-CAM overlay
        lime_overlay: LIME explanation
        ig_overlay: Integrated Gradients overlay
        predicted_class: Predicted class index
        confidence: Prediction confidence
        class_probs: Probabilities for all classes
        save_path: Optional path to save figure
        
    Returns:
        Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Grad-CAM
    axes[0, 1].imshow(gradcam_overlay)
    axes[0, 1].set_title('Grad-CAM', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # LIME
    axes[1, 0].imshow(lime_overlay)
    axes[1, 0].set_title('LIME Explanation', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Integrated Gradients
    axes[1, 1].imshow(ig_overlay)
    axes[1, 1].set_title('Integrated Gradients', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Add prediction info as suptitle
    pred_text = f"Predicted: {CLASS_NAMES[predicted_class]} (Confidence: {confidence*100:.1f}%)"
    fig.suptitle(pred_text, fontsize=16, fontweight='bold', y=0.98)
    
    # Add class probabilities as text
    prob_text = "Class Probabilities:\n"
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, class_probs)):
        prob_text += f"{name}: {prob*100:.1f}%"
        if i < len(CLASS_NAMES) - 1:
            prob_text += " | "
    
    fig.text(0.5, 0.02, prob_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def calculate_explanation_metrics(heatmap, threshold=0.5):
    """
    Calculate quantitative metrics for explanation
    
    Args:
        heatmap: Attribution heatmap (H, W) in [0, 1] range
        threshold: Threshold for important regions
        
    Returns:
        Dictionary of metrics
    """
    # Attribution coverage (percentage of image contributing)
    coverage = (heatmap > threshold).sum() / heatmap.size * 100
    
    # Peak activation location
    peak_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    # Attribution statistics
    metrics = {
        'attribution_coverage_percent': float(coverage),
        'peak_activation_location': {
            'y': int(peak_idx[0]),
            'x': int(peak_idx[1])
        },
        'mean_attribution': float(np.mean(heatmap)),
        'max_attribution': float(np.max(heatmap)),
        'std_attribution': float(np.std(heatmap)),
        'attribution_above_threshold': float((heatmap > threshold).sum())
    }
    
    return metrics


def get_top_regions(heatmap, num_regions=5, region_size=32):
    """
    Get bounding boxes of top contributing regions
    
    Args:
        heatmap: Attribution heatmap (H, W)
        num_regions: Number of top regions to identify
        region_size: Size of region window
        
    Returns:
        List of tuples (x, y, width, height, importance)
    """
    h, w = heatmap.shape
    regions = []
    
    # Create a copy to mark already selected regions
    heatmap_copy = heatmap.copy()
    
    for _ in range(num_regions):
        # Find max in remaining areas
        peak_idx = np.unravel_index(np.argmax(heatmap_copy), heatmap_copy.shape)
        peak_y, peak_x = peak_idx
        
        # Calculate region bounds
        x_start = max(0, peak_x - region_size // 2)
        y_start = max(0, peak_y - region_size // 2)
        x_end = min(w, x_start + region_size)
        y_end = min(h, y_start + region_size)
        
        # Calculate importance (mean attribution in region)
        importance = heatmap[y_start:y_end, x_start:x_end].mean()
        
        regions.append((x_start, y_start, x_end - x_start, y_end - y_start, float(importance)))
        
        # Mark this region as processed
        heatmap_copy[y_start:y_end, x_start:x_end] = 0
    
    return regions


def visualize_with_bounding_boxes(image, heatmap, num_regions=5, region_size=32):
    """
    Visualize image with bounding boxes around important regions
    
    Args:
        image: Original image (H, W, C)
        heatmap: Attribution heatmap (H, W)
        num_regions: Number of regions to highlight
        region_size: Size of region window
        
    Returns:
        Image with bounding boxes
    """
    # Ensure image is in correct format
    if image.max() <= 1.0:
        image_vis = (image * 255).astype(np.uint8).copy()
    else:
        image_vis = image.astype(np.uint8).copy()
    
    # Resize heatmap if needed
    if heatmap.shape != image.shape[:2]:
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    else:
        heatmap_resized = heatmap
    
    # Get top regions
    regions = get_top_regions(heatmap_resized, num_regions, region_size)
    
    # Draw bounding boxes
    for i, (x, y, w, h, importance) in enumerate(regions):
        # Color based on importance (red = most important)
        color = (255, int(255 * (1 - importance)), 0)
        thickness = 2 if i == 0 else 1
        
        # Draw rectangle
        cv2.rectangle(image_vis, (x, y), (x + w, y + h), color, thickness)
        
        # Add label
        label = f"#{i+1}: {importance:.2f}"
        cv2.putText(image_vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.4, color, 1, cv2.LINE_AA)
    
    return image_vis


def numpy_to_base64(image_array):
    """
    Convert numpy array to base64 string for web display
    
    Args:
        image_array: Image as numpy array
        
    Returns:
        Base64 encoded string with data URI prefix
    """
    # Ensure uint8
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)
    
    # Convert to PIL Image
    if len(image_array.shape) == 2:
        image_pil = Image.fromarray(image_array)
    else:
        image_pil = Image.fromarray(image_array)
    
    # Save to buffer
    buffer = io.BytesIO()
    image_pil.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    
    return f"data:image/png;base64,{image_base64}"


def fig_to_base64(fig):
    """
    Convert matplotlib figure to base64 string
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64 encoded string with data URI prefix
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    
    return f"data:image/png;base64,{image_base64}"

