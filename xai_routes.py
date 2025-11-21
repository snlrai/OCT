"""
XAI Routes for Flask Web Application
Provides /explain endpoint for generating explainable AI visualizations
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import traceback

# Import XAI utilities
from xai_utils import (
    preprocess_image_for_xai,
    generate_gradcam,
    generate_integrated_gradients,
    generate_lime_explanation,
    apply_colormap_on_image,
    create_comparison_plot,
    calculate_explanation_metrics,
    visualize_with_bounding_boxes,
    numpy_to_base64,
    fig_to_base64,
    CLASS_NAMES
)

# Create Blueprint
xai_bp = Blueprint('xai', __name__)

# Global variables for models (will be set by main app)
classification_model = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
XAI_CONFIG = {
    'upload_folder': 'uploads',
    'xai_output_folder': 'xai_explanations',
    'allowed_extensions': {'png', 'jpg', 'jpeg'}
}

# Create directories
os.makedirs(XAI_CONFIG['xai_output_folder'], exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in XAI_CONFIG['allowed_extensions']


def set_classification_model(model):
    """Set the classification model for XAI explanations"""
    global classification_model
    classification_model = model
    print(f"✓ Classification model set for XAI routes")


@xai_bp.route('/explain', methods=['POST'])
def explain():
    """
    Generate XAI explanations for uploaded image
    
    Returns JSON with:
        - predicted_class
        - confidence
        - class_probabilities
        - gradcam_image (base64)
        - lime_image (base64)
        - integrated_gradients_image (base64)
        - comparison_image (base64)
        - gradcam_bbox_image (base64)
        - metrics
    """
    # Check if model is loaded
    if classification_model is None:
        return jsonify({'error': 'Classification model not loaded'}), 500
    
    # Check if file was uploaded
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
        filepath = os.path.join(XAI_CONFIG['upload_folder'], filename)
        file.save(filepath)
        
        print(f"\n{'='*70}")
        print(f"Generating XAI explanations for: {filename}")
        print(f"{'='*70}")
        
        # Preprocess image
        image_tensor, original_image = preprocess_image_for_xai(filepath)
        image_tensor = image_tensor.to(DEVICE)
        
        # Resize original for overlays
        original_resized = cv2.resize(original_image, (224, 224))
        
        # 1. Generate Grad-CAM
        print("[1/3] Generating Grad-CAM...")
        gradcam_heatmap, pred_class, confidence, class_probs = generate_gradcam(
            classification_model, image_tensor
        )
        gradcam_overlay = apply_colormap_on_image(original_resized, gradcam_heatmap, alpha=0.4)
        gradcam_bbox = visualize_with_bounding_boxes(original_resized, gradcam_heatmap, num_regions=5)
        
        print(f"  Predicted: {CLASS_NAMES[pred_class]} (Confidence: {confidence*100:.1f}%)")
        
        # 2. Generate Integrated Gradients
        print("[2/3] Generating Integrated Gradients...")
        ig_map, _, _, _ = generate_integrated_gradients(classification_model, image_tensor, n_steps=50)
        ig_overlay = apply_colormap_on_image(original_resized, ig_map, alpha=0.4)
        
        # 3. Generate LIME Explanation
        print("[3/3] Generating LIME explanation...")
        lime_image, lime_mask, _, _, _ = generate_lime_explanation(
            classification_model, image_tensor, original_image, 
            num_samples=500, num_features=5
        )
        
        # Calculate metrics
        print("Calculating metrics...")
        gradcam_metrics = calculate_explanation_metrics(gradcam_heatmap)
        ig_metrics = calculate_explanation_metrics(ig_map)
        
        # Create comparison plot
        print("Creating comparison visualization...")
        fig = create_comparison_plot(
            original_resized, gradcam_overlay, lime_image, ig_overlay,
            pred_class, confidence, class_probs
        )
        
        # Convert images to base64
        original_base64 = numpy_to_base64(original_resized)
        gradcam_base64 = numpy_to_base64(gradcam_overlay)
        lime_base64 = numpy_to_base64(lime_image)
        ig_base64 = numpy_to_base64(ig_overlay)
        gradcam_bbox_base64 = numpy_to_base64(gradcam_bbox)
        comparison_base64 = fig_to_base64(fig)
        
        # Save results to disk
        base_name = os.path.splitext(filename)[0]
        output_prefix = os.path.join(XAI_CONFIG['xai_output_folder'], f"{base_name}_{CLASS_NAMES[pred_class]}")
        
        cv2.imwrite(f"{output_prefix}_gradcam.png", cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_prefix}_lime.png", cv2.cvtColor(lime_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_prefix}_ig.png", cv2.cvtColor(ig_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{output_prefix}_gradcam_bbox.png", cv2.cvtColor(gradcam_bbox, cv2.COLOR_RGB2BGR))
        
        print(f"✓ Results saved to {XAI_CONFIG['xai_output_folder']}/")
        print(f"{'='*70}\n")
        
        # Prepare response
        response = {
            'success': True,
            'predicted_class': CLASS_NAMES[pred_class],
            'confidence': float(confidence),
            'class_probabilities': {
                CLASS_NAMES[i]: float(class_probs[i]) for i in range(len(CLASS_NAMES))
            },
            'images': {
                'original': original_base64,
                'gradcam': gradcam_base64,
                'lime': lime_base64,
                'integrated_gradients': ig_base64,
                'gradcam_with_boxes': gradcam_bbox_base64,
                'comparison': comparison_base64
            },
            'metrics': {
                'gradcam': gradcam_metrics,
                'integrated_gradients': ig_metrics
            },
            'interpretation': {
                'gradcam_description': 'Regions highlighted in red have the strongest influence on the prediction. The model focuses on these areas when making its decision.',
                'lime_description': 'Highlighted boundaries show important image regions (superpixels) that contribute to the classification.',
                'ig_description': 'Attribution map showing pixel-level contributions. Brighter areas indicate higher importance for the prediction.',
                'clinical_notes': f'The model predicted {CLASS_NAMES[pred_class]} with {confidence*100:.1f}% confidence. ' +
                                 f'Peak attention is located at ({gradcam_metrics["peak_activation_location"]["x"]}, {gradcam_metrics["peak_activation_location"]["y"]}). ' +
                                 f'{gradcam_metrics["attribution_coverage_percent"]:.1f}% of the image contributes to the decision.'
            },
            'message': f'XAI explanations generated successfully for {CLASS_NAMES[pred_class]} prediction!'
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error during XAI explanation: {e}")
        traceback.print_exc()
        return jsonify({'error': f'XAI explanation failed: {str(e)}'}), 500


@xai_bp.route('/explain/batch', methods=['POST'])
def explain_batch():
    """
    Generate XAI explanations for multiple images
    
    Expects:
        - files: List of image files
    
    Returns:
        List of explanation results
    """
    if classification_model is None:
        return jsonify({'error': 'Classification model not loaded'}), 500
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    
    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    errors = []
    
    for file in files:
        if file.filename == '' or not allowed_file(file.filename):
            errors.append({'filename': file.filename, 'error': 'Invalid file'})
            continue
        
        try:
            # Save and process
            filename = secure_filename(file.filename)
            filepath = os.path.join(XAI_CONFIG['upload_folder'], filename)
            file.save(filepath)
            
            # Preprocess
            image_tensor, original_image = preprocess_image_for_xai(filepath)
            image_tensor = image_tensor.to(DEVICE)
            
            # Generate only Grad-CAM for batch (faster)
            gradcam_heatmap, pred_class, confidence, class_probs = generate_gradcam(
                classification_model, image_tensor
            )
            
            original_resized = cv2.resize(original_image, (224, 224))
            gradcam_overlay = apply_colormap_on_image(original_resized, gradcam_heatmap, alpha=0.4)
            
            # Calculate metrics
            metrics = calculate_explanation_metrics(gradcam_heatmap)
            
            results.append({
                'filename': filename,
                'predicted_class': CLASS_NAMES[pred_class],
                'confidence': float(confidence),
                'gradcam_image': numpy_to_base64(gradcam_overlay),
                'metrics': metrics
            })
            
        except Exception as e:
            errors.append({'filename': file.filename, 'error': str(e)})
    
    return jsonify({
        'success': True,
        'total_processed': len(results),
        'total_errors': len(errors),
        'results': results,
        'errors': errors
    })


@xai_bp.route('/explain/health', methods=['GET'])
def xai_health():
    """Health check for XAI endpoints"""
    return jsonify({
        'status': 'healthy',
        'xai_enabled': classification_model is not None,
        'device': str(DEVICE),
        'supported_methods': ['Grad-CAM', 'LIME', 'Integrated Gradients']
    })


# Helper function to integrate with main app.py
def register_xai_routes(app, model):
    """
    Register XAI routes with the main Flask app
    
    Args:
        app: Flask application instance
        model: Trained classification model
    """
    global DEVICE
    try:
        set_classification_model(model)
        app.register_blueprint(xai_bp)
        print(f"✓ XAI routes registered on device: {DEVICE}")
        print(f"  Available endpoints:")
        print(f"    POST /explain - Generate XAI explanations")
        print(f"    POST /explain/batch - Batch explanations")
        print(f"    GET  /explain/health - XAI health check")
    except Exception as e:
        print(f"❌ Error registering XAI routes: {e}")
        raise

