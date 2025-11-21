// OCT Retina Analysis - JavaScript Functionality

let selectedFile = null;
let currentMode = null; // 'segment' or 'classify'

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadButton = document.getElementById('uploadButton');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const segmentButton = document.getElementById('segmentButton');
const classifyButton = document.getElementById('classifyButton');
const resetButton = document.getElementById('resetButton');
const loadingSpinner = document.getElementById('loadingSpinner');
const loadingText = document.getElementById('loadingText');
const segmentationResults = document.getElementById('segmentationResults');
const classificationResults = document.getElementById('classificationResults');
const errorAlert = document.getElementById('errorAlert');
const errorMessage = document.getElementById('errorMessage');
const analyzeAnotherButton = document.getElementById('analyzeAnotherButton');
const classifyAnotherButton = document.getElementById('classifyAnotherButton');

// Event Listeners
uploadButton.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
segmentButton.addEventListener('click', () => analyzeImage('segment'));
classifyButton.addEventListener('click', () => analyzeImage('classify'));
resetButton.addEventListener('click', resetUpload);
analyzeAnotherButton.addEventListener('click', resetUpload);
classifyAnotherButton.addEventListener('click', resetUpload);

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Handle File Selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!allowedTypes.includes(file.type)) {
        showError('Invalid file type. Please upload a PNG, JPG, or JPEG image.');
        return;
    }
    
    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size too large. Maximum size is 16MB.');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewSection.style.display = 'block';
        hideError();
    };
    reader.readAsDataURL(file);
}

// Analyze Image (Segment or Classify)
async function analyzeImage(mode) {
    if (!selectedFile) {
        showError('Please select an image first.');
        return;
    }
    
    currentMode = mode;
    
    // Show loading
    previewSection.style.display = 'none';
    segmentationResults.style.display = 'none';
    classificationResults.style.display = 'none';
    loadingSpinner.style.display = 'block';
    
    if (mode === 'segment') {
        loadingText.textContent = 'Segmenting image... Please wait';
    } else {
        loadingText.textContent = 'Classifying image... Please wait';
    }
    
    hideError();
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        // Send request to appropriate endpoint
        const endpoint = mode === 'segment' ? '/segment' : '/classify';
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // Display results
            if (mode === 'segment') {
                displaySegmentationResults(data);
            } else {
                displayClassificationResults(data);
            }
        } else {
            showError(data.error || `${mode === 'segment' ? 'Segmentation' : 'Classification'} failed. Please try again.`);
            previewSection.style.display = 'block';
        }
    } catch (error) {
        console.error('Error:', error);
        showError('An error occurred. Please try again.');
        previewSection.style.display = 'block';
    } finally {
        loadingSpinner.style.display = 'none';
    }
}

// Display Segmentation Results
function displaySegmentationResults(data) {
    // Set images
    document.getElementById('segOriginalImage').src = data.original_image;
    document.getElementById('segmentedImage').src = data.segmented_mask;
    document.getElementById('overlayImage').src = data.overlay_image;
    
    // Display training metrics if available
    if (data.training_metrics) {
        document.getElementById('segTrainingMetrics').src = data.training_metrics;
    }
    
    // Display class distribution
    displaySegmentationDistribution(data.class_distribution);
    
    // Show results section
    segmentationResults.style.display = 'block';
    
    // Scroll to results
    segmentationResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display Segmentation Distribution
function displaySegmentationDistribution(distribution) {
    const container = document.getElementById('segClassDistribution');
    container.innerHTML = '';
    
    // Sort by percentage (descending)
    const sortedClasses = Object.entries(distribution).sort((a, b) => b[1] - a[1]);
    
    sortedClasses.forEach(([className, percentage]) => {
        const classBar = document.createElement('div');
        classBar.className = 'class-bar';
        
        const label = document.createElement('div');
        label.className = 'class-label';
        label.textContent = className;
        
        const barContainer = document.createElement('div');
        barContainer.className = 'bar-container';
        
        const barFill = document.createElement('div');
        barFill.className = 'bar-fill';
        barFill.style.width = '0%';
        
        const barPercentage = document.createElement('span');
        barPercentage.className = 'bar-percentage';
        barPercentage.textContent = `${percentage.toFixed(2)}%`;
        
        barFill.appendChild(barPercentage);
        barContainer.appendChild(barFill);
        classBar.appendChild(label);
        classBar.appendChild(barContainer);
        container.appendChild(classBar);
        
        // Animate bar
        setTimeout(() => {
            barFill.style.width = `${Math.min(percentage, 100)}%`;
        }, 100);
    });
}

// Display Classification Results
function displayClassificationResults(data) {
    // Set original image
    document.getElementById('clsOriginalImage').src = data.original_image;
    
    // Set classification message
    document.getElementById('classificationMessage').textContent = data.message;
    
    // Display prediction
    const predictionResult = document.getElementById('predictionResult');
    predictionResult.innerHTML = `
        <div class="prediction-badge ${data.predicted_class.toLowerCase()}">
            ${data.predicted_class}
        </div>
        <p class="prediction-confidence">Confidence: ${(data.class_probabilities[data.predicted_class] * 100).toFixed(1)}%</p>
    `;
    
    // Display class probabilities
    displayClassProbabilities(data.class_probabilities, data.predicted_class);
    
    // Display confusion matrix
    if (data.confusion_matrix) {
        document.getElementById('confusionMatrix').src = data.confusion_matrix;
    }
    
    // Display training history
    if (data.training_history) {
        document.getElementById('trainingHistory').src = data.training_history;
    }
    
    // Display model summary
    if (data.model_summary) {
        displayModelSummary(data.model_summary);
    }
    
    // Show results section
    classificationResults.style.display = 'block';
    
    // Scroll to results
    classificationResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display Class Probabilities
function displayClassProbabilities(probabilities, predictedClass) {
    const container = document.getElementById('classProbabilities');
    container.innerHTML = '';
    
    // Sort by probability (descending)
    const sortedClasses = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
    
    sortedClasses.forEach(([className, probability]) => {
        const isPredicted = className === predictedClass;
        
        const classBar = document.createElement('div');
        classBar.className = `probability-bar ${isPredicted ? 'predicted' : ''}`;
        
        const label = document.createElement('div');
        label.className = 'class-label';
        label.innerHTML = `
            ${className}
            ${isPredicted ? '<span class="predicted-tag">✓ Predicted</span>' : ''}
        `;
        
        const barContainer = document.createElement('div');
        barContainer.className = 'bar-container';
        
        const barFill = document.createElement('div');
        barFill.className = 'bar-fill';
        barFill.style.width = '0%';
        
        const barPercentage = document.createElement('span');
        barPercentage.className = 'bar-percentage';
        barPercentage.textContent = `${(probability * 100).toFixed(2)}%`;
        
        barFill.appendChild(barPercentage);
        barContainer.appendChild(barFill);
        classBar.appendChild(label);
        classBar.appendChild(barContainer);
        container.appendChild(classBar);
        
        // Animate bar
        setTimeout(() => {
            barFill.style.width = `${Math.min(probability * 100, 100)}%`;
        }, 100);
    });
}

// Display Model Summary
function displayModelSummary(summary) {
    const container = document.getElementById('modelSummary');
    container.innerHTML = '';
    
    const summaryData = [
        { label: 'Model Architecture', value: summary.model_name || 'N/A' },
        { label: 'Number of Classes', value: summary.num_classes || 'N/A' },
        { label: 'Image Size', value: `${summary.img_size}×${summary.img_size}` || 'N/A' },
        { label: 'Training Samples', value: summary.training_samples || 'N/A' },
        { label: 'Validation Accuracy', value: summary.best_val_accuracy ? `${summary.best_val_accuracy.toFixed(2)}%` : 'N/A' },
        { label: 'Test Accuracy', value: summary.test_accuracy ? `${summary.test_accuracy.toFixed(2)}%` : 'N/A' },
        { label: 'Total Parameters', value: summary.total_parameters ? summary.total_parameters.toLocaleString() : 'N/A' },
    ];
    
    const grid = document.createElement('div');
    grid.className = 'summary-grid';
    
    summaryData.forEach(item => {
        const summaryItem = document.createElement('div');
        summaryItem.className = 'summary-item';
        summaryItem.innerHTML = `
            <span class="summary-label">${item.label}:</span>
            <span class="summary-value">${item.value}</span>
        `;
        grid.appendChild(summaryItem);
    });
    
    container.appendChild(grid);
}

// Download Image
function downloadImage(imageId, filename) {
    const image = document.getElementById(imageId);
    const link = document.createElement('a');
    link.href = image.src;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Reset Upload
function resetUpload() {
    selectedFile = null;
    currentMode = null;
    fileInput.value = '';
    previewImage.src = '';
    uploadArea.style.display = 'block';
    previewSection.style.display = 'none';
    segmentationResults.style.display = 'none';
    classificationResults.style.display = 'none';
    loadingSpinner.style.display = 'none';
    hideError();
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Show Error
function showError(message) {
    errorMessage.textContent = message;
    errorAlert.style.display = 'flex';
    
    // Auto hide after 8 seconds
    setTimeout(hideError, 8000);
}

// Hide Error
function hideError() {
    errorAlert.style.display = 'none';
}

// Check server health on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        if (!data.segmentation_model_loaded && !data.classification_model_loaded) {
            showError('Warning: No models loaded. Please check the console.');
        } else if (!data.segmentation_model_loaded) {
            console.warn('Segmentation model not loaded');
        } else if (!data.classification_model_loaded) {
            console.warn('Classification model not loaded');
        }

        console.log('Server health:', data);
    } catch (error) {
        console.error('Could not check server health:', error);
    }
});

/* ===================================
   XAI (Explainable AI) Functions
   =================================== */

// Add event listener for Explain button
const explainButton = document.getElementById('explainButton');
const xaiResults = document.getElementById('xaiResults');
const xaiAnotherButton = document.getElementById('xaiAnotherButton');

if (explainButton) {
    explainButton.addEventListener('click', generateXAIExplanations);
}

if (xaiAnotherButton) {
    xaiAnotherButton.addEventListener('click', resetUpload);
}

// Generate XAI Explanations
async function generateXAIExplanations() {
    if (!selectedFile) {
        showError('No file selected. Please upload an image first.');
        return;
    }

    // Hide classification results and show loading
    classificationResults.style.display = 'none';
    loadingSpinner.style.display = 'flex';
    document.getElementById('loadingText').textContent = 'Generating AI explanations... This may take a minute';
    hideError();

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/explain', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            displayXAIResults(data);
        } else {
            throw new Error(data.error || 'XAI explanation failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showError(`Failed to generate explanations: ${error.message}`);
        classificationResults.style.display = 'block';
    } finally {
        loadingSpinner.style.display = 'none';
    }
}

// Display XAI Results
function displayXAIResults(data) {
    // Populate summary
    const xaiSummary = document.getElementById('xaiSummary');
    xaiSummary.innerHTML = `
        <div class="summary-row">
            <span class="label">Predicted Class:</span>
            <span class="value highlight">${data.predicted_class}</span>
        </div>
        <div class="summary-row">
            <span class="label">Confidence:</span>
            <span class="value">${(data.confidence * 100).toFixed(1)}%</span>
        </div>
        <div class="summary-row">
            <span class="label">Attribution Coverage (Grad-CAM):</span>
            <span class="value">${data.metrics.gradcam.attribution_coverage_percent.toFixed(1)}%</span>
        </div>
        <div class="summary-row">
            <span class="label">Peak Focus Location:</span>
            <span class="value">(${data.metrics.gradcam.peak_activation_location.x}, ${data.metrics.gradcam.peak_activation_location.y})</span>
        </div>
    `;

    // Populate images
    document.getElementById('xaiGradcam').src = data.images.gradcam;
    document.getElementById('xaiGradcamBbox').src = data.images.gradcam_with_boxes;
    document.getElementById('xaiLime').src = data.images.lime;
    document.getElementById('xaiIG').src = data.images.integrated_gradients;
    document.getElementById('xaiComparison').src = data.images.comparison;

    // Populate metrics
    displayXAIMetrics(data.metrics, data.interpretation);

    // Update message
    document.getElementById('xaiMessage').textContent = data.message;

    // Show XAI results
    xaiResults.style.display = 'block';
    
    // Smooth scroll to XAI results
    xaiResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Display XAI Metrics
function displayXAIMetrics(metrics, interpretation) {
    const metricsContainer = document.getElementById('xaiMetrics');
    metricsContainer.innerHTML = '';

    const metricsData = [
        { 
            label: 'Grad-CAM Attribution Coverage', 
            value: `${metrics.gradcam.attribution_coverage_percent.toFixed(1)}%`,
            description: 'Percentage of image contributing to the decision'
        },
        { 
            label: 'Grad-CAM Peak Location', 
            value: `(${metrics.gradcam.peak_activation_location.x}, ${metrics.gradcam.peak_activation_location.y})`,
            description: 'Location of highest importance'
        },
        { 
            label: 'Grad-CAM Mean Attribution', 
            value: metrics.gradcam.mean_attribution.toFixed(3),
            description: 'Average attribution strength'
        },
        { 
            label: 'Integrated Gradients Coverage', 
            value: `${metrics.integrated_gradients.attribution_coverage_percent.toFixed(1)}%`,
            description: 'Percentage of pixels with significant attribution'
        },
        { 
            label: 'IG Mean Attribution', 
            value: metrics.integrated_gradients.mean_attribution.toFixed(3),
            description: 'Average pixel-level contribution'
        },
        { 
            label: 'IG Max Attribution', 
            value: metrics.integrated_gradients.max_attribution.toFixed(3),
            description: 'Maximum pixel contribution'
        }
    ];

    metricsData.forEach(item => {
        const metricItem = document.createElement('div');
        metricItem.className = 'xai-metric-item';
        metricItem.innerHTML = `
            <div class="xai-metric-label">${item.label}</div>
            <div class="xai-metric-value">${item.value}</div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem;">
                ${item.description}
            </div>
        `;
        metricsContainer.appendChild(metricItem);
    });

    // Add interpretation note
    if (interpretation && interpretation.clinical_notes) {
        const noteDiv = document.createElement('div');
        noteDiv.style.gridColumn = '1 / -1';
        noteDiv.style.marginTop = '1rem';
        noteDiv.style.padding = '1rem';
        noteDiv.style.background = 'var(--bg-secondary)';
        noteDiv.style.borderRadius = '8px';
        noteDiv.style.borderLeft = '4px solid var(--secondary-color)';
        noteDiv.innerHTML = `
            <strong style="color: var(--primary-color);">Clinical Notes:</strong><br>
            ${interpretation.clinical_notes}
        `;
        metricsContainer.appendChild(noteDiv);
    }
}