"""
Common utility functions for image processing, visualization, and model operations.
This module provides reusable functions to keep the main scripts clean.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import segmentation_models_pytorch as smp

from config import (
    CLASS_COLORS, ALLOWED_CLASS_IDS, NUM_CLASSES, get_display_name, get_alias_color, rgb_to_bgr
)


def preprocess_image(image, target_size=(512, 512)):
    """
    Preprocess image for model input
    
    Args:
        image (PIL.Image): Input image
        target_size (tuple): Target size for resizing
        
    Returns:
        tuple: (processed_image_array, input_tensor)
    """
    # Convert PIL to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image_resized = image.resize(target_size, Image.BILINEAR)
    
    # Transform for model (ImageNet normalization)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image_resized).unsqueeze(0)
    
    return np.array(image_resized), input_tensor


def predict_segmentation_mask(model, input_tensor, device='cuda'):
    """
    Predict segmentation mask using model
    
    Args:
        model: Trained segmentation model
        input_tensor: Preprocessed input tensor
        device: Device to run inference on
        
    Returns:
        numpy.ndarray: Predicted mask
    """
    if torch.cuda.is_available() and device == 'cuda':
        input_tensor = input_tensor.cuda()
    
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    return mask


def create_segmentation_overlay(image, mask, alpha=0.6, defect_visibility=None):
    """
    Create color overlay for segmentation results
    
    Args:
        image: Input image (RGB numpy array)
        mask: Segmentation mask
        alpha: Overlay transparency (0-1)
        defect_visibility: Dict of defect visibility settings
        
    Returns:
        numpy.ndarray: Image with overlay
    """
    # Convert image to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = np.zeros_like(image_bgr)
    
    # Apply colors for each class
    for class_id, color in CLASS_COLORS.items():
        if class_id == 0:  # Skip background
            continue
            
        # Check if this defect should be visible
        if defect_visibility is not None:
            display_name = get_display_name(class_id)
            if not defect_visibility.get(display_name, True):
                continue
        
        # Get appropriate color (alias or regular)
        overlay_color = get_alias_color(class_id)
        bgr_color = rgb_to_bgr(overlay_color)
        overlay[mask == class_id] = bgr_color
    
    # Blend images
    result = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)
    
    # Convert back to RGB
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def create_bounding_box_visualization(image, mask, present_classes, 
                                    pixel_threshold=1000, defect_visibility=None):
    """
    Create bounding box visualization for detected defects
    
    Args:
        image: Input image (RGB numpy array)
        mask: Segmentation mask
        present_classes: List of classes present in mask
        pixel_threshold: Minimum pixels to show defect
        defect_visibility: Dict of defect visibility settings
        
    Returns:
        numpy.ndarray: Image with bounding boxes
    """
    annotated_image = image.copy()
    
    # Draw bounding boxes for each defect class
    for class_id in present_classes:
        if class_id == 0 or class_id not in ALLOWED_CLASS_IDS:
            continue
        pixel_coverage = np.sum(mask == class_id)
        if pixel_coverage <= pixel_threshold:
            continue
            
        # Get display name
        label = get_display_name(class_id)
        
        # Check visibility
        if defect_visibility and not defect_visibility.get(label, True):
            continue
        
        # Use uniform color for bounding boxes
        bbox_color = (0, 255, 0)  # Green
        
        # Find contours and draw bounding boxes
        mask_bin = (mask == class_id).astype(np.uint8)
        if np.count_nonzero(mask_bin) == 0:
            continue
            
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:  # Ignore tiny boxes
                continue
            cv2.rectangle(annotated_image, (x, y), (x+w, y+h), bbox_color, 2)
            cv2.putText(annotated_image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, bbox_color, 2, cv2.LINE_AA)
    
    return annotated_image


def add_class_labels_to_image(image, mask, present_classes, 
                             pixel_threshold=1000, defect_visibility=None):
    """
    Add class labels to image margin
    
    Args:
        image: Input image (RGB numpy array)
        mask: Segmentation mask
        present_classes: List of classes present in mask
        pixel_threshold: Minimum pixels to show defect
        defect_visibility: Dict of defect visibility settings
        
    Returns:
        numpy.ndarray: Image with labels
    """
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    y0, dy = 30, 30
    label_count = 0
    
    for class_id in present_classes:
        if class_id == 0 or class_id not in ALLOWED_CLASS_IDS:
            continue
        pixel_coverage = np.sum(mask == class_id)
        if pixel_coverage <= pixel_threshold:
            continue
            
        # Get display name
        label = get_display_name(class_id)
        
        # Check visibility
        if defect_visibility is not None:
            if not defect_visibility.get(label, True):
                continue
        
        y = y0 + label_count * dy
        # Use white text for readability
        text_color = (255, 255, 255)
        bgr_color = rgb_to_bgr(text_color)
        cv2.putText(annotated, label, (20, y), font, font_scale, bgr_color, thickness, cv2.LINE_AA)
        label_count += 1
    
    return annotated


def analyze_segmentation_results(mask, present_classes, pixel_threshold=1000, defect_visibility=None):
    """
    Analyze segmentation results and generate statistics
    
    Args:
        mask: Segmentation mask
        present_classes: List of classes present in mask
        pixel_threshold: Minimum pixels to include defect
        defect_visibility: Dict of defect visibility settings
        
    Returns:
        dict: Statistics for each defect type
    """
    stats = {}
    total_pixels = mask.shape[0] * mask.shape[1]
    
    for class_id in present_classes:
        if class_id == 0 or class_id not in ALLOWED_CLASS_IDS:
            continue
        class_pixels = np.sum(mask == class_id)
        if class_pixels <= pixel_threshold:
            continue
            
        percentage = (class_pixels / total_pixels) * 100
        
        # Get display name and color
        label = get_display_name(class_id)
        color = get_alias_color(class_id)
        
        # Check visibility
        if defect_visibility is not None:
            if not defect_visibility.get(label, True):
                continue
        
        # Combine stats for aliased defects
        if label in stats:
            stats[label]['pixels'] += class_pixels
            stats[label]['percentage'] += percentage
        else:
            stats[label] = {
                'pixels': class_pixels,
                'percentage': percentage,
                'color': color
            }
    
    return stats


def create_model_from_architecture(architecture, encoder_name, num_classes=NUM_CLASSES):
    """
    Create segmentation model based on architecture and encoder
    
    Args:
        architecture: Model architecture name
        encoder_name: Encoder name
        num_classes: Number of output classes
        
    Returns:
        torch.nn.Module: Created model
    """
    if architecture == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )
    elif architecture == 'fpn':
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )
    elif architecture == 'linknet':
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )
    elif architecture == 'pspnet':
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )
    else:  # deeplabv3plus (default)
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=num_classes
        )
    
    return model


def load_trained_model(weights_path, architecture=None, encoder_name=None):
    """
    Load trained model with automatic architecture detection
    
    Args:
        weights_path: Path to model weights
        architecture: Model architecture (auto-detect if None)
        encoder_name: Encoder name (auto-detect if None)
        
    Returns:
        tuple: (model, device, architecture, encoder_name)
    """
    from config import detect_architecture_from_filename
    
    # Auto-detect if not specified
    if architecture is None or encoder_name is None:
        architecture, encoder_name = detect_architecture_from_filename(weights_path)
    
    # Create model
    model = create_model_from_architecture(architecture, encoder_name)
    
    # Load weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model, device, architecture, encoder_name


def filter_detected_defects(present_classes, mask, pixel_threshold=1000):
    """
    Filter detected defects based on pixel threshold
    
    Args:
        present_classes: List of classes present in mask
        mask: Segmentation mask
        pixel_threshold: Minimum pixels to include defect
        
    Returns:
        list: Filtered list of defect names
    """
    detected_defects = []
    
    for class_id in present_classes:
        if class_id == 0 or class_id not in ALLOWED_CLASS_IDS:
            continue
        
        # Check if defect meets pixel threshold
        pixel_coverage = np.sum(mask == class_id)
        if pixel_coverage <= pixel_threshold:
            continue
        
        # Get display name
        display_name = get_display_name(class_id)
        
        if display_name not in detected_defects:
            detected_defects.append(display_name)
    
    return detected_defects


# ============================================================================
# IMAGE I/O UTILITIES
# ============================================================================

def save_image_as_bytes(image, format='PNG'):
    """
    Convert image to bytes for download
    
    Args:
        image: PIL Image or numpy array
        format: Image format (PNG, JPEG)
        
    Returns:
        bytes: Image as bytes
    """
    from io import BytesIO
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    buf = BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def generate_analysis_report(defect_stats, filename="", processing_date=""):
    """
    Generate text analysis report
    
    Args:
        defect_stats: Dictionary of defect statistics
        filename: Original filename
        processing_date: Processing date
        
    Returns:
        str: Formatted report
    """
    report = f"""
Bridge Defect Detection Report
=============================

Image: {filename}
Processing Date: {processing_date}

Defects Detected: {len(defect_stats)}

Detailed Analysis:
"""
    
    for defect_name, stats in defect_stats.items():
        report += f"\n- {defect_name}: {stats['percentage']:.2f}% coverage ({stats['pixels']:,} pixels)"
    
    if len(defect_stats) == 0:
        report += "\nNo defects detected in this image."
    
    return report
