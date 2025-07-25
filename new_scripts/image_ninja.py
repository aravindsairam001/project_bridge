import matplotlib
matplotlib.use('Agg')
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os

# Import centralized configuration
from config import (
    CLASS_COLORS, CLASS_LABELS, ALLOWED_CLASS_IDS, NUM_CLASSES,
    detect_architecture_from_filename, get_display_name, rgb_to_bgr
)

# Load the model - Auto-detect architecture from filename
def load_model(weights_path):
    """Load model and auto-detect architecture from filename"""
    
    # Auto-detect architecture and encoder from filename
    architecture, encoder_name = detect_architecture_from_filename(weights_path)
    
    print(f"[INFO] Detected architecture: {architecture.upper()}")
    print(f"[INFO] Detected encoder: {encoder_name}")
    
    # Create model based on detected architecture
    if architecture == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=NUM_CLASSES
        )
    elif architecture == 'fpn':
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=NUM_CLASSES
        )
    elif architecture == 'linknet':
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=NUM_CLASSES
        )
    elif architecture == 'pspnet':
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=NUM_CLASSES
        )
    else:  # deeplabv3plus
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=3,
            classes=NUM_CLASSES
        )
    
    model.load_state_dict(torch.load(weights_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    return model

# Preprocess image
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Predict mask
def predict(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize((512, 512), Image.BILINEAR)
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)[0]
        mask = torch.argmax(output, dim=0).numpy()
    return np.array(image_resized), mask

# Overlay mask
def overlay(image, mask, alpha=0.5):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = np.zeros_like(image)
    
    # Define alias colors for consistent visualization
    ALIAS_COLORS_OVERLAY = {
        4: (255, 165, 0),   # Honeycombing (Cavity) - Orange
        5: (255, 165, 0),   # Honeycombing (Hollowareas) - Orange  
        10: (0, 255, 255),  # Leaching (Weathering) - Cyan
        11: (0, 255, 255)   # Leaching (Efflorescence) - Cyan
    }
    
    for class_id, color in CLASS_COLORS.items():
        if class_id in ALIAS_COLORS_OVERLAY:
            # Use alias color for aliased defects
            overlay[mask == class_id] = ALIAS_COLORS_OVERLAY[class_id]
        else:
            # Use original color for non-aliased classes
            overlay[mask == class_id] = color
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

# Only process these classes (same as LABEL_MAP in min_class.py)
ALLOWED_CLASS_IDS = set([1,2,3,4,5,6,7,8,9,10,11])

# Run

if __name__ == "__main__":
    model_path = "dacl10k_unetplusplus_efficientnet_b5_ver1.pth"
    test_image_path = "input_image.jpg"  # <-- your test image
    output_path = "prediction_overlay.jpg"

    model = load_model(model_path)
    image, mask = predict(model, test_image_path)
    result = overlay(image, mask)

    # Annotate present defect class names on the overlay
    present_classes = np.unique(mask)
    present_labels = []
    
    # Map class id to readable label
    CLASS_LABELS = {
        0: "Background",
        1: "Rust", 
        2: "ACrack",
        3: "WConccor", 
        4: "Cavity",
        5: "Hollowareas",
        6: "Spalling",
        7: "Rockpocket",
        8: "ExposedRebars",
        9: "Crack",
        10: "Weathering",
        11: "Efflorescence"
    }
    
    # Aliases for output
    DEFECT_ALIASES = {
        4: "Honeycombing",   # Cavity
        5: "Honeycombing",   # Hollowareas
        10: "Leaching",      # Weathering
        11: "Leaching"       # Efflorescence
    }
    
    # Alias colors - same color for aliased defects
    ALIAS_COLORS = {
        "Honeycombing": (255, 165, 0),  # Orange for both Cavity and Hollowareas
        "Leaching": (0, 255, 255)       # Cyan for both Weathering and Efflorescence
    }
    for class_id in present_classes:
        if class_id == 0 or class_id not in ALLOWED_CLASS_IDS:
            continue  # skip background and unwanted classes
        # Calculate pixel coverage for this defect
        pixel_coverage = np.sum(mask == class_id)
        if pixel_coverage <= 1000:
            continue  # skip small regions
        # Use alias if available
        if class_id in DEFECT_ALIASES:
            label = DEFECT_ALIASES[class_id]
        else:
            label = CLASS_LABELS.get(class_id, f"Class {class_id}")
        present_labels.append(label)

    # Visualization mode: 'overlay' or 'bbox'
    VIS_MODE = 'bbox'  # Change to 'overlay' for color overlay only

    annotated = result.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    y0, dy = 30, 30

    label_positions = {}
    for i, class_id in enumerate(present_classes):
        if class_id == 0 or class_id not in ALLOWED_CLASS_IDS:
            continue  # skip background and unwanted classes
        pixel_coverage = np.sum(mask == class_id)
        if pixel_coverage <= 1000:
            continue
        if class_id in DEFECT_ALIASES:
            label = DEFECT_ALIASES[class_id]
            color = ALIAS_COLORS.get(label, CLASS_COLORS.get(class_id, (255,255,255)))
        else:
            label = CLASS_LABELS.get(class_id, f"Class {class_id}")
            color = CLASS_COLORS.get(class_id, (255,255,255))
        y = y0 + (len(label_positions)) * dy
        label_positions[class_id] = (20, y)
        cv2.putText(annotated, label, (20, y), font, font_scale, color, 2, cv2.LINE_AA)

    if VIS_MODE == 'bbox':
        # Draw bounding boxes for each defect class
        for class_id in present_classes:
            if class_id == 0 or class_id not in ALLOWED_CLASS_IDS:
                continue  # skip background and unwanted classes
            pixel_coverage = np.sum(mask == class_id)
            if pixel_coverage <= 1000:
                continue
            # Get color based on alias if available
            if class_id in DEFECT_ALIASES:
                label = DEFECT_ALIASES[class_id]
                color = ALIAS_COLORS.get(label, CLASS_COLORS.get(class_id, (255,255,255)))
            else:
                label = CLASS_LABELS.get(class_id, f"Class {class_id}")
                color = CLASS_COLORS.get(class_id, (255,255,255))
            mask_bin = (mask == class_id).astype(np.uint8)
            if np.count_nonzero(mask_bin) == 0:
                continue
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
                # Draw label above the box with alias color
                cv2.putText(annotated, label, (x, y-10), font, 0.6, color, 2, cv2.LINE_AA)

    cv2.imwrite(output_path, annotated)
    print(f"[✓] Saved overlay to: {output_path}")
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title("Defect Segmentation")
    plt.axis("off")
    # plt.show()