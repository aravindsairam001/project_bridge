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

# Same class-color mapping as mask generation (19 classes + background)
CLASS_COLORS = {
    0: (0, 0, 0),           # 0: background (black)
    1: (255, 0, 0),         # 1: Wetspot (bright red)
    2: (0, 255, 0),         # 2: Rust (bright green)
    3: (0, 0, 255),         # 3: EJoint (bright blue)
    4: (255, 255, 0),       # 4: ACrack (yellow)
    5: (255, 0, 255),       # 5: WConccor (magenta)
    6: (0, 255, 255),       # 6: Cavity (cyan)
    7: (255, 128, 0),       # 7: Hollowareas (orange)
    8: (128, 0, 255),       # 8: JTape (purple)
    9: (0, 255, 128),       # 9: Spalling (spring green)
    10: (255, 0, 128),      # 10: Rockpocket (rose)
    11: (128, 255, 0),      # 11: ExposedRebars (lime)
    12: (0, 128, 255),      # 12: Crack (azure)
    13: (255, 255, 128),    # 13: Restformwork (light yellow)
    14: (255, 128, 255),    # 14: Drainage (pink)
    15: (128, 255, 255),    # 15: Weathering (light cyan)
    16: (255, 128, 128),    # 16: Bearing (light red)
    17: (128, 255, 128),    # 17: Graffiti (light green)
    18: (128, 128, 255),    # 18: PEquipment (light blue)
    19: (255, 128, 192),    # 19: Efflorescence (light pink)
}

# Load the model
def load_model(weights_path):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101", 
        encoder_weights='imagenet',  # match training
        in_channels=3,
        classes=20  # match NUM_CLASSES in training
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
    for class_id, color in CLASS_COLORS.items():
        overlay[mask == class_id] = color
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

# Run

if __name__ == "__main__":
    model_path = "dacl10k_ninja_new.pth"
    test_image_path = "input_image.jpg"  # <-- your test image
    output_path = "prediction_overlay.jpg"

    model = load_model(model_path)
    image, mask = predict(model, test_image_path)
    result = overlay(image, mask)

    # Annotate present defect class names on the overlay
    present_classes = np.unique(mask)
    present_labels = []
    # Map class id to readable label with aliasing
    CLASS_LABELS = [
        "Background", "Wetspot", "Rust", "EJoint", "ACrack", "WConccor", "Cavity", "Hollowareas",
        "JTape", "Spalling", "Rockpocket", "ExposedRebars", "Crack", "Restformwork", "Drainage",
        "Weathering", "Bearing", "Graffiti", "PEquipment", "Efflorescence"
    ]
    # Aliases for output
    DEFECT_ALIASES = {
        6: "Honeycombing",   # Cavity
        7: "Honeycombing",   # Hollowareas
        15: "Leaching",      # Weathering
        19: "Leaching"       # Efflorescence
    }
    for class_id in present_classes:
        if class_id == 0:
            continue  # skip background
        # Calculate pixel coverage for this defect
        pixel_coverage = np.sum(mask == class_id)
        if pixel_coverage <= 1000:
            continue  # skip small regions
        # Use alias if available
        if class_id in DEFECT_ALIASES:
            label = DEFECT_ALIASES[class_id]
        else:
            label = CLASS_LABELS[class_id] if class_id < len(CLASS_LABELS) else f"Class {class_id}"
        present_labels.append(label)

    # Visualization mode: 'overlay' or 'bbox'
    VIS_MODE = 'bbox'  # Change to 'overlay' for color overlay only

    annotated = result.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    y0, dy = 30, 30

    label_positions = {}
    for i, class_id in enumerate(present_classes):
        if class_id == 0:
            continue  # skip background
        pixel_coverage = np.sum(mask == class_id)
        if pixel_coverage <= 1000:
            continue
        if class_id in DEFECT_ALIASES:
            label = DEFECT_ALIASES[class_id]
        else:
            label = CLASS_LABELS[class_id] if class_id < len(CLASS_LABELS) else f"Class {class_id}"
        y = y0 + (len(label_positions)) * dy
        label_positions[class_id] = (20, y)
        color = CLASS_COLORS.get(class_id, (255,255,255))
        cv2.putText(annotated, label, (20, y), font, font_scale, color, 2, cv2.LINE_AA)

    if VIS_MODE == 'bbox':
        # Draw bounding boxes for each defect class
        for class_id in present_classes:
            if class_id == 0:
                continue  # skip background
            pixel_coverage = np.sum(mask == class_id)
            if pixel_coverage <= 1000:
                continue
            color = CLASS_COLORS.get(class_id, (255,255,255))
            mask_bin = (mask == class_id).astype(np.uint8)
            if np.count_nonzero(mask_bin) == 0:
                continue
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
                # Optionally, draw label above the box
                if class_id in DEFECT_ALIASES:
                    label = DEFECT_ALIASES[class_id]
                else:
                    label = CLASS_LABELS[class_id] if class_id < len(CLASS_LABELS) else f"Class {class_id}"
                cv2.putText(annotated, label, (x, y-10), font, 0.6, color, 2, cv2.LINE_AA)

    cv2.imwrite(output_path, annotated)
    print(f"[✓] Saved overlay to: {output_path}")
    plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    plt.title("Defect Segmentation")
    plt.axis("off")
    # plt.show()