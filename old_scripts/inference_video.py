import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import os

# Color and label maps
CLASS_COLORS = {
    0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0),
    5: (255, 0, 255), 6: (0, 255, 255), 7: (255, 128, 0), 8: (128, 0, 255), 9: (0, 255, 128),
    10: (255, 0, 128), 11: (128, 255, 0), 12: (0, 128, 255), 13: (255, 255, 128),
    14: (255, 128, 255), 15: (128, 255, 255), 16: (255, 128, 128), 17: (128, 255, 128),
    18: (128, 128, 255), 19: (255, 128, 192)
}
CLASS_LABELS = [
    "Background", "Wetspot", "Rust", "EJoint", "ACrack", "WConccor", "Cavity", "Hollowareas",
    "JTape", "Spalling", "Rockpocket", "ExposedRebars", "Crack", "Restformwork", "Drainage",
    "Weathering", "Bearing", "Graffiti", "PEquipment", "Efflorescence"
]

def load_model(weights_path, device='cuda'):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights='imagenet',
        in_channels=3,
        classes=20
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(model, image, device='cuda'):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_img = pil_img.resize((512, 512), Image.BILINEAR)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)[0]
        mask = torch.argmax(output, dim=0).cpu().numpy()
    return np.array(pil_img), mask

def overlay(image, mask, alpha=0.5):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = np.zeros_like(image)
    for class_id, color in CLASS_COLORS.items():
        overlay[mask == class_id] = color
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

def process_video(video_path, output_path, model_path, device='cuda'):
    model = load_model(model_path, device)
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image, mask = predict(model, frame, device)
        result = overlay(image, mask)
        # Optionally, annotate present classes (as in image script)
        present_classes = np.unique(mask)
        y0, dy = 30, 30
        for i, class_id in enumerate(present_classes):
            if class_id == 0:
                continue
            label = CLASS_LABELS[class_id] if class_id < len(CLASS_LABELS) else f"Class {class_id}"
            color = CLASS_COLORS.get(class_id, (255,255,255))
            y = y0 + i * dy
            cv2.putText(result, label, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        if out is None:
            h, w = result.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
        out.write(result)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")
    cap.release()
    if out:
        out.release()
    print(f"[✓] Saved annotated video to: {output_path}")

if __name__ == "__main__":
    video_path = "/home/aero360/Downloads/3542708341-preview.mp4"  # Path to your input video
    output_path = "output_annotated_video.mp4"
    model_path = "dacl10k15_deeplabv3plus.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    process_video(video_path, output_path, model_path, device)
