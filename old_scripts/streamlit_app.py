from datetime import datetime
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os
import io

# Configure page
st.set_page_config(
    page_title="Bridge Defect Detection",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class colors and labels (same as inference_image.py)
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

@st.cache_resource
def load_model(weights_path, encoder_name="resnet34"):
    """Load the trained model with caching to avoid reloading"""
    try:
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,  # Use selected encoder
            encoder_weights='imagenet',
            in_channels=3,
            classes=20  # 20 classes including background
        )
        
        # Load weights
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_image(image, target_size=(512, 512)):
    """Preprocess image for model input"""
    # Convert PIL to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image_resized = image.resize(target_size, Image.BILINEAR)
    
    # Transform for model
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image_resized).unsqueeze(0)
    
    return np.array(image_resized), input_tensor

def predict_mask(model, input_tensor, device):
    """Predict segmentation mask"""
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    return mask

def create_overlay(image, mask, alpha=0.6):
    """Create overlay of mask on original image"""
    # Convert image to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = np.zeros_like(image_bgr)
    
    # Apply colors for each class
    for class_id, color in CLASS_COLORS.items():
        # Convert RGB to BGR for OpenCV
        bgr_color = (color[2], color[1], color[0])
        overlay[mask == class_id] = bgr_color
    
    # Blend images
    result = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)
    
    # Convert back to RGB for display
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

def create_annotated_image(image, mask, present_classes):
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    y0, dy = 30, 30
    label_count = 0
    for class_id in present_classes:
        if class_id == 0:
            continue
        pixel_coverage = np.sum(mask == class_id)
        if pixel_coverage <= 1000:
            continue
        label = DEFECT_ALIASES.get(class_id, CLASS_LABELS[class_id] if class_id < len(CLASS_LABELS) else f"Class {class_id}")
        y = y0 + label_count * dy
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        bgr_color = (int(color[2]), int(color[1]), int(color[0]))
        cv2.putText(annotated, label, (20, y), font, font_scale, bgr_color, thickness, cv2.LINE_AA)
        label_count += 1
    return annotated

def analyze_defects(mask, present_classes):
    stats = {}
    total_pixels = mask.shape[0] * mask.shape[1]
    for class_id in present_classes:
        if class_id == 0:
            continue
        class_pixels = np.sum(mask == class_id)
        if class_pixels <= 1000:
            continue
        percentage = (class_pixels / total_pixels) * 100
        label = DEFECT_ALIASES.get(class_id, CLASS_LABELS[class_id] if class_id < len(CLASS_LABELS) else f"Class {class_id}")
        stats[label] = {
            'pixels': class_pixels,
            'percentage': percentage,
            'color': CLASS_COLORS.get(class_id, (255, 255, 255))
        }
    return stats

def main():
    st.title("🌉 Bridge Defect Detection System")
    st.markdown("Upload an image of a bridge to detect and visualize structural defects using AI.")
    
    # Sidebar for model selection and parameters
    st.sidebar.header("Settings")
    
    # Model architecture selection
    st.sidebar.subheader("Model Configuration")
    encoder_name = st.sidebar.selectbox(
        "Encoder Architecture",
        ["resnet34", "resnet50", "resnet101"],
        index=0,  # Default to resnet34
        help="Select the encoder architecture used in your trained model"
    )
    
    # Image processing size
    image_size = st.sidebar.selectbox(
        "Processing Image Size",
        [512],
        index=0,  # Default to 512
        help="Image size used during training (should match your model)"
    )
    
    # Model path selection
    model_path = st.sidebar.selectbox(
        "Model Path", 
        ["dacl10k_ninja.pth", "dacl10k_ninja_new.pth", "dacl10k_resnet50.pth"],
        help="Path to the trained model file"
    )
    
    # Visualization mode
    vis_mode = st.sidebar.radio(
        "Visualization Mode",
        ["Color Overlay", "Bounding Box"],
        index=0,
        help="Choose how to visualize detected defects"
    )

    # Overlay transparency (only for overlay mode)
    alpha = 0.6
    if vis_mode == "Color Overlay":
        alpha = st.sidebar.slider(
            "Overlay Transparency", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.6, 
            step=0.1,
            help="Transparency of the segmentation overlay"
        )

    # Show/hide annotations
    show_annotations = st.sidebar.checkbox("Show Class Labels", value=True)
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a bridge image for defect detection"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image")
        
        # Check if model exists
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Please train the model first or check the path.")
            return
        
        # Load model
        with st.spinner("Loading model..."):
            model, device = load_model(model_path, encoder_name)
        
        if model is None:
            return
        
        # Process image
        with st.spinner("Processing image..."):
            # Preprocess
            processed_image, input_tensor = preprocess_image(image, target_size=(image_size, image_size))
            
            # Predict
            mask = predict_mask(model, input_tensor, device)
            
            # Get present classes
            present_classes = np.unique(mask)
            
            # Create visualizations
            if vis_mode == "Color Overlay":
                overlay_image = create_overlay(processed_image, mask, alpha)
                if show_annotations:
                    annotated_image = create_annotated_image(overlay_image, mask, present_classes)
                else:
                    annotated_image = overlay_image
            else:  # Bounding Box mode
                annotated_image = processed_image.copy()
                import cv2
                # Draw bounding boxes for each defect class
                for class_id in present_classes:
                    if class_id == 0:
                        continue  # skip background
                    mask_bin = (mask == class_id).astype(np.uint8)
                    if np.count_nonzero(mask_bin) == 0:
                        continue
                    pixel_coverage = np.sum(mask == class_id)
                    if pixel_coverage <= 1000:
                        continue
                    color = CLASS_COLORS.get(class_id, (255,255,255))
                    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # Ignore tiny bounding boxes
                        if w < 10 or h < 10:
                            continue
                        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 2)
                # Show class labels on the left margin if enabled
                if show_annotations:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    y0, dy = 30, 30
                    label_count = 0
                    for class_id in present_classes:
                        if class_id == 0:
                            continue
                        pixel_coverage = np.sum(mask == class_id)
                        if pixel_coverage <= 1000:
                            continue
                        label = DEFECT_ALIASES.get(class_id, CLASS_LABELS[class_id] if class_id < len(CLASS_LABELS) else f"Class {class_id}")
                        y = y0 + label_count * dy
                        color = CLASS_COLORS.get(class_id, (255,255,255))
                        bgr_color = (int(color[2]), int(color[1]), int(color[0]))
                        cv2.putText(annotated_image, label, (20, y), font, font_scale, bgr_color, thickness, cv2.LINE_AA)
                        label_count += 1
        
        with col2:
            st.subheader("Defect Detection Results")
            st.image(annotated_image, caption="Detected Defects")
        
        # Analysis section
        st.subheader("📊 Defect Analysis")
        
        # Get defect statistics
        defect_stats = analyze_defects(mask, present_classes)
        
        if len(defect_stats) == 0:
            st.success("✅ No defects detected in this image!")
        else:
            st.warning(f"⚠️ {len(defect_stats)} type(s) of defects detected!")
            
            # Create columns for statistics
            cols = st.columns(min(3, len(defect_stats)))
            
            for idx, (defect_name, stats) in enumerate(defect_stats.items()):
                col_idx = idx % 3
                with cols[col_idx]:
                    # Create a colored box for the defect
                    color = stats['color']
                    st.markdown(
                        f"""
                        <div style="
                            background-color: rgb({color[0]}, {color[1]}, {color[2]});
                            padding: 10px;
                            border-radius: 5px;
                            margin: 5px 0;
                            color: {'white' if sum(color) < 400 else 'black'};
                        ">
                            <strong>{defect_name}</strong><br>
                            Coverage: {stats['percentage']:.2f}%<br>
                            Pixels: {stats['pixels']:,}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Detailed table
            st.subheader("Detailed Statistics")
            import pandas as pd
            
            df_data = []
            for defect_name, stats in defect_stats.items():
                df_data.append({
                    'Defect Type': defect_name,
                    'Coverage (%)': f"{stats['percentage']:.2f}%",
                    'Pixels': f"{stats['pixels']:,}",
                    'Severity': 'High' if stats['percentage'] > 5 else 'Medium' if stats['percentage'] > 1 else 'Low'
                })
            
            if df_data:
                df = pd.DataFrame(df_data)
                st.dataframe(df)
        
        # Download options
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Convert result to bytes for download
            result_pil = Image.fromarray(annotated_image)
            buf = io.BytesIO()
            result_pil.save(buf, format='PNG')
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Annotated Image",
                data=byte_im,
                file_name="defect_detection_result.png",
                mime="image/png"
            )
        
        with col2:
            # Create and download report
            report = f"""
Bridge Defect Detection Report
=============================

Image: {uploaded_file.name}
Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Defects Detected: {len(defect_stats)}

Detailed Analysis:
"""
            for defect_name, stats in defect_stats.items():
                report += f"\n- {defect_name}: {stats['percentage']:.2f}% coverage ({stats['pixels']:,} pixels)"
            
            if len(defect_stats) == 0:
                report += "\nNo defects detected in this image."
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name="defect_report.txt",
                mime="text/plain"
            )
    
    # Instructions
    else:
        st.info("👆 Please upload an image to get started!")
        
        st.markdown("""
        ### Instructions:
        1. **Upload an image** of a bridge structure using the file uploader above
        2. **Adjust settings** in the sidebar if needed (overlay transparency, annotations)
        3. **View results** showing detected defects with color-coded overlays
        4. **Analyze statistics** to understand the severity and coverage of defects
        5. **Download results** including annotated images and analysis reports
        
        ### Supported Defect Types:
        - **Wetspot** - Water-related damage
        - **Rust** - Metal corrosion
        - **Cracks** - Structural cracks (various types)
        - **Spalling** - Concrete surface deterioration
        - **Exposed Rebars** - Visible reinforcement bars
        - **Cavity** - Hollow areas in concrete
        - **Weathering** - Environmental damage
        - And many more...
        """)

if __name__ == "__main__":
    main()
