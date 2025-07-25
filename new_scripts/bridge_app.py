from datetime import datetime
import warnings
import os
import io

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*torch.classes.*")

import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# Import centralized configuration
from config import (
    CLASS_COLORS, 
    ALLOWED_CLASS_IDS, NUM_CLASSES,
    detect_architecture_from_filename, get_architecture_display_info,
    get_alias_color, get_display_name, rgb_to_bgr
)

# Configure page
st.set_page_config(
    page_title="Bridge Defect Detection",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model(weights_path, architecture=None, encoder_name=None):
    """Load model with auto-detection or specified architecture"""
    try:
        # Auto-detect architecture from filename if not specified
        if architecture is None or encoder_name is None:
            architecture, encoder_name = detect_architecture_from_filename(weights_path)
        
        st.info(f"🏗️ Loading {architecture.upper()} model with {encoder_name} encoder...")
        
        # Create model based on architecture
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
        
        # Load weights
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        st.success(f"✅ Model loaded: {architecture.upper()} + {encoder_name}")
        return model, device, architecture, encoder_name
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

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

def create_overlay(image, mask, alpha=0.6, defect_visibility=None):
    """Create overlay with alias colors for consistent visualization"""
    # Convert image to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = np.zeros_like(image_bgr)
    
    # Apply colors for each class
    for class_id, color in CLASS_COLORS.items():
        if class_id == 0:  # Always show background
            continue
            
        # Check if this defect should be visible
        if defect_visibility is not None:
            display_name = get_display_name(class_id)
            if not defect_visibility.get(display_name, True):
                continue  # Skip this defect if not visible
        
        # Get appropriate color (alias or regular)
        overlay_color = get_alias_color(class_id)
        bgr_color = rgb_to_bgr(overlay_color)
        overlay[mask == class_id] = bgr_color
    
    # Blend images
    result = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)
    
    # Convert back to RGB for display
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

def create_annotated_image(image, mask, present_classes, pixel_threshold=1000, defect_visibility=None):
    """Create annotated image with alias-based labeling"""
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    y0, dy = 30, 30
    label_count = 0
    
    for class_id in present_classes:
        if class_id == 0 or class_id not in ALLOWED_CLASS_IDS:
            continue  # skip background and unwanted classes
        pixel_coverage = np.sum(mask == class_id)
        if pixel_coverage <= pixel_threshold:
            continue  # skip small regions
            
        # Get display name (with alias if available)
        label = get_display_name(class_id)
        
        # Check if this defect should be visible
        if defect_visibility is not None:
            if not defect_visibility.get(label, True):
                continue  # Skip this defect if not visible
        
        y = y0 + label_count * dy
        # Use uniform white color for all text labels
        uniform_text_color = (255, 255, 255)  # White text
        bgr_color = rgb_to_bgr(uniform_text_color)
        cv2.putText(annotated, label, (20, y), font, font_scale, bgr_color, thickness, cv2.LINE_AA)
        label_count += 1
    return annotated

def analyze_defects(mask, present_classes, pixel_threshold=1000, defect_visibility=None):
    """Analyze defects with alias-based grouping and filtering"""
    stats = {}
    total_pixels = mask.shape[0] * mask.shape[1]
    
    for class_id in present_classes:
        if class_id == 0 or class_id not in ALLOWED_CLASS_IDS:
            continue  # skip background and unwanted classes
        class_pixels = np.sum(mask == class_id)
        if class_pixels <= pixel_threshold:
            continue  # skip small regions
            
        percentage = (class_pixels / total_pixels) * 100
        
        # Get display name and color (with alias if available)
        label = get_display_name(class_id)
        color = get_alias_color(class_id)
        
        # Check if this defect should be visible
        if defect_visibility is not None:
            if not defect_visibility.get(label, True):
                continue  # Skip this defect if not visible
        
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

def main():
    st.title("🌉 Bridge Defect Detection System")
    st.markdown("Upload an image of a bridge to detect and visualize structural defects using AI.")
    
    # Sidebar for model selection and parameters
    st.sidebar.header("🔧 Model Settings")
    
    # Model path selection with new architectures
    available_models = []
    model_dir = "."
    for file in os.listdir(model_dir):
        if file.endswith('.pth'):
            available_models.append(file)
    
    if not available_models:
        available_models = [
            "No models found. Please train a model first.",
        ]
    
    model_path = st.sidebar.selectbox(
        "📁 Model File", 
        available_models,
        help="Select your trained model file"
    )
    
    # Display model info
    if model_path:
        arch_info = get_architecture_display_info(detect_architecture_from_filename(model_path)[0])
        st.sidebar.info(arch_info)
    
    st.sidebar.header("🎛️ Visualization Settings")
    
    # Visualization mode
    vis_mode = st.sidebar.radio(
        "🎨 Visualization Mode",
        ["Color Overlay", "Bounding Box"],
        index=0,
        help="Choose how to visualize detected defects"
    )

    # Overlay transparency (only for overlay mode)
    alpha = 0.6
    if vis_mode == "Color Overlay":
        alpha = st.sidebar.slider(
            "🌈 Overlay Transparency", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3, 
            step=0.1,
            help="Transparency of the segmentation overlay"
        )

    # Show/hide annotations
    show_annotations = st.sidebar.checkbox("🏷️ Show Class Labels", value=True)
    
    # Pixel threshold
    pixel_threshold = st.sidebar.slider(
        "🔍 Minimum Defect Size (pixels)",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Ignore defects smaller than this pixel count"
    )
    
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
        
        # Load model with auto-detection
        with st.spinner("🔄 Loading model..."):
            model, device, architecture, encoder_name = load_model(model_path)
        
        if model is None:
            return
        
        # Process image
        with st.spinner("🔮 Processing image..."):
            # Preprocess (always use 512x512 for consistency)
            processed_image, input_tensor = preprocess_image(image, target_size=(512, 512))
            
            # Predict
            mask = predict_mask(model, input_tensor, device)
            
            # Get present classes
            present_classes = np.unique(mask)
        
        # Get detected defects for dynamic visibility controls
        detected_defects = []
        for class_id in present_classes:
            if class_id == 0 or class_id not in ALLOWED_CLASS_IDS:
                continue
            # Check if defect meets pixel threshold
            pixel_coverage = np.sum(mask == class_id)
            if pixel_coverage <= pixel_threshold:
                continue
            
            # Get display name (with alias if available)
            display_name = get_display_name(class_id)
            
            if display_name not in detected_defects:
                detected_defects.append(display_name)

        # Create dynamic defect visibility controls in sidebar
        defect_visibility = {}
        if detected_defects:
            st.sidebar.subheader("🔍 Detected Defects Visibility")
            st.sidebar.write(f"Found {len(detected_defects)} defect type(s)")
            
            # Create checkboxes for each detected defect
            for defect_name in sorted(detected_defects):
                defect_visibility[defect_name] = st.sidebar.checkbox(
                    f"{defect_name}", 
                    value=True,
                    help=f"Toggle visibility of {defect_name} defects"
                )
            
            # Quick select buttons for detected defects
            sidebar_col1, sidebar_col2 = st.sidebar.columns(2)
            with sidebar_col1:
                if st.sidebar.button("🔄 Show All", help="Show all detected defects"):
                    st.experimental_rerun()
            with sidebar_col2:
                if st.sidebar.button("❌ Hide All", help="Hide all detected defects"):
                    for key in defect_visibility:
                        defect_visibility[key] = False
        else:
            st.sidebar.info("No defects detected above threshold")
            # Set empty visibility dict when no defects detected
            defect_visibility = {}

        # Create visualizations
        if vis_mode == "Color Overlay":
            overlay_image = create_overlay(processed_image, mask, alpha, defect_visibility)
            if show_annotations:
                annotated_image = create_annotated_image(overlay_image, mask, present_classes, pixel_threshold, defect_visibility)
            else:
                annotated_image = overlay_image
        else:  # Bounding Box mode
            annotated_image = processed_image.copy()
            
            # Draw bounding boxes for each defect class
            for class_id in present_classes:
                if class_id == 0 or class_id not in ALLOWED_CLASS_IDS:
                    continue  # skip background and unwanted classes
                pixel_coverage = np.sum(mask == class_id)
                if pixel_coverage <= pixel_threshold:
                    continue  # skip small regions
                    
                # Get display name (with alias if available)
                label = get_display_name(class_id)
                
                # Check if this defect should be visible
                if defect_visibility and not defect_visibility.get(label, True):
                    continue  # Skip this defect if not visible
                
                # Use uniform color for bounding boxes
                uniform_bbox_color = (0, 255, 0)  # Green bounding boxes
                
                mask_bin = (mask == class_id).astype(np.uint8)
                if np.count_nonzero(mask_bin) == 0:
                    continue
                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Ignore tiny bounding boxes
                    if w < 10 or h < 10:
                        continue
                    cv2.rectangle(annotated_image, (x, y), (x+w, y+h), uniform_bbox_color, 2)
                    # Add label above the box with uniform color
                    cv2.putText(annotated_image, label, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, uniform_bbox_color, 2, cv2.LINE_AA)
            
            # Show class labels on the left margin if enabled
            if show_annotations:
                annotated_image = create_annotated_image(annotated_image, mask, present_classes, pixel_threshold, defect_visibility)
        
        with col2:
            st.subheader("🔍 Defect Detection Results")
            st.image(annotated_image, caption="Detected Defects")
            
            # Display model info
            st.info(f"🏗️ **Model**: {architecture.upper()} + {encoder_name}")
            st.info(f"🔍 **Pixel Threshold**: {pixel_threshold:,} pixels")
        
        # Analysis section
        st.subheader("📊 Defect Analysis")
        
        # Get defect statistics
        defect_stats = analyze_defects(mask, present_classes, pixel_threshold, defect_visibility)
        
        # Show visibility info
        if defect_visibility:
            visible_defects = [name for name, visible in defect_visibility.items() if visible]
            if len(visible_defects) < len(defect_visibility):
                st.info(f"👁️ Currently showing: {', '.join(visible_defects) if visible_defects else 'None'}")
        
        if len(defect_stats) == 0:
            if defect_visibility and len([name for name, visible in defect_visibility.items() if visible]) == 0:
                st.info("ℹ️ All detected defects are hidden. Enable some defects in the sidebar to see analysis.")
            else:
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
        
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
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
        
        with download_col2:
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
        ### 🚀 Instructions:
        1. **📁 Upload an image** of a bridge structure using the file uploader above
        2. **🔧 Adjust settings** in the sidebar if needed (model, visualization mode, thresholds)
        3. **👁️ View results** showing detected defects with color-coded overlays or bounding boxes
        4. **📊 Analyze statistics** to understand the severity and coverage of defects
        5. **💾 Download results** including annotated images and analysis reports
        """)
        
        # ### 🏗️ Supported Defect Types:
        # - **🔴 Rust** - Metal corrosion and oxidation
        # - **⚡ ACrack/Crack** - Structural cracks of various types
        # - **🟣 WConccor** - Concrete corrosion 
        # - **🟠 Honeycombing** - Hollow areas and cavities in concrete
        # - **🟢 Spalling** - Concrete surface deterioration
        # - **🟡 Rockpocket** - Air voids in concrete
        # - **🟢 ExposedRebars** - Visible reinforcement bars
        # - **🔵 Leaching** - Environmental damage and efflorescence
                    
        # ### 🎯 Model Features:
        # - **Multi-Architecture Support**: UNet++, FPN, LinkNet, PSPNet, DeepLabV3Plus
        # - **Advanced Encoders**: EfficientNet, ResNeXt, SE-ResNeXt
        # - **Intelligent Filtering**: Configurable pixel thresholds
        # - **Alias Grouping**: Related defects grouped with consistent colors
        # - **High Accuracy**: Optimized for bridge infrastructure inspection
        # 
        
        # # Model performance info
        # st.markdown("""
        # ### 📈 Performance Highlights:
        # - **🏆 UNet++**: Best for fine crack detection
        # - **🔍 FPN**: Excellent for multi-scale defects  
        # - **⚡ LinkNet**: Fastest processing
        # - **🧠 PSPNet**: Superior contextual understanding
        # - **🎯 EfficientNet**: Optimal accuracy/efficiency balance
        # """)

    # # Color Legend
    # with st.expander("🎨 Defect Color Legend", expanded=False):
    #     legend_cols = st.columns(3)
        
    #     defect_info = [
    #         ("Rust", CLASS_COLORS[1], "Metal corrosion"),
    #         ("ACrack", CLASS_COLORS[2], "Structural cracks"), 
    #         ("WConccor", CLASS_COLORS[3], "Concrete corrosion"),
    #         ("Honeycombing", ALIAS_COLORS["Honeycombing"], "Cavities & hollow areas"),
    #         ("Spalling", CLASS_COLORS[6], "Surface deterioration"),
    #         ("Rockpocket", CLASS_COLORS[7], "Air voids"),
    #         ("ExposedRebars", CLASS_COLORS[8], "Visible reinforcement"),
    #         ("Crack", CLASS_COLORS[9], "General cracks"),
    #         ("Leaching", ALIAS_COLORS["Leaching"], "Environmental damage")
    #     ]
        
    #     for idx, (name, color, desc) in enumerate(defect_info):
    #         col_idx = idx % 3
    #         with legend_cols[col_idx]:
    #             st.markdown(
    #                 f"""
    #                 <div style="
    #                     background-color: rgb({color[0]}, {color[1]}, {color[2]});
    #                     padding: 8px;
    #                     border-radius: 4px;
    #                     margin: 2px 0;
    #                     color: {'white' if sum(color) < 400 else 'black'};
    #                     text-align: center;
    #                     font-size: 12px;
    #                 ">
    #                     <strong>{name}</strong><br>
    #                     <small>{desc}</small>
    #                 </div>
    #                 """,
    #                 unsafe_allow_html=True
    #             )

if __name__ == "__main__":
    main()
