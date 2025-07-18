## 🚀 Repository Update Notice

**This repository has been updated and moved to a new location:**
**[https://github.com/aravindsairam001/updated_project_bridge](https://github.com/aravindsairam001/updated_project_bridge)**

For the latest code, improvements, and updates, please visit the new repository.

---

# DACL10K Defect Segmentation - DeepLabV3+ ResNet101

A deep learning project for concrete defect detection and segmentation using DeepLabV3+ architecture (with ResNet101 encoder) trained on the DACL10K dataset.

## Overview

This project implements semantic segmentation for concrete infrastructure defect detection. The model can identify and segment 19 different types of concrete defects including cracks, spalling, graffiti, rust, and other structural issues.

## Dataset

**DACL10K** - A large-scale dataset for concrete defect classification and segmentation containing:
- 19 concrete defect classes + background
- High-resolution images of concrete structures
- Pixel-level annotations for precise segmentation

### Dataset Download

**Official Website:** [https://dacl.ai/workshop.html](https://dacl.ai/workshop.html)

To use this project, you need to download the DACL10K dataset:

1. **Visit the official website**: Navigate to [dacl.ai/workshop.html](https://dacl.ai/workshop.html)
2. **Download the dataset**: Follow the instructions on the website to download the complete dataset
3. **Extract the dataset**: Place the downloaded files in the `dacl10k_dataset/` directory
4. **Verify structure**: Ensure your dataset follows this structure:
   ```
   dacl10k_dataset/
   ├── annotations/            # JSON annotation files
   │   ├── train/
   │   └── validation/
   ├── images/                 # Original images
   │   ├── train/
   │   ├── validation/
   │   └── testdev/
   └── (additional dataset files)
   ```

### Defect Classes

1. Wetspot
2. Rust
3. EJoint (Expansion Joint)
4. ACrack (Active Crack)
5. WConccor (Water Concrete Corrosion)
6. Cavity
7. Hollowareas
8. JTape (Joint Tape)
9. Spalling
10. Rockpocket
11. ExposedRebars
12. Crack
13. Restformwork (Residual Formwork)
14. Drainage
15. Weathering
16. Bearing
17. Graffiti
18. PEquipment (Protective Equipment)
19. Efflorescence

## Project Structure

```
project_bridge/
├── README.md                    # This file
├── json_to_mask.py             # Convert LabelMe JSON annotations to masks
├── train_dacl10k.py            # Training script for DeepLabV3+ model
├── inference_image.py          # Single image inference with visualization
├── inference_video.py          # Video processing for defect detection
├── test_mask.py                # Utility to test and visualize generated masks
├── check_dataset.py            # Dataset validation utilities
├── extract_labels.py           # Label extraction utilities
├── segformer.py                # Alternative SegFormer implementation
└── dacl10k_dataset/            # Dataset directory
    ├── annotations/            # JSON annotation files
    │   ├── train/
    │   └── validation/
    ├── images/                 # Original images
    │   ├── train/
    │   ├── validation/
    │   └── testdev/
    ├── masks_val/              # Validation masks
    └── masks_train/            # Training masks
```

## Installation

### Requirements

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install opencv-python
pip install albumentations
pip install matplotlib
pip install pillow
pip install numpy
pip install tqdm
```

### Alternative Installation

```bash
pip install -r requirements.txt
```

## Usage

**Prerequisites:** Before starting, ensure you have downloaded the DACL10K dataset from [dacl.ai/workshop.html](https://dacl.ai/workshop.html) and placed it in the `dacl10k_dataset/` directory as described in the Dataset section above.

### 1. Dataset Preparation

#### Convert JSON Annotations to Masks

If you have LabelMe JSON annotations, convert them to training masks:

```bash
python json_to_mask.py
```

This script:
- Reads JSON files from `dacl10k_dataset/annotations/train/`
- Generates PNG masks in `dacl10k_dataset/masks_train/`
- Maps class names to integer IDs (1-19, 0 for background)
- Creates colored visualization masks for inspection

### 2. Training

Train the DeepLabV3+ model:

```bash
python train_dacl10k.py
```


**Training Configuration:**
- **Architecture:** DeepLabV3+ with ResNet101 encoder
- **Input Size:** 512×512 pixels
- **Batch Size:** 4
- **Epochs:** 20
- **Optimizer:** Adam (lr=1e-4)
- **Loss Function:** CrossEntropyLoss
- **Data Augmentation:** Horizontal flip, brightness/contrast adjustment

**Model will be saved as:** `dacl10k_ninja_new.pth`

### 3. Inference

#### Single Image Inference

```bash
python inference_image.py
```

Features:
- Loads pre-trained model
- Processes single images
- Overlays segmentation results with distinct colors
- Annotates detected defect classes
- Saves annotated results

**Modify the script to change:**
```python
model_path = "dacl10k15_deeplabv3plus.pth"  # Your model path
test_image_path = "path/to/your/image.jpg"   # Input image
output_path = "prediction_overlay.jpg"        # Output path
```

#### Video Processing

```bash
python inference_video.py
```

Features:
- Frame-by-frame defect detection
- Real-time annotation overlay
- Defect class labeling
- Output as annotated video

**Modify the script to change:**
```python
video_path = "input_video.mp4"              # Input video
output_path = "output_annotated_video.mp4"   # Output video
model_path = "dacl10k15_deeplabv3plus.pth"   # Model path
```

-## Model Architecture

**DeepLabV3+** with the following specifications:
- **Encoder:** ResNet101 (ImageNet pre-trained)
- **Decoder:** DeepLabV3+ decoder with ASPP
- **Input Channels:** 3 (RGB)
- **Output Classes:** 20 (19 defects + background)
- **Input Resolution:** 512×512

## Streamlit Web UI

An interactive Streamlit app is included for easy defect detection and visualization.

### Launch the App

```bash
streamlit run streamlit_app.py
```

### Features
- Upload bridge images for instant defect segmentation
- Choose between color overlay and bounding box visualization
- Toggle class label display
- Download annotated images and detailed analysis reports
- View per-class statistics and coverage

### Usage
1. Run the command above
2. Open the provided local URL in your browser
3. Upload an image and view results interactively

**Note:** The app uses the latest trained model (by default: `dacl10k_ninja_new.pth`). You can change the model path in `streamlit_app.py` if needed.
- **Decoder:** DeepLabV3+ decoder with ASPP
- **Input Channels:** 3 (RGB)
- **Output Classes:** 20 (19 defects + background)
- **Input Resolution:** 512×512

## Output Features

The model generates comprehensive defect segmentation results with advanced visualization capabilities:

### Sample Output
![Defect Segmentation Results](prediction_overlay.jpg)

*Example output showing detected concrete defects with color-coded segmentation overlays and class annotations*

### Output Characteristics

#### 1. **Segmentation Overlay**
- **Precise Pixel-level Detection**: Each defect pixel is accurately classified
- **Multi-class Segmentation**: Simultaneous detection of multiple defect types
- **Color-coded Visualization**: Each defect class has a distinct, bright color for easy identification
- **Transparency Control**: Semi-transparent overlays preserve original image details

#### 2. **Class Annotation System**
- **Automatic Labeling**: Detected defect classes are automatically listed
- **Color Coordination**: Class labels match their corresponding segmentation colors
- **Clean Layout**: Professional annotation placement at image margins
- **Comprehensive Coverage**: All 19 defect types clearly distinguished

#### 3. **Visual Quality Features**
- **Image Overlay**: Maintains original image with overlay
- **Distinct Colors**: Carefully selected color palette for maximum contrast
- **Professional Presentation**: Publication-ready visualization format
- **Multiple Output Formats**: Supports various image formats (JPG, PNG)

#### 4. **Detection Capabilities**
- **Crack Detection**: Identifies both active and passive cracks
- **Surface Defects**: Detects spalling, weathering, and surface deterioration
- **Material Issues**: Recognizes rust, corrosion, and material degradation
- **Structural Elements**: Identifies joints, equipment, and construction features
- **Environmental Damage**: Detects graffiti, staining, and environmental effects

#### 5. **Technical Output Specifications**
- **Confidence-based Segmentation**: Only high-confidence predictions are visualized
- **Multi-defect Support**: Single image can contain multiple defect types
- **Real-time Processing**: Optimized for both batch and real-time inference

### Output Files Generated
- **Segmented Images**: Color-coded defect overlay images
- **Prediction Maps**: Raw probability maps for each class
- **Annotated Results**: Images with class labels and color legends
- **Video Outputs**: Frame-by-frame annotated video sequences (for video input)

## Data Processing Pipeline

### 1. Annotation Format
- **Input:** LabelMe JSON format with polygon annotations
- **Structure:**
  ```json
  {
    "imageHeight": 1440,
    "imageWidth": 1920,
    "shapes": [
      {
        "label": "Graffiti",
        "points": [[x1,y1], [x2,y2], ...],
        "shape_type": "polygon"
      }
    ]
  }
  ```

### 2. Mask Generation
- Converts polygon annotations to pixel-wise masks
- Each defect class gets unique integer ID (1-19)
- Background pixels = 0
- Handles multiple defects per image
- Generates both grayscale masks and colored visualizations

### 3. Training Data Format
- **Images:** RGB JPEG files (resized to 512×512)
- **Masks:** Grayscale PNG files with class IDs
- **Augmentations:** Random flips, brightness/contrast changes
- **Normalization:** ImageNet statistics

## Visualization Features

### Color Mapping
Each defect class has a distinct bright color for easy identification:
- **Wetspot:** Bright Red
- **Rust:** Bright Green  
- **Graffiti:** Light Green
- **Spalling:** Spring Green
- **Crack:** Azure
- And 15 other distinct colors...

### Annotation Features
- **Class Labels:** Listed at image margin
- **Color-coded:** Each class uses its unique color
- **Clean Visualization:** Professional overlay presentation

## Pre-trained Model

### Download
The pre-trained model `dacl10k_ninja_new_.pth` should be placed in the project root directory.

### Model Specifications
- **Training Dataset:** DACL10K
- **Architecture:** DeepLabV3+ (ResNet34)
- **Input Size:** 512×512×3
- **Output:** 20-channel probability map
- **Performance:** Optimized for concrete defect detection

### Usage with Pre-trained Model
```python
import torch
import segmentation_models_pytorch as smp

# Load model
model = smp.DeepLabV3Plus(
    encoder_name="resnet101",
    encoder_weights='imagenet',
    in_channels=3,
    classes=20
)
model.load_state_dict(torch.load('dacl10k_ninja_new'))
model.eval()
```

## Performance Optimization

### GPU Support
- Automatic GPU detection and utilization
- CUDA-optimized inference
- Fallback to CPU if GPU unavailable

### Batch Processing
- Efficient video frame processing
- Progress tracking for long videos
- Memory-optimized operations

## Utilities

### Mask Validation
```bash
python test_mask.py
```
- Checks generated masks
- Displays unique class values
- Color visualization

### Dataset Checking
```bash
python check_dataset.py
```
- Validates dataset structure
- Checks for missing files
- Reports dataset statistics

## Troubleshooting

### Common Issues

1. **Empty Masks**
   - Check label names match exactly (case-sensitive)
   - Verify JSON format is correct
   - Ensure polygon coordinates are valid

2. **CUDA Errors**
   - Check GPU memory availability
   - Reduce batch size if needed
   - Use CPU fallback: `device='cpu'`

3. **Video Processing Issues**
   - Ensure input video format is supported
   - Check available disk space for output
   - Verify video codec compatibility

### Dependencies Issues
```bash
# If OpenCV issues:
pip install opencv-python-headless

# If segmentation-models-pytorch issues:
pip install timm
```

## Extending the Project

### Adding New Defect Classes
1. Update `LABEL_MAP` in `json_to_mask.py`
2. Add corresponding colors in `CLASS_COLORS`
3. Update `CLASS_LABELS` array
4. Retrain model with updated class count

### Custom Data
1. Prepare annotations in LabelMe JSON format
2. Update dataset paths in training script
3. Modify class mappings as needed
4. Run the full pipeline

## License

This project is for research and educational purposes. Please check the DACL10K dataset license for commercial usage restrictions.

## Contributing

1. Fork the repository
2. Create feature branch
3. Make improvements
4. Submit pull request

## Citation

If you use this work, please cite the DACL10K dataset:

```bibtex
@dataset{dacl10k,
  author={Johannes Flotzinger and Philipp J. Rosch and Thomas Braml},
  title={dacl10k: Benchmark for Semantic Bridge Damage Segmentation},
  year={2023},
  url={https://dacl.ai/workshop.html}
}
```

---
