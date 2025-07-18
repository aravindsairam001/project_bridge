# Bridge Defect Detection Streamlit App

A web-based interface for detecting and analyzing bridge defects using deep learning.

## Features

- **Upload Images**: Drag and drop or browse to upload bridge images
- **Real-time Detection**: AI-powered defect detection using trained segmentation models
- **Visual Analysis**: Color-coded overlay showing different types of defects
- **Statistical Analysis**: Detailed statistics about defect coverage and severity
- **Export Results**: Download annotated images and analysis reports
- **Interactive Controls**: Adjust overlay transparency and display options

## Setup and Installation

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r streamlit_requirements.txt
```

### 2. Train or Obtain Model

Make sure you have a trained model file. You can either:

- Train a new model: `python train_dacl10k.py`
- Use an existing model file (e.g., `dacl10k_ninja.pth`)

### 3. Run the Application

Choose one of these methods:

#### Method 1: Using Python launcher (Recommended)
```bash
python launch_app.py
```

#### Method 2: Using bash script
```bash
./run_app.sh
```

#### Method 3: Direct Streamlit command
```bash
streamlit run streamlit_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Usage Guide

### 1. Upload Image
- Click "Choose an image..." or drag and drop an image file
- Supported formats: JPG, JPEG, PNG
- The original image will be displayed in the left column

### 2. Configure Settings (Sidebar)
- **Model Path**: Specify the path to your trained model file
- **Overlay Transparency**: Adjust how transparent the defect overlay appears
- **Show Class Labels**: Toggle whether to display defect type labels on the image

### 3. View Results
- The processed image with defect overlays appears in the right column
- Different defect types are color-coded for easy identification

### 4. Analyze Statistics
- View the total number of defect types detected
- See coverage percentages and pixel counts for each defect type
- Review severity levels (High/Medium/Low based on coverage)

### 5. Download Results
- **Annotated Image**: Download the processed image with defect overlays
- **Analysis Report**: Download a text report with detailed statistics

## Detected Defect Types

The system can detect 19 different types of bridge defects:

| Class ID | Defect Type | Description |
|----------|-------------|-------------|
| 1 | Wetspot | Water-related damage |
| 2 | Rust | Metal corrosion |
| 3 | EJoint | Expansion joint issues |
| 4 | ACrack | Alligator cracking |
| 5 | WConccor | Washouts/concrete corrosion |
| 6 | Cavity | Hollow areas in concrete |
| 7 | Hollowareas | Hollow areas |
| 8 | JTape | Joint tape damage |
| 9 | Spalling | Concrete surface deterioration |
| 10 | Rockpocket | Rock pocket formation |
| 11 | ExposedRebars | Visible reinforcement bars |
| 12 | Crack | General structural cracks |
| 13 | Restformwork | Residual formwork |
| 14 | Drainage | Drainage issues |
| 15 | Weathering | Environmental damage |
| 16 | Bearing | Bearing-related defects |
| 17 | Graffiti | Graffiti/vandalism |
| 18 | PEquipment | Protective equipment issues |
| 19 | Efflorescence | Mineral deposits |

## Color Coding

Each defect type is assigned a unique color for easy visualization:
- **Red**: Wetspot
- **Green**: Rust  
- **Blue**: Expansion Joint
- **Yellow**: Alligator Crack
- **Magenta**: Washouts/Concrete Corrosion
- **Cyan**: Cavity
- *And more...*

## Troubleshooting

### Common Issues

1. **Model file not found**
   - Ensure the model file path is correct in the sidebar
   - Train a model using `python train_dacl10k.py` if needed

2. **Dependencies missing**
   - Run `pip install -r streamlit_requirements.txt`
   - Use the Python launcher which can auto-install dependencies

3. **CUDA/GPU issues**
   - The app automatically detects if CUDA is available
   - Falls back to CPU processing if GPU is not available

4. **Large image processing is slow**
   - Images are automatically resized to 720x720 for processing
   - Consider using smaller input images for faster processing

### Performance Tips

- Use images with resolution around 720x720 for optimal performance
- Enable GPU acceleration if available (CUDA)
- Close other applications to free up memory for processing

## Technical Details

- **Framework**: Streamlit for web interface
- **Model**: DeepLabV3Plus with ResNet50 backbone
- **Input Size**: 720x720 pixels (automatically resized)
- **Output**: 20-class segmentation (background + 19 defect types)
- **Post-processing**: Color overlay generation and statistical analysis

## File Structure

```
project_bridge/
├── streamlit_app.py          # Main Streamlit application
├── launch_app.py             # Python launcher script
├── run_app.sh               # Bash launcher script
├── streamlit_requirements.txt # Dependencies for Streamlit app
├── train_dacl10k.py         # Model training script
├── dacl10k_ninja.pth        # Trained model file
└── README_streamlit.md      # This documentation
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure the model file exists and is accessible
4. Check the terminal output for error messages
