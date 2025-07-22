# Developer Documentation: Bridge Defect Detection Project

## Overview
This project provides a complete pipeline for bridge defect detection using semantic segmentation. It covers:
- Converting DACL10K Ninja dataset JSON annotations to segmentation masks
- Training a custom segmentation model
- Running a Streamlit web app for inference and visualization

The codebase is modular, with configuration and utility logic centralized for easy maintenance and extension.

---

## Project Structure
```
project_bridge/
│
├── dataset/                # Place the DACL10K dataset here (download from https://datasetninja.com/dacl10k)
├── new_scripts/            # All main scripts and modules
│   ├── bridge_app.py       # Streamlit web app for inference
│   ├── dacl10k_ninja_to_mask.py # Converts JSON annotations to masks
│   ├── train_ninja.py      # Model training script
│   ├── config.py           # Centralized configuration (classes, colors, mappings)
│   ├── utils.py            # Utility functions (preprocessing, visualization, etc.)
│   └── ...                 # Other helper scripts
├── requirements.txt        # Core dependencies for training and preprocessing
├── streamlit_requirements.txt # Additional dependencies for Streamlit app
└── docs/                   # Developer documentation (this file)
```

---

## 1. Dataset Preparation
- Download the DACL10K dataset from [datasetninja.com/dacl10k](https://datasetninja.com/dacl10k).
- The dataset folder **must be named** `dacl10k_ninja` and placed inside a `dataset/` folder at the project root.
- The required structure is:
  ```
  dataset/
    dacl10k_ninja/
      train/
        images/
        masks/
      val/
        images/
        masks/
  ```

> **Note:**
> - The `train` and `val` folders must each contain two subfolders: `images` (for input images) and `masks` (for segmentation masks).
> - The mask generation script will create the `masks` folders if they do not exist.
> - Ensure the folder and file names match this structure exactly for the training and inference scripts to work correctly.

---

## 2. Mask Generation
Run the following command to convert JSON annotations to segmentation masks:
```bash
python new_scripts/dacl10k_ninja_to_mask.py
```
- This will create `masks_train`, `masks_val`, and `masks_test` folders inside each split directory.
- The script maps each defect class in the JSON to a unique integer in the mask (see `DACL10K_LABEL_MAP` in the script).

---

## 3. Model Training
Train a custom segmentation model using the generated masks:
```bash
python new_scripts/train_ninja.py
```
- The script uses configuration at the top for architecture, encoder, and hyperparameters.
- Model checkpoints are saved as `.pth` files in the project root (e.g., `dacl10k_unetplusplus_efficientnet_b7_ver1.pth`).
- You can adjust architecture and encoder in the script header.

---

## 4. Running the Streamlit App
1. Move the best `.pth` model file into the `new_scripts/` folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r streamlit_requirements.txt
   ```
3. Run the app locally:
   ```bash
   streamlit run new_scripts/bridge_app.py
   ```
   Or, to run the app on your local WiFi network at port 8501 (accessible from other devices on the same network):
   ```bash
   streamlit run new_scripts/bridge_app.py --server.address=0.0.0.0 --server.port=8501
   ```
   Then open `http://<your-local-ip>:8501` in a browser on any device connected to the same WiFi.
4. Upload a bridge image and select your model in the sidebar.

---

## 5. Codebase Walkthrough

### `config.py`
- Centralizes all class mappings, color codes, and label aliases.
- Ensures consistency across mask generation, training, and inference.
- Example: `CLASS_COLORS`, `CLASS_LABELS`, `DEFECT_ALIASES`, and helper functions for display and color conversion.

### `utils.py`
- Contains reusable functions for image preprocessing, mask prediction, and visualization.
- Used by both training and inference scripts to avoid code duplication.
- Example: `preprocess_image`, `predict_segmentation_mask`, color conversion utilities.

### `dacl10k_ninja_to_mask.py`
- Converts DACL10K Ninja JSON annotations to PNG masks.
- Maps each object in the JSON to a class ID using `DACL10K_LABEL_MAP`.
- Handles missing or unknown classes gracefully.
- Can be extended to generate colorized masks for visualization.

### `train_ninja.py`
- Loads images and masks, applies augmentations, and trains a segmentation model.
- Supports multiple architectures (UNet++, FPN, DeepLabV3+, etc.) and encoders.
- Uses Dice + CrossEntropy loss for robust training.
- Saves the best model checkpoint based on validation loss.
- Includes recommendations for architecture selection based on GPU memory.

### `bridge_app.py`
- Streamlit web app for interactive defect detection.
- Loads a trained model, processes uploaded images, and visualizes results.
- Sidebar allows model selection, visualization mode, and defect filtering.
- Provides downloadable annotated images and analysis reports.

---

## 6. Best Practices & Tips
- Always use the same class mappings and color codes across all scripts (via `config.py`).
- Check the requirements files for all dependencies. If you add new packages, update these files.
- For new defect types or dataset changes, update `config.py` and re-generate masks.
- For large models, ensure you have enough GPU memory or reduce batch size/encoder size.

---

## 7. Extending the Project
- Add new architectures or encoders in `train_ninja.py` and `bridge_app.py`.
- Add new visualization or analysis features in `utils.py` and `bridge_app.py`.
- Update `config.py` for new classes, colors, or aliases.

---

## 8. Troubleshooting
- **Missing masks:** Ensure you run the mask generation script after placing the dataset.
- **Model not found in Streamlit app:** Move the `.pth` file into `new_scripts/`.
- **Dependency errors:** Double-check both requirements files and install all packages.
- **CUDA errors:** Lower batch size or use a smaller encoder/model.

---

## 9. References
- [DACL10K Dataset](https://datasetninja.com/dacl10k)
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

For further questions or to contribute, please refer to the code comments and follow the modular structure for new features.
