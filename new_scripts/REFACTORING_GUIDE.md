# Code Refactoring Documentation

## 🔄 **Refactoring Summary**

The codebase has been refactored to improve maintainability, reduce code duplication, and create a cleaner architecture. This document outlines the changes and how to use the new structure.

---

## 📁 **New File Structure**

### **Core Configuration Files**
- **`config.py`** - Centralized configuration for classes, colors, and model settings
- **`utils.py`** - Common utility functions for image processing and model operations

### **Refactored Scripts**
- **`bridge_app.py`** - Streamlit web application (now uses centralized config)
- **`image_ninja.py`** - Single image inference script (refactored)
- **`json_mask_min_class.py`** - JSON to mask conversion (refactored)

---

## 🔧 **Key Improvements**

### **1. Centralized Configuration (`config.py`)**

**Before:** Each script had its own class definitions:
```python
# In bridge_app.py, image_ninja.py, json_mask_min_class.py, etc.
CLASS_COLORS = {
    0: (0, 0, 0),           # 0: background (black)
    1: (255, 0, 0),         # 1: Rust (bright red)
    # ... repeated in every file
}
```

**After:** Single source of truth:
```python
# In config.py only
from config import CLASS_COLORS, CLASS_LABELS, NUM_CLASSES
```

### **2. Utility Functions (`utils.py`)**

**Before:** Duplicate functions in multiple scripts:
```python
# Repeated in multiple files
def preprocess_image(image, target_size=(512, 512)):
    # Same code copied everywhere
```

**After:** Reusable utilities:
```python
# In utils.py only
from utils import preprocess_image, create_segmentation_overlay
```

### **3. Auto-Detection Features**

**Before:** Manual architecture detection in each script:
```python
# Repeated pattern in multiple files
filename = os.path.basename(weights_path).lower()
if 'unetplusplus' in filename:
    architecture = 'unetplusplus'
elif 'fpn' in filename:
    architecture = 'fpn'
# ... long repetitive code
```

**After:** Centralized function:
```python
from config import detect_architecture_from_filename
architecture, encoder_name = detect_architecture_from_filename(weights_path)
```

---

## 🎯 **How to Use the New Structure**

### **1. Basic Configuration**

```python
# Import what you need
from config import (
    CLASS_COLORS, CLASS_LABELS, NUM_CLASSES, ALLOWED_CLASS_IDS,
    detect_architecture_from_filename, print_dataset_info
)

# Print current configuration
print_dataset_info()

# Auto-detect model architecture
architecture, encoder_name = detect_architecture_from_filename("my_model.pth")
```

### **2. Using Utilities**

```python
from utils import (
    preprocess_image, predict_segmentation_mask, 
    create_segmentation_overlay, analyze_segmentation_results
)

# Load and preprocess image
processed_img, input_tensor = preprocess_image(pil_image)

# Create overlay
overlay = create_segmentation_overlay(image, mask, alpha=0.6)

# Analyze results
stats = analyze_segmentation_results(mask, present_classes)
```

### **3. Model Loading**

```python
from utils import load_trained_model

# Auto-detect and load model
model, device, arch, encoder = load_trained_model("my_model.pth")
```

---

## 🔧 **Configuration Options**

### **Class Set Selection**

Edit `config.py` to choose which classes to use:

```python
# In config.py
USE_MINIMAL_SET = True   # Use 11 important classes
USE_MINIMAL_SET = False  # Use all 19 DACL10K classes
```

### **Color Customization**

All colors are defined in `config.py`:
```python
# Modify colors in config.py
MINIMAL_CLASS_COLORS = {
    0: (0, 0, 0),         # Background
    1: (255, 0, 0),       # Rust - red
    2: (255, 255, 0),     # ACrack - yellow
    # ... customize as needed
}
```

---

## 📊 **Benefits of Refactoring**

### **✅ Maintainability**
- **Single source of truth** for class definitions
- **Easy to modify** colors and labels globally
- **Consistent behavior** across all scripts

### **✅ Code Reusability** 
- **No more code duplication**
- **Shared utility functions** 
- **Standardized model loading**

### **✅ Flexibility**
- **Easy switching** between minimal/full class sets
- **Auto-detection** of model architectures
- **Configurable thresholds** and parameters

### **✅ Testing & Debugging**
- **Centralized validation** functions
- **Easy to test** individual components
- **Better error handling**

---

## 🚀 **Migration Guide**

If you have existing scripts, here's how to migrate them:

### **Step 1: Replace Class Definitions**

**Old:**
```python
CLASS_COLORS = { ... }
CLASS_LABELS = { ... }
```

**New:**
```python
from config import CLASS_COLORS, CLASS_LABELS
```

### **Step 2: Replace Utility Functions**

**Old:**
```python
def preprocess_image(...):
    # Your implementation
```

**New:**
```python
from utils import preprocess_image
```

### **Step 3: Use Auto-Detection**

**Old:**
```python
# Manual architecture detection
filename = weights_path.lower()
if 'unetplusplus' in filename:
    # ... manual logic
```

**New:**
```python
from config import detect_architecture_from_filename
architecture, encoder = detect_architecture_from_filename(weights_path)
```

---

## 🔍 **Testing the Refactoring**

### **Verify Configuration**
```bash
cd new_scripts
python3 config.py
```

### **Test Imports**
```bash
python3 -c "from config import CLASS_COLORS; print('Config OK')"
python3 -c "from utils import preprocess_image; print('Utils OK')"
```

### **Run Refactored Scripts**
```bash
python3 bridge_app.py  # Should work as before
python3 image_ninja.py your_model.pth your_image.jpg
```

---

## 🛠 **Additional Improvements Made**

### **1. Better Documentation**
- Comprehensive docstrings for all functions
- Clear parameter descriptions
- Usage examples in comments

### **2. Error Handling**
- Configuration validation functions
- Better error messages
- Graceful fallbacks

### **3. Performance Optimizations**
- Reduced redundant imports
- More efficient color conversions
- Streamlined processing pipelines

### **4. Extensibility**
- Easy to add new architectures
- Simple to modify class mappings
- Flexible utility functions

---

## 📝 **Next Steps**

### **Recommended Actions:**
1. ✅ **Test the refactored scripts** with your existing models
2. ✅ **Update any custom scripts** to use the new imports  
3. ✅ **Consider using utilities** in other parts of your workflow
4. ✅ **Customize colors/classes** in `config.py` as needed

### **Future Enhancements:**
- Add configuration file (YAML/JSON) support
- Create visualization presets
- Add more model architectures
- Implement advanced post-processing utilities

---

**The refactored codebase is now cleaner, more maintainable, and easier to extend! 🎉**
