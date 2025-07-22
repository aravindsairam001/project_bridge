"""
DACL10K Bridge Defect Detection - Configuration Module
====================================================

Centralized configuration for class definitions, colors, and mappings.
This module ensures consistency across all scripts and makes maintenance easier.

Author: Bridge Defect Detection Project
Date: 2025-07-22
"""

# ============================================================================
# DATASET LABEL MAPPINGS
# ============================================================================

# Full DACL10K dataset labels (19 classes)
DACL10K_FULL_LABEL_MAP = {
    'alligator crack': 1,
    'bearing': 2,
    'cavity': 3,
    'crack': 4,
    'drainage': 5,
    'efflorescence': 6,
    'expansion joint': 7,
    'exposed rebars': 8,
    'graffiti': 9,
    'hollowareas': 10,
    'joint tape': 11,
    'protective equipment': 12,
    'restformwork': 13,
    'rockpocket': 14,
    'rust': 15,
    'spalling': 16,
    'washouts/concrete corrosion': 17,
    'weathering': 18,
    'wetspot': 19,
}

# Original DACL10K mapping (alternative format)
DACL10K_ORIGINAL_LABEL_MAP = {
    "Wetspot": 1,
    "Rust": 2,
    "EJoint": 3,
    "ACrack": 4,
    "WConccor": 5,
    "Cavity": 6,
    "Hollowareas": 7,
    "JTape": 8,
    "Spalling": 9,
    "Rockpocket": 10,
    "ExposedRebars": 11,
    "Crack": 12,
    "Restformwork": 13,
    "Drainage": 14,
    "Weathering": 15,
    "Bearing": 16,
    "Graffiti": 17,
    "PEquipment": 18,
    "Efflorescence": 19
}

# Minimal class subset (11 important classes for focused training)
MINIMAL_LABEL_MAP = {
    'rust': 1,
    'alligator crack': 2,  # ACrack equivalent
    'washouts/concrete corrosion': 3,  # WConccor equivalent
    'cavity': 4,
    'hollowareas': 5,
    'spalling': 6,
    'rockpocket': 7,
    'exposed rebars': 8,
    'crack': 9,
    'weathering': 10,
    'efflorescence': 11
}

# ============================================================================
# CONFIGURATION FLAGS
# ============================================================================

# Set this to choose which label mapping to use
USE_MINIMAL_SET = True  # Set to False to use all 19 classes
USE_ORIGINAL_DACL10K = False  # Set to True to use original DACL10K format

# Active label mapping (automatically selected based on flags)
if USE_ORIGINAL_DACL10K:
    ACTIVE_LABEL_MAP = DACL10K_ORIGINAL_LABEL_MAP
    NUM_CLASSES = 20  # 19 defect classes + background
elif USE_MINIMAL_SET:
    ACTIVE_LABEL_MAP = MINIMAL_LABEL_MAP
    NUM_CLASSES = 12  # 11 defect classes + background
else:
    ACTIVE_LABEL_MAP = DACL10K_FULL_LABEL_MAP
    NUM_CLASSES = 20  # 19 defect classes + background

# ============================================================================
# COLOR MAPPINGS
# ============================================================================

# Color mapping for minimal class subset (RGB format)
MINIMAL_CLASS_COLORS = {
    0: (0, 0, 0),           # 0: background (black)
    1: (255, 0, 0),         # 1: Rust (bright red)
    2: (255, 255, 0),       # 2: ACrack (yellow)
    3: (255, 0, 255),       # 3: WConccor (magenta)
    4: (0, 255, 255),       # 4: Cavity (cyan)
    5: (255, 128, 0),       # 5: Hollowareas (orange)
    6: (0, 255, 128),       # 6: Spalling (spring green)
    7: (255, 0, 128),       # 7: Rockpocket (rose)
    8: (128, 255, 0),       # 8: ExposedRebars (lime)
    9: (0, 128, 255),       # 9: Crack (azure)
    10: (128, 255, 255),    # 10: Weathering (light cyan)
    11: (255, 128, 192),    # 11: Efflorescence (light pink)
}

# Color mapping for full DACL10K dataset (19 classes + background)
FULL_CLASS_COLORS = {
    0: (0, 0, 0),           # 0: background
    1: (255, 0, 0),         # 1: Wetspot/Alligator crack
    2: (0, 255, 0),         # 2: Rust/Bearing
    3: (0, 0, 255),         # 3: EJoint/Cavity
    4: (255, 255, 0),       # 4: ACrack/Crack
    5: (255, 0, 255),       # 5: WConccor/Drainage
    6: (0, 255, 255),       # 6: Cavity/Efflorescence
    7: (255, 128, 0),       # 7: Hollowareas/Expansion joint
    8: (128, 255, 0),       # 8: JTape/Exposed rebars
    9: (0, 255, 128),       # 9: Spalling/Graffiti
    10: (255, 0, 128),      # 10: Rockpocket/Hollowareas
    11: (128, 0, 255),      # 11: ExposedRebars/Joint tape
    12: (0, 128, 255),      # 12: Crack/Protective equipment
    13: (255, 255, 128),    # 13: Restformwork
    14: (128, 255, 255),    # 14: Drainage/Rockpocket
    15: (255, 128, 255),    # 15: Weathering/Rust
    16: (128, 128, 255),    # 16: Bearing/Spalling
    17: (255, 128, 128),    # 17: Graffiti/Washouts
    18: (128, 255, 128),    # 18: PEquipment/Weathering
    19: (192, 192, 192),    # 19: Efflorescence/Wetspot
}

# Active color mapping (automatically selected based on configuration)
CLASS_COLORS = MINIMAL_CLASS_COLORS if USE_MINIMAL_SET else FULL_CLASS_COLORS

# ============================================================================
# CLASS LABELS
# ============================================================================

# Human-readable labels for minimal class subset
MINIMAL_CLASS_LABELS = {
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

# Human-readable labels for full DACL10K dataset
FULL_CLASS_LABELS = {
    0: "Background",
    1: "Wetspot",
    2: "Rust",
    3: "EJoint",
    4: "ACrack",
    5: "WConccor",
    6: "Cavity",
    7: "Hollowareas",
    8: "JTape",
    9: "Spalling",
    10: "Rockpocket",
    11: "ExposedRebars",
    12: "Crack",
    13: "Restformwork",
    14: "Drainage",
    15: "Weathering",
    16: "Bearing",
    17: "Graffiti",
    18: "PEquipment",
    19: "Efflorescence"
}

# Active class labels (automatically selected based on configuration)
CLASS_LABELS = MINIMAL_CLASS_LABELS if USE_MINIMAL_SET else FULL_CLASS_LABELS

# ============================================================================
# DEFECT ALIASES AND GROUPINGS
# ============================================================================

# Aliases for output with consistent colors (for UI display)
DEFECT_ALIASES = {
    4: "Honeycombing",   # Cavity
    5: "Honeycombing",   # Hollowareas
    10: "Leaching",      # Weathering
    11: "Leaching"       # Efflorescence
}

# Alias colors - same color for aliased defects
ALIAS_COLORS = {
    "Honeycombing": (255, 165, 0),  # Orange
    "Leaching": (0, 255, 255)       # Cyan
}

# Define alias colors for consistent visualization in overlays
ALIAS_COLORS_OVERLAY = {
    4: (255, 165, 0),   # Honeycombing (Cavity) - Orange
    5: (255, 165, 0),   # Honeycombing (Hollowareas) - Orange  
    10: (0, 255, 255),  # Leaching (Weathering) - Cyan
    11: (0, 255, 255)   # Leaching (Efflorescence) - Cyan
}

# ============================================================================
# ALLOWED CLASS SETS
# ============================================================================

# Only process these classes (based on active configuration)
ALLOWED_CLASS_IDS = set(range(1, NUM_CLASSES))

# ============================================================================
# MODEL ARCHITECTURE CONFIGURATIONS
# ============================================================================

# Supported architectures and their properties
MODEL_ARCHITECTURES = {
    'deeplabv3plus': {
        'display_name': 'DeepLabV3Plus',
        'emoji': '🔧',
        'description': 'Balanced performance and accuracy'
    },
    'unetplusplus': {
        'display_name': 'UNet++',
        'emoji': '🏆',
        'description': 'Best for fine crack detection'
    },
    'fpn': {
        'display_name': 'FPN',
        'emoji': '🔍',
        'description': 'Excellent for multi-scale defects'
    },
    'linknet': {
        'display_name': 'LinkNet',
        'emoji': '⚡',
        'description': 'Fastest processing'
    },
    'pspnet': {
        'display_name': 'PSPNet',
        'emoji': '🧠',
        'description': 'Superior contextual understanding'
    }
}

# Supported encoders and their properties
MODEL_ENCODERS = {
    'resnet50': {'display_name': 'ResNet50', 'params': '25M'},
    'resnet101': {'display_name': 'ResNet101', 'params': '44M'},
    'resnext101_32x8d': {'display_name': 'ResNeXt101', 'params': '88M'},
    'se_resnext101_32x4d': {'display_name': 'SE-ResNeXt101', 'params': '49M'},
    'efficientnet-b3': {'display_name': 'EfficientNet-B3', 'params': '12M'},
    'efficientnet-b4': {'display_name': 'EfficientNet-B4', 'params': '19M'},
    'efficientnet-b5': {'display_name': 'EfficientNet-B5', 'params': '30M'},
    'efficientnet-b6': {'display_name': 'EfficientNet-B6', 'params': '43M'},
    'efficientnet-b7': {'display_name': 'EfficientNet-B7', 'params': '66M'},
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_class_info():
    """Return current class configuration info"""
    return {
        'num_classes': NUM_CLASSES,
        'use_minimal_set': USE_MINIMAL_SET,
        'allowed_class_ids': ALLOWED_CLASS_IDS,
        'label_map': ACTIVE_LABEL_MAP,
        'class_colors': CLASS_COLORS,
        'class_labels': CLASS_LABELS
    }

def print_dataset_info():
    """Print current dataset configuration"""
    print("🔧 Current Configuration:")
    print(f"   Using {'minimal' if USE_MINIMAL_SET else 'full'} class set")
    print(f"   Total classes: {NUM_CLASSES} ({NUM_CLASSES-1} defect classes + background)")
    print(f"   Allowed class IDs: {sorted(ALLOWED_CLASS_IDS)}")
    print("\n📊 Class Mapping:")
    for label, class_id in sorted(ACTIVE_LABEL_MAP.items(), key=lambda x: x[1]):
        print(f"   {class_id}: {label}")
    print(f"\n💻 Training Configuration:")
    print(f"   NUM_CLASSES = {NUM_CLASSES}")
    print(f"   ALLOWED_CLASS_IDS = set({sorted(ALLOWED_CLASS_IDS)})")

def detect_architecture_from_filename(filename):
    """Auto-detect model architecture from filename"""
    filename = filename.lower()
    
    # Detect architecture
    if 'unetplusplus' in filename:
        architecture = 'unetplusplus'
    elif 'fpn' in filename:
        architecture = 'fpn'
    elif 'linknet' in filename:
        architecture = 'linknet'
    elif 'pspnet' in filename:
        architecture = 'pspnet'
    else:
        architecture = 'deeplabv3plus'  # default
    
    # Detect encoder
    if 'efficientnet_b' in filename:
        if 'b7' in filename:
            encoder_name = 'efficientnet-b7'
        elif 'b6' in filename:
            encoder_name = 'efficientnet-b6'
        elif 'b5' in filename:
            encoder_name = 'efficientnet-b5'
        elif 'b4' in filename:
            encoder_name = 'efficientnet-b4'
        elif 'b3' in filename:
            encoder_name = 'efficientnet-b3'
        else:
            encoder_name = 'efficientnet-b5'  # default EfficientNet
    elif 'resnext101' in filename:
        if 'se_resnext101_32x4d' in filename:
            encoder_name = 'se_resnext101_32x4d'
        else:
            encoder_name = 'resnext101_32x8d'
    elif 'resnet50' in filename:
        encoder_name = 'resnet50'
    elif 'resnet101' in filename:
        encoder_name = 'resnet101'
    else:
        encoder_name = 'resnet101'  # fallback
    
    return architecture, encoder_name

def get_architecture_display_info(architecture):
    """Get display information for architecture"""
    if architecture in MODEL_ARCHITECTURES:
        info = MODEL_ARCHITECTURES[architecture]
        return f"{info['emoji']} {info['display_name']}"
    return f"🔧 {architecture.upper()}"

def rgb_to_bgr(rgb_color):
    """Convert RGB color to BGR for OpenCV"""
    return (rgb_color[2], rgb_color[1], rgb_color[0])

def get_alias_color(class_id):
    """Get alias color if available, otherwise return class color"""
    if class_id in ALIAS_COLORS_OVERLAY:
        return ALIAS_COLORS_OVERLAY[class_id]
    return CLASS_COLORS.get(class_id, (255, 255, 255))

def get_display_name(class_id):
    """Get display name (with alias if available)"""
    if class_id in DEFECT_ALIASES:
        return DEFECT_ALIASES[class_id]
    return CLASS_LABELS.get(class_id, f"Class {class_id}")

# ============================================================================
# VALIDATION
# ============================================================================

def validate_configuration():
    """Validate that the current configuration is consistent"""
    errors = []
    
    # Check that number of classes matches active label map
    expected_classes = len(ACTIVE_LABEL_MAP) + 1  # +1 for background
    if NUM_CLASSES != expected_classes:
        errors.append(f"NUM_CLASSES ({NUM_CLASSES}) doesn't match active label map size ({expected_classes})")
    
    # Check that class colors exist for all classes
    for class_id in range(NUM_CLASSES):
        if class_id not in CLASS_COLORS:
            errors.append(f"Missing color for class {class_id}")
    
    # Check that class labels exist for all classes
    for class_id in range(NUM_CLASSES):
        if class_id not in CLASS_LABELS:
            errors.append(f"Missing label for class {class_id}")
    
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("✅ Configuration is valid")
        return True

# ============================================================================
# INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    print("DACL10K Bridge Defect Detection - Configuration")
    print("=" * 50)
    print_dataset_info()
    print("\n" + "=" * 50)
    print("🔍 Validating Configuration...")
    validate_configuration()
