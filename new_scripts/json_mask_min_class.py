import os
import json
import numpy as np
import cv2
from tqdm import tqdm

# Import centralized configuration
from config import (
    ACTIVE_LABEL_MAP, NUM_CLASSES, ALLOWED_CLASS_IDS, 
    MINIMAL_LABEL_MAP, DACL10K_FULL_LABEL_MAP as LABEL_MAP, USE_MINIMAL_SET,
    print_dataset_info, validate_configuration
)

def convert_labelme_json(json_path, out_mask_path):
    with open(json_path) as f:
        data = json.load(f)

    # Support both 'height'/'width' and 'size' dict
    if 'height' in data and 'width' in data:
        height = data['height']
        width = data['width']
    elif 'size' in data:
        height = data['size']['height']
        width = data['size']['width']
    else:
        raise ValueError("Cannot find image size in JSON.")

    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill mask based on objects
    for obj in data.get('objects', []):
        label = obj['classTitle'].strip().lower()  # Convert to lowercase for matching
        if label not in ACTIVE_LABEL_MAP:
            if not USE_MINIMAL_SET:  # Only show warning when using full set
                print(f"Warning: Label '{label}' not in LABEL_MAP, skipping.")
            continue
        points = np.array(obj['points']['exterior'], dtype=np.int32)
        class_id = ACTIVE_LABEL_MAP[label]
        cv2.fillPoly(mask, [points], class_id)

    cv2.imwrite(out_mask_path, mask)

def main():
    # Print dataset configuration info
    print_dataset_info()
    
    json_dir = input("\nEnter JSON directory path: ").strip()
    if not os.path.isdir(json_dir):
        print(f"Error: Directory '{json_dir}' does not exist.")
        return

    # Create output directory
    output_dir = f"{json_dir}_masks"
    os.makedirs(output_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in '{json_dir}'")
        return

    print(f"\nFound {len(json_files)} JSON files. Converting...")
    
    for json_file in tqdm(json_files, desc="Converting"):
        json_path = os.path.join(json_dir, json_file)
        mask_filename = json_file.replace('.json', '.png')
        mask_path = os.path.join(output_dir, mask_filename)
        
        try:
            convert_labelme_json(json_path, mask_path)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print(f"\nConversion complete! Masks saved to: {output_dir}")
    print(f"Class configuration: {'MINIMAL' if USE_MINIMAL_SET else 'FULL'}")
    print(f"Total classes for training: {len(ACTIVE_LABEL_MAP) + 1}")

if __name__ == "__main__":
    main()