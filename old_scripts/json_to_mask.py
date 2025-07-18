import os
import json
import numpy as np
import cv2
from tqdm import tqdm


# Map class names to integer IDs
LABEL_MAP = {'Wetspot': 1, 
             'Rust': 2,
             'EJoint': 3,
             'ACrack': 4,
             'WConccor': 5,
             'Cavity': 6,
             'Hollowareas': 7,
             'JTape': 8,
             'Spalling': 9,
             'Rockpocket': 10,
             'ExposedRebars': 11,
             'Crack': 12,
             'Restformwork': 13,
             'Drainage': 14,
             'Weathering': 15,
             'Bearing': 16,
             'Graffiti': 17,
             'PEquipment': 18,
             'Efflorescence': 19
             }

def convert_labelme_json(json_path, out_mask_path):
    with open(json_path) as f:
        data = json.load(f)

    height = data['imageHeight']
    width = data['imageWidth']
    mask = np.zeros((height, width), dtype=np.uint8)

    for shape in data['shapes']:
        label = shape['label'].strip()
        points = np.array(shape['points'], dtype=np.int32)
        if label not in LABEL_MAP:
            print(f"Warning: Label '{label}' not in LABEL_MAP, skipping.")
            continue

        class_id = LABEL_MAP[label]
        cv2.fillPoly(mask, [points], class_id)

    cv2.imwrite(out_mask_path, mask)

    # # Optional: Save a colorized version of the mask for visualization
    # # Define a color for each class (0 is background)
    # colormap = np.array([
    #     [0, 0, 0],         # 0: background
    #     [128, 0, 0],       # 1: Wetspot
    #     [0, 128, 0],       # 2: Rust
    #     [128, 128, 0],     # 3: EJoint
    #     [0, 0, 128],       # 4: ACrack
    #     [128, 0, 128],     # 5: WConccor
    #     [0, 128, 128],     # 6: Cavity
    #     [128, 128, 128],   # 7: Hollowareas
    #     [64, 0, 0],        # 8: JTape
    #     [192, 0, 0],       # 9: Spalling
    #     [64, 128, 0],      # 10: Rockpocket
    #     [192, 128, 0],     # 11: ExposedRebars
    #     [64, 0, 128],      # 12: Crack
    #     [192, 0, 128],     # 13: Restformwork
    #     [64, 128, 128],    # 14: Drainage
    #     [192, 128, 128],   # 15: Weathering
    #     [0, 64, 0],        # 16: Bearing
    #     [128, 64, 0],      # 17: Graffiti
    #     [0, 192, 0],       # 18: PEquipment
    #     [128, 192, 0],     # 19: Efflorescence
    # ], dtype=np.uint8)

    # mask_color = colormap[mask]
    # color_out_path = out_mask_path.replace('.png', '_color.png')
    # cv2.imwrite(color_out_path, cv2.cvtColor(mask_color, cv2.COLOR_RGB2BGR))

# Example usage
json_dir = 'dacl10k_dataset/annotations/validation' 
mask_dir = 'dacl10k_dataset/masks_val'

os.makedirs(mask_dir, exist_ok=True)

json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
for json_file in tqdm(json_files, desc="Generating masks"):
    base = os.path.splitext(json_file)[0]
    json_path = os.path.join(json_dir, json_file)
    out_path = os.path.join(mask_dir, f"{base}.png")
    convert_labelme_json(json_path, out_path)

print(f"[✓] All masks saved to: {mask_dir}")