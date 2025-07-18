import cv2
import numpy as np
import matplotlib.pyplot as plt

mask = cv2.imread('dacl10k_dataset/masks_train/dacl10k_v2_train_0000.png', 0)
print('Unique values in mask:', np.unique(mask))

# Create a color map for up to 20 classes (0 is background)
colormap = np.array([
    [0, 0, 0],         # 0: background
    [128, 0, 0],       # 1
    [0, 128, 0],       # 2
    [128, 128, 0],     # 3
    [0, 0, 128],       # 4
    [128, 0, 128],     # 5
    [0, 128, 128],     # 6
    [128, 128, 128],   # 7
    [64, 0, 0],        # 8
    [192, 0, 0],       # 9
    [64, 128, 0],      # 10
    [192, 128, 0],     # 11
    [64, 0, 128],      # 12
    [192, 0, 128],     # 13
    [64, 128, 128],    # 14
    [192, 128, 128],   # 15
    [0, 64, 0],        # 16
    [128, 64, 0],      # 17
    [0, 192, 0],       # 18
    [128, 192, 0],     # 19
], dtype=np.uint8)

# Map mask values to colors
mask_color = colormap[mask]

plt.figure(figsize=(10, 8))
plt.imshow(mask_color)
plt.title('Mask Visualization (colored by class)')
plt.axis('off')
plt.show()