import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import signal
import sys
import logging

# Set up logging
logging.basicConfig(filename='training_interrupt.log', level=logging.INFO, format='%(asctime)s %(message)s')

def handle_kill_signal(signum, frame):
    reason = f"Training interrupted or killed (signal {signum})"
    print(f"\n[!] {reason}")
    logging.info(reason)
    sys.exit(1)

signal.signal(signal.SIGINT, handle_kill_signal)   # Ctrl+C
signal.signal(signal.SIGTERM, handle_kill_signal)  # kill command

# -------- CONFIG --------
NUM_CLASSES = 20  #(background + 19 defect classes)
BATCH_SIZE = 4
NUM_EPOCHS = 20
IMAGE_SIZE = 512
DATASET_DIR = 'dacl10k_Dataset'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_SAVE_PATH = 'dacl10k_resnet50.pth'
# ------------------------

# -------- Dataset --------
class DACL10KDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir 
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.long()

# -------- Transforms --------
def get_train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# -------- Training Function --------
def train_one_epoch(model, dataloader, loss_fn, optimizer):
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, desc="Train", leave=False)
    for images, masks in loop:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)

# -------- Validation Function --------
def validate(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# -------- Main --------
def main():
    # Paths
    train_img_dir = os.path.join(DATASET_DIR, "train", "images")
    train_mask_dir = os.path.join(DATASET_DIR, "train", "masks")
    val_img_dir = os.path.join(DATASET_DIR, "val", "images")
    val_mask_dir = os.path.join(DATASET_DIR, "val", "masks")

    # Datasets & Loaders
    train_ds = DACL10KDataset(train_img_dir, train_mask_dir, transform=get_train_transform())
    val_ds = DACL10KDataset(val_img_dir, val_mask_dir, transform=get_val_transform())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        classes=NUM_CLASSES
    ).to(DEVICE)

    # Loss functions
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(mode='multiclass')
    def combined_loss(pred, target):
        return ce_loss(pred, target) + dice_loss(pred, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        train_loss = train_one_epoch(model, train_loader, combined_loss, optimizer)
        val_loss = validate(model, val_loader, combined_loss)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"[✓] Saved best model at epoch {epoch+1}")

if __name__ == "__main__":
    main()