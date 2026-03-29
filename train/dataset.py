"""
Dataset preparation and loading for drowsiness detection (Eye State).
Supports automatic downloading of the MRL Eye Dataset.
"""

import os
import glob
import urllib.request
import zipfile
import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MRL Eye Dataset direct URL
MRL_URL = "http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip"

class MRLEyeDataset(Dataset):
    """
    PyTorch Dataset for the MRL Eye Dataset.
    The MRL dataset uses a specific naming convention:
    subjectID_imageID_gender_glasses_eyeState_reflections_lightingConditions_sensorID.png
    eyeState: 0 for closed, 1 for open.
    """
    
    def __init__(self, data_dir="data/mrlEyes", split="train", transform=None, download=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        if download:
            self.download_and_extract()
            
        self.image_paths = []
        self.labels = []
        
        self.load_dataset()
        
    def download_and_extract(self):
        """Downloads and extracts the MRL Eye dataset if not present."""
        os.makedirs(self.data_dir, exist_ok=True)
        zip_path = self.data_dir / "mrlEyes_2018_01.zip"
        
        if not zip_path.exists() and not (self.data_dir / "mrlEyes_2018_01").exists():
            logger.info("Downloading MRL Eye Dataset (~2GB). This may take a while...")
            try:
                urllib.request.urlretrieve(MRL_URL, zip_path)
                logger.info("Download completed.")
            except Exception as e:
                logger.error(f"Failed to download dataset: {e}")
                return

        if zip_path.exists() and not (self.data_dir / "mrlEyes_2018_01").exists():
            logger.info("Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            logger.info("Extraction completed.")
            
    def load_dataset(self):
        """Parse filenames to extract eye state labels."""
        # Find all pngs in extracted dataset
        search_path = self.data_dir / "mrlEyes_2018_01" / "**" / "*.png"
        all_files = glob.glob(str(search_path), recursive=True)
        
        if not all_files:
            logger.warning(f"No images found in {self.data_dir}. Please run with download=True.")
            return
            
        # Split into train/test (80/20) deterministically
        all_files = sorted(all_files)
        split_idx = int(len(all_files) * 0.8)
        
        files_to_use = all_files[:split_idx] if self.split == "train" else all_files[split_idx:]
        
        for file_path in files_to_use:
            filename = os.path.basename(file_path)
            parts = filename.split('_')
            
            # Sanity check on filename format
            if len(parts) >= 8:
                eye_state = int(parts[4])  # 0: closed, 1: open
                self.image_paths.append(file_path)
                self.labels.append(eye_state)
                
        logger.info(f"Loaded {len(self.image_paths)} images for {self.split} split.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        # Return as float tensor label for BCEWithLogitsLoss
        return image, torch.tensor([label], dtype=torch.float32)

class BinaryImageFolder(Dataset):
    """Wrapper to convert integer labels into float tensors for BCEWithLogitsLoss."""
    def __init__(self, img_folder):
        self.img_folder = img_folder
    def __len__(self):
        return len(self.img_folder)
    def __getitem__(self, idx):
        img, label = self.img_folder[idx]
        return img, torch.tensor([float(label)], dtype=torch.float32)

def get_dataloaders(data_dir="data/mrlEyes", batch_size=64, download=True):
    """Creates training and validation dataloaders with standard augmentations."""
    
    # Normalization mean/std for MobileNet/ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                                     
    train_transform = transforms.Compose([
        transforms.Resize((64, 128)),  # W=128, H=64 matching config.EYE_ROI_SIZE
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Use ImageFolder wrapper if the dataset is formatted in sub-folders (like Kaggle CEW)
    if 'kaggle' in str(data_dir).lower() or 'cew' in str(data_dir).lower():
        from torchvision.datasets import ImageFolder
        
        # Check if zip gave us train/test splits inside
        train_path = Path(data_dir) / "train" if (Path(data_dir) / "train").exists() else Path(data_dir)
        val_path = Path(data_dir) / "test" if (Path(data_dir) / "test").exists() else train_path
        
        base_train = ImageFolder(root=train_path, transform=train_transform)
        base_val = ImageFolder(root=val_path, transform=val_transform)
        
        # We need to wrap it so it returns torch.tensor([label]) instead of an integer for BCEWithLogitsLoss
        train_dataset = BinaryImageFolder(base_train)
        val_dataset = BinaryImageFolder(base_val)
        logger.info(f"Loaded {len(train_dataset)} Kaggle training images and {len(val_dataset)} validation images.")
    else:
        train_dataset = MRLEyeDataset(data_dir=data_dir, split="train", transform=train_transform, download=download)
        val_dataset = MRLEyeDataset(data_dir=data_dir, split="val", transform=val_transform, download=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test downloading and loading
    logger.info("Testing Dataset Loader...")
    train_loader, val_loader = get_dataloaders(download=True)
    if train_loader.dataset:
        images, labels = next(iter(train_loader))
        logger.info(f"Batch Image shape: {images.shape}")
        logger.info(f"Batch Label shape: {labels.shape}")
