"""
Training pipeline for the Eye State CNN using MobileNetV2.
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm

# Add current directory to path so imports work perfectly whether run from root or inside train/
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EyeStateMobileNetV2(nn.Module):
    """
    MobileNetV2 modified for binary classification (open vs closed eye).
    It accepts input shapes as small as 64x128.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.model = mobilenet_v2(weights=weights)
        
        # Replace classifier for binary classification (1 output with BCEWithLogitsLoss)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1)
        )
        
    def forward(self, x):
        return self.model(x)

def train_model(data_dir="data/mrlEyes", epochs=15, batch_size=128, lr=1e-3, save_dir="models", dry_run=False):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Using device: {device}")
    
    # Enable automatic mixed precision if using CUDA
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    
    try:
        train_loader, val_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size, download=False)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return
        
    if len(train_loader.dataset) == 0:
        logger.warning("Dataset empty. Please ensure dataset is downloaded.")
        if not dry_run:
            return
            
    model = EyeStateMobileNetV2(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{epochs} ---")
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc="Training") if not dry_run else []
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Autocast for mixed precision
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            
            if dry_run: break
            
        if not dry_run:
            train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        val_pbar = tqdm(val_loader, desc="Validation") if not dry_run else []
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item() * images.size(0)
                
                predicted = (torch.sigmoid(outputs) >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if dry_run: break
                
        if not dry_run:
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct / total
            
            logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            scheduler.step(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(save_dir, "mobilenet_v2_eye_state.pth"))
                logger.info(f"🏆 Saved new best model with Acc: {best_acc:.4f}")
        
        if dry_run:
            logger.info("Dry run completed successfully.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Eye State CNN")
    parser.add_argument("--data_dir", type=str, default="data/mrlEyes", help="Path to MRL dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dry_run", action="store_true", help="Run a single mock batch for testing")
    args = parser.parse_args()
    
    train_model(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, dry_run=args.dry_run)
