import os
import time
import argparse
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import get_dataloaders
from train import EyeStateMobileNetV2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model_path="../models/mobilenet_v2_eye_state.pth", data_dir="data/mrlEyes", batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"📊 Evaluating on {device}")
    
    try:
        _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size, download=False)
    except Exception as e:
        logger.error(f"Failed to load validation dataset: {e}")
        return
        
    model = EyeStateMobileNetV2(pretrained=False).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"✅ Loaded weights from {model_path}")
    else:
        logger.warning(f"⚠️ Weights not found at {model_path}. Evaluating random initialization.")
        
    model.eval()
    
    all_preds = []
    all_labels = []
    
    start_time = time.time()
    total_samples = 0
    total_latency = 0.0
    
    if len(test_loader.dataset) == 0:
        logger.warning("Test loader is empty.")
        return
        
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.cpu().numpy().flatten()
            
            # Measure latency
            t0 = time.time()
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                outputs = model(images)
            t1 = time.time()
            
            total_latency += (t1 - t0)
            total_samples += images.size(0)
            
            preds = (torch.sigmoid(outputs) >= 0.5).cpu().numpy().flatten()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    total_time = time.time() - start_time
    avg_latency_ms = (total_latency / total_samples) * 1000
    fps = total_samples / total_time
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    logger.info(f"\n====== Performance Metrics ======")
    logger.info(f"Accuracy : {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall   : {rec:.4f}")
    logger.info(f"F1-Score : {f1:.4f}")
    logger.info(f"Avg Model Latency/Sample: {avg_latency_ms:.2f} ms")
    logger.info(f"Approximate FPS: {fps:.2f}")
    logger.info(f"=================================\n")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Closed (0)', 'Open (1)'], 
                yticklabels=['Closed (0)', 'Open (1)'])
    plt.title('Eye State CNN Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    cm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'confusion_matrix.png')
    plt.savefig(cm_path)
    logger.info(f"Saved confusion matrix to {cm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Eye State CNN")
    parser.add_argument("--model_path", type=str, default="../models/mobilenet_v2_eye_state.pth")
    parser.add_argument("--data_dir", type=str, default="data/mrlEyes")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()
    
    evaluate_model(model_path=args.model_path, data_dir=args.data_dir, batch_size=args.batch_size)
