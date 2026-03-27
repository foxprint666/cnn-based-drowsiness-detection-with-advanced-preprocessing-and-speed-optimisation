"""
CNN Inference with temporal sampling & EAR calculation.
"""

import cv2
import numpy as np
import queue
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EyeStateCNN(nn.Module):
    """Lightweight CNN for eye state (open/closed)."""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # Binary: open/closed
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class InferencePipeline:
    """CNN inference with temporal sampling every 3 frames."""
    
    def __init__(self, config, input_queue: queue.Queue, output_queue: queue.Queue):
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.is_running = False
        
        # Load model
        self.device = torch.device("cuda" if config.USE_CUDA and torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.model.eval()
        
        self.frame_count = 0
        self.closed_frame_count = 0
        
    def _load_model(self):
        """Load or create model."""
        model = EyeStateCNN().to(self.device)
        
        # Load pretrained weights if available
        try:
            state_dict = torch.load(self.config.MODEL_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info(f"✅ Model loaded from {self.config.MODEL_PATH}")
        except FileNotFoundError:
            logger.warning(f"⚠️  Model not found at {self.config.MODEL_PATH}, using random init")
        
        return model
    
    def run(self):
        """Main inference loop."""
        logger.info("🧠 Inference thread started")
        self.is_running = True
        
        try:
            while self.is_running:
                try:
                    data = self.input_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                self.frame_count += 1
                
                # **TEMPORAL SAMPLING: Run heavy inference every 3rd frame**
                if self.frame_count % self.config.INFERENCE_INTERVAL == 0:
                    left_eye = data['left_eye']
                    right_eye = data['right_eye']
                    
                    # Normalize
                    left_eye_tensor = torch.from_numpy(left_eye).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                    right_eye_tensor = torch.from_numpy(right_eye).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                    
                    with torch.no_grad():
                        left_pred = self.model(left_eye_tensor.to(self.device))
                        right_pred = self.model(right_eye_tensor.to(self.device))
                    
                    # Get probabilities
                    left_open = torch.softmax(left_pred, dim=1)[0, 0].item()
                    right_open = torch.softmax(right_pred, dim=1)[0, 0].item()
                    
                    # EAR equivalent (simplified)
                    avg_open_prob = (left_open + right_open) / 2.0
                    
                    # Track closed frames
                    if avg_open_prob < self.config.EAR_THRESHOLD:
                        self.closed_frame_count += 1
                    else:
                        self.closed_frame_count = 0
                    
                    # Determine drowsiness
                    is_drowsy = self.closed_frame_count >= self.config.CLOSED_FRAMES_THRESHOLD
                    
                    output_data = {
                        'frame_id': data['frame_id'],
                        'original_frame': data['original_frame'],
                        'roi_frame': data['roi_frame'],
                        'left_eye_open_prob': left_open,
                        'right_eye_open_prob': right_open,
                        'avg_open_prob': avg_open_prob,
                        'closed_frames': self.closed_frame_count,
                        'is_drowsy': is_drowsy,
                    }
                    
                    try:
                        self.output_queue.put_nowait(output_data)
                    except queue.Full:
                        logger.debug("Inference output queue full")
                
        except Exception as e:
            logger.error(f"❌ Inference error: {e}")
        finally:
            self.is_running = False
            logger.info("🧠 Inference thread stopped")