"""
CNN Inference with temporal sampling & EAR/MAR/Pose tracking.
"""

import cv2
import numpy as np
import queue
import logging
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

logger = logging.getLogger(__name__)

class EyeStateMobileNetV2(nn.Module):
    """MobileNetV2 architecture adapted for binary classification."""
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v2(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1)
        )
    def forward(self, x):
        return self.model(x)


class InferencePipeline:
    """CNN inference with temporal sampling every 3 frames."""
    
    def __init__(self, config, input_queue: queue.Queue, output_queue: queue.Queue):
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.is_running = False
        
        # Load model & Device Handling
        self.device = torch.device("cuda" if config.USE_CUDA and torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.model.eval()
        logger.info(f"🧠 Inference initialized on: {self.device}")
        
        self.frame_count = 0
        self.closed_frame_count = 0
        self.yawning_frame_count = 0
        self.distracted_frame_count = 0
        
        self.last_left_open = 1.0
        self.last_right_open = 1.0
        
        # Normalization mean and std
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
    def _load_model(self):
        """Load or create model."""
        model = EyeStateMobileNetV2().to(self.device)
        try:
            state_dict = torch.load(self.config.MODEL_PATH, map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info(f"✅ Model loaded from {self.config.MODEL_PATH}")
        except FileNotFoundError:
            logger.warning(f"⚠️ Model not found at {self.config.MODEL_PATH}, using random init")
        return model
    
    def run(self):
        """Main inference loop."""
        logger.info("🧠 Inference thread started")
        self.is_running = True
        
        try:
            while self.is_running:
                try:
                    data = self.input_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                self.frame_count += 1
                
                left_eye = data['left_eye']
                right_eye = data['right_eye']
                
                # Use cached probabilities if inference skipped
                left_open, right_open = self.last_left_open, self.last_right_open
                
                if data['face_detected'] and left_eye is not None and right_eye is not None:
                    # **TEMPORAL SAMPLING: Run heavy inference every N frames**
                    if self.frame_count % self.config.INFERENCE_INTERVAL == 0:
                        
                        # Preprocess & Format as RGB for MobileNetV2
                        l_rgb = cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)
                        r_rgb = cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)
                        
                        l_tensor = torch.from_numpy(l_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                        r_tensor = torch.from_numpy(r_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                        
                        l_tensor = (l_tensor.to(self.device) - self.mean) / self.std
                        r_tensor = (r_tensor.to(self.device) - self.mean) / self.std
                        
                        # Forward pass with Automatic Mixed Precision (AMP)
                        with torch.no_grad():
                            with torch.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                                left_pred = self.model(l_tensor)
                                right_pred = self.model(r_tensor)
                        
                        # Sigmoid for BCEWithLogits output
                        left_open = torch.sigmoid(left_pred).item()
                        right_open = torch.sigmoid(right_pred).item()
                        
                        # Cache for intervening frames
                        self.last_left_open = left_open
                        self.last_right_open = right_open
                
                # EAR equivalent (averaged open probability)
                avg_open_prob = (left_open + right_open) / 2.0
                
                # ------ STATE TRACKING ------
                
                # 1. Distraction (Head Pose)
                # Check instantaneous distraction for reliable eye inference
                is_currently_distracted = (abs(data['pitch']) > self.config.PITCH_THRESHOLD or \
                                           abs(data['yaw']) > self.config.YAW_THRESHOLD or \
                                           abs(data['roll']) > self.config.ROLL_THRESHOLD)
                
                if is_currently_distracted:
                    self.distracted_frame_count += 1
                else:
                    self.distracted_frame_count = max(0, self.distracted_frame_count - 1)
                    
                is_distracted = self.distracted_frame_count >= self.config.HEAD_POSE_FRAMES_THRESHOLD

                # 2. Closed Eyes (Drowsiness)
                # GUARD: Only track eye closure if NOT currently turning head AND face is detected
                if not is_currently_distracted and data.get('face_detected', False):
                    if avg_open_prob < self.config.EAR_THRESHOLD:
                        self.closed_frame_count += 1
                    else:
                        self.closed_frame_count = max(0, self.closed_frame_count - 2) # quick decay
                else:
                    # If turning head OR NO face detected, slowly decay drowsiness count
                    self.closed_frame_count = max(0, self.closed_frame_count - 1)
                    
                is_drowsy = self.closed_frame_count >= self.config.CLOSED_FRAMES_THRESHOLD
                
                # 3. Yawning (MAR)
                if data['mar'] > self.config.MAR_THRESHOLD:
                    self.yawning_frame_count += 1
                else:
                    self.yawning_frame_count = max(0, self.yawning_frame_count - 1)
                    
                is_yawning = self.yawning_frame_count >= self.config.YAWNING_FRAMES_THRESHOLD
                
                # Prepare output
                output_data = data.copy()
                output_data.update({
                    'left_eye_open_prob': left_open,
                    'right_eye_open_prob': right_open,
                    'avg_open_prob': avg_open_prob,
                    'closed_frames': self.closed_frame_count,
                    'is_drowsy': is_drowsy,
                    'is_yawning': is_yawning,
                    'is_distracted': is_distracted
                })
                
                try:
                    # Non-blocking put, prevent deadlocks
                    if self.output_queue.full():
                        try: self.output_queue.get_nowait()
                        except queue.Empty: pass
                    self.output_queue.put_nowait(output_data)
                except Exception as e:
                    logger.debug(f"Inference queue error: {e}")
                
        except Exception as e:
            logger.error(f"❌ Inference error: {e}")
        finally:
            self.is_running = False
            # Clear queue gracefully
            while not self.input_queue.empty():
                try: self.input_queue.get_nowait()
                except queue.Empty: break
            logger.info("🧠 Inference thread stopped cleanly")