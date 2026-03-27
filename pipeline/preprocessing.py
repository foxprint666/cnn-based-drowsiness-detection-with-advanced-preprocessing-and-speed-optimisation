"""
Preprocessing pipeline: ROI detection, stabilization, enhancement.
"""

import cv2
import numpy as np
import queue
import logging
import mediapipe as mp
from collections import deque

logger = logging.getLogger(__name__)


class KalmanROI:
    """Lightweight Kalman filter for ROI stabilization."""
    
    def __init__(self, process_noise, measurement_noise):
        self.kf = cv2.KalmanFilter(4, 2)  # State: (x, y, vx, vy); Measure: (x, y)
        
        # Transition matrix
        dt = 1.0
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Covariance matrices
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        
        # Initial state
        self.kf.statePost = np.array([[0], [0], [0], [0]], dtype=np.float32)
        self.initialized = False
    
    def predict(self):
        """Predict next ROI center."""
        predicted = self.kf.predict()
        return int(predicted[0, 0]), int(predicted[1, 0])
    
    def update(self, x, y):
        """Update with measurement."""
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kf.correct(measurement)
        self.initialized = True


class PreprocessingPipeline:
    """ROI → Stabilization → Enhancement → Eye Extraction."""
    
    def __init__(self, config, frame_queue: queue.Queue, output_queue: queue.Queue):
        self.config = config
        self.frame_queue = frame_queue
        self.output_queue = output_queue
        self.is_running = False
        
        # MediaPipe Face Detection
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(
            model_selection=1,  # 1 = full range (slower but accurate)
            min_detection_confidence=config.FACE_CONFIDENCE
        )
        
        # Kalman filter for stabilization
        self.kalman_roi = KalmanROI(config.KALMAN_PROCESS_NOISE, config.KALMAN_MEASUREMENT_NOISE)
        
        # Optical flow tracker
        self.prev_gray = None
        self.prev_roi = None
        
        self.frame_count = 0
        
    def run(self):
        """Main preprocessing loop."""
        logger.info("🔧 Preprocessing thread started")
        self.is_running = True
        
        try:
            while self.is_running:
                try:
                    frame_id, frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                self.frame_count += 1
                h, w = frame.shape[:2]
                
                # **STEP 1: Face & Eye ROI Detection (MediaPipe)**
                if self.frame_count % self.config.FACE_DETECTION_INTERVAL == 0:
                    results = self.face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    if results.detections:
                        det = results.detections[0]  # Strongest detection
                        bbox = det.location_data.relative_bounding_box
                        
                        # Extract face ROI
                        x = max(0, int(bbox.xmin * w) - 10)
                        y = max(0, int(bbox.ymin * h) - 10)
                        x2 = min(w, int((bbox.xmin + bbox.width) * w) + 10)
                        y2 = min(h, int((bbox.ymin + bbox.height) * h) + 10)
                        
                        face_roi = frame[y:y2, x:x2]
                        self.prev_roi = (x, y, x2, y2)
                        self.kalman_roi.update(x + (x2-x)//2, y + (y2-y)//2)
                
                # **STEP 2: Virtual Gimbal - Optical Flow Stabilization**
                if self.prev_roi and self.prev_gray is not None:
                    x, y, x2, y2 = self.prev_roi
                    curr_gray = cv2.cvtColor(frame[y:y2, x:x2], cv2.COLOR_BGR2GRAY)
                    
                    # Optical flow (sparse)
                    flow = cv2.calcOpticalFlowFarneback(
                        self.prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    
                    # Estimate translation
                    tx, ty = np.mean(flow, axis=(0, 1))
                    
                    # Warp for stabilization
                    M = np.float32([[1, 0, -tx], [0, 1, -ty]])
                    frame = cv2.warpAffine(frame, M, (w, h))
                    
                    self.prev_gray = curr_gray
                else:
                    self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Crop to ROI if available
                if self.prev_roi:
                    x, y, x2, y2 = self.prev_roi
                    roi_frame = frame[y:y2, x:x2]
                else:
                    roi_frame = frame
                
                # **STEP 3: Illumination Enhancement**
                roi_frame = self._enhance_illumination(roi_frame)
                
                # **STEP 4: Super-Resolution (Optional)**
                if self.config.ENABLE_SRCNN:
                    roi_frame = self._apply_srcnn(roi_frame)
                
                # **STEP 5: Extract Eye Regions**
                left_eye, right_eye = self._extract_eye_regions(roi_frame)
                
                # Output for inference
                output_data = {
                    'frame_id': frame_id,
                    'original_frame': frame,
                    'roi_frame': roi_frame,
                    'left_eye': left_eye,
                    'right_eye': right_eye,
                }
                
                try:
                    self.output_queue.put_nowait(output_data)
                except queue.Full:
                    logger.debug("Preprocessing output queue full")
                    
        except Exception as e:
            logger.error(f"❌ Preprocessing error: {e}")
        finally:
            self.is_running = False
            logger.info("🔧 Preprocessing thread stopped")
    
    def _enhance_illumination(self, img):
        """Apply composite enhancement: CLAHE + Gamma + Multi-scale Retinex."""
        # Convert to LAB
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE on L channel
        clahe = cv2.createCLAHE(
            clipLimit=self.config.CLAHE_CLIP_LIMIT,
            tileGridSize=(self.config.CLAHE_TILE_SIZE, self.config.CLAHE_TILE_SIZE)
        )
        l = clahe.apply(l)
        
        # Gamma correction
        inv_gamma = 1.0 / self.config.GAMMA_CORRECTION
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
        l = cv2.LUT(l, table)
        
        # Merge back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def _apply_srcnn(self, img):
        """Placeholder for SRCNN upscaling."""
        # In production, use TensorRT or ONNX Runtime
        return cv2.resize(img, None, fx=self.config.SRCNN_SCALE, fy=self.config.SRCNN_SCALE)
    
    def _extract_eye_regions(self, roi_frame):
        """Extract left and right eye regions (simplified)."""
        h, w = roi_frame.shape[:2]
        
        # Approximate eye locations (top-left & top-right)
        left_eye = roi_frame[h//4:h//2, w//6:w//3]
        right_eye = roi_frame[h//4:h//2, 2*w//3:5*w//6]
        
        # Resize to standard size
        left_eye = cv2.resize(left_eye, self.config.EYE_ROI_SIZE)
        right_eye = cv2.resize(right_eye, self.config.EYE_ROI_SIZE)
        
        return left_eye, right_eye