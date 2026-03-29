"""
Preprocessing pipeline: Tasks API FaceLandmarker for precise tracking & Yawn/Pose.
"""

import cv2
import numpy as np
import queue
import logging
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """Uses the Modern MediaPipe Tasks API for 3D landmarks, Blendshapes, and Transform Matrices."""
    
    def __init__(self, config, frame_queue: queue.Queue, output_queue: queue.Queue):
        self.config = config
        self.frame_queue = frame_queue
        self.output_queue = output_queue
        self.is_running = False
        
        # 1. Download official Task model if missing
        model_path = os.path.join("models", "face_landmarker.task")
        os.makedirs("models", exist_ok=True)
        if not os.path.exists(model_path):
            logger.info("Downloading FaceLandmarker task model... (This is a one-time download)")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            
        # 2. Configure Modern Tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
            
        logger.info("Initializing FaceLandmarker Tasks API... This will flawlessly load on Python 3.14+\n")
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        self.frame_count = 0
        
    def run(self):
        logger.info("🔧 Preprocessing thread started")
        self.is_running = True
        
        try:
            while self.is_running:
                try:
                    frame_id, frame = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                self.frame_count += 1
                h, w = frame.shape[:2]
                
                # Format for MP Tasks
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Synchronous execute
                detection_result = self.detector.detect(mp_image)
                
                mar = 0.0
                pitch, yaw, roll = 0.0, 0.0, 0.0
                left_eye_crop = None
                right_eye_crop = None
                roi_frame = frame
                face_detected = False
                
                if detection_result.face_landmarks:
                    face_detected = True
                    landmarks = detection_result.face_landmarks[0]
                    
                    # 1. Extract Yawning from internal neural Blendshapes (JawOpen)
                    if detection_result.face_blendshapes:
                        blendshapes = detection_result.face_blendshapes[0]
                        for category in blendshapes:
                            if category.category_name == 'jawOpen':
                                mar = category.score
                                break
                                
                    # 2. Extract Head Pose Matrix directly from internal solver
                    if detection_result.facial_transformation_matrixes:
                        matrix = detection_result.facial_transformation_matrixes[0]
                        pitch, yaw, roll = self._extract_pose_from_matrix(matrix)
                        
                    # 3. Precise Eye Cropping
                    left_eye_crop, right_eye_crop = self._extract_eyes(frame, landmarks, w, h)
                
                # Pack dictionary for Central Logic
                output_data = {
                    'frame_id': frame_id,
                    'original_frame': frame,
                    'roi_frame': roi_frame,
                    'left_eye': left_eye_crop,
                    'right_eye': right_eye_crop,
                    'mar': mar,
                    'pitch': pitch,
                    'yaw': yaw,
                    'roll': roll,
                    'face_detected': face_detected
                }
                
                try:
                    if self.output_queue.full():
                        try: self.output_queue.get_nowait()
                        except: pass
                    self.output_queue.put_nowait(output_data)
                except Exception as e:
                    pass
                    
        except Exception as e:
            logger.error(f"❌ Preprocessing error: {e}")
        finally:
            self.is_running = False
            self.detector.close()
            while not self.frame_queue.empty():
                try: self.frame_queue.get_nowait()
                except: break
            logger.info("🔧 Preprocessing thread stopped cleanly")

    def _extract_pose_from_matrix(self, matrix):
        """Converts internal MediaPipe 4x4 matrix into Pitch, Yaw, Roll."""
        r_mat = matrix[:3, :3]
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(r_mat)
        return angles[0], angles[1], angles[2]
        
    def _extract_eyes(self, frame, landmarks, w, h):
        """Extract precisely aligned bounding boxes around eyes."""
        # Standard landmarks for right & left eye boundaries contour
        r_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        l_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        def crop_polygon(indices):
            # x and y represent normalized landmarks.
            pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices])
            x_min, y_min = np.min(pts, axis=0) - 10
            x_max, y_max = np.max(pts, axis=0) + 10
            
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            
            crop = frame[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                crop = np.zeros((self.config.EYE_ROI_SIZE[1], self.config.EYE_ROI_SIZE[0], 3), dtype=np.uint8)
            else:
                crop = cv2.resize(crop, self.config.EYE_ROI_SIZE)
            return crop
            
        right_eye = crop_polygon(r_indices)
        left_eye = crop_polygon(l_indices)
        return left_eye, right_eye