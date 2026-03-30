"""Configuration and hyperparameters."""

class Config:
    # Camera settings
    CAMERA_ID = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    QUEUE_MAXSIZE = 2  # Keep pipeline responsive
    
    # ROI & Face Detection (MediaPipe)
    FACE_CONFIDENCE = 0.5
    FACE_DETECTION_INTERVAL = 5  # Re-detect every 5 frames
    
    # Stabilization (Kalman Filter)
    KALMAN_PROCESS_NOISE = 1e-4
    KALMAN_MEASUREMENT_NOISE = 1e-2
    OPTICAL_FLOW_TEMPLATE_SIZE = 50
    
    # Image Enhancement
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = 8
    GAMMA_CORRECTION = 0.5
    
    # Super-Resolution
    SRCNN_SCALE = 2
    ENABLE_SRCNN = False  # Set True if GPU available
    
    # CNN Inference
    EYE_ROI_SIZE = (128, 64)
    INFERENCE_INTERVAL = 3  # Every 3 frames
    MODEL_PATH = "models/mobilenet_v2_eye_state.pth"
    
    # Drowsiness Detection
    EAR_THRESHOLD = 0.2
    CLOSED_FRAMES_THRESHOLD = 20  # ~0.7s at 30 FPS
    DROWSY_FRAMES_THRESHOLD = 15
    
    # New Thresholds (Yawning, Head Pose via Tasks API Blendshapes & Matrix)
    MAR_THRESHOLD = 0.35  # Yawning (jawOpen blendshape, 0.0 - 1.0)
    YAWNING_FRAMES_THRESHOLD = 20
    PITCH_THRESHOLD = 45.0  # Degrees
    YAW_THRESHOLD = 45.0
    ROLL_THRESHOLD = 45.0
    HEAD_POSE_FRAMES_THRESHOLD = 15
    
    # Alerting
    ALERT_SOUND_DURATION = 0.5  # seconds
    DISPLAY_DEBUG_WINDOW = True
    DEBUG_WINDOW_SIZE = (320, 240)
    
    # Hardware
    USE_CUDA = False  # Set True if NVIDIA GPU available
    NUM_THREADS = 4