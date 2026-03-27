# Driver Drowsiness Detection Pipeline

A real-time driver drowsiness detection system using computer vision and deep learning. The pipeline processes camera frames through multiple stages: face detection, eye tracking, drowsiness classification, and alerting.

## Features

- **Real-time Processing**: Multi-threaded async pipeline with parallel processing
- **Face Detection**: MediaPipe-based face detection with ROI stabilization
- **Eye Tracking**: Optical flow-based eye landmark tracking with Kalman filtering
- **Image Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization) and gamma correction
- **Drowsiness Detection**: CNN-based eye state classification using EAR (Eye Aspect Ratio)
- **Alert System**: Configurable audio and visual alerts

## Pipeline Architecture

```
Camera Capture → Preprocessing → Inference → Alerting
     ↓               ↓              ↓           ↓
  Frame Q       Processed Q    Inference Q   Alerts
```

### Threads
1. **CameraThread**: Captures frames from camera
2. **PreprocessThread**: Face detection, eye tracking, image enhancement
3. **InferenceThread**: CNN-based drowsiness classification
4. **AlertThread**: Manages alerts based on inference results

## Requirements

```
opencv-python>=4.5.0
numpy>=1.19.0
mediapipe>=0.8.0
torch>=1.9.0
playsound>=1.3.0
```

Install dependencies:
```bash
pip install opencv-python numpy mediapipe torch playsound
```

## Usage

```bash
python main.py
```

## Configuration

Edit `config.py` to customize:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `CAMERA_ID` | Camera device ID | 0 |
| `CAMERA_WIDTH` | Frame width | 640 |
| `CAMERA_HEIGHT` | Frame height | 480 |
| `EAR_THRESHOLD` | Eye aspect ratio threshold | 0.2 |
| `CLOSED_FRAMES_THRESHOLD` | Frames for closed eye detection | 15 |
| `USE_CUDA` | Enable GPU acceleration | False |

## Project Structure

```
.
├── config.py                 # Configuration and hyperparameters
├── main.py                   # Main pipeline orchestration
├── pipeline/
│   ├── capture.py            # Camera capture module
│   ├── preprocessing.py      # Face detection & image enhancement
│   ├── inference.py          # CNN inference pipeline
│   └── alerting.py           # Alert management
└── models/                   # Model weights (add your models here)
```

## License

MIT License
