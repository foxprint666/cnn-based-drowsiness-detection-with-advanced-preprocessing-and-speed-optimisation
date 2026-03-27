"""
Real-Time Driver Drowsiness Detection Pipeline
Main async orchestration with parallel processing threads.
"""

import cv2
import numpy as np
import threading
import queue
import time
import logging
from collections import deque
from pathlib import Path
import sys

# Local imports
from config import Config
from pipeline.capture import CameraCapture
from pipeline.preprocessing import PreprocessingPipeline
from pipeline.inference import InferencePipeline
from pipeline.alerting import AlertManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrowsinessDetectionPipeline:
    """
    Async pipeline orchestrator with three parallel threads:
    - Camera Capture → Preprocessing → Inference → Alerting
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.is_running = False
        
        # Thread-safe queues
        self.frame_queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        self.processed_queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        self.inference_queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        
        # Modules
        self.capture = CameraCapture(config, self.frame_queue)
        self.preprocessor = PreprocessingPipeline(config, self.frame_queue, self.processed_queue)
        self.inferencer = InferencePipeline(config, self.processed_queue, self.inference_queue)
        self.alerter = AlertManager(config, self.inference_queue)
        
        # Metrics
        self.fps_deque = deque(maxlen=30)
        self.last_time = time.time()
        
    def start(self):
        """Launch all threads."""
        self.is_running = True
        logger.info("🚀 Starting Drowsiness Detection Pipeline")
        
        threads = [
            threading.Thread(target=self.capture.run, daemon=True, name="CameraThread"),
            threading.Thread(target=self.preprocessor.run, daemon=True, name="PreprocessThread"),
            threading.Thread(target=self.inferencer.run, daemon=True, name="InferenceThread"),
            threading.Thread(target=self.alerter.run, daemon=True, name="AlertThread"),
        ]
        
        for thread in threads:
            thread.start()
        
        # Main monitoring loop
        self._monitor_loop()
    
    def _monitor_loop(self):
        """Monitor FPS, queue health, and graceful shutdown."""
        try:
            while self.is_running:
                current_time = time.time()
                elapsed = current_time - self.last_time
                
                if elapsed >= 1.0:  # Every 1 second
                    avg_fps = len(self.fps_deque) / (elapsed + 1e-6)
                    queue_health = {
                        'frame_q': self.frame_queue.qsize(),
                        'process_q': self.processed_queue.qsize(),
                        'infer_q': self.inference_queue.qsize(),
                    }
                    
                    logger.info(f"FPS: {avg_fps:.1f} | Queue Health: {queue_health}")
                    self.last_time = current_time
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("⛔ Shutdown signal received")
            self.stop()
    
    def stop(self):
        """Graceful shutdown."""
        self.is_running = False
        self.capture.stop()
        logger.info("✅ Pipeline stopped")


if __name__ == "__main__":
    cfg = Config()
    pipeline = DrowsinessDetectionPipeline(cfg)
    pipeline.start()