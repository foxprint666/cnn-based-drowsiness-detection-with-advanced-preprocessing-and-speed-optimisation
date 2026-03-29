"""
Advanced Real-Time Driver Drowsiness Detection Pipeline
Main async orchestration with robust parallel processing threads.
"""

import threading
import queue
import time
import logging
from collections import deque

from config import Config
from pipeline.capture import CameraCapture
from pipeline.preprocessing import PreprocessingPipeline
from pipeline.inference import InferencePipeline
from pipeline.alerting import AlertManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrowsinessDetectionPipeline:
    """
    Async pipeline orchestrator with four parallel threads:
    - Camera Capture → Preprocessing → Inference → Alerting
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.is_running = False
        self.stop_event = threading.Event()
        
        # Thread-safe queues
        self.frame_queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        self.processed_queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        self.inference_queue = queue.Queue(maxsize=config.QUEUE_MAXSIZE)
        
        # Pipeline modules
        self.capture = CameraCapture(config, self.frame_queue)
        self.preprocessor = PreprocessingPipeline(config, self.frame_queue, self.processed_queue)
        self.inferencer = InferencePipeline(config, self.processed_queue, self.inference_queue)
        self.alerter = AlertManager(config, self.inference_queue)
        
        self.last_time = time.time()
        
    def start(self):
        """Launch all threads safely."""
        self.is_running = True
        self.stop_event.clear()
        
        # Pre-set components to prevent race conditions during startup validation
        self.capture.is_running = True
        self.preprocessor.is_running = True
        self.inferencer.is_running = True
        self.alerter.is_running = True
        
        logger.info("🚀 Starting Advanced Drowsiness Detection Pipeline...")
        
        self.threads = [
            threading.Thread(target=self._run_module, args=(self.capture,), daemon=True, name="CaptureThread"),
            threading.Thread(target=self._run_module, args=(self.preprocessor,), daemon=True, name="PreprocessThread"),
            threading.Thread(target=self._run_module, args=(self.inferencer,), daemon=True, name="InferenceThread"),
            threading.Thread(target=self._run_module, args=(self.alerter,), daemon=True, name="AlertThread"),
        ]
        
        for t in self.threads:
            t.start()
            
        self._monitor_loop()
        
    def _run_module(self, module):
        module.run()
        
    def _monitor_loop(self):
        """Monitor queue health and manage graceful breakdown."""
        try:
            while self.is_running and not self.stop_event.is_set():
                
                # Check if UI window was closed by user
                if not self.alerter.is_running:
                    self.stop()
                    break
                    
                current_time = time.time()
                elapsed = current_time - self.last_time
                
                if elapsed >= 1.0:  # Every 1 second
                    q_health = f"Q:[Frames:{self.frame_queue.qsize()} Preps:{self.processed_queue.qsize()} Infer:{self.inference_queue.qsize()}]"
                    logger.info(f"Pipeline Health | {q_health}")
                    self.last_time = current_time
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("⛔ Keyboard Interrupt received")
        finally:
            if self.is_running:
                self.stop()
            
    def stop(self):
        """Safely shutdown all modules and prevent deadlocks by flushing queues."""
        logger.info("Initiating safe shutdown...")
        self.is_running = False
        self.stop_event.set()
        
        # Stop modules
        self.capture.stop()
        self.preprocessor.is_running = False
        self.inferencer.is_running = False
        self.alerter.is_running = False
        
        # Flush all queues to unblock threads hanging on .put() or .get()
        for q in [self.frame_queue, self.processed_queue, self.inference_queue]:
            while not q.empty():
                try: 
                    q.get_nowait()
                except queue.Empty: 
                    break
                    
        logger.info("✅ Pipeline gracefully stopped.")

if __name__ == "__main__":
    cfg = Config()
    pipeline = DrowsinessDetectionPipeline(cfg)
    pipeline.start()