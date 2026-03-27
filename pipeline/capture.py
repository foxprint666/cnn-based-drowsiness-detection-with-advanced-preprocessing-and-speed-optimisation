"""Camera capture thread - minimalist, non-blocking."""

import cv2
import queue
import logging
import time

logger = logging.getLogger(__name__)


class CameraCapture:
    """Continuously captures frames from laptop camera."""
    
    def __init__(self, config, frame_queue: queue.Queue):
        self.config = config
        self.frame_queue = frame_queue
        self.cap = None
        self.is_running = False
        self.frame_count = 0
        
    def run(self):
        """Main capture loop."""
        try:
            self.cap = cv2.VideoCapture(self.config.CAMERA_ID)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
            
            logger.info(f"📷 Camera initialized: {self.config.CAMERA_WIDTH}x{self.config.CAMERA_HEIGHT}")
            self.is_running = True
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("⚠️ Failed to read frame")
                    continue
                
                self.frame_count += 1
                
                # Non-blocking queue put (drop frame if queue full)
                try:
                    self.frame_queue.put_nowait((self.frame_count, frame.copy()))
                except queue.Full:
                    logger.debug("Frame dropped (queue full)")
                    
        except Exception as e:
            logger.error(f"❌ Camera error: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Release camera."""
        self.is_running = False
        if self.cap:
            self.cap.release()
            logger.info("📷 Camera released")