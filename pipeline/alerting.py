"""Alert management and visualization."""

import cv2
import queue
import logging
import time
import platform
import os

logger = logging.getLogger(__name__)


class AlertManager:
    """Handles drowsiness alerts and debug visualization."""
    
    def __init__(self, config, input_queue: queue.Queue):
        self.config = config
        self.input_queue = input_queue
        self.is_running = False
        self.last_alert_time = 0
        self.alert_cooldown = 1.0  # seconds
        
    def run(self):
        """Main alerting loop."""
        logger.info("🔔 Alert thread started")
        self.is_running = True
        
        try:
            while self.is_running:
                try:
                    data = self.input_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if data['is_drowsy']:
                    self._trigger_alert()
                
                if self.config.DISPLAY_DEBUG_WINDOW:
                    self._display_debug_window(data)
                
        except Exception as e:
            logger.error(f"❌ Alert error: {e}")
        finally:
            cv2.destroyAllWindows()
            self.is_running = False
            logger.info("🔔 Alert thread stopped")
    
    def _trigger_alert(self):
        """Sound alarm if cooldown elapsed."""
        current_time = time.time()
        
        if current_time - self.last_alert_time >= self.alert_cooldown:
            logger.warning("🚨 DROWSINESS DETECTED!")
            self._play_sound()
            self.last_alert_time = current_time
    
    def _play_sound(self):
        """Cross-platform system beep."""
        try:
            if platform.system() == "Windows":
                import winsound
                # Frequency (Hz), Duration (ms)
                winsound.Beep(1000, int(self.config.ALERT_SOUND_DURATION * 1000))
            else:
                # Unix/Linux/macOS
                os.system(f"beep -f 1000 -l {int(self.config.ALERT_SOUND_DURATION * 1000)}")
        except Exception as e:
            logger.warning(f"Could not play sound: {e}")
    
    def _display_debug_window(self, data):
        """Show ROI, eye regions, and metrics."""
        frame = data['original_frame'].copy()
        roi = data['roi_frame']
        
        # Draw info
        h, w = frame.shape[:2]
        y_pos = 30
        
        text_data = [
            f"Frame: {data['frame_id']}",
            f"L-Eye Open: {data['left_eye_open_prob']:.2f}",
            f"R-Eye Open: {data['right_eye_open_prob']:.2f}",
            f"Closed Frames: {data['closed_frames']}",
            f"Status: {'🛑 DROWSY' if data['is_drowsy'] else '✅ ALERT'}",
        ]
        
        for text in text_data:
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 25
        
        # Resize for display
        display = cv2.resize(frame, self.config.DEBUG_WINDOW_SIZE)
        cv2.imshow("Drowsiness Detection", display)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.is_running = False