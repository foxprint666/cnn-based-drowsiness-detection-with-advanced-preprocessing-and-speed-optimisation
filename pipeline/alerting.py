"""
Alert management and visualization, with new Head Pose & Yawning states.
"""

import cv2
import queue
import logging
import time
import platform
import os

logger = logging.getLogger(__name__)


class AlertManager:
    """Handles visual display and sound alerts for various distracted states."""
    
    def __init__(self, config, input_queue: queue.Queue):
        self.config = config
        self.input_queue = input_queue
        self.is_running = False
        self.last_alert_time = 0.0
        self.alert_cooldown = 1.0  # seconds
        
    def run(self):
        logger.info("🔔 Alert thread started")
        self.is_running = True
        
        try:
            while self.is_running:
                try:
                    data = self.input_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Check for critical errors requiring sound
                if data['is_drowsy'] or data['is_distracted']:
                    self._trigger_alert()
                
                if self.config.DISPLAY_DEBUG_WINDOW:
                    self._display_debug_window(data)
                
        except Exception as e:
            logger.error(f"❌ Alert error: {e}")
        finally:
            cv2.destroyAllWindows()
            self.is_running = False
            # Clear queue
            while not self.input_queue.empty():
                try: self.input_queue.get_nowait()
                except: break
            logger.info("🔔 Alert thread stopped cleanly")
    
    def _trigger_alert(self):
        current_time = time.time()
        
        if current_time - self.last_alert_time >= self.alert_cooldown:
            logger.warning("🚨 ALERT TRIGGERED (Drowsy or Distracted)!")
            self._play_sound()
            self.last_alert_time = current_time
    
    def _play_sound(self):
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, int(self.config.ALERT_SOUND_DURATION * 1000))
            else:
                os.system(f"beep -f 1000 -l {int(self.config.ALERT_SOUND_DURATION * 1000)}")
        except Exception as e:
            pass
    
    def _display_debug_window(self, data):
        frame = data['original_frame'].copy()
        
        h, w = frame.shape[:2]
        y_pos = 30
        
        # Determine overall status text - prioritize Distraction as per user feedback
        if data.get('is_distracted', False):
            status = "⚠️ DISTRACTED"
            color = (0, 165, 255) # Orange
        elif data.get('is_drowsy', False):
            status = "🛑 ASLEEP!"
            color = (0, 0, 255)
        elif data['is_yawning']:
            status = "🥱 YAWNING"
            color = (0, 255, 255) # Yellow
        else:
            status = "✅ FOCUSED"
            color = (0, 255, 0)
            
        text_data = [
            f"Status: {status}",
            f"Frame: {data['frame_id']}",
            f"Avg Eye Open: {data['avg_open_prob']:.2f}",
            f"MAR (Yawn): {data['mar']:.2f}",
            f"Pose (P/Y/R): {data['pitch']:.0f}/{data['yaw']:.0f}/{data['roll']:.0f}"
        ]
        
        # Draw text
        for idx, text in enumerate(text_data):
            # Highlight status differently
            c = color if "Status" in text else (0, 255, 0)
            font_scale = 0.7 if "Status" in text else 0.5
            thickness = 2 if "Status" in text else 1
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, c, thickness)
            y_pos += 25
            
        # Draw eyes overlay locally if desired (for debugging)
        if data['left_eye'] is not None and data['right_eye'] is not None:
            l_eye_img = cv2.resize(data['left_eye'], (100, 50))
            r_eye_img = cv2.resize(data['right_eye'], (100, 50))
            # Put them in top right corner
            frame[10:60, w-210:w-110] = l_eye_img
            frame[10:60, w-100:w-0] = r_eye_img
            cv2.putText(frame, "Eye Crops", (w-180, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw ROI Box
        # In Preprocessing we set prev_roi. Let's just draw an indicator if face detected.
        if data['face_detected']:
            cv2.putText(frame, "Face Tracked", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "NO FACE DETECTED", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        # Resize for display
        display = cv2.resize(frame, self.config.DEBUG_WINDOW_SIZE) if self.config.DEBUG_WINDOW_SIZE != (w,h) else frame
        cv2.imshow("Drowsiness Detection AI", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.is_running = False