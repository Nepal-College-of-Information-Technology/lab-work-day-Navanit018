import cv2
import time
from deepface import DeepFace
import threading
import queue
import numpy as np

class HighFPSEmotionDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.emotion_history = []
        self.current_emotion = "Neutral"
        self.current_confidence = 0
        self.current_emotions = {}
        self.face_region = None
        
        # Performance optimization settings
        self.processing_interval = 0.3  # Process every 0.3 seconds
        self.last_processing_time = 0
        self.frame_queue = queue.Queue(maxsize=1)
        self.latest_frame = None
        self.processing = False
        self.fps = 0
        self.frame_count = 0
        self.fps_update_time = time.time()
        
        # Camera optimization
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
        
        # Reduce processing resolution for faster analysis
        self.process_width = 320
        
        print("Initializing High-FPS Emotion Detector...")
        
    def process_frame_async(self, frame):
        """Process frame in a separate thread to avoid blocking"""
        if self.processing or self.frame_queue.full():
            return
            
        try:
            # Reduce resolution for faster processing
            small_frame = cv2.resize(frame, (self.process_width, 
                                           int(frame.shape[0] * self.process_width / frame.shape[1])))
            
            self.frame_queue.put(small_frame)
        except:
            pass
    
    def emotion_analysis_worker(self):
        """Worker thread for emotion analysis"""
        while True:
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    break
                    
                self.processing = True
                analysis = DeepFace.analyze(
                    frame, 
                    actions=['emotion'], 
                    enforce_detection=False, 
                    silent=True,
                    detector_backend='opencv'  # Faster than default
                )
                
                if analysis:
                    if isinstance(analysis, list):
                        face_data = analysis[0]
                    else:
                        face_data = analysis
                    
                    self.current_emotions = face_data['emotion']
                    self.current_emotion = face_data['dominant_emotion']
                    self.current_confidence = self.current_emotions[self.current_emotion]
                    self.face_region = face_data['region']
                    
                    # Scale region back to original frame size
                    if self.face_region:
                        scale_x = 640 / self.process_width
                        scale_y = 480 / (frame.shape[0])
                        self.face_region['x'] = int(self.face_region['x'] * scale_x)
                        self.face_region['y'] = int(self.face_region['y'] * scale_y)
                        self.face_region['w'] = int(self.face_region['w'] * scale_x)
                        self.face_region['h'] = int(self.face_region['h'] * scale_y)
                    
                    self.emotion_history.append(self.current_emotion)
                    if len(self.emotion_history) > 15:
                        self.emotion_history.pop(0)
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Analysis error: {e}")
            finally:
                self.processing = False
                self.frame_queue.task_done()
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        current_time = time.time()
        time_diff = current_time - self.fps_update_time
        
        if time_diff >= 1.0:  # Update FPS every second
            self.fps = self.frame_count / time_diff
            self.frame_count = 0
            self.fps_update_time = current_time
    
    def draw_optimized_ui(self, frame):
        """Draw optimized UI elements"""
        # Draw face bounding box
        if self.face_region:
            x, y, w, h = self.face_region.values()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Simple emotion text
            text = f"{self.current_emotion} ({self.current_confidence:.0f}%)"
            cv2.putText(frame, text, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # FPS display (always visible)
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Current emotion
        emotion_text = f"Emotion: {self.current_emotion}"
        cv2.putText(frame, emotion_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Processing status
        status = "PROCESSING..." if self.processing else "READY"
        status_color = (0, 0, 255) if self.processing else (0, 255, 0)
        cv2.putText(frame, f"Status: {status}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Recent history (simplified)
        if len(self.emotion_history) > 0:
            recent = ", ".join(self.emotion_history[-3:])
            cv2.putText(frame, f"Recent: {recent}", (10, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit | 'c' to capture", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_lightweight_bars(self, frame):
        """Draw lightweight emotion bars"""
        if not self.current_emotions:
            return
            
        bar_width = 80
        bar_height = 12
        start_x = frame.shape[1] - bar_width - 10
        start_y = 10
        
        # Show only top 5 emotions for performance
        top_emotions = sorted(self.current_emotions.items(), 
                             key=lambda x: x[1], reverse=True)[:5]
        
        for i, (emotion, confidence) in enumerate(top_emotions):
            y_pos = start_y + (i * (bar_height + 3))
            
            # Emotion name (abbreviated)
            emotion_short = emotion[:4] if emotion != 'surprise' else 'surp'
            cv2.putText(frame, emotion_short, (start_x - 25, y_pos + bar_height - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Confidence bar
            bar_fill = int((confidence / 100) * bar_width)
            color = self.get_emotion_color(emotion)
            cv2.rectangle(frame, (start_x, y_pos), 
                         (start_x + bar_width, y_pos + bar_height), (50, 50, 50), -1)
            cv2.rectangle(frame, (start_x, y_pos), 
                         (start_x + bar_fill, y_pos + bar_height), color, -1)
            
            # Confidence percentage
            cv2.putText(frame, f"{confidence:.0f}%", (start_x + bar_width + 5, y_pos + bar_height - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    def get_emotion_color(self, emotion):
        """Get color for each emotion"""
        colors = {
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'surprise': (255, 255, 0), # Cyan
            'fear': (255, 0, 255),     # Magenta
            'disgust': (0, 255, 0),    # Green
            'neutral': (255, 255, 255) # White
        }
        return colors.get(emotion, (255, 255, 255))
    
    def run(self):
        """Main loop for high-FPS emotion detection"""
        # Start analysis thread
        analysis_thread = threading.Thread(target=self.emotion_analysis_worker, daemon=True)
        analysis_thread.start()
        
        print("High-FPS Emotion Detection Started!")
        print("=== PERFORMANCE OPTIMIZATIONS ===")
        print("✓ Lower processing resolution")
        print("✓ Async background processing")
        print("✓ Reduced UI complexity")
        print("✓ Optimized camera settings")
        print("✓ Faster face detection backend")
        print("================================")
        
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read from webcam")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            self.latest_frame = frame.copy()
            
            # Calculate FPS
            self.calculate_fps()
            
            # Process emotion analysis at intervals (non-blocking)
            current_time = time.time()
            if current_time - self.last_processing_time >= self.processing_interval:
                self.process_frame_async(frame)
                self.last_processing_time = current_time
            
            # Draw optimized UI
            self.draw_optimized_ui(frame)
            self.draw_lightweight_bars(frame)
            
            # Display frame
            cv2.imshow('High-FPS Emotion Detection', frame)
            
            # Calculate actual processing time for this frame
            processing_time = time.time() - start_time
            actual_fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Capture frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"capture_{timestamp}_{self.current_emotion}.jpg"
                cv2.imwrite(filename, self.latest_frame)
                print(f"Capture saved: {filename}")
            elif key == ord('1'):
                # Increase processing speed (lower quality)
                self.processing_interval = min(0.5, self.processing_interval + 0.1)
                print(f"Slower processing: {1/self.processing_interval:.1f} FPS analysis")
            elif key == ord('2'):
                # Decrease processing speed (higher quality)
                self.processing_interval = max(0.1, self.processing_interval - 0.1)
                print(f"Faster processing: {1/self.processing_interval:.1f} FPS analysis")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("High-FPS emotion detection stopped.")

# Run the detector
if __name__ == "__main__":
    detector = HighFPSEmotionDetector()
    detector.run()