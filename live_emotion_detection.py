import cv2
import time
from deepface import DeepFace
import numpy as np

class LiveEmotionDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.emotion_history = []
        self.current_emotion = "Neutral"
        self.current_confidence = 0
        self.current_emotions = {}
        self.face_region = None
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        self.last_analysis_time = 0
        
        # Camera settings for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
    def draw_emotion_info(self, frame, emotion_data):
        """Draw emotion information on the frame"""
        if emotion_data and 'region' in emotion_data:
            region = emotion_data['region']
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']
            
            # Extract coordinates safely
            x = region.get('x', 0)
            y = region.get('y', 0)
            w = region.get('w', 0)
            h = region.get('h', 0)
            
            # Draw face bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw emotion text with background
            text = f"{emotion}: {confidence:.1f}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (x, y - text_size[1] - 10), 
                         (x + text_size[0], y), (0, 255, 0), -1)
            cv2.putText(frame, text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            return emotion, confidence
        return None, 0
    
    def draw_emotion_bars(self, frame, emotions):
        """Draw emotion confidence bars"""
        if not emotions:
            return
            
        bar_width = 100
        bar_height = 15
        start_x = frame.shape[1] - bar_width - 10
        start_y = 10
        
        # Show only top emotions for performance
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:6]
        
        for i, (emotion, confidence) in enumerate(top_emotions):
            y_pos = start_y + (i * (bar_height + 5))
            
            # Emotion name
            emotion_short = emotion[:4] if emotion != 'surprise' else 'surp'
            cv2.putText(frame, emotion_short, (start_x - 30, y_pos + bar_height - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Confidence bar
            bar_fill = int((confidence / 100) * bar_width)
            color = self.get_emotion_color(emotion)
            cv2.rectangle(frame, (start_x, y_pos), 
                         (start_x + bar_width, y_pos + bar_height), (50, 50, 50), -1)
            cv2.rectangle(frame, (start_x, y_pos), 
                         (start_x + bar_fill, y_pos + bar_height), color, -1)
            
            # Confidence percentage
            cv2.putText(frame, f"{confidence:.1f}%", (start_x + bar_width + 5, y_pos + bar_height - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
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
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        current_time = time.time()
        time_diff = current_time - self.last_time
        
        if time_diff >= 1.0:
            self.fps = self.frame_count / time_diff
            self.frame_count = 0
            self.last_time = current_time
    
    def run(self):
        """Main loop for live emotion detection"""
        print("=== Live Emotion Detection Started ===")
        print("Instructions:")
        print("- Make sure you're in a well-lit area")
        print("- Face the camera directly")
        print("- Press 'q' to quit")
        print("- Press 'c' to capture current frame")
        print("- Press '1' to slow down analysis (more accurate)")
        print("- Press '2' to speed up analysis (less accurate)")
        print("======================================")
        
        processing_interval = 0.5  # Process every 0.5 seconds
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read from webcam")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Calculate FPS
            self.calculate_fps()
            
            current_time = time.time()
            
            # Process emotion analysis at intervals
            if current_time - self.last_analysis_time >= processing_interval:
                try:
                    analysis = DeepFace.analyze(
                        frame, 
                        actions=['emotion'], 
                        enforce_detection=False, 
                        silent=True
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
                        
                        # Draw emotion info on frame
                        self.draw_emotion_info(frame, {
                            'emotion': self.current_emotion,
                            'confidence': self.current_confidence,
                            'region': self.face_region
                        })
                        
                        # Update emotion history
                        self.emotion_history.append(self.current_emotion)
                        if len(self.emotion_history) > 10:
                            self.emotion_history.pop(0)
                        
                        self.last_analysis_time = current_time
                        
                except Exception as e:
                    # Skip analysis errors and continue
                    pass
            
            # Draw emotion bars
            self.draw_emotion_bars(frame, self.current_emotions)
            
            # Display FPS and current emotion
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Emotion: {self.current_emotion}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {self.current_confidence:.1f}%", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show recent emotion history
            if self.emotion_history:
                history_text = f"Recent: {', '.join(self.emotion_history[-3:])}"
                cv2.putText(frame, history_text, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Display instructions
            cv2.putText(frame, "Press 'q': Quit | 'c': Capture | '1/2': Speed", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Show the frame
            cv2.imshow('Live Emotion Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Capture current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"emotion_capture_{timestamp}_{self.current_emotion}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Capture saved: {filename}")
            elif key == ord('1'):
                # Slow down analysis (more accurate)
                processing_interval = min(1.0, processing_interval + 0.1)
                print(f"Analysis speed: {1/processing_interval:.1f} FPS")
            elif key == ord('2'):
                # Speed up analysis (less accurate)
                processing_interval = max(0.2, processing_interval - 0.1)
                print(f"Analysis speed: {1/processing_interval:.1f} FPS")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("Emotion detection stopped.")

def main():
    detector = LiveEmotionDetector()
    detector.run()

if __name__ == "__main__":
    main()