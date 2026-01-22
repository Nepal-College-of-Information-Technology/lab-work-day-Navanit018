import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import time
from deepface import DeepFace
import threading
import os

class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Emotion Detector Pro")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Application variables
        self.cap = None
        self.is_running = False
        self.current_emotion = "Neutral"
        self.current_confidence = 0
        self.emotion_history = []
        self.photo_path = None
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Create the user interface
        self.create_ui()
        
    def create_ui(self):
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Camera and controls
        self.create_camera_panel(main_frame)
        
        # Right panel - Analytics and information
        self.create_analytics_panel(main_frame)
        
    def create_camera_panel(self, parent):
        # Camera frame
        camera_frame = ttk.LabelFrame(parent, text="Live Camera Feed", padding=15)
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video display
        self.video_label = tk.Label(camera_frame, background='#000000', 
                                   relief='sunken', borderwidth=2)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Camera controls
        controls_frame = ttk.Frame(camera_frame)
        controls_frame.pack(fill=tk.X, pady=15)
        
        # Start/Stop buttons
        self.start_btn = ttk.Button(controls_frame, text="🎥 Start Camera", 
                                   command=self.start_camera, width=15)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(controls_frame, text="⏹️ Stop Camera", 
                                  command=self.stop_camera, state=tk.DISABLED, width=15)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.capture_btn = ttk.Button(controls_frame, text="📸 Capture", 
                                     command=self.capture_photo, state=tk.DISABLED, width=15)
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        # FPS display
        self.fps_label = ttk.Label(controls_frame, text="FPS: 0", foreground='#3498db')
        self.fps_label.pack(side=tk.RIGHT, padx=10)
        
    def create_analytics_panel(self, parent):
        # Analytics frame
        analytics_frame = ttk.LabelFrame(parent, text="Emotion Analytics", padding=15)
        analytics_frame.pack(side=tk.RIGHT, fill=tk.BOTH, ipadx=10)
        
        # Current emotion display
        self.create_emotion_display(analytics_frame)
        
        # Emotion statistics
        self.create_statistics(analytics_frame)
        
        # Emotion history
        self.create_history(analytics_frame)
        
        # Image analysis section
        self.create_image_analysis(analytics_frame)
        
    def create_emotion_display(self, parent):
        emotion_frame = ttk.LabelFrame(parent, text="Current Emotion", padding=10)
        emotion_frame.pack(fill=tk.X, pady=5)
        
        self.emotion_label = tk.Label(emotion_frame, text="NEUTRAL", 
                                     font=('Arial', 28, 'bold'), 
                                     foreground='#3498db',
                                     background='#ecf0f1')
        self.emotion_label.pack(pady=15, fill=tk.X)
        
        self.confidence_label = tk.Label(emotion_frame, text="Confidence: 0%", 
                                        font=('Arial', 14),
                                        foreground='#7f8c8d')
        self.confidence_label.pack(pady=5)
        
    def create_statistics(self, parent):
        stats_frame = ttk.LabelFrame(parent, text="Emotion Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create progress bars for each emotion
        self.emotion_bars = {}
        emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
        
        for emotion in emotions:
            emotion_row = ttk.Frame(stats_frame)
            emotion_row.pack(fill=tk.X, pady=3)
            
            # Emotion name
            name_label = ttk.Label(emotion_row, text=emotion.title(), width=12)
            name_label.pack(side=tk.LEFT)
            
            # Progress bar
            progress = ttk.Progressbar(emotion_row, orient=tk.HORIZONTAL, 
                                      length=180, mode='determinate')
            progress.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            # Percentage label
            value_label = ttk.Label(emotion_row, text="0%", width=6)
            value_label.pack(side=tk.RIGHT)
            
            self.emotion_bars[emotion] = {
                'bar': progress,
                'label': value_label,
                'name_label': name_label
            }
            
            # Set emotion-specific colors
            self.set_emotion_color(emotion)
    
    def set_emotion_color(self, emotion):
        colors = {
            'happy': '#f39c12',    # Orange
            'sad': '#3498db',      # Blue
            'angry': '#e74c3c',    # Red
            'surprise': '#9b59b6', # Purple
            'fear': '#34495e',     # Dark blue
            'disgust': '#27ae60',  # Green
            'neutral': '#95a5a6'   # Gray
        }
        
        if emotion in colors:
            style_name = f"{emotion}.Horizontal.TProgressbar"
            style = ttk.Style()
            style.configure(style_name, troughcolor='#ecf0f1', background=colors[emotion])
            self.emotion_bars[emotion]['bar'].configure(style=style_name)
            self.emotion_bars[emotion]['name_label'].configure(foreground=colors[emotion])
    
    def create_history(self, parent):
        history_frame = ttk.LabelFrame(parent, text="Emotion History", padding=10)
        history_frame.pack(fill=tk.BOTH, pady=5)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(history_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.history_text = tk.Text(text_frame, height=8, width=35, 
                                   font=('Consolas', 10),
                                   bg='#f8f9fa', fg='#2c3e50')
        
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, 
                                 command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=scrollbar.set)
        
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_image_analysis(self, parent):
        image_frame = ttk.LabelFrame(parent, text="Image Analysis", padding=10)
        image_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(image_frame, text="📁 Open Image", 
                  command=self.open_image).pack(fill=tk.X, pady=2)
        
        ttk.Button(image_frame, text="🔍 Analyze Image", 
                  command=self.analyze_image).pack(fill=tk.X, pady=2)
        
        self.image_path_label = ttk.Label(image_frame, text="No image selected", 
                                         foreground='#95a5a6', font=('Arial', 9))
        self.image_path_label.pack(fill=tk.X, pady=5)
        
    def start_camera(self):
        """Start the camera and emotion detection"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not access webcam. Please check if it's connected.")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.NORMAL)
        
        # Start video processing in separate thread
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()
        
        messagebox.showinfo("Camera Started", "Live emotion detection is now active!")
        
    def stop_camera(self):
        """Stop the camera and clean up"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.DISABLED)
        
        # Clear video feed
        self.video_label.configure(image='')
        self.fps_label.config(text="FPS: 0")
        
    def process_video(self):
        """Process video frames and detect emotions"""
        last_analysis_time = 0
        
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Calculate FPS
            self.calculate_fps()
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            current_time = time.time()
            
            # Analyze emotion every 0.5 seconds to avoid overloading
            if current_time - last_analysis_time > 0.5:
                try:
                    analysis = DeepFace.analyze(frame, actions=['emotion'], 
                                              enforce_detection=False, silent=True)
                    
                    if analysis:
                        if isinstance(analysis, list):
                            face_data = analysis[0]
                        else:
                            face_data = analysis
                        
                        emotions = face_data['emotion']
                        self.current_emotion = face_data['dominant_emotion']
                        self.current_confidence = emotions[self.current_emotion]
                        region = face_data['region']
                        
                        # Update UI in main thread
                        self.root.after(0, self.update_emotion_display, emotions)
                        
                        # Draw face rectangle on frame
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Add emotion text
                        text = f"{self.current_emotion} ({self.current_confidence:.1f}%)"
                        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 255, 0), 2)
                        
                        last_analysis_time = current_time
                        
                except Exception as e:
                    # Skip analysis errors
                    pass
            
            # Convert frame for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            
            # Update video feed in main thread
            self.root.after(0, self.update_video_feed, photo)
            
            time.sleep(0.03)  # Control frame rate
            
    def calculate_fps(self):
        """Calculate and update FPS"""
        self.frame_count += 1
        current_time = time.time()
        time_diff = current_time - self.last_fps_time
        
        if time_diff >= 1.0:
            self.fps = self.frame_count / time_diff
            self.frame_count = 0
            self.last_fps_time = current_time
            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
            
    def update_video_feed(self, photo):
        """Update the video feed in the GUI"""
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # Keep a reference
        
    def update_emotion_display(self, emotions):
        """Update emotion information in the GUI"""
        # Update current emotion with color
        emotion_colors = {
            'happy': '#f39c12',    # Orange
            'sad': '#3498db',      # Blue
            'angry': '#e74c3c',    # Red
            'surprise': '#9b59b6', # Purple
            'fear': '#34495e',     # Dark blue
            'disgust': '#27ae60',  # Green
            'neutral': '#95a5a6'   # Gray
        }
        
        color = emotion_colors.get(self.current_emotion, '#3498db')
        self.emotion_label.configure(text=self.current_emotion.upper(), foreground=color)
        self.confidence_label.configure(text=f"Confidence: {self.current_confidence:.1f}%")
        
        # Update progress bars
        for emotion, confidence in emotions.items():
            if emotion in self.emotion_bars:
                self.emotion_bars[emotion]['bar']['value'] = confidence
                self.emotion_bars[emotion]['label']['text'] = f"{confidence:.1f}%"
        
        # Update history
        timestamp = time.strftime("%H:%M:%S")
        history_entry = f"[{timestamp}] {self.current_emotion}: {self.current_confidence:.1f}%\n"
        self.history_text.insert(tk.END, history_entry)
        self.history_text.see(tk.END)
        
        # Keep only last 15 entries
        lines = self.history_text.get(1.0, tk.END).split('\n')
        if len(lines) > 15:
            self.history_text.delete(1.0, tk.END)
            self.history_text.insert(tk.END, '\n'.join(lines[-15:]))
        
    def capture_photo(self):
        """Capture and save the current frame"""
        if self.cap and self.is_running:
            ret, frame = self.cap.read()
            if ret:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"emotion_capture_{timestamp}_{self.current_emotion}.jpg"
                cv2.imwrite(filename, frame)
                messagebox.showinfo("Photo Captured", f"Photo saved as:\n{filename}")
                
    def open_image(self):
        """Open an image file for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.photo_path = file_path
            self.image_path_label.configure(
                text=f"Selected: {os.path.basename(file_path)}"
            )
            
    def analyze_image(self):
        """Analyze emotion in selected image"""
        if not self.photo_path:
            messagebox.showwarning("Warning", "Please select an image file first.")
            return
            
        try:
            analysis = DeepFace.analyze(self.photo_path, actions=['emotion'], silent=True)
            
            if analysis:
                if isinstance(analysis, list):
                    face_data = analysis[0]
                else:
                    face_data = analysis
                
                emotions = face_data['emotion']
                dominant_emotion = face_data['dominant_emotion']
                confidence = emotions[dominant_emotion]
                
                # Create results message
                result_text = f"📊 Image Analysis Results\n\n"
                result_text += f"🎭 Dominant Emotion: {dominant_emotion}\n"
                result_text += f"📈 Confidence: {confidence:.1f}%\n\n"
                result_text += "Detailed Analysis:\n"
                
                for emotion, score in emotions.items():
                    result_text += f"• {emotion.title()}: {score:.1f}%\n"
                
                messagebox.showinfo("Analysis Complete", result_text)
                
        except Exception as e:
            messagebox.showerror("Analysis Error", 
                               f"Could not analyze the image:\n{str(e)}")
            
    def on_closing(self):
        """Handle application closing"""
        self.stop_camera()
        self.root.destroy()

def main():
    """Main function to start the application"""
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()