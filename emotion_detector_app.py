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
        self.root.title("AI Emotion Detector")
        self.root.geometry("1200x700")
        self.root.configure(bg='#2c3e50')
        
        # Variables
        self.cap = None
        self.is_running = False
        self.current_emotion = "Neutral"
        self.current_confidence = 0
        self.emotion_history = []
        self.photo_path = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Camera feed
        left_frame = ttk.LabelFrame(main_frame, text="Live Camera Feed", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = ttk.Label(left_frame, background='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Camera controls
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.capture_btn = ttk.Button(control_frame, text="Capture Photo", command=self.capture_photo, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Analytics
        right_frame = ttk.LabelFrame(main_frame, text="Emotion Analytics", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0), ipadx=10)
        
        # Current emotion display
        current_emotion_frame = ttk.LabelFrame(right_frame, text="Current Emotion", padding=10)
        current_emotion_frame.pack(fill=tk.X, pady=5)
        
        self.emotion_label = ttk.Label(current_emotion_frame, text="NEUTRAL", 
                                      font=('Arial', 24, 'bold'), foreground='#3498db')
        self.emotion_label.pack(pady=10)
        
        self.confidence_label = ttk.Label(current_emotion_frame, text="Confidence: 0%", 
                                         font=('Arial', 12))
        self.confidence_label.pack()
        
        # Emotion statistics
        stats_frame = ttk.LabelFrame(right_frame, text="Emotion Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create progress bars for each emotion
        self.emotion_bars = {}
        emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
        
        for emotion in emotions:
            emotion_frame = ttk.Frame(stats_frame)
            emotion_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(emotion_frame, text=emotion.title(), width=10).pack(side=tk.LEFT)
            
            progress = ttk.Progressbar(emotion_frame, orient=tk.HORIZONTAL, length=150, mode='determinate')
            progress.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            value_label = ttk.Label(emotion_frame, text="0%", width=5)
            value_label.pack(side=tk.RIGHT)
            
            self.emotion_bars[emotion] = {'bar': progress, 'label': value_label}
        
        # Emotion history
        history_frame = ttk.LabelFrame(right_frame, text="Emotion History", padding=10)
        history_frame.pack(fill=tk.BOTH, pady=5)
        
        self.history_text = tk.Text(history_frame, height=8, width=30, font=('Arial', 10))
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=scrollbar.set)
        
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Image analysis section
        image_frame = ttk.LabelFrame(right_frame, text="Image Analysis", padding=10)
        image_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(image_frame, text="Analyze Image", command=self.analyze_image).pack(fill=tk.X, pady=5)
        ttk.Button(image_frame, text="Open Image", command=self.open_image).pack(fill=tk.X, pady=5)
        
        self.image_path_label = ttk.Label(image_frame, text="No image selected", foreground='gray')
        self.image_path_label.pack(fill=tk.X)
        
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not access webcam")
            return
        
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.NORMAL)
        
        # Start video processing in separate thread
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()
        
    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.DISABLED)
        
        # Clear video feed
        self.video_label.configure(image='')
        
    def process_video(self):
        last_analysis_time = 0
        
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
            
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
                        
                        # Draw face rectangle
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Add emotion text
                        text = f"{self.current_emotion} ({self.current_confidence:.1f}%)"
                        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 255, 0), 2)
                        
                        last_analysis_time = current_time
                        
                except Exception as e:
                    print(f"Analysis error: {e}")
            
            # Convert frame for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            
            # Update video feed in main thread
            self.root.after(0, self.update_video_feed, photo)
            
            time.sleep(0.03)  # Control frame rate
            
    def update_video_feed(self, photo):
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # Keep a reference
        
    def update_emotion_display(self, emotions):
        # Update current emotion
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
        
        # Keep only last 20 entries
        lines = self.history_text.get(1.0, tk.END).split('\n')
        if len(lines) > 20:
            self.history_text.delete(1.0, tk.END)
            self.history_text.insert(tk.END, '\n'.join(lines[-20:]))
        
    def capture_photo(self):
        if self.cap and self.is_running:
            ret, frame = self.cap.read()
            if ret:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"emotion_capture_{timestamp}_{self.current_emotion}.jpg"
                cv2.imwrite(filename, frame)
                messagebox.showinfo("Photo Captured", f"Photo saved as: {filename}")
                
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            self.photo_path = file_path
            self.image_path_label.configure(text=os.path.basename(file_path))
            
    def analyze_image(self):
        if not self.photo_path:
            messagebox.showwarning("Warning", "Please select an image first")
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
                
                # Show results
                result_text = f"Analysis Results:\n\n"
                result_text += f"Dominant Emotion: {dominant_emotion}\n"
                result_text += f"Confidence: {confidence:.1f}%\n\n"
                result_text += "All Emotions:\n"
                
                for emotion, score in emotions.items():
                    result_text += f"{emotion}: {score:.1f}%\n"
                
                messagebox.showinfo("Image Analysis Results", result_text)
                
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Could not analyze image: {str(e)}")
            
    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()import tkinter as tk
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
        self.root.title("AI Emotion Detector")
        self.root.geometry("1200x700")
        self.root.configure(bg='#2c3e50')
        
        # Variables
        self.cap = None
        self.is_running = False
        self.current_emotion = "Neutral"
        self.current_confidence = 0
        self.emotion_history = []
        self.photo_path = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Camera feed
        left_frame = ttk.LabelFrame(main_frame, text="Live Camera Feed", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.video_label = ttk.Label(left_frame, background='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Camera controls
        control_frame = ttk.Frame(left_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.capture_btn = ttk.Button(control_frame, text="Capture Photo", command=self.capture_photo, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Analytics
        right_frame = ttk.LabelFrame(main_frame, text="Emotion Analytics", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0), ipadx=10)
        
        # Current emotion display
        current_emotion_frame = ttk.LabelFrame(right_frame, text="Current Emotion", padding=10)
        current_emotion_frame.pack(fill=tk.X, pady=5)
        
        self.emotion_label = ttk.Label(current_emotion_frame, text="NEUTRAL", 
                                      font=('Arial', 24, 'bold'), foreground='#3498db')
        self.emotion_label.pack(pady=10)
        
        self.confidence_label = ttk.Label(current_emotion_frame, text="Confidence: 0%", 
                                         font=('Arial', 12))
        self.confidence_label.pack()
        
        # Emotion statistics
        stats_frame = ttk.LabelFrame(right_frame, text="Emotion Statistics", padding=10)
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create progress bars for each emotion
        self.emotion_bars = {}
        emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
        
        for emotion in emotions:
            emotion_frame = ttk.Frame(stats_frame)
            emotion_frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(emotion_frame, text=emotion.title(), width=10).pack(side=tk.LEFT)
            
            progress = ttk.Progressbar(emotion_frame, orient=tk.HORIZONTAL, length=150, mode='determinate')
            progress.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            value_label = ttk.Label(emotion_frame, text="0%", width=5)
            value_label.pack(side=tk.RIGHT)
            
            self.emotion_bars[emotion] = {'bar': progress, 'label': value_label}
        
        # Emotion history
        history_frame = ttk.LabelFrame(right_frame, text="Emotion History", padding=10)
        history_frame.pack(fill=tk.BOTH, pady=5)
        
        self.history_text = tk.Text(history_frame, height=8, width=30, font=('Arial', 10))
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=scrollbar.set)
        
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Image analysis section
        image_frame = ttk.LabelFrame(right_frame, text="Image Analysis", padding=10)
        image_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(image_frame, text="Analyze Image", command=self.analyze_image).pack(fill=tk.X, pady=5)
        ttk.Button(image_frame, text="Open Image", command=self.open_image).pack(fill=tk.X, pady=5)
        
        self.image_path_label = ttk.Label(image_frame, text="No image selected", foreground='gray')
        self.image_path_label.pack(fill=tk.X)
        
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not access webcam")
            return
        
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.NORMAL)
        
        # Start video processing in separate thread
        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()
        
    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.DISABLED)
        
        # Clear video feed
        self.video_label.configure(image='')
        
    def process_video(self):
        last_analysis_time = 0
        
        while self.is_running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                break
            
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
                        
                        # Draw face rectangle
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Add emotion text
                        text = f"{self.current_emotion} ({self.current_confidence:.1f}%)"
                        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 255, 0), 2)
                        
                        last_analysis_time = current_time
                        
                except Exception as e:
                    print(f"Analysis error: {e}")
            
            # Convert frame for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)
            
            # Update video feed in main thread
            self.root.after(0, self.update_video_feed, photo)
            
            time.sleep(0.03)  # Control frame rate
            
    def update_video_feed(self, photo):
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # Keep a reference
        
    def update_emotion_display(self, emotions):
        # Update current emotion
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
        
        # Keep only last 20 entries
        lines = self.history_text.get(1.0, tk.END).split('\n')
        if len(lines) > 20:
            self.history_text.delete(1.0, tk.END)
            self.history_text.insert(tk.END, '\n'.join(lines[-20:]))
        
    def capture_photo(self):
        if self.cap and self.is_running:
            ret, frame = self.cap.read()
            if ret:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"emotion_capture_{timestamp}_{self.current_emotion}.jpg"
                cv2.imwrite(filename, frame)
                messagebox.showinfo("Photo Captured", f"Photo saved as: {filename}")
                
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            self.photo_path = file_path
            self.image_path_label.configure(text=os.path.basename(file_path))
            
    def analyze_image(self):
        if not self.photo_path:
            messagebox.showwarning("Warning", "Please select an image first")
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
                
                # Show results
                result_text = f"Analysis Results:\n\n"
                result_text += f"Dominant Emotion: {dominant_emotion}\n"
                result_text += f"Confidence: {confidence:.1f}%\n\n"
                result_text += "All Emotions:\n"
                
                for emotion, score in emotions.items():
                    result_text += f"{emotion}: {score:.1f}%\n"
                
                messagebox.showinfo("Image Analysis Results", result_text)
                
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Could not analyze image: {str(e)}")
            
    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()