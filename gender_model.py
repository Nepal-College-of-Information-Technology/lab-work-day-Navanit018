# ================================================
# NEURA VISION X - WITH GENDER & AGE DETECTION
# Final Year Project 2025 - Ultimate Version
# Author: [Your Name] | Roll No: [Your Roll]
# ================================================

from ultralytics import YOLO
from PIL import Image, ImageTk
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread
import time
import torch
import numpy as np

# ================== GENDER & AGE MODEL ==================
class GenderAgeDetector:
    def __init__(self):
        # Pretrained lightweight model (FairFace or SSR-Net style)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model.fc = torch.nn.Linear(512, 2 + 8)  # 2 gender + 8 age groups
        # Download this model once: https://huggingface.co/serengil/deepface
        # Or use deepface (easiest)
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            self.use_deepface = True
            print("[INFO] DeepFace loaded for Gender & Age")
        except:
            self.use_deepface = False
            print("[WARNING] DeepFace not found. Install with: pip install deepface")

    def predict(self, face_img):
        if not self.use_deepface:
            return "Unknown", "??"
        try:
            result = self.deepface.analyze(face_img, actions=['gender', 'age'], enforce_detection=False, silent=True)[0]
            gender = result['dominant_gender']
            age = str(result['age'])
            return gender, age
        except:
            return "Unknown", "??"

gender_age_detector = GenderAgeDetector()

# ================== MAIN APP ==================
class NeuraVisionX:
    def __init__(self, root):
        self.root = root
        self.root.title("NEURA VISION X - Gender & Age Detection")
        self.root.configure(bg="#0a001a")
        self.root.state('zoomed')

        self.running = False
        self.cap = None
        self.latest_frame = None
        self.frame_ready = False
        self.person_count = 0
        self.confidence = tk.DoubleVar(value=0.50)
        self.last_time = time.time()

        print("[INFO] Loading YOLOv8-n...")
        self.model = YOLO("yolov8n.pt")
        print("[INFO] System Ready!")

        self.build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.safe_exit)

    def build_ui(self):
        main = tk.Frame(self.root, bg="#0a001a")
        main.pack(fill="both", expand=True, padx=50, pady=30)

        tk.Label(main, text="NEURA VISION X", font=("Orbitron", 60, "bold"),
                 fg="#00ffff", bg="#0a001a").pack(pady=(20, 10))
        tk.Label(main, text="Real-Time Object + Gender + Age Detection", 
                 font=("Segoe UI", 20), fg="#88ccff", bg="#0a001a").pack(pady=(0, 40))

        video_frame = tk.Frame(main, bg="#1a0033", bd=10, relief="flat", highlightbackground="#00ffff", highlightthickness=4)
        video_frame.pack(pady=20)
        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack()

        control = tk.Frame(main, bg="#1a0033", bd=10, relief="flat", highlightbackground="#ff00ff", highlightthickness=3)
        control.pack(pady=40, padx=200, fill="x")

        tk.Label(control, text="AI CONTROL PANEL", font=("Orbitron", 22, "bold"),
                 fg="#ff00ff", bg="#1a0033").pack(pady=(20, 20))

        btns = tk.Frame(control, bg="#1a0033")
        btns.pack(pady=20)
        self.start_btn = tk.Button(btns, text="START AI", command=self.toggle,
                                   font=("Orbitron", 18, "bold"), bg="#00ff99", fg="black",
                                   width=18, height=2, relief="raised", bd=8)
        self.start_btn.pack(side="left", padx=40)
        tk.Button(btns, text="EXIT", command=self.safe_exit,
                  font=("Orbitron", 18, "bold"), bg="#ff3366", fg="white",
                  width=18, height=2, relief="raised", bd=8).pack(side="left", padx=40)

        stats = tk.Frame(control, bg="#1a0033")
        stats.pack(pady=30)
        tk.Label(stats, text="Confidence", font=("Segoe UI", 14), fg="#00ffff", bg="#1a0033").grid(row=0, column=0, columnspan=2)
        ttk.Scale(stats, from_=0.1, to=0.9, variable=self.confidence, length=500).grid(row=1, column=0, columnspan=2, pady=10)
        tk.Label(stats, textvariable=self.confidence, font=("Arial", 16), fg="#00ffcc", bg="#1a0033").grid(row=0, column=2, padx=20)

        self.fps_label = tk.Label(stats, text="FPS: --", font=("Orbitron", 30, "bold"), fg="#00ff00", bg="#1a0033")
        self.fps_label.grid(row=2, column=0, pady=20, padx=50)
        self.person_label = tk.Label(stats, text="PERSONS: 0", font=("Orbitron", 30, "bold"), fg="#ff00ff", bg="#1a0033")
        self.person_label.grid(row=2, column=1, pady=20, padx=50)

        self.status = tk.Label(main, text="Status: Ready | Gender & Age Detection Active", 
                               font=("Segoe UI", 16), fg="#cccccc", bg="#0a001a")
        self.status.pack(pady=30)

        tk.Label(main, text="© 2025 NEURA VISION X | YOLOv8 + DeepFace | Final Project", 
                 font=("Segoe UI", 11), fg="#555577", bg="#0a001a").pack(side="bottom", pady=20)

    def toggle(self):
        if not self.running:
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        self.running = True
        self.start_btn.config(text="STOP AI", bg="#ff3366", fg="white")
        self.status.config(text="LIVE • Gender & Age Detection Active", fg="#00ff99")
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Thread(target=self.capture_loop, daemon=True).start()
        self.update_display()

    def stop_detection(self):
        self.running = False
        self.start_btn.config(text="START AI", bg="#00ff99", fg="black")
        self.status.config(text="Status: Stopped", fg="#ffaa00")
        if self.cap:
            self.cap.release()

    def capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame.copy()
                self.frame_ready = True

    def process_frame(self, frame):
        results = self.model(frame, conf=self.confidence.get(), verbose=False)[0]
        annotated = frame.copy()

        person_count = 0
        for box in results.boxes:
            if int(box.cls) == 0:  # Person class
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    gender, age = gender_age_detector.predict(face)
                    label = f"{gender} {age}"
                    color = (0, 255, 255) if gender == "Male" else (255, 105, 180)
                else:
                    label = "Person"
                    color = (0, 255, 0)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 3)

        # Update stats
        self.person_count = person_count
        now = time.time()
        self.fps = 1.0 / (now - self.last_time + 1e-5)
        self.last_time = now

        # Overlay
        cv2.putText(annotated, f"FPS: {self.fps:.1f}", (30, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,255), 5)
        cv2.putText(annotated, f"PERSONS: {person_count}", (30, 120), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255,0,255), 4)

        return annotated

    def update_display(self):
        if not self.running:
            return
        if self.frame_ready:
            frame = self.latest_frame.copy()
            self.frame_ready = False
            frame = cv2.resize(frame, (1000, 620))
            result = self.process_frame(frame)

            img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            photo = ImageTk.PhotoImage(img)
            self.video_label.config(image=photo)
            self.video_label.image = photo

            self.fps_label.config(text=f"FPS: {self.fps:.1f}")
            self.person_label.config(text=f"PERSONS: {self.person_count}")

        self.root.after(16, self.update_display)

    def safe_exit(self):
        if messagebox.askyesno("Exit", "Shut down NEURA VISION X?"):
            self.running = False
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.root.destroy()

# ====================== LAUNCH ======================
if __name__ == "__main__":
    root = tk.Tk()
    app = NeuraVisionX(root)
    root.mainloop()