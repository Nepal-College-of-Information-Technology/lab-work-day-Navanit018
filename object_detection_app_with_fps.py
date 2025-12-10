# object_detection_app_with_fps.py

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image, ImageTk
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from threading import Thread
import queue
import time

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Live Object Detection Pro")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e1e1e")

        self.running = False
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.confidence_threshold = tk.DoubleVar(value=0.6)

        # FPS variables
        self.fps = 0
        self.prev_time = 0

        # Load Model
        print("Loading PyTorch Model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.model.to(self.device)
        if self.device.type == 'cuda':
            self.model.half()
        self.model.eval()
        self.transform = weights.transforms()

        # COCO Classes & Colors
        self.COCO_CLASSES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                              'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                              'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                              'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                              'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                              'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        self.COLORS = np.random.uniform(80, 255, size=(len(self.COCO_CLASSES), 3))

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        title = tk.Label(self.root, text="Real-Time Object Detection Pro", font=("Helvetica", 22, "bold"),
                         fg="#00ff41", bg="#1e1e1e")
        title.pack(pady=15)

        self.video_label = tk.Label(self.root, bg="black", width=640, height=480)
        self.video_label.pack(pady=10)

        control_frame = tk.Frame(self.root, bg="#1e1e1e")
        control_frame.pack(pady=15)

        self.start_btn = tk.Button(control_frame, text="Start Detection", command=self.toggle_detection,
                                   bg="#00ff41", fg="black", font=("Helvetica", 14, "bold"), width=18, height=2)
        self.start_btn.pack(side=tk.LEFT, padx=25)

        tk.Label(control_frame, text="Confidence:", fg="white", bg="#1e1e1e", font=("Helvetica", 12)).pack(side=tk.LEFT, padx=10)
        slider = ttk.Scale(control_frame, from_=0.3, to=0.95, orient=tk.HORIZONTAL, variable=self.confidence_threshold, length=200)
        slider.pack(side=tk.LEFT, padx=10)
        tk.Label(control_frame, textvariable=self.confidence_threshold, fg="#00ff41", bg="#1e1e1e", font=("Helvetica", 12)).pack(side=tk.LEFT)

        self.status = tk.Label(self.root, text="Status: Ready", fg="#888888", bg="#1e1e1e", font=("Helvetica", 12))
        self.status.pack(pady=10)

        footer = tk.Label(self.root, text="PyTorch • OpenCV • Real-time AI Detection", fg="#555555", bg="#1e1e1e", font=("Helvetica", 10))
        footer.pack(side=tk.BOTTOM, pady=10)

    def toggle_detection(self):
        if not self.running:
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        self.running = True
        self.start_btn.config(text="Stop Detection", bg="#ff4444", fg="white")
        self.status.config(text="Status: Running • Detecting objects in real-time!", fg="#00ff41")
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.prev_time = time.time()
        Thread(target=self.video_loop, daemon=True).start()
        self.update_frame()

    def stop_detection(self):
        self.running = False
        self.start_btn.config(text="Start Detection", bg="#00ff41", fg="black")
        self.status.config(text="Status: Stopped", fg="#ff4444")
        if self.cap:
            self.cap.release()

    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                self.frame_queue.put(frame)

    def calculate_fps(self):
        current_time = time.time()
        self.fps = 1.0 / (current_time - self.prev_time + 1e-6)
        self.prev_time = current_time
        return self.fps

    def detect_and_draw(self, frame):
        # FPS Calculation
        fps = self.calculate_fps()

        # Detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        if self.device.type == 'cuda':
            img_tensor = img_tensor.half()

        with torch.no_grad():
            prediction = self.model(img_tensor)[0]

        threshold = self.confidence_threshold.get()

        # Draw boxes
        for i in range(len(prediction['boxes'])):
            score = prediction['scores'][i].item()
            if score > threshold:
                box = prediction['boxes'][i].cpu().numpy().astype(int)
                label_id = prediction['labels'][i].item()
                color = tuple(map(int, self.COLORS[label_id]))
                label = self.COCO_CLASSES[label_id]

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                text = f"{label} {score:.2f}"
                cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw FPS (clean background box)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)   # black shadow
        cv2.putText(frame, fps_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)   # green text

        return frame

    def update_frame(self):
        if self.running:
            try:
                frame = self.frame_queue.get_nowait()
                frame = cv2.resize(frame, (640, 480))
                result_frame = self.detect_and_draw(frame)
                img = Image.fromarray(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
                photo = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=photo)
                self.video_label.image = photo
            except:
                pass
            self.root.after(10, self.update_frame)

    def on_closing(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

# =============== RUN APP ===============
if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()