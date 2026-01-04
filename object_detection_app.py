# object_detection_app.py
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image
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
        self.root.title("AI Live Object Detection - Powered by PyTorch")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1e1e1e")

        # Variables
        self.running = False
        self.stream = None
        self.thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.confidence_threshold = tk.DoubleVar(value=0.6)

        # Load Model
        print("Loading AI Model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using: {self.device}")
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.model.to(self.device)
        if self.device.type == 'cuda':
            self.model.half()
        self.model.eval()
        self.transform = weights.transforms()

        # COCO Classes
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
        self.COLORS = np.random.uniform(50, 255, size=(len(self.COCO_CLASSES), 3))

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="Real-Time Object Detection", font=("Helvetica", 20, "bold"),
                         fg="#00ff00", bg="#1e1e1e")
        title.pack(pady=10)

        # Video Frame
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(pady=10)

        # Controls Frame
        control_frame = tk.Frame(self.root, bg="#1e1e1e")
        control_frame.pack(pady=10)

        # Start/Stop Button
        self.start_btn = tk.Button(control_frame, text="Start Detection", command=self.toggle_detection,
                                   bg="#00ff00", fg="black", font=("Helvetica", 12, "bold"), width=15, height=2)
        self.start_btn.pack(side=tk.LEFT, padx=20)

        # Confidence Slider
        tk.Label(control_frame, text="Confidence Threshold:", fg="white", bg="#1e1e1e", font=("Helvetica", 10)).pack(side=tk.LEFT, padx=10)
        slider = ttk.Scale(control_frame, from_=0.3, to=0.95, orient=tk.HORIZONTAL, variable=self.confidence_threshold)
        slider.pack(side=tk.LEFT, padx=10)
        tk.Label(control_frame, textvariable=self.confidence_threshold, fg="#00ff00", bg="#1e1e1e").pack(side=tk.LEFT)

        # Status
        self.status = tk.Label(self.root, text="Status: Ready | Press Start to begin", fg="#888", bg="#1e1e1e", font=("Helvetica", 10))
        self.status.pack(pady=10)

        # Footer
        footer = tk.Label(self.root, text="Made with PyTorch + OpenCV | © 2025", fg="#555", bg="#1e1e1e")
        footer.pack(side=tk.BOTTOM, pady=10)

    def toggle_detection(self):
        if not self.running:
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        self.running = True
        self.start_btn.config(text="Stop Detection", bg="#ff4444", fg="white")
        self.status.config(text="Status: Running... Detecting objects live!", fg="#00ff00")
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.thread = Thread(target=self.video_loop, daemon=True)
        self.thread.start()
        self.update_frame()

    def stop_detection(self):
        self.running = False
        self.start_btn.config(text="Start Detection", bg="#00ff00", fg="black")
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

    def detect_and_draw(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        if self.device.type == 'cuda':
            img_tensor = img_tensor.half()

        with torch.no_grad():
            prediction = self.model(img_tensor)[0]

        threshold = self.confidence_threshold.get()
        for i in range(len(prediction['boxes'])):
            score = prediction['scores'][i].cpu().item()
            if score > threshold:
                box = prediction['boxes'][i].cpu().numpy().astype(int)
                label = prediction['labels'][i].cpu().item()
                color = self.COLORS[label]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                text = f"{self.COCO_CLASSES[label]} {score:.2f}"
                cv2.putText(frame, text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def update_frame(self):
        if self.running:
            try:
                frame = self.frame_queue.get_nowait()
                frame = self.detect_and_draw(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                from PIL import ImageTk
                photo = ImageTk.PhotoImage(img)
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

# ================ RUN THE APP ================
if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)

    root.mainloop()
