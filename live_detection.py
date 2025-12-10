# live_object_detection_final.py   ← Save as this name

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from PIL import Image
import cv2
import time
import numpy as np
from threading import Thread
import queue

# ==================== COCO CLASSES ====================
COCO_CLASSES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
                'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
                'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
                'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
                'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

COLORS = np.random.uniform(0, 255, size=(len(COCO_CLASSES), 3))

# ==================== MODEL ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights)
model.to(device)
if device.type == 'cuda':
    model.half()                    # 2–3× faster on GPU
model.eval()

# ==================== THREADED CAMERA ====================
class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)  # CAP_DSHOW = no lag on Windows
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.q = queue.Queue(maxsize=2)

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                continue
            if self.q.full():
                self.q.get()           # drop old frame
            self.q.put((grabbed, frame))

    def read(self):
        return self.q.get()

    def stop(self):
        self.stopped = True
        self.stream.release()

# ==================== DETECTION ====================
def detect(frame_bgr):
    # BGR → RGB → PIL → Tensor
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(rgb)
    img_tensor = weights.transforms()(img_pil).unsqueeze(0).to(device)

    if device.type == 'cuda':
        img_tensor = img_tensor.half()

    with torch.no_grad():
        prediction = model(img_tensor)[0]
    return prediction

# ==================== DRAW ====================
def draw_boxes(frame, pred, threshold=0.6):
    boxes = pred['boxes'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()

    for i in range(len(boxes)):
        if scores[i] < threshold:
            continue
        x1, y1, x2, y2 = map(int, boxes[i])
        color = tuple(map(int, COLORS[labels[i]]))
        text = f"{COCO_CLASSES[labels[i]]} {scores[i]:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# ==================== MAIN ====================
def main():
    print("Live Object Detection – Press 'q' to quit")
    stream = WebcamStream(src=0).start()
    time.sleep(1)

    prev_time = time.time()

    while True:
        grabbed, frame = stream.read()
        if not grabbed:
            continue

        # Resize for speed (optional)
        frame = cv2.resize(frame, (640, 480))

        # Detect
        prediction = detect(frame)

        # Draw
        result = draw_boxes(frame.copy(), prediction, threshold=0.6)

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(result, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Show
        cv2.imshow("PyTorch Live Object Detection", result)

        if cv2.waitKey(1) == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()
    print("Finished.")

if __name__ == "__main__":
    main()