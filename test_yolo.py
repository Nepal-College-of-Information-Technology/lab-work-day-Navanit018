import torch

# Load YOLOv5 model
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("YOLOv5 loaded successfully!")
