import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision import models

###########################################
# 1. LOAD GENDER CLASSIFICATION MODEL
###########################################

class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Male/Female

    def forward(self, x):
        return self.model(x)

# Load trained model (.pth file you downloaded)
gender_model_path = "gender_model_simple.pth"

gender_model = GenderClassifier()
gender_model.load_state_dict(torch.load(gender_model_path, map_location="cpu"))
gender_model.eval()

# Preprocessing for gender model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

###########################################
# 2. LOAD YOLOv5 FOR PERSON DETECTION
###########################################
yolo = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

###########################################
# 3. LIVE CAMERA DETECTION
###########################################
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 detection
    results = yolo(frame)
    detections = results.xyxy[0]

    for det in detections:
        x1, y1, x2, y2, conf, cls = det

        if int(cls) == 0:  # Class 0 = person
            # Draw person box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)

            # Crop person for gender model
            person_crop = frame[int(y1):int(y2), int(x1):int(x2)]

            if person_crop.size > 0:
                img = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                img_tensor = transform(img).unsqueeze(0)

                # Predict gender
                with torch.no_grad():
                    output = gender_model(img_tensor)
                    pred = torch.argmax(output, dim=1).item()

                gender = "Male" if pred == 0 else "Female"

                cv2.putText(frame, gender, (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Object + Gender Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
