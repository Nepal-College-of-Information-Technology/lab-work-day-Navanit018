# detect.py
import torch
import torchvision
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

from utils import draw_boxes

# Load object detector
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()

# Load gender model
gender_model_path = "gender_model_simple.pth"
gender_model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(224*224*3, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 2)
)
gender_model.load_state_dict(torch.load(gender_model_path, map_location="cpu"))
gender_model.eval()

gender_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def detect_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transforms.ToTensor()(img)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes = outputs["boxes"].numpy()
    labels = outputs["labels"].numpy()
    scores = outputs["scores"].numpy()

    # only keep detections > 0.6
    keep = scores >= 0.6
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

    # -------------------
    # Gender prediction
    # -------------------
    img_cv = cv2.imread(image_path)
    genders = []

    for i, box in enumerate(boxes):
        if labels[i] == 1:  # person class
            x1, y1, x2, y2 = map(int, box)
            face = img_cv[y1:y2, x1:x2]

            if face.size == 0:
                genders.append("unknown")
                continue

            face_tensor = gender_tf(face).unsqueeze(0)
            with torch.no_grad():
                pred = gender_model(face_tensor)
            gender = "male" if pred.argmax() == 0 else "female"
            genders.append(gender)
        else:
            genders.append(None)

    output_img = draw_boxes(img_cv.copy(), boxes, labels, scores, genders)
    cv2.imwrite("output.jpg", output_img)
    print("Saved output.jpg")

detect_image("sample.jpg")
