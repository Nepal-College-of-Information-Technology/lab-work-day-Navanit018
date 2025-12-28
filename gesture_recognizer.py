import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

class GestureRecognizer:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, landmarks):
        landmarks = np.array(landmarks).reshape(1, -1)  # Reshape for model input
        prediction = self.model.predict(landmarks)
        return np.argmax(prediction)  # Return class index with highest probability

    def load_gesture_data(self, file_path):
        data = pd.read_csv(file_path)
        return data  # You can process the data further if needed