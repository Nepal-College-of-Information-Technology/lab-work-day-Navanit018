import cv2
import mediapipe as mp
from gesture_recognizer import GestureRecognizer
from app_launcher import AppLauncher

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize other components
gesture_recognizer = GestureRecognizer('models/gesture_recognition_model.h5')
app_launcher = AppLauncher()

# Capture Video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks for gesture recognition
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Append x, y, z coordinates

            gesture_id = gesture_recognizer.predict(landmarks)  # Gesture prediction

            # Launch application based on gesture
            if gesture_id == 0:  # Example: gesture for launching Notepad
                app_launcher.open_application("notepad.exe")
            elif gesture_id == 1:  # Example: gesture for launching Browser
                app_launcher.open_application("chrome.exe")

            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture-Based App Launcher", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()