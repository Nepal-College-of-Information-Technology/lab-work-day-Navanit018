import cv2
import mediapipe as mp
import pyautogui

import numpy as np

# Screen size
screen_width, screen_height = pyautogui.size()

# Camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Smoothing
smoothening = 7
prev_x, prev_y = 0, 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Index finger tip
            x1, y1 = lm_list[8][1], lm_list[8][2]
            # Thumb tip
            x2, y2 = lm_list[4][1], lm_list[4][2]

            # Convert coordinates
            screen_x = np.interp(x1, (0, 640), (0, screen_width))
            screen_y = np.interp(y1, (0, 480), (0, screen_height))

            # Smooth movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Click gesture (thumb + index close)
            distance = np.hypot(x2 - x1, y2 - y1)
            if distance < 30:
                pyautogui.click()
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
