import cv2
from fer import FER

# Initialize the detector
detector = FER()

# Start webcam
cap = cv2.VideoCapture(0)  # 0 for default camera

print("Starting real-time emotion detection...")
print("Press 'q' to quit")

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Detect emotions (this can be slow, so we'll process every other frame)
    try:
        emotions = detector.detect_emotions(frame)
        
        # Draw results on frame
        for face in emotions:
            x, y, w, h = face["box"]
            emotions_dict = face["emotions"]
            
            # Get dominant emotion
            dominant_emotion = max(emotions_dict, key=emotions_dict.get)
            confidence = emotions_dict[dominant_emotion]
            
            # Only show if confidence is reasonable
            if confidence > 0.2:
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Put emotion text
                text = f"{dominant_emotion}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Print to console
                print(f"Detected: {dominant_emotion} ({confidence:.2f})")
    
    except Exception as e:
        print(f"Error in detection: {e}")
    
    # Display the frame
    cv2.imshow('Real-Time Emotion Detection', frame)
    
    # Break loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Webcam released. Goodbye!")