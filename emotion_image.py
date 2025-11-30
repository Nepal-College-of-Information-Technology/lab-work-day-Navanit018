import cv2
from deepface import DeepFace
import numpy as np

# Load your image
image_path = "person.jpg"  # Make sure this image exists
image = cv2.imread(image_path)

if image is None:
    print("Error: Could not load image. Please check the file path.")
else:
    try:
        # Analyze the image for emotions
        analysis = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        
        # Print all analysis results
        print("Analysis Results:")
        print(analysis)
        
        # Extract the first face analysis (if multiple faces, it returns a list)
        if isinstance(analysis, list):
            face_data = analysis[0]
        else:
            face_data = analysis
        
        # Get emotion results
        emotions = face_data['emotion']
        dominant_emotion = face_data['dominant_emotion']
        region = face_data['region']
        
        print(f"\nDominant Emotion: {dominant_emotion}")
        print("All Emotions:")
        for emotion, score in emotions.items():
            print(f"  {emotion}: {score:.2f}%")
        
        # Draw on the image
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Put emotion text
        text = f"{dominant_emotion}"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow("Facial Expression Recognition", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during analysis: {e}")