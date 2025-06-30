import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import numpy as np

# Load OpenCV's pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # type: ignore


# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Create a figure for displaying results
fig, ax = plt.subplots()
plt.ion()  # Turn on interactive mode

running = True  # Flag to control loop

def handle_close(event):
    """Handles window close event."""
    global running
    running = False  # Stops the loop when the figure is closed

# Attach close event handler
fig.canvas.mpl_connect('close_event', handle_close)

while running:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        break  # Stop if frame capture fails

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.2, minNeighbors=6, minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]  # Extract face region

        try:
            # Perform emotion analysis on detected face
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']  # Extract dominant emotion
        except Exception as e:
            emotion = "Unknown"  # Default if emotion detection fails

        # Draw rectangle around face and label with emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

    # Convert frame to RGB for Matplotlib display
    rgb_display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Update Matplotlib display
    ax.clear()
    ax.imshow(rgb_display_frame)
    ax.axis('off')
    plt.pause(0.001)

    # If figure is closed manually, exit loop
    if not plt.get_fignums():
        running = False

# Release resources
cap.release()
plt.close()
