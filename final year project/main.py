import cv2
import numpy as np
from datetime import datetime
from firebase_utils import mark_attendance

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model.yml")

# Load label map
label_map = {}
with open("labels.txt", "r") as f:
    for line in f:
        id, name = line.strip().split(":")
        label_map[int(id)] = name

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting real-time recognition. Press 'q' to exit.")

# Track cooldown for each person
last_marked_time = {}
COOLDOWN = 30  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    current_time = datetime.now()

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        id, confidence = recognizer.predict(face_img)

        if confidence < 70:
            name = label_map[id]

            # Cooldown logic
            if name not in last_marked_time or (current_time - last_marked_time[name]).total_seconds() > COOLDOWN:
                print(f"[DEBUG] Marking attendance for: {name}")
                mark_attendance(name)
                last_marked_time[name] = current_time

            cv2.putText(frame, f"{name} ({int(confidence)}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("Real-Time Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Attendance session ended.")
