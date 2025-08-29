import cv2
import mediapipe as mp
import numpy as np
import face_recognition
import pickle
import time
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

# -------------------- Firebase Initialization --------------------
cred = credentials.Certificate("firebase_admin_sdk.json")  # Update with your filename
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://your-project-id-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

# -------------------- Load Known Face Encodings --------------------
with open('encodings.pickle', 'rb') as file:
    data = pickle.load(file)
known_encodings = data['encodings']
known_names = data['names']

# -------------------- Initialize MediaPipe FaceMesh --------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# -------------------- Blink Detection Function --------------------
def is_blinking(landmarks):
    left_eye = [33, 160, 158, 133, 153, 144]
    right_eye = [362, 385, 387, 263, 373, 380]

    def eye_aspect_ratio(eye_points):
        vertical1 = np.linalg.norm(np.array(landmarks[eye_points[1]]) - np.array(landmarks[eye_points[5]]))
        vertical2 = np.linalg.norm(np.array(landmarks[eye_points[2]]) - np.array(landmarks[eye_points[4]]))
        horizontal = np.linalg.norm(np.array(landmarks[eye_points[0]]) - np.array(landmarks[eye_points[3]]))
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear

    leftEAR = eye_aspect_ratio(left_eye)
    rightEAR = eye_aspect_ratio(right_eye)
    avg_EAR = (leftEAR + rightEAR) / 2.0
    return avg_EAR < 0.25

# -------------------- Initialize Video Capture --------------------
video = cv2.VideoCapture(0)
attendance_marked = set()

print("[INFO] Starting camera...")

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            if is_blinking(landmarks):
                print("[INFO] Blink Detected – Liveness Confirmed")

                face_locations = face_recognition.face_locations(small_frame)
                face_encodings = face_recognition.face_encodings(small_frame, face_locations)

                for face_encoding, face_location in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]

                        if name not in attendance_marked:
                            now = datetime.now()
                            time_str = now.strftime("%H:%M:%S")
                            date_str = now.strftime("%Y-%m-%d")

                            ref = db.reference(f'/Attendance/{date_str}/{name}')
                            ref.set({
                                'name': name,
                                'time': time_str
                            })

                            print(f"[MARKED] {name} at {time_str}")
                            attendance_marked.add(name)

                        # Draw rectangle
                        top, right, bottom, left = [v * 4 for v in face_location]
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    else:
                        print("[WARNING] Unknown face detected.")

            else:
                print("[WARNING] No blink detected – Spoof suspected.")

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

video.release()
cv2.destroyAllWindows()
