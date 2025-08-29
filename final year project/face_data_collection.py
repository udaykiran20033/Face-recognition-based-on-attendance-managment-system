import cv2
import os

dataset_path = 'face_data'
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
camera = cv2.VideoCapture(0)
user_id = input("Enter your name or ID: ").strip()

print("[INFO] Starting face collection. Look at the camera...")

count = 0
while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_img = gray[y:y+h, x:x+w]
        face_filename = os.path.join(dataset_path, f"{user_id}_{count}.jpg")
        cv2.imwrite(face_filename, face_img)
        count += 1

    cv2.imshow("Face Data Collection", frame)

    if count >= 100 or cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("[INFO] Face data collection completed!")
camera.release()
cv2.destroyAllWindows()
