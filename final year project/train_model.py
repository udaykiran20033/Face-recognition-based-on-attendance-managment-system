import cv2
import os
import numpy as np

dataset_path = 'face_data'
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces = []
labels = []
label_map = {}
current_id = 0

print("[INFO] Starting training process...")

# Scan all images in dataset
for filename in os.listdir(dataset_path):
    if filename.endswith(".jpg"):
        path = os.path.join(dataset_path, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"[WARNING] Unable to read image: {path}")
            continue

        name = filename.split("_")[0]
        if name not in label_map:
            label_map[name] = current_id
            current_id += 1

        id = label_map[name]

        faces.append(img)
        labels.append(id)

print(f"[INFO] Total faces collected: {len(faces)}")
print(f"[INFO] Label map: {label_map}")

if len(faces) == 0:
    print("[ERROR] No face images found for training. Exiting.")
    exit()

# Train and save
print("[INFO] Training the model...")
face_recognizer.train(faces, np.array(labels))
face_recognizer.write("trained_model.yml")

with open("labels.txt", "w") as f:
    for name, id in label_map.items():
        f.write(f"{id}:{name}\n")

print("[SUCCESS] Model saved as 'trained_model.yml' and labels saved as 'labels.txt'")
