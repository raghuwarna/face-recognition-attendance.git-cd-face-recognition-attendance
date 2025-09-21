# face-recognition-attendance.git-cd-face-recognition-attendance
The Face Recognition Attendance System is an automated system designed to record attendance by identifying individuals through their facial features. Unlike traditional attendance methods using manual entry or ID cards, this system uses computer vision and machine learning techniques to detect and recognize faces in real time.
#_face_dataset.py
import cv2
import os

# Path to save dataset
dataset_path = "dataset"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Load OpenCV's built-in face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Input user ID and Name
face_id = input("\nEnter user ID (numeric) and press <Enter>: ")
name = input("Enter user name and press <Enter>: ")

print("\n[INFO] Initializing face capture. Look at the camera ...")

# Start webcam
cam = cv2.VideoCapture(0)
count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f"{dataset_path}/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:  # ESC key to exit
        break
    elif count >= 50:  # Capture 50 face samples
        break

print("\n[INFO] Exiting and cleaning up ...")
cam.release()
cv2.destroyAllWindows()



#_face_training.py
import cv2
import numpy as np
from PIL import Image
import os

dataset_path = "dataset"
trainer_path = "trainer"
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    for imagePath in image_paths:
        PIL_img = Image.open(imagePath).convert('L')  # grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)
    return face_samples, ids

print("\n[INFO] Training faces. It will take a few seconds ...")
faces, ids = get_images_and_labels(dataset_path)
recognizer.train(faces, np.array(ids))

recognizer.write(f"{trainer_path}/trainer.yml")
print(f"\n[INFO] {len(np.unique(ids))} faces trained. Model saved at {trainer_path}/trainer.yml")


#_face_recognition.py
import cv2
import pandas as pd
import os
from datetime import datetime

trainer_path = "trainer"
attendance_path = "Attendance"
if not os.path.exists(attendance_path):
    os.makedirs(attendance_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(f"{trainer_path}/trainer.yml")
cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = {}  # Map user IDs to names

# Build name dictionary from dataset filenames
for file in os.listdir("dataset"):
    parts = file.split(".")
    if len(parts) >= 3:
        names[int(parts[1])] = parts[0]  # maps ID -> "User"

cam = cv2.VideoCapture(0)

print("\n[INFO] Starting face recognition. Press 'q' to quit ...")

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 60:
            name = names.get(id, "Unknown")
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = f"{round(100 - confidence)}%"

        cv2.putText(img, str(name), (x+5, y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence_text), (x+5, y+h-5), font, 1, (255,255,0), 1)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        if name != "Unknown":
            # Record attendance
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
            file_path = os.path.join(attendance_path, "Attendance.csv")

            if not os.path.exists(file_path):
                df = pd.DataFrame(columns=["Id", "Name", "DateTime"])
                df.to_csv(file_path, index=False)

            df = pd.read_csv(file_path)
            if not ((df["Id"] == id) & (df["DateTime"].str.contains(now.strftime("%Y-%m-%d")))).any():
                df.loc[len(df)] = [id, name, dt_string]
                df.to_csv(file_path, index=False)

    cv2.imshow('camera', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

print("\n[INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
