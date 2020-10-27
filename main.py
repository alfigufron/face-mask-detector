from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import numpy as np
import os, cv2

path = f"{os.path.dirname(cv2.__file__)}/data/haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(path)

model = load_model('./model.h5')

video_capture = cv2.VideoCapture(0)

while True:
  ret, frame = video_capture.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  face = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minSize=(10, 10),
    flags=cv2.CASCADE_SCALE_IMAGE
  )

  face_list = []
  predicts = []

  for (x, y, w, h) in face:
    face_frame = frame[y:y+h, x:x+w]
    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
    face_frame = cv2.resize(face_frame, (224, 224))
    face_frame = img_to_array(face_frame)
    face_frame = np.expand_dims(face_frame, axis=0)
    face_frame = preprocess_input(face_frame)
    face_list.append(face_frame)

    if len(face_list) > 0:
      predicts = model.predict(face_list)
    
    for predict in predicts:
      (mask, without_mask) = predict

    label = "Menggunakan Masker" if mask > without_mask else "Tanpa Masker"
    color = (0, 255, 0) if label == "Menggunakan Masker" else (0, 0, 255)

    percent = "{:.2f}".format(max(mask, without_mask) * 100)
    label = f"{label}: {percent}%"

    cv2.putText(frame, label, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
  
  cv2.imshow('Capture', frame)
  key = cv2.waitKey(20)
  if key == 27:
    break

video_capture.release()
cv2.destroyAllWindows()
