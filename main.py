from mtcnn import MTCNN
import cv2, os

cascPath = os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
videoCapture = cv2.VideoCapture(0)

while (True):
  ret, frame = videoCapture.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(60, 60),
    flags=cv2.CASCADE_SCALE_IMAGE)

  for (x, y, w, h) in faces:
    cv2.rectangle(
      frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
  
  cv2.imshow("Face Detector", frame)
  if cv2.waitKey(25) & 0XFF == ord('q'):
    break

videoCapture.release()
cv2.destroyAllWindow()