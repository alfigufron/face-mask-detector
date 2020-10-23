import cv2, imutils, os

pathCascade = f"{os.path.dirname(cv2.__file__)}/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(pathCascade)

capture = cv2.VideoCapture(0)

if not (capture.isOpened()):
  print("Not Opened Video Device")

while True:
  ret, frame = capture.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  face = faceCascade.detectMultiScale(gray)

  for (x, y, w, h) in face:
    cv2.rectangle(
      frame, (x, y), (x+w, y+h), (255, 255, 255), 1.4
    )

  frame = imutils.resize(frame, width=640, height=480)
  cv2.imshow('Video Capture', frame)

  key = cv2.waitKey(20)
  if key == 27:
    break

capture.release()
cv2.destroyAllWindows()