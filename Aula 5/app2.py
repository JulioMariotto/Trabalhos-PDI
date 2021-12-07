import numpy as np;
import cv2 as cv;

face_classifier = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_classifier = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

image = cv.imread('image.jpg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)

for(x,y,w,h) in faces:
	cv.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = image[y:y+h, x:x+w]
	eyes = eye_classifier.detectMultiScale(roi_gray)
	for(ex,ey,ew,eh) in eyes:
		cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

cv.imshow('imagem', image)
cv.waitKey(0)
cv.destroyAllWindows()