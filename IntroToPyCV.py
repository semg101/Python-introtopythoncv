import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default');

#We are going to capture images from our webcam and process it and detect the face

#To capture the face from the webcam, we need a video captured object
#You can try any number (the number that will work in your case)
cam = cv2.VideoCapture(0);

#We are going to capture the frames one by one and detect the face, then we will show it in the window
while(True):
	#To capture the image is cam.read()
	#cam.read will return one status variable ret and the captured image img
	ret, img = cam.read();

	#The image is a coloured image but the classifier can work only with a grayscale image
	#We need to convert our coloured image to a grayscale image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);

	#We've got our grayscale image, we can detect faces from that
	#detectMultiScale(gray, 1.3, 5) will detect all the faces in the current frame and then return the coordinates of the faces
	#We add some parameters to increase the accuracy
	faces = faceDetect.detectMultiScale(gray,1.0, 2, 5, 1);

	#The variable faces is holding a large number of faces, we are going to hold each face and draw a rectangle on it
	for(x, y, w, h) in faces:
		#We need to draw a rectangle on the current image
		#The initial point of the rectangle will be (x, y) and the end point of the rectangle will be (x+w, y+h)
		#and the colour of the rectangle will be (0, 255, 0) and the tickness of the rectangle will be 2
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2);

	#We need to show the image in another window
	#cv2.imshow("Face", img) will create a window with a window name "Face" and image img
	cv2.imshow("Face", img);

	#We need to give a wait command otherwise the opencv will not work
	if(cv2.waitKey(100) == ord('q')):
		break;

#We need to release the cam because we are done
#cam.release();	

#We need to destroy all the windows
#cv2.destroyAllWindows();	