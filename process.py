
# # import the necessary packages
# import argparse
# import cv2
 
# # initialize the list of reference points and boolean indicating
# # whether cropping is being performed or not
# refPt = []
# cropping = False
 
# def click_and_crop(event, x, y, flags, param):
# 	# grab references to the global variables
# 	global refPt, cropping
 
# 	# if the left mouse button was clicked, record the starting
# 	# (x, y) coordinates and indicate that cropping is being
# 	# performed
# 	if event == cv2.EVENT_LBUTTONDOWN:
# 		refPt = [(x, y)]
# 		cropping = True
 
# 	# check to see if the left mouse button was released
# 	elif event == cv2.EVENT_LBUTTONUP:
# 		# record the ending (x, y) coordinates and indicate that
# 		# the cropping operation is finished
# 		refPt.append((x, y))
# 		cropping = False
 
# 		# draw a rectangle around the region of interest
# 		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
# 		cv2.imshow("image", image)

# # construct the argument parser and parse the arguments

# cap = cv2.VideoCapture("1.mp4")

 
# # load the image, clone it, and setup the mouse callback function
# image = cv2.imread(args["image"])
# clone = image.copy()
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", click_and_crop)
 
# # keep looping until the 'q' key is pressed
# while True:
# 	# display the image and wait for a keypress
# 	ret,  = cap.read()
# 	cv2.imshow("image", image)
# 	key = cv2.waitKey(1) & 0xFF
 
# 	# if the 'r' key is pressed, reset the cropping region
# 	if key == ord("r"):
# 		image = clone.copy()
 
# 	# if the 'c' key is pressed, break from the loop
# 	elif key == ord("c"):
# 		break
 
# # if there are two reference points, then crop the region of interest
# # from teh image and display it
# if len(refPt) == 2:
# 	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
# 	cv2.imshow("ROI", roi)
# 	cv2.waitKey(0)
 
# # close all open windows
# cv2.destroyAllWindows()

import numpy as np
import cv2
import cv2.cv as cv
import time

cap = cv2.VideoCapture("1.mp4")

boxes = []

def on_mouse(event, x, y, flags, params):
	if event == cv.CV_EVENT_LBUTTONDOWN:
		print 'Start Mouse Position: '+str(x)+', '+str(y)
		sbox = [x, y]
		boxes.append(sbox)

	elif event == cv.CV_EVENT_LBUTTONUP:
		print 'End Mouse Position: '+str(x)+', '+str(y)
		ebox = [x, y]
		boxes.append(ebox)

def wait(time):
	if cv2.waitKey(time) & 0xFF == ord('q'):
		# true if break
		return true


last_size = 0
while(True):
	ret, frame = cap.read()
	print ret

	cv2.namedWindow('frame')
	cv.SetMouseCallback('frame', on_mouse, 0)
	cv2.imshow('frame', frame)
	if wait(0):
		break
	if len(boxes) != last_size:
		# cv2.rectangle(frame, (1,1), (10, 10), (0,255,0), 2)
		crop = frame[boxes[-2][1]:boxes[-1][1], boxes[-2][0]:boxes[-1][0]]
		cv2.imshow('frame', crop)
		last_size = len(boxes)
	print boxes
	if wait(0):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

