import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

from pandas import read_csv
import cv2
from os import listdir
from os.path import isfile, join, isdir
import threading
import time

def wait(time):
	if cv2.waitKey(time) & 0xFF == ord('q'):
		# true if break
		return True

def split_into_rgb_channels(image):
	red = image[:,:,2]
	green = image[:,:,1]
	blue = image[:,:,0]
	return red, green, blue

def find_if_close(cnt1, cnt2):
	row1, row2 = cnt1.shape[0], cnt2.shape[0]
	for i in xrange(row1):
		for j in xrange(row2):
			dist = np.linalg.norm(cnt1[i] - cnt2[j])
			if abs(dist) < 30:
				return True
			elif i == row1 - 1 and j == row2 - 1:
				return False

def readDirs(source):
	return [f for f in listdir(source) if isdir(join(source, f))]

def preprocRed(pic, resize_to = (15, 30), tresh = 220, scale = 10):
	red, green, blue = split_into_rgb_channels(pic)
	gray = red - green
	gray = cv2.resize(gray, resize_to)
	gray = cv2.blur(gray, (3, 3))
	shape = gray.shape
	for i in range(shape[0]):
		for j in range(shape[1]):
			if gray[i, j] <= 0:
				gray[i, j] = 0
			if gray[i, j] > 255:
				gray[i, j] = 255
	scaled = cv2.resize(gray, (shape[1]*scale, shape[0]*scale))
	return gray, scaled

def preprocGreen(pic, resize_to = (15, 30), tresh = 0, scale = 10):
	red, green, blue = split_into_rgb_channels(pic)
	gray = 2*green - red - 2*(green - blue)
	gray = cv2.resize(gray, resize_to)
	gray = cv2.blur(gray, (3, 3))
	shape = gray.shape
	for i in range(shape[0]):
		for j in range(shape[1]):
			if gray[i, j] <= tresh:
				gray[i, j] = 0
			if gray[i, j] > 255:
				gray[i, j] = 255
	scaled = cv2.resize(gray, (shape[1]*scale, shape[0]*scale))
	return gray, scaled
	

def train(train_source_red, train_source_green):
	train_green = read_csv(train_source_green)
	train_red = read_csv(train_source_red)
	tmp = open(train_source_red)
	feature_count = None
	for line in tmp:
		feature_count = len(line.split(","))
		break

	trainX_green = np.asarray(train_green[range(1, feature_count)])
	trainY_green = np.asarray(train_green[[0]]).ravel()
	trainX_red = np.asarray(train_red[range(1, feature_count)])
	trainY_red = np.asarray(train_red[[0]]).ravel()

	clf_green = RandomForestClassifier(n_estimators=85)
	clf_red = RandomForestClassifier(n_estimators=85)
	clf_green.fit(trainX_green, trainY_green)
	clf_red.fit(trainX_red, trainY_red)
	return clf_green, clf_red

def predict(image, grad_score, clf, c_type):
	gray = None
	scaled = None
	if c_type is "red":
		gray, scaled = preprocRed(image)
	else:
		gray, scaled = preprocGreen(image)
	s = gray.shape

	features = []
	h_w = float(s[0]) / float(s[1])
	features.append(h_w)
	features.append(grad_score)

	for i in range(s[0]):
	    for j in range(s[1]):
	    	features.append(gray[i, j])
	return clf.predict(features)



def find_traffic_lights(t, clf, light_type="green"):	
	s = t.shape
	aspect = max(2, s[0] / 300)
	t = cv2.resize(t, (s[1] / aspect, s[0] / aspect))
	reserved = t.copy()

	img_gray = cv2.cvtColor(reserved, cv2.COLOR_BGR2GRAY)
	img = cv2.cvtColor(t, cv2.COLOR_RGB2HSV)
	img = cv2.blur(img, (3, 3))

	lower = None
	upper = None
	if light_type is "red":
		lower = np.array([115, 130, 150], dtype="uint8")
		upper = np.array([130, 255, 255], dtype="uint8")
	else:
		if light_type is "green":
			lower = np.array([25, 100, 150], dtype="uint8")
			upper = np.array([70, 255, 255], dtype="uint8")

	mask = cv2.inRange(img, lower, upper)
	output = cv2.bitwise_and(img, img, mask=mask)

	morphres = cv2.threshold(output, 50, 255, cv2.THRESH_BINARY)[1]
	shape = cv2.getStructuringElement(0, (3, 3))
	morphres = cv2.morphologyEx(morphres, cv2.MORPH_DILATE, shape)
	morphres = cv2.morphologyEx(morphres, cv2.MORPH_ERODE, shape)
	morphres = cv2.morphologyEx(morphres, cv2.MORPH_DILATE, shape)
	gray = cv2.cvtColor(morphres, cv2.IMREAD_GRAYSCALE)

	gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	edged = cv2.Canny(gray, 10, 100)

	contours, hier = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, 2)

	LENGTH = len(contours)
	status = np.zeros((LENGTH, 1))

	for i, cnt1 in enumerate(contours):
		x = i
		if i != LENGTH - 1:
			for j, cnt2 in enumerate(contours[i + 1:]):
				x = x + 1
				dist = find_if_close(cnt1, cnt2)
				if dist == True:
					val = min(status[i], status[x])
					status[x] = status[i] = val
				else:
					if status[x] == status[i]:
						status[x] = i + 1

	unified = []
	maximum = None
	if len(status) > 0:
		maximum = int(status.max()) + 1
	else:
		maximum = 0
	for i in xrange(maximum):
		pos = np.where(status == i)[0]
		if pos.size != 0:
			cont = np.vstack(contours[i] for i in pos)
			hull = cv2.convexHull(cont)
			unified.append(hull)

	detected = []
	for c in unified:
		x, y, w, h = cv2.boundingRect(c)
		(xc, yc), radius = cv2.minEnclosingCircle(c)
		center = (int(xc), int(yc))
		radius = int(radius)
		s = w * h
		v = radius * radius * 3.14
		detected.append([x, y, w, h, center, radius])

	sobelx = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, ksize=3)
	strel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
	vertical = cv2.morphologyEx(sobelx, cv2.MORPH_OPEN, strel)

	i = 0
	result = []
	for rect in detected:
		x = rect[0]
		y = rect[1]
		w = int(rect[2])
		h = int(rect[3])
		crop = t[y: y + h, x: x + w]
		crop_under = vertical[y + h: y + 5 * h, x - 0.5 * w: x + 1.5 * w]
		crop_under_shape = crop_under.shape
		crop_shape = crop.shape
		if crop_shape[0] > 0 and crop_shape[1] > 0 and crop_under_shape[0] > 0 and crop_under_shape[1] > 0:
			mean_under = cv2.mean(crop_under)[0]/(crop_under_shape[0] * crop_under_shape[1])
			predicted = predict(crop, mean_under, clf_green if light_type is "green" else clf_red, light_type)
			i += 1

			if predicted == True:
				result.append([light_type, x, y, w, h])
	return result



video = "video_glasses/20140704_180308_946.mp4"
clf_green, clf_red = train("csv_data/merged_green.csv", "csv_data/merged_red.csv")
cap = cv2.VideoCapture(video)
video_length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
print "Frame count: " + str(video_length)
video_fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
print "Video FPS: " + str(video_fps)
print "Video length: " + str(video_length/video_fps) + " seconds"

f_count = 0
step = 2
frame_time = 0

while (f_count < video_length):
	ret, frame = cap.read()
	if f_count % step != 0:
			f_count += 1
			continue

	start = time.time()
	rectangles = []
	green = find_traffic_lights(frame, clf_green, "green")
	red = find_traffic_lights(frame, clf_red, "red")
	rectangles = green + red
	s = frame.shape
	aspect = max(2, s[0] / 300)
	frame = cv2.resize(frame, (s[1] / aspect, s[0] / aspect))
	print len(rectangles)
	for rect in rectangles:
		light_type = rect[0]
		x = rect[1]
		y = rect[2]
		w = rect[3]
		h = rect[4]
		cv2.rectangle(frame, (x, y), (x + w, y + h), 
			(0, 255 if light_type=="green" else 0, 255 if light_type=="red" else 0), 1)
	cv2.imshow("frame", frame)
	f_count += 1
	end = time.time()
	frame_time += end - start
	if f_count > 5:
		print "Average Frame time: " + str((f_count/step)/frame_time)
	wait(0)