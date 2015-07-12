import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import threading


def wait(time):
	if cv2.waitKey(time) & 0xFF == ord('q'):
		# true if break
		return true


def split_into_rgb_channels(image):
	red = image[:, :, 2]
	green = image[:, :, 1]
	blue = image[:, :, 0]
	return red, green, blue


def readVideos(source):
	source_files = [f for f in listdir(source) if isfile(
		join(source, f))]
	full_path = []
	name = []
	for source_name in source_files:
		if source_name[0] is not ".":
			source_path = source + "/" + source_name
			full_path.append(source_path)
			name.append(source_name)
	return name, full_path


def find_if_close(cnt1, cnt2):
	row1, row2 = cnt1.shape[0], cnt2.shape[0]
	for i in xrange(row1):
		for j in xrange(row2):
			dist = np.linalg.norm(cnt1[i] - cnt2[j])
			if abs(dist) < 30:
				return True
			elif i == row1 - 1 and j == row2 - 1:
				return False


def detect(t, f_count, debug, csv_file, light_type="green", video_name="test"):
	dir_to_save = "lights/" + video_name[:5] + "/"
	s = t.shape
	aspect = max(2, s[0] / 300)
	# print(aspect)

	t = cv2.resize(t, (s[1] / aspect, s[0] / aspect))
	reserved = t.copy()
	# cv2.imshow("morphres", t)
	# cv2.waitKey()

	img_gray = cv2.cvtColor(reserved, cv2.COLOR_BGR2GRAY)
	# cv2.imshow("img_gray", img_gray)
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

	# if debug:
	# 	cv2.imshow("bitwise_and_mask", output)

	morphres = cv2.threshold(output, 50, 255, cv2.THRESH_BINARY)[1]
	shape = cv2.getStructuringElement(0, (3, 3))
	morphres = cv2.morphologyEx(morphres, cv2.MORPH_DILATE, shape)
	# morphres = cv2.morphologyEx(morphres, cv2.MORPH_DILATE,shape)
	morphres = cv2.morphologyEx(morphres, cv2.MORPH_ERODE, shape)
	# if light_type == "green":
	# 	morphres = cv2.morphologyEx(morphres, cv2.MORPH_ERODE,shape)
	# morphres = cv2.morphologyEx(morphres, cv2.MORPH_ERODE,shape)
	morphres = cv2.morphologyEx(morphres, cv2.MORPH_DILATE, shape)
	gray = cv2.cvtColor(morphres, cv2.IMREAD_GRAYSCALE)
	# if light_type == "red" and debug:
	# 	cv2.imshow("filtered " + light_type, gray)

	gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	edged = cv2.Canny(gray, 10, 100)

	# cnt , _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
	contours, hier = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, 2)

	# print cnt

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

	if debug:
		for c in unified:
			x, y, w, h = cv2.boundingRect(c)
			(xc, yc), radius = cv2.minEnclosingCircle(c)
			# cv2.rectangle(reserved,(x,y),(x+w,y+h),(255,255,0),1)

	candidates = []
	for c in unified:
		x, y, w, h = cv2.boundingRect(c)
		(xc, yc), radius = cv2.minEnclosingCircle(c)
		center = (int(xc), int(yc))
		radius = int(radius)
		s = w * h
		v = radius * radius * 3.14
		# print(s/v)
		# if s / v > 0.9:
		candidates.append([x, y, w, h, center, radius])
			# cv2.rectangle(t,(x,y),(x+w,y+w),(0,255,0),1)
			# cv2.circle(reserved,center,radius,(0,255,0),1)

	# print len(candidates)

	filtered = []
	filtered = candidates

	widths = sorted([item[2] for item in filtered])
	# print widths
	median_width = None
	if len(widths) > 0:
		median_width = widths[int(len(widths) * 0.8)]
	else:
		median_width = 10

	detected = []
	detected = filtered
	# for rect in filtered:
	# 	if rect[2] <= median_width * 3:
	# 		detected.append(rect)

	sobelx = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, ksize=3)
	strel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
	vertical = cv2.morphologyEx(sobelx, cv2.MORPH_OPEN, strel)
	if debug:
		for c in unified:
			x, y, w, h = cv2.boundingRect(c)
			(xc, yc), radius = cv2.minEnclosingCircle(c)
			cv2.rectangle(vertical, (x, y), (x + w, y + h), (255, 255, 0), 1)
			cv2.rectangle(vertical, (x - int(0.5 * w), y + h),
						  (x + int(1.5 * w), y + 5 * h), (255, 255, 0), 1)
		# cv2.imshow("vertical", vertical)
	i = 0
	for rect in detected:
		x = rect[0]  # - int(rect[2] / 2.5)
		y = rect[1]  # - int(rect[3] * 1.5)
		w = int(rect[2])  # * 2)
		h = int(rect[3])  # * 3.5)
		crop = t[y: y + h, x: x + w]
		crop_under = vertical[y + h: y + 5 * h, x - 0.5 * w: x + 1.5 * w]
		crop_under_shape = crop_under.shape
		crop_shape = crop.shape
		if crop_shape[0] > 0 and crop_shape[1] > 0 and crop_under_shape[0] > 0 and crop_under_shape[1] > 0:
			mean_under = cv2.mean(
				crop_under)[0] / (crop_under_shape[0] * crop_under_shape[1])
			# print "Mean under: " + str(mean_under)
			if not debug and float(h) >= float(w):
				file_name = light_type + "_frame" + str(f_count) + "_" + str(i)
				# print dir_to_save + file_name + ".jpg"
				cv2.imwrite(dir_to_save + file_name + ".jpg", crop)
				csv_file.write(file_name + "," + str(mean_under) + "\n")
			if float(h) >= float(w):
				i += 1
				cv2.rectangle(reserved, (x, y), (x + w, y + h),
							  (0, 255 if light_type is "green" else 0, 0 if light_type is "green" else 255), 1)
	print "Detected " + light_type + ": " + str(i)
	if debug:
		cv2.imshow(light_type, reserved)
		# cv2.waitKey()


one_video = False
debug = False


names = []
sources = []

if one_video:
	name = "20140704_180308_946.mp4"
	# name = "2014-07-15 11.21.49.mp4"
	# name = "20140309_130324_133.mp4"
	names = [name]
	sources = ["video_glasses/" + name]
else:
	names, sources = readVideos("video_glasses")


for i in range(len(sources)):
	if i < 46:
		continue
	if not debug:
		os.system("rm -rf " + "lights/" + names[i][-9:-4])
		os.system("mkdir " + "lights/" + names[i][-9:-4])

	cap = cv2.VideoCapture(sources[i])
	video_length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	video_fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
	f_count = 0
	dir_to_save = "lights/" + names[i][-9:-4]
	csv_file = None
	if not debug:
		csv_file = open(dir_to_save + '/pre_markup.csv', 'w')

	step = max(int(video_length / (video_fps * 7)), 10)
	while (f_count < video_length):
		# print step
		ret, frame = cap.read()
		if f_count % step != 0:
			f_count += 1
			continue
		print "Video " + names[i][-9:-4] + "; Number: " + str(i) + " of " + str(len(sources)) + "; FPS: " + str(video_fps) + "; Frame: " + str(f_count) + " of " + str(video_length)
		e1 = threading.Event()
		e2 = threading.Event()

		# init threads
		if not debug:
			t1 = threading.Thread(
				target=detect, args=(frame, f_count, debug, csv_file, "red", names[i][-9:-4]))
			t2 = threading.Thread(
				target=detect, args=(frame, f_count, debug, csv_file, "green", names[i][-9:-4]))
			t1.start()
			t2.start()
			t1.join()
			t2.join()
		else:
			detect(frame, f_count, debug, csv_file, "red", names[i][-9:-4])
			# detect(frame, f_count, debug, csv_file, "green", names[i][-9:-4])

		f_count += 1
		if debug:
			if wait(0):
				break
	csv_file.close()
