import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join


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


names, pathes = readVideos("video_glasses")

time = 0

for video in pathes:
	cap = cv2.VideoCapture(video)
	video_length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	video_fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
	print video + " length: " + str(video_length/video_fps)
	time += video_length/video_fps

print "Total time: " + str(time/60) + " minutes"
