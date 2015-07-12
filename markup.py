import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join, isdir
import sys

def readDir(source):
	source_files = [f for f in listdir(source) if isfile(
		join(source, f))]
	images = []
	names = []
	for source_name in source_files:
		if ".csv" not in source_name:
			source_path = source + "/" + source_name
			img = cv2.imread(source_path)
			images.append(img)
			names.append(source_name)
	return images, names

def readDirs(source):
	return [f for f in listdir(source) if isdir(join(source, f))]

def wait(time = 0):
	key = cv2.waitKey(time)
	if key & 0xFF == ord('q'):
		return "quit"
	if key & 0xFF == ord('1'):
		return "true"
	if key & 0xFF == ord('2'):
		return "false"
	return "skip"

def split_into_rgb_channels(image):
	red = image[:,:,2]
	green = image[:,:,1]
	blue = image[:,:,0]
	return red, green, blue

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

one_video = False
lights = "lights/"

if one_video:
	name = "2_190"
	dirs = [name]
else:
	dirs = readDirs(lights)
	print dirs

count = 0
for name in dirs:
	source = lights + name
	markup_dir = "csv_data/"
	pre_markup = source + "/pre_markup.csv"
	pre_markup = open(pre_markup)
	pre_features = {}
	for line in pre_markup:
		tmp = line.split(",")
		tmp[1] = tmp[1][:-2]
		pre_features[tmp[0]] = tmp[1]
	pre_markup.close()
	csv_file_green = open(markup_dir + name + '_train_' + 'green' + '.csv', 'w')
	csv_file_red = open(markup_dir + name + '_train_' + 'red' + '.csv', 'w')
	images, names = readDir(source)
	for i in range(len(images)):
		gray = None
		scaled = None
		win_name = None
		csv_file = None
		if "red" in names[i]:
			gray, scaled = preprocRed(images[i])	
			win_name = "red"
			csv_file = csv_file_red
		else:
			gray, scaled = preprocGreen(images[i])
			win_name = "green"
			csv_file = csv_file_green
		shape = gray.shape
		for_show = cv2.resize(images[i], (150, 300))
		im_shape = images[i].shape
		h_w = float(im_shape[0]) / float(im_shape[1])
		cv2.imshow(win_name, for_show)
		cv2.imshow(win_name + " processed", scaled)
		value = wait()
		if value is "quit":
			print "quit"
			sys.exit()
			break
		if value is "skip":
			print "skip"
			continue
		csv_file.write(str(value))
		csv_file.write("," + str(h_w))
		csv_file.write("," + str(float(pre_features[names[i].split(".")[0]])))
		for k in range(shape[0]):
			for j in range(shape[1]):
				csv_file.write("," + str(gray[k, j]))
		csv_file.write("\n")
		print "Video " + name + " " + str(count) + " of " + str(len(dirs) - 1)
		print "Frame " + str(i) + " of " + str(len(images)) + " : " + str(value)
	count += 1

	csv_file_green.close()
	csv_file_red.close()