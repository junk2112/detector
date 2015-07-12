import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn import svm
import neurolab as nl
from sklearn.ensemble import RandomForestClassifier


import cPickle as pickle
from math import sqrt
from pybrain.datasets.supervised import SupervisedDataSet as SDS
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

from pandas import read_csv
import cv2
from os import listdir
from os.path import isfile, join, isdir

def split_into_rgb_channels(image):
	red = image[:,:,2]
	green = image[:,:,1]
	blue = image[:,:,0]
	return red, green, blue

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
			# gray[i, j] = 255 - gray[i, j]
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
			# gray[i, j] = 255 - gray[i, j]
	scaled = cv2.resize(gray, (shape[1]*scale, shape[0]*scale))
	# cv2.imshow("gray", scaled)
	# cv2.waitKey()
	return gray, scaled
	return gray, scaled

def nn(train_source, test_source, validation=False, v_size=0.5):

	hidden_size = 100
	epochs = 600

	# load data
	train = read_csv(train_source)
	tmp = open(train_source)
	feature_count = None
	for line in tmp:
		feature_count = len(line.split(","))
		break

	trainX = np.asarray(train[range(1, feature_count)])
	trainY = np.asarray(train[[0]]).ravel()
	# print "All Data size: " + str(len(trainX))
	testX = None
	testY = None

	if validation:
		# --- CROSS VALIDATION ---
		trainX, testX, trainY, testY = cross_validation.train_test_split(
			trainX, trainY, test_size=v_size, random_state=0)
	else:
		# --- TEST DATA ---
		test = read_csv(test_source)
		testX = np.asarray(test[range(1, feature_count)])
		testY = np.asarray(test[[0]]).ravel()

	# print testX
	# print testY
	input_size = len(trainX[0])
	target_size = 1
	print input_size
	print target_size
	# prepare dataset

	ds = SDS( input_size, target_size )
	ds.setField( 'input', trainX )
	ds.setField( 'target', [[item] for item in trainY] )

	# init and train

	net = buildNetwork( input_size, hidden_size, target_size, bias = True )
	trainer = BackpropTrainer(net, ds)

	print "training for {} epochs...".format(epochs)

	for i in range( epochs ):
		mse = trainer.train()
		rmse = sqrt(mse)
		print "training RMSE, epoch {}: {}".format(i + 1, rmse)
		
	# pickle.dump( net, open( output_model_file, 'wb' ))

def make_test(train_source, test_source, light_type=None, validation=False, v_size=0.5, estimators=85):
	train = read_csv(train_source)
	tmp = open(train_source)
	feature_count = None
	for line in tmp:
		feature_count = len(line.split(","))
		break

	trainX = np.asarray(train[range(1, feature_count)])
	trainY = np.asarray(train[[0]]).ravel()
	# print "All Data size: " + str(len(trainX))
	testX = None
	testY = None

	if validation:
		# --- CROSS VALIDATION ---
		trainX, testX, trainY, testY = cross_validation.train_test_split(
			trainX, trainY, test_size=v_size, random_state=0)
	else:
		# --- TEST DATA ---
		test = read_csv(test_source)
		testX = np.asarray(test[range(1, feature_count)])
		testY = np.asarray(test[[0]]).ravel()
	if len(testX) < 100:
		return 0
	print "Train size: " + str(len(trainX))
	print "Test size: " + str(len(testX))

	# --- KNN ---
	# clf = KNeighborsClassifier(metric='minkowski', n_neighbors=1, p=2)

	# --- SVM ---
	# clf = svm.SVC()
	# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
	# gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
	# shrinking=True, tol=0.001, verbose=False)

	# --- Random Forest ---
	clf = RandomForestClassifier(n_estimators=estimators)


	clf.fit_transform(trainX, trainY)

	true_false = 0
	true_true = 0
	false_true = 0
	false_false = 0
	true = 0
	false = 0
	for i in range(len(testY)):
		answer = clf.predict(testX[i])
		if testY[i] == True:
			true += 1
		else:
			false += 1
		# print str(answer[0]) + " " + str(testY[i])
		if answer[0] == True and testY[i] == False:
			true_false += 1
		if answer[0] == True and testY[i] == True:
			true_true += 1
		if answer[0] == False and testY[i] == False:
			false_false += 1
		if answer[0] == False and testY[i] == True:
			false_true += 1
	if validation:
		if true > 0:
			print light_type + " true_true (precision): " + str(float(true_true)/float(true))
			print light_type + " false_true: " + str(float(false_true)/float(true))
		if false > 0:
			print light_type + " true_false: " + str(float(true_false)/float(false))
			print light_type + " false_false (precision): " + str(float(false_false)/float(false))

	result = clf.score(testX, testY)
	print "Main precision for " + light_type + ": " + str(result)
	return result


def leave_one_out(light_type):
	dirs = readDirs("csv_tests")
	average = 0
	ignored = 0

	for i in range(len(dirs)):
		# print "csv_tests/" + dirs[i] + "/test_" + light_type + ".csv"
		precision = make_test("csv_tests/" + dirs[i] + "/train_" + light_type + ".csv", 
			"csv_tests/" + dirs[i] + "/test_" + light_type + ".csv", light_type, False, 0.5)
		if precision == 0:
			ignored += 1
		average += precision
	print light_type
	print "Average: " + str(average/len(dirs))
	print "Ignored: " + str(ignored)
	average /= (len(dirs) - ignored)
	print "Average with ignored: " + str(average)

make_test("csv_data/merged_green.csv", "csv_data/39.38_train.csv", "green", True, 0.3, 85)
make_test("csv_data/merged_red.csv", "csv_data/39.38_train.csv", "red", True, 0.3, 85)


# DATASET 2 -- GOOGLE GLASS
# leave_one_out - green: 0.83
# leave_one_out - red: 0.75
# 
# Train size: 729
# Test size: 313
# green true_true (precision): 0.731092436975
# green false_true: 0.268907563025
# green true_false: 0.0567010309278
# green false_false (precision): 0.943298969072
# Main precision for green: 0.862619808307
# Train size: 2803
# Test size: 1202
# red true_true (precision): 0.900630914826
# red false_true: 0.0993690851735
# red true_false: 0.167253521127
# red false_false (precision): 0.832746478873
# Main precision for red: 0.868552412646

# leave_one_out("green")
# leave_one_out("red")







