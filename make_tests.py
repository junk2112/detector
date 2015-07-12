from pandas import read_csv
import numpy as np
import os
from os import listdir
from os.path import isfile, join, isdir

def readDir(source):
	source_files = [f for f in listdir(source) if isfile(
		join(source, f))]
	names_green = []
	names_red = []
	for source_name in source_files:
		if "merged" in source_name:
			continue
		source_path = source + "/" + source_name
		if "green" in source_name:
			names_green.append(source_path)
		else:
			names_red.append(source_path)
	return names_green, names_red

csv_files_green, csv_files_red = readDir("csv_data")
tests_dir = "csv_tests/"

def make_tests(light_type, csv_files, tests_dir):
	for i in range(len(csv_files)):
		# os.system("rm -rf " + tests_dir + str(i))
		os.system("mkdir " + tests_dir + str(i))
		train = tests_dir + str(i) + "/train_" + light_type + ".csv"
		test = tests_dir + str(i) + "/test_" + light_type + ".csv"
		csv_train = open(train, "w")
		csv_test = open(test, "w")

		for j in range(len(csv_files)):
			for line in open(csv_files[j]):
				if i == j:
					csv_test.write(line)
				else:
					csv_train.write(line)

		csv_train.close()
		csv_test.close


os.system("rm -rf " + tests_dir)
os.system("mkdir " + tests_dir)

make_tests("green", csv_files_green, tests_dir)
make_tests("red", csv_files_red, tests_dir)

