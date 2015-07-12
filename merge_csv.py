from pandas import read_csv
import numpy as np
import os
from os import listdir
from os.path import isfile, join

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

def merge(result, csv_files):
	os.system("rm " + result)
	
	# os.system("touch " + result)
	csv = open(result, "w")

	i = 0
	for name in csv_files:
		for line in open(name):
			if i < 3:
				i += 1
				continue
			csv.write(line)    
			i += 1
	csv.close()

csv_files_green, csv_files_red = readDir("csv_data")
merge("csv_data/merged_green.csv", csv_files_green)
merge("csv_data/merged_red.csv", csv_files_red)



