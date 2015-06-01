import cv2
import os

# name = "video/7.mp4"
folder_name = name.split(".")[0] + "_frames"
# print folder_name
os.system("rm -rf " + folder_name)
os.system("mkdir " + folder_name)

cap = cv2.VideoCapture(name)


count = 0
while(True):
	ret, frame = cap.read()
	cv2.imwrite(folder_name+"/img"+str(count)+".jpg", frame)
	count += 1