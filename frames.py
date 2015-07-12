import cv2
import os

name = "video_night/20140608_225701_011.mp4"
folder_name = "_frames"
# print folder_name
os.system("rm -rf " + folder_name)
os.system("mkdir " + folder_name)

cap = cv2.VideoCapture(name)


count = 0
step = 5
while(True):
	ret, frame = cap.read()
	if count % step != 0:
		count += 1
		continue
	if count > 100:
		break
	cv2.imwrite(folder_name+"/img"+str(count)+".jpg", frame)
	count += 1