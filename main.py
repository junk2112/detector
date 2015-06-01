import cv2
import numpy as np
import os
# 0, 1, 2, 3, 4,      5
# x, y, w, h, center, radius
# def contains(rect1, rect2, e):
# 	if (abs(rect1[0] - rect2[0]) <= e) and (abs(rect1[1] - rect2[1]) <= e):
# 		if ((abs(rect2[0] + rect2[2] -  rect1[0] - rect1[2]) <= e) and 
# 					(abs(rect2[1] + rect2[3] -  rect1[1] - rect1[3]) <= e)):
# 			return True

def wait(time):
	if cv2.waitKey(time) & 0xFF == ord('q'):
		# true if break
		return true

def split_into_rgb_channels(image):
	red = image[:,:,2]
	green = image[:,:,1]
	blue = image[:,:,0]
	return red, green, blue

def find_if_close(cnt1,cnt2):
	row1,row2 = cnt1.shape[0],cnt2.shape[0]
	for i in xrange(row1):
		for j in xrange(row2):
			dist = np.linalg.norm(cnt1[i]-cnt2[j])
			if abs(dist) < 30:
				return True
			elif i==row1-1 and j==row2-1:
				  return False

count = 0
cap = cv2.VideoCapture("video/6.mp4")
dir_to_save = "red6";
os.system("rm -rf " + dir_to_save)
os.system("mkdir " + dir_to_save)
while (True):

	ret, t = cap.read()
	# t = cv2.imread("./1_frames/img40.jpg")
	# t = cv2.imread("./2_frames/img11.jpg")
	# t = cv2.imread("./6_frames/img141.jpg")
	# t = cv2.imread("video/7_frames/img240.jpg")
	# t = cv2.imread("4.jpg")
	s = t.shape
	aspect = max(0,2,s[0]/300)
	# print(aspect)

	t = cv2.resize(t,(s[1]/aspect,s[0]/aspect))
	reserved = t
	# cv2.imshow("morphres", t)
	# cv2.waitKey()

	img = cv2.cvtColor(t,cv2.COLOR_RGB2HSV)

	img = cv2.blur(img,(3,3))

	# t = cv2.blur(t,(7,7))
	# red, green, blue = split_into_rgb_channels(t)
	# # cv2.imshow("red", red)
	# # cv2.imshow("green", green)
	# # cv2.imshow("blue", blue)
	# for_red = red - green - blue
	# # for_red = cv2.threshold(for_red, 200, 255, cv2.THRESH_BINARY)[1]
	# cv2.imshow("for_red", for_red)
	# if wait(0):
	# 	break 

	lower = np.array([115, 130, 150], dtype = "uint8")
	upper = np.array([130, 255, 255], dtype = "uint8")
	mask = cv2.inRange(img, lower, upper)




	# cv2.imshow("morphres", mask)
	# cv2.waitKey()

	output = cv2.bitwise_and(img, img, mask = mask)


	# cv2.imshow("morphres", output)
	# cv2.waitKey()

	morphres = cv2.threshold(output, 50, 255, cv2.THRESH_BINARY )[1]

	# cv2.imshow("morphres", morphres)
	# cv2.waitKey()

	shape = cv2.getStructuringElement(0, (3,3))
	morphres = cv2.morphologyEx(morphres, cv2.MORPH_DILATE,shape)
	# morphres = cv2.morphologyEx(morphres, cv2.MORPH_DILATE,shape)

	# cv2.imshow("morphres", morphres)
	# cv2.waitKey()

	morphres = cv2.morphologyEx(morphres, cv2.MORPH_ERODE,shape)
	# morphres = cv2.morphologyEx(morphres, cv2.MORPH_ERODE,shape)
	# morphres = cv2.morphologyEx(morphres, cv2.MORPH_ERODE,shape)

	# cv2.imshow("morphres", morphres)
	# cv2.waitKey()

	morphres = cv2.morphologyEx(morphres, cv2.MORPH_DILATE,shape)

	# cv2.imshow("morphres", morphres)
	# cv2.waitKey()

	r = cv2.cvtColor(morphres,cv2.IMREAD_GRAYSCALE)
	# cv2.imshow("qwe",r)
	# cv2.waitKey()



	cv2.imwrite("qwe.jpg",r)
	file = cv2.imread("qwe.jpg",cv2.IMREAD_GRAYSCALE)
	gray = cv2.bilateralFilter(file, 11, 17, 17)
	edged = cv2.Canny(gray, 10, 100)

	# cnt , _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
	contours,hier = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,2)

	# print cnt

	LENGTH = len(contours)
	status = np.zeros((LENGTH,1))

	for i,cnt1 in enumerate(contours):
	    x = i    
	    if i != LENGTH-1:
	        for j,cnt2 in enumerate(contours[i+1:]):
	            x = x+1
	            dist = find_if_close(cnt1,cnt2)
	            if dist == True:
	                val = min(status[i],status[x])
	                status[x] = status[i] = val
	            else:
	                if status[x]==status[i]:
	                    status[x] = i+1

	unified = []
	maximum = None
	if len(status) > 0:
		maximum = int(status.max())+1
	else:
		maximum = 0
	for i in xrange(maximum):
	    pos = np.where(status==i)[0]
	    if pos.size != 0:
	        cont = np.vstack(contours[i] for i in pos)
	        hull = cv2.convexHull(cont)
	        unified.append(hull)

	candidates = []
	for c in unified:
		x,y,w,h = cv2.boundingRect(c)
		(xc,yc),radius = cv2.minEnclosingCircle(c)
		center = (int(xc),int(yc))
		radius = int(radius)
		s = w*h
		v = radius*radius*3.14
		print(s/v)
		if s/v > 0.7:
			candidates.append([x,y,w,w,center,radius])
			# cv2.rectangle(t,(x,y),(x+w,y+w),(0,255,0),1)
			# cv2.circle(t,center,radius,(0,255,0),1)

	# print len(candidates)


	filtered = []
	e = 50
	for i in range(len(candidates)):
		for j in range(len(candidates)):
			if i >= j:
				continue
			if abs(candidates[i][0] - candidates[j][0]) <= e and  abs(candidates[i][1] - candidates[j][1]) <= e:
				if candidates[i][2] > candidates[j][2] and candidates[i] not in filtered:
					filtered.append(candidates[i])
				else:
					filtered.append(candidates[j])
			else:
				if candidates[i] not in filtered:
					filtered.append(candidates[i])
				if candidates[j] not in filtered:
					filtered.append(candidates[j])
	# print len(filtered)

	widths = sorted([item[2] for item in filtered])
	# print widths
	median_width = None
	if len(widths) > 0:
		median_width = widths[len(widths)/2]
	else:
		median_width = 0

	detected = []
	for rect in filtered:
		# if rect[2] >= median_width:
		detected.append(rect)
	
	for rect in detected:
		x = rect[0] - int(rect[2]/2.5)
		y = rect[1] - int(rect[3])
		w = int(rect[2] * 2)
		h = int(rect[3] * 3.5)
		crop = t[y : y + h, x : x + w]
		cv2.imwrite(dir_to_save + "/light" + str(count) + ".jpg", crop)
		cv2.rectangle(reserved, (x, y), (x + w, y + h), (0,255,0), 1)
		count += 1
		# cv2.imshow("crop", crop)

	print len(detected)

	# cv2.imshow("asd", reserved)
	# if wait(0):
	# 	break
